"""Plain PyTorch LSTM baseline -- built BEFORE any foundation model.

Two jobs: give the project a working end-to-end pipeline early, and give RQ1 the benchmark
that the hybrid must beat. Deliberately simple; sophistication here would muddy the
comparison rather than strengthen it.

Trained with pinball (quantile) loss at p10/p50/p90 so it is directly comparable to the
foundation models on the same metric. Logged to MLflow with the same param/metric names
Phase 4 uses, so runs sit side by side in one table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

from forecasting.base import DEFAULT_QUANTILES, ForecastResult

logger = logging.getLogger(__name__)


@dataclass
class LSTMConfig:
    window: int = 60
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 30
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    seed: int = 42
    val_fraction: float = 0.2
    # Stop when validation loss has not improved for this many epochs. Financial series are
    # low signal-to-noise, so an unregularized LSTM overfits fast.
    patience: int = 5


def pinball_loss(
    pred: torch.Tensor, target: torch.Tensor, quantiles: tuple[float, ...]
) -> torch.Tensor:
    """Mean pinball loss across quantiles. Also RQ1's forecast-quality metric.

    For quantile q the loss is max(q*e, (q-1)*e) where e = y - yhat: under-prediction is
    penalized q, over-prediction 1-q. That asymmetry is what makes the model learn a genuine
    quantile rather than a conditional mean.
    """
    target = target.unsqueeze(-1) if target.dim() == pred.dim() - 1 else target
    errors = target - pred
    q = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device)
    return torch.maximum(q * errors, (q - 1.0) * errors).mean()


class QuantileLSTM(nn.Module):
    """LSTM encoder with one output head per quantile.

    Quantiles are parameterized as a base plus CUMULATIVE non-negative increments
    (softplus), so p10 <= p50 <= p90 holds by construction rather than being left to the
    loss. Crossed quantiles would make the CVaR objective in Phase 5a incoherent -- the
    "tail" would not be the tail -- so this is enforced structurally.
    """

    def __init__(self, n_features: int, config: LSTMConfig) -> None:
        super().__init__()
        self.n_quantiles = len(config.quantiles)

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.base = nn.Linear(config.hidden_size, 1)
        self.increments = nn.Linear(config.hidden_size, max(self.n_quantiles - 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        last = hidden[-1]

        base = self.base(last)                                   # (batch, 1) -- lowest quantile
        if self.n_quantiles == 1:
            return base

        steps = nn.functional.softplus(self.increments(last))    # strictly positive
        return torch.cat([base, base + torch.cumsum(steps, dim=1)], dim=1)


@dataclass
class _Scaler:
    """Standardization fitted on TRAIN ONLY -- fitting on the full series would leak the
    test distribution's mean and variance into training."""

    mean: np.ndarray = field(default_factory=lambda: np.zeros(1))
    std: np.ndarray = field(default_factory=lambda: np.ones(1))

    def fit(self, x: np.ndarray) -> "_Scaler":
        self.mean = x.mean(axis=0)
        self.std = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def build_sequences(
    features: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    window: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Sliding windows per symbol.

    Windows never straddle a symbol boundary -- a sequence mixing AAPL's history with MSFT's
    would be meaningless, and grouping is the only thing preventing it.
    """
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    stamps: list[pd.Timestamp] = []

    for _, group in features.groupby("symbol", sort=False):
        group = group.sort_values("timestamp")
        values = group[feature_cols].to_numpy(dtype=np.float32)
        target = group[target_col].to_numpy(dtype=np.float32)
        times = pd.to_datetime(group["timestamp"]).to_numpy()

        for end in range(window, len(group)):
            if np.isnan(target[end]):
                continue
            sequences.append(values[end - window : end])
            targets.append(target[end])
            stamps.append(times[end])

    if not sequences:
        return np.empty((0, window, len(feature_cols)), dtype=np.float32), np.empty(0), pd.DatetimeIndex([])

    return np.stack(sequences), np.array(targets, dtype=np.float32), pd.DatetimeIndex(stamps)


class BaselineLSTMForecaster:
    """Forecaster implementation wrapping QuantileLSTM."""

    name = "baseline_lstm"

    def __init__(self, config: LSTMConfig | None = None) -> None:
        self.config = config or LSTMConfig()
        self.version = "untrained"
        self.model: QuantileLSTM | None = None
        self.scaler = _Scaler()
        self.feature_cols: list[str] = []

    def fit(
        self,
        features: pd.DataFrame,
        *,
        horizon: int,
        target_col: str = "target_return",
        log_to_mlflow: bool = True,
        **kwargs: object,
    ) -> None:
        """Train with an MLflow run recording window/hidden/lr/horizon and val MAE/RMSE/pinball."""
        from features.feature_store import feature_columns

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        if target_col not in features.columns:
            raise ValueError(
                f"{target_col} not in features. Call features.feature_store.add_targets(horizon=...) first."
            )

        self.feature_cols = [c for c in feature_columns(features) if c in features.columns]
        if not self.feature_cols:
            raise ValueError("no usable feature columns")

        X, y, _ = build_sequences(features, self.feature_cols, target_col, self.config.window)
        if len(X) < 10:
            raise ValueError(f"only {len(X)} sequences available; need at least 10 to train")

        # CHRONOLOGICAL split, never random: a shuffled split would let the model train on
        # rows dated after its validation rows, which is look-ahead by another name.
        split = int(len(X) * (1 - self.config.val_fraction))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        flat = X_train.reshape(-1, X_train.shape[-1])
        self.scaler.fit(flat)
        X_train = self.scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

        self.model = QuantileLSTM(len(self.feature_cols), self.config)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        train_x = torch.tensor(np.nan_to_num(X_train), dtype=torch.float32)
        train_y = torch.tensor(y_train, dtype=torch.float32)
        val_x = torch.tensor(np.nan_to_num(X_val), dtype=torch.float32)
        val_y = torch.tensor(y_val, dtype=torch.float32)

        best_val, best_state, stagnant = float("inf"), None, 0
        history: list[dict[str, float]] = []

        for epoch in range(self.config.epochs):
            self.model.train()
            permutation = torch.randperm(len(train_x))
            epoch_loss = 0.0

            for start in range(0, len(train_x), self.config.batch_size):
                idx = permutation[start : start + self.config.batch_size]
                optimizer.zero_grad()
                loss = pinball_loss(self.model(train_x[idx]), train_y[idx], self.config.quantiles)
                loss.backward()
                # Financial series produce occasional extreme returns; without clipping a
                # single outlier batch can blow up the LSTM's recurrent weights.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.detach().item() * len(idx)

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(val_x) if len(val_x) else None
                val_loss = float(pinball_loss(val_pred, val_y, self.config.quantiles)) if val_pred is not None else float("nan")

            history.append({"epoch": epoch, "train_loss": epoch_loss / max(len(train_x), 1), "val_pinball": val_loss})

            if val_loss < best_val - 1e-9:
                best_val, stagnant = val_loss, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                stagnant += 1
                if stagnant >= self.config.patience:
                    logger.info("early stopping at epoch %d (best val pinball %.6f)", epoch, best_val)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        metrics = self._validation_metrics(val_x, val_y)
        self.version = f"lstm-w{self.config.window}-h{horizon}"

        if log_to_mlflow:
            self._log_run(horizon, metrics, history)

        logger.info("baseline LSTM trained: %s", metrics)

    def _validation_metrics(self, val_x: torch.Tensor, val_y: torch.Tensor) -> dict[str, float]:
        if self.model is None or len(val_x) == 0:
            return {}
        self.model.eval()
        with torch.no_grad():
            pred = self.model(val_x).numpy()

        median_idx = int(np.argmin(np.abs(np.array(self.config.quantiles) - 0.5)))
        point = pred[:, median_idx]
        actual = val_y.numpy()

        return {
            "val_mae": float(np.mean(np.abs(actual - point))),
            "val_rmse": float(np.sqrt(np.mean((actual - point) ** 2))),
            "val_pinball": float(
                pinball_loss(torch.tensor(pred), torch.tensor(actual), self.config.quantiles)
            ),
        }

    def _log_run(self, horizon: int, metrics: dict[str, float], history: list[dict[str, float]]) -> None:
        """Log to MLflow. Never fatal -- a tracking outage must not lose a trained model."""
        try:
            import mlflow

            mlflow.set_tracking_uri("sqlite:///artifacts/mlflow.db")
            mlflow.set_experiment("rq1_forecast_quality")

            with mlflow.start_run(run_name=f"baseline_lstm_h{horizon}"):
                mlflow.log_params(
                    {
                        "model": self.name, "window": self.config.window,
                        "hidden_size": self.config.hidden_size, "num_layers": self.config.num_layers,
                        "lr": self.config.lr, "horizon": horizon, "seed": self.config.seed,
                        "n_features": len(self.feature_cols),
                    }
                )
                mlflow.log_metrics(metrics)
                for row in history:
                    mlflow.log_metric("epoch_val_pinball", row["val_pinball"], step=int(row["epoch"]))
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow logging failed (model is still trained): %s", exc)

    def predict_quantiles(
        self,
        features: pd.DataFrame,
        *,
        horizon: int,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        target_col: str = "target_return",
        **kwargs: object,
    ) -> ForecastResult:
        if self.model is None:
            raise RuntimeError("call fit() before predict_quantiles()")

        working = features.copy()
        if target_col not in working.columns:
            working[target_col] = 0.0      # unused at inference; keeps build_sequences uniform

        X, _, stamps = build_sequences(working, self.feature_cols, target_col, self.config.window)
        if len(X) == 0:
            raise ValueError(
                f"not enough history to build a {self.config.window}-step window"
            )

        X = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        self.model.eval()
        with torch.no_grad():
            values = self.model(torch.tensor(np.nan_to_num(X), dtype=torch.float32)).numpy()

        symbol = str(working["symbol"].iloc[0]) if "symbol" in working.columns else "unknown"
        return ForecastResult(
            symbol=symbol, horizon=horizon, quantiles=self.config.quantiles,
            values=values, timestamps=stamps,
            model_name=self.name, model_version=self.version,
        )
