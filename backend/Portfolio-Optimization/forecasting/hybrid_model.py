"""The hybrid TimesFM/Chronos + LSTM-MLP forecasting engine.

This is what the TAF commits to: "develop the hybrid TimesFM + LSTM/MLP forecasting
engine". ONE fusion model, not two parallel options that get compared and discarded.

    final_forecast = base_forecast + residual_head(technical, sentiment, base_forecast, asset_class)

METHODOLOGY RATIONALE (belongs in the dissertation's methodology chapter)
------------------------------------------------------------------------
A time-series foundation model is pretrained on raw univariate series. It captures
seasonality, trend and level shifts extremely well, but it has no channel for
instrument-specific exogenous information -- it cannot know that RSI is at 80 or that
FinBERT scored today's headlines at -0.7. Residual correction is the standard way to combine
a strong general forecaster with feature-rich covariates it cannot exploit natively: the
foundation model supplies the prior, and a small supervised head learns the systematic part
of its error as a function of the covariates.

Two properties make this preferable to fine-tuning the base model on covariates directly:
  1. The base model stays frozen (or LoRA-adapted), so the hybrid can never do worse than
     the base by more than the head's capacity -- and with zero-init it starts exactly equal.
  2. The decomposition is interpretable. `residual` is a first-class quantity, so Phase 7 can
     report how much of the hybrid's advantage came from covariates rather than the base --
     directly answering RQ1.

On this platform it is also the *only* covariate path: TimesFM's XReg needs jax[cuda], which
has no Windows wheels. See timesfm_adapter.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from forecasting.base import DEFAULT_QUANTILES, ForecastResult, Forecaster, enforce_non_crossing
from forecasting.residual_head import ASSET_CLASS_ORDER, ResidualHead, ResidualHeadConfig, asset_class_index

logger = logging.getLogger(__name__)


@dataclass
class HybridConfig:
    base_model: str = "timesfm"
    head: ResidualHeadConfig | None = None
    freeze_base: bool = True
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    window: int = 60
    log_to_mlflow: bool = True
    # Walk-forward controls, used only for single-row (foundation) bases -- see
    # _walk_forward_base. min_context is the shortest prefix worth forecasting from;
    # walk_forward_points caps how many model calls each symbol costs.
    min_context: int = 64
    walk_forward_points: int = 200
    # Cuts sent to the base model per forward pass. Only used when the base exposes
    # predict_quantiles_batch; the LSTM returns every in-sample row from one call and
    # never reaches the walk.
    base_batch_size: int = 32


class HybridForecaster:
    """Frozen/LoRA foundation model + trained residual head, as one Forecaster."""

    name = "hybrid"

    def __init__(self, base: Forecaster, config: HybridConfig | None = None) -> None:
        self.base = base
        self.config = config or HybridConfig()
        self.head_config = self.config.head or ResidualHeadConfig(quantiles=self.config.quantiles)
        self.head: ResidualHead | None = None
        self.feature_cols: list[str] = []
        self.version = "untrained"
        self.horizon: int | None = None
        self.n_asset_classes = len(ASSET_CLASS_ORDER)

    def fit(
        self,
        features: pd.DataFrame,
        *,
        horizon: int,
        target_col: str = "target_return",
        log_to_mlflow: bool | None = None,
        **kwargs: object,
    ) -> None:
        """Train the head on the base model's residuals.

        Base forecasts are generated WALK-FORWARD so the head never trains on a base
        prediction that saw its own target. Training the head on in-sample base forecasts
        would teach it to correct an error pattern that does not exist out of sample, and
        RQ1 would report an improvement that evaporates in the backtest.
        """
        from features.feature_store import feature_columns
        from forecasting.baseline_lstm import pinball_loss

        torch.manual_seed(self.head_config.seed)
        np.random.seed(self.head_config.seed)

        if target_col not in features.columns:
            raise ValueError(f"{target_col} missing; call add_targets(horizon=...) first")

        self.feature_cols = [c for c in feature_columns(features) if c in features.columns]

        # Fit the base FIRST. For a zero-shot foundation adapter this is a documented no-op,
        # but for a trainable base (the LSTM) it is required -- otherwise predict_quantiles
        # raises and the head has nothing to learn a residual from. Calling it unconditionally
        # is what lets HybridForecaster accept any Forecaster rather than only frozen ones.
        try:
            self.base.fit(features, horizon=horizon, log_to_mlflow=False)
        except TypeError:
            self.base.fit(features, horizon=horizon)      # adapters without the kwarg

        base_forecasts = self._walk_forward_base(features, horizon=horizon)
        if base_forecasts.empty:
            raise RuntimeError(
                f"base model {self.base.name!r} produced no forecasts, so the residual head "
                "has nothing to learn from. Check that the feature table covers more rows "
                "than the base model's context window."
            )

        merged = features.merge(base_forecasts, on=["symbol", "timestamp"], how="inner")
        merged = merged.dropna(subset=[target_col])
        if merged.empty:
            joined = features.merge(base_forecasts, on=["symbol", "timestamp"], how="inner")
            raise RuntimeError(
                f"no overlap between base forecasts and targets: "
                f"{len(base_forecasts)} base forecast row(s) joined to {len(joined)} feature "
                f"row(s), of which 0 had a non-null {target_col}. A base that returns one row "
                f"stamped at the last input timestamp always lands on a NaN target, because "
                f"the final {horizon} rows of each symbol have no realised future yet."
            )

        X, base_matrix, asset_idx, y, _ = self._cut_sequences(features, base_forecasts, target_col,
                                                              require_target=True)
        if len(X) < 10:
            raise ValueError(f"only {len(X)} sequences; need at least 10")

        # THE TARGET IS THE RESIDUAL, not the return: what the base model got wrong.
        residual_target = y - base_matrix[:, self._median_col()]

        self.head = ResidualHead(len(self.feature_cols), len(ASSET_CLASS_ORDER), self.head_config)
        optimizer = torch.optim.Adam(self.head.parameters(), lr=self.head_config.lr)

        x_t = torch.tensor(np.nan_to_num(X), dtype=torch.float32)
        base_t = torch.tensor(np.nan_to_num(base_matrix), dtype=torch.float32)
        asset_t = torch.tensor(asset_idx, dtype=torch.long)
        y_t = torch.tensor(residual_target, dtype=torch.float32)

        for epoch in range(self.head_config.epochs):
            self.head.train()
            permutation = torch.randperm(len(x_t))
            for start in range(0, len(x_t), self.head_config.batch_size):
                idx = permutation[start : start + self.head_config.batch_size]
                optimizer.zero_grad()
                residual = self.head(x_t[idx], base_t[idx], asset_t[idx])
                loss = pinball_loss(base_t[idx] + residual, y_t[idx] + base_t[idx, self._median_col()],
                                    self.config.quantiles)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
                optimizer.step()

        self.version = f"hybrid-{self.base.name}-h{horizon}"
        self.horizon = horizon
        should_log = self.config.log_to_mlflow if log_to_mlflow is None else log_to_mlflow
        if should_log:
            self._log_run(horizon)

        logger.info("hybrid head trained on %d sequences over base=%s", len(X), self.base.name)

    def _median_col(self) -> int:
        return int(np.argmin(np.abs(np.asarray(self.config.quantiles) - 0.5)))

    def _walk_forward_base(self, features: pd.DataFrame, *, horizon: int) -> pd.DataFrame:
        """Base-model forecasts generated without seeing their own targets.

        The foundation adapters return a SINGLE row per call, stamped at the last input
        timestamp. Under add_targets(horizon=h) that row's target is always NaN -- the last
        h rows of every symbol have no realised future yet -- so calling predict_quantiles
        once per symbol produced exactly the rows that cannot be trained on. That is why the
        hybrid died with "no overlap between base forecasts and targets" on every foundation
        base while passing with the LSTM, which returns many in-sample rows instead.

        So walk the cut point forward: forecast from group[:t+1] and stamp the result at t.
        That forecast predicts the return realised over the following `horizon` steps, which
        is exactly target_return at t, and the base still never sees its own target.

        A multi-row adapter needs none of this -- one call already yields a forecast per
        timestamp -- and re-calling it per cut point would be quadratic for no gain, so it
        keeps the single call.
        """
        qcols = [f"base_p{int(q * 100)}" for q in self.config.quantiles]
        rename = {f"p{int(q * 100)}": f"base_p{int(q * 100)}" for q in self.config.quantiles}
        frames = []

        for symbol, group in features.groupby("symbol", sort=False):
            group = group.sort_values("timestamp").reset_index(drop=True)

            try:
                probe = self.base.predict_quantiles(group, horizon=horizon)
            except Exception as exc:  # noqa: BLE001 - one symbol must not kill training
                logger.warning("base forecast failed for %s: %s", symbol, exc)
                continue

            probe_frame = probe.to_frame().rename(columns=rename)

            if len(probe_frame) > 1:
                frames.append(probe_frame[["symbol", "timestamp", *qcols]])
                continue

            # Single-row adapter: walk the cut point forward.
            # Stop `horizon` rows short of the end -- beyond that the target does not exist.
            last_cut = len(group) - horizon
            cuts = list(range(self.config.min_context, last_cut))
            if not cuts:
                logger.warning(
                    "%s: %d rows is too short for a %d-step horizon after a %d-row minimum "
                    "context; no walk-forward points",
                    symbol, len(group), horizon, self.config.min_context,
                )
                continue

            # Cap the number of model calls. A foundation model costs one forward pass per
            # cut, so an uncapped walk over a multi-year series is hours of GPU time for a
            # head that converges on far fewer points.
            if len(cuts) > self.config.walk_forward_points:
                stride = len(cuts) // self.config.walk_forward_points
                cuts = cuts[::stride][: self.config.walk_forward_points]

            # One call per cut is a full pass over a 200M+ parameter model, and there are
            # hundreds of cuts per symbol on every fold of a backtest -- the hybrid was
            # taking hours and looked hung. Both foundation adapters have always accepted a
            # LIST of series, so the cuts go through in batches. Each series is normalised
            # against its own statistics, so this changes speed and nothing else.
            batch_fn = getattr(self.base, "predict_quantiles_batch", None)
            batch = max(1, self.config.base_batch_size) if batch_fn else 1

            rows = []
            for start in range(0, len(cuts), batch):
                chunk = cuts[start : start + batch]
                windows = [group.iloc[: cut + 1] for cut in chunk]
                try:
                    if batch_fn:
                        results = batch_fn(windows, horizon=horizon)
                    else:
                        results = [
                            self.base.predict_quantiles(window, horizon=horizon)
                            for window in windows
                        ]
                except Exception as exc:  # noqa: BLE001
                    logger.debug("base forecast failed for %s near cut %d: %s",
                                 symbol, chunk[0], exc)
                    continue

                for cut, result in zip(chunk, results):
                    row = result.to_frame().rename(columns=rename).iloc[-1]
                    rows.append(
                        {
                            "symbol": symbol,
                            # Stamp at the CUT, not at whatever the adapter labelled its own
                            # last input row, so the join lands on the row whose target this
                            # forecast is actually predicting.
                            "timestamp": group.loc[cut, "timestamp"],
                            **{c: float(row[c]) for c in qcols},
                        }
                    )

                if start and start % (batch * 20) == 0:
                    logger.info("  %s: %d/%d walk-forward points", symbol, start, len(cuts))

            if rows:
                frames.append(pd.DataFrame(rows))
            else:
                logger.warning("%s: every walk-forward forecast failed", symbol)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _cut_sequences(
        self, features: pd.DataFrame, base_forecasts: pd.DataFrame, target_col: str, *, require_target: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """One training/inference example per base forecast: the `window` CONSECUTIVE
        feature rows before the cut (from the full history), the base quantiles at the cut,
        the asset class, and the target at the cut.

        Windows are cut from the full feature table, not from the rows that happen to carry
        a base forecast. A foundation base is walked forward with a stride, so its forecast
        rows are not consecutive days; windowing over those rows would train the LSTM on
        sequences with gaps of several sessions while serving reads consecutive sessions --
        the same weights reading two different kinds of input.

        Convention (shared with build_sequences and predict_latest): the window is rows
        [cut - window, cut), the base forecast and target belong to the cut row itself.
        """
        quantile_cols = [f"base_p{int(q * 100)}" for q in self.config.quantiles]
        xs, bases, assets, ys, stamps = [], [], [], [], []
        forecasts = (base_forecasts.drop_duplicates(["symbol", "timestamp"], keep="last")
                     .set_index(["symbol", "timestamp"]))
        for symbol, group in features.groupby("symbol", sort=False):
            group = group.sort_values("timestamp").reset_index(drop=True)
            values = group[self.feature_cols].to_numpy(dtype=np.float32)
            targets = group[target_col].to_numpy(dtype=np.float32) if target_col in group else                 np.full(len(group), np.nan, dtype=np.float32)
            asset = asset_class_index(group["asset_class"].iloc[0]) if "asset_class" in group else 0
            for pos, stamp in enumerate(group["timestamp"]):
                if pos < self.config.window or (symbol, stamp) not in forecasts.index:
                    continue
                if require_target and np.isnan(targets[pos]):
                    continue
                xs.append(values[pos - self.config.window : pos])
                bases.append(forecasts.loc[(symbol, stamp), quantile_cols].to_numpy(dtype=np.float32))
                assets.append(asset)
                ys.append(targets[pos])
                stamps.append(stamp)
        if not xs:
            empty = np.empty((0, len(quantile_cols)), dtype=np.float32)
            return (np.empty((0, self.config.window, len(self.feature_cols)), dtype=np.float32), empty,
                    np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), pd.DatetimeIndex([]))
        return (np.stack(xs), np.stack(bases), np.array(assets, dtype=np.int64),
                np.array(ys, dtype=np.float32), pd.DatetimeIndex(stamps))

    def predict_quantiles(
        self,
        features: pd.DataFrame,
        *,
        horizon: int,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        **kwargs: object,
    ) -> ForecastResult:
        if self.head is None:
            raise RuntimeError("call fit() before predict_quantiles()")

        decomposed = self.decompose(features, horizon=horizon)
        quantile_cols = [f"final_p{int(q * 100)}" for q in self.config.quantiles]

        return ForecastResult(
            symbol=str(features["symbol"].iloc[0]) if "symbol" in features.columns else "unknown",
            horizon=horizon,
            quantiles=self.config.quantiles,
            values=enforce_non_crossing(decomposed[quantile_cols].to_numpy()),
            timestamps=pd.DatetimeIndex(decomposed["timestamp"]),
            model_name=self.name,
            model_version=self.version,
        )

    def decompose(self, features: pd.DataFrame, *, horizon: int) -> pd.DataFrame:
        """Base forecast, residual correction and final side by side -- the RQ1 ablation.

        Exposing the decomposition is what lets Phase 7 answer "how much of the hybrid's
        advantage came from the covariates?" rather than just "the hybrid is better".
        """
        if self.head is None:
            raise RuntimeError("call fit() before decompose()")

        base_forecasts = self._walk_forward_base(features, horizon=horizon)
        if base_forecasts.empty:
            raise RuntimeError("base model produced no overlapping forecasts")
        X, base_matrix, asset_idx, _, stamps = self._cut_sequences(
            features, base_forecasts, "target_return", require_target=False)
        if len(X) == 0:
            raise RuntimeError("base model produced no overlapping forecasts")

        self.head.eval()
        with torch.no_grad():
            residual = self.head(
                torch.tensor(np.nan_to_num(X), dtype=torch.float32),
                torch.tensor(np.nan_to_num(base_matrix), dtype=torch.float32),
                torch.tensor(asset_idx, dtype=torch.long),
            ).numpy()

        final = base_matrix + residual
        out = pd.DataFrame({"timestamp": stamps})
        for i, q in enumerate(self.config.quantiles):
            label = int(q * 100)
            out[f"base_p{label}"] = base_matrix[:, i]
            out[f"residual_p{label}"] = residual[:, i]
            out[f"final_p{label}"] = final[:, i]
        return out

    def _log_run(self, horizon: int) -> None:
        try:
            import mlflow

            mlflow.set_tracking_uri("sqlite:///artifacts/mlflow.db")
            mlflow.set_experiment("rq1_forecast_quality")
            with mlflow.start_run(run_name=f"hybrid_{self.base.name}_h{horizon}"):
                mlflow.log_params(
                    {
                        "model": self.name, "base_model": self.base.name, "horizon": horizon,
                        "window": self.config.window, "head_hidden": self.head_config.hidden_size,
                        "head_lr": self.head_config.lr, "freeze_base": self.config.freeze_base,
                        "zero_init_output": self.head_config.zero_init_output,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow logging failed (model is still trained): %s", exc)

    # ----------------------------------------------------------------------------------
    # Serving: one forecast from the latest window, and a portable checkpoint
    # ----------------------------------------------------------------------------------

    def predict_latest(self, features: pd.DataFrame, *, horizon: int) -> dict[str, object]:
        """Forecast from the end of `features` (one symbol): base quantiles from ONE base call
        on the full history, the head's correction from the last `window` feature rows.

        decompose() walks the base forward over the whole history, which is what training
        and RQ1 need and far too slow for a request. Returns base, residual and final so the
        caller can show how much of the forecast came from the covariates.
        """
        if self.head is None:
            raise RuntimeError("no trained head; load a checkpoint or call fit()")
        group = features.sort_values("timestamp").reset_index(drop=True)
        if len(group) <= self.config.window:
            raise ValueError(f"need more than {self.config.window} feature rows, have {len(group)}")

        base = self.base.predict_quantiles(group, horizon=horizon, quantiles=self.config.quantiles)
        base_values = np.asarray(base.values[-1], dtype=np.float32)

        missing = [c for c in self.feature_cols if c not in group.columns]
        if missing:
            # A column the head was trained on but this environment cannot build: zero is
            # the value the enrichment block itself uses for "absent", so this matches training.
            logger.warning("features missing at inference, zero-filled: %s", missing)
            group = group.assign(**{c: 0.0 for c in missing})
        # The cut is the last row; its window is the `window` rows before it (_cut_sequences).
        window = group[self.feature_cols].iloc[-self.config.window - 1 : -1].to_numpy(dtype=np.float32)
        asset = asset_class_index(group["asset_class"].iloc[-1]) if "asset_class" in group else 0
        asset = asset if asset < self.n_asset_classes else 0

        self.head.eval()
        with torch.no_grad():
            residual = self.head(
                torch.tensor(np.nan_to_num(window)[None, ...]),
                torch.tensor(np.nan_to_num(base_values)[None, ...]),
                torch.tensor([asset], dtype=torch.long),
            ).numpy()[0]
        final = enforce_non_crossing((base_values + residual)[None, :])[0]
        return {
            "quantiles": self.config.quantiles,
            "base": base_values.tolist(),
            "residual": residual.tolist(),
            "final": final.tolist(),
            "as_of": pd.Timestamp(group["timestamp"].iloc[-1]),
        }

    def save(self, directory) -> None:  # noqa: ANN001 - str | Path
        """Head weights + the metadata needed to rebuild it. The base is NOT saved: it is a
        public foundation checkpoint (or a LoRA adapter saved by finetune_lora)."""
        import json
        from dataclasses import asdict
        from pathlib import Path

        if self.head is None:
            raise RuntimeError("nothing to save; fit() first")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.head.state_dict(), path / "head.pt")
        meta = {
            "base_model": self.base.name,
            "horizon": self.horizon,
            "window": self.config.window,
            "quantiles": list(self.config.quantiles),
            "feature_cols": self.feature_cols,
            "n_asset_classes": self.n_asset_classes,
            "asset_class_order": list(ASSET_CLASS_ORDER[: self.n_asset_classes]),
            "head_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self.head_config).items()},
            "version": self.version,
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, directory, base: Forecaster) -> "HybridForecaster":  # noqa: ANN001
        import json
        from pathlib import Path

        path = Path(directory)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        head_config = ResidualHeadConfig(**{**meta["head_config"], "quantiles": tuple(meta["quantiles"])})
        config = HybridConfig(base_model=meta["base_model"], head=head_config,
                              quantiles=tuple(meta["quantiles"]), window=int(meta["window"]), log_to_mlflow=False)
        model = cls(base, config)
        model.feature_cols = list(meta["feature_cols"])
        model.horizon = meta.get("horizon")
        model.version = meta.get("version", "loaded")
        model.n_asset_classes = int(meta["n_asset_classes"])
        model.head = ResidualHead(len(model.feature_cols), model.n_asset_classes, head_config)
        model.head.load_state_dict(torch.load(path / "head.pt", map_location="cpu", weights_only=True))
        model.head.eval()
        return model
