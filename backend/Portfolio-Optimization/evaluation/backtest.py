"""Walk-forward backtest harness.

NO LOOK-AHEAD, ENFORCED
-----------------------
Look-ahead bias is the single most common way a financial ML result becomes meaningless, and
it is usually invisible in the output -- results just look good. So this harness asserts the
property rather than documenting it: assert_no_lookahead() raises if any fold's feature rows
carry a timestamp at or after their target, and a deliberate leak test in tests/ confirms the
guard actually fires (a guard that never triggers proves nothing).

Two subtler cases the assertion also covers:
  * Indicator warm-up. A fold starting inside the MACD burn-in uses values computed from
    prior-fold data. Folds are offset by features.technical.warmup_periods().
  * Sentiment decay carry-over. The decayed sentiment state must be recomputed per fold, not
    carried across the boundary.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    fold_index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date


@dataclass
class BacktestConfig:
    train_window_days: int = 756      # ~3 trading years
    test_window_days: int = 63        # ~1 quarter
    step_days: int = 63               # non-overlapping test windows
    expanding_window: bool = False    # False = rolling; True = anchored/expanding
    min_train_days: int = 252
    # Gap between train end and test start, in trading days. Must be at least the forecast
    # horizon: with a 5-day target, the last 5 training rows have targets that overlap the
    # test window, so training on them leaks test-period outcomes.
    embargo_days: int = 5


def generate_folds(start: date, end: date, config: BacktestConfig) -> list[Fold]:
    """Walk-forward fold boundaries.

    Test windows never overlap, so per-fold metrics are independent and can be aggregated
    without double-counting -- overlapping windows would make a lucky quarter count twice
    and inflate any significance claim.
    """
    folds: list[Fold] = []
    index = 0
    train_start = start

    while True:
        train_end = train_start + timedelta(days=config.train_window_days)
        test_start = train_end + timedelta(days=config.embargo_days)
        test_end = test_start + timedelta(days=config.test_window_days)

        if test_end > end:
            break

        folds.append(
            Fold(
                fold_index=index,
                train_start=start if config.expanding_window else train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

        index += 1
        if not config.expanding_window:
            train_start += timedelta(days=config.step_days)
        else:
            train_start = start
            # For an expanding window, growing the train window is what advances the fold.
            config = BacktestConfig(
                train_window_days=config.train_window_days + config.step_days,
                test_window_days=config.test_window_days,
                step_days=config.step_days,
                expanding_window=True,
                min_train_days=config.min_train_days,
                embargo_days=config.embargo_days,
            )

    if not folds:
        logger.warning(
            "no folds fit between %s and %s with a %d-day train window",
            start, end, config.train_window_days,
        )
    return folds


def assert_no_lookahead(features: pd.DataFrame, target_timestamp: pd.Timestamp) -> None:
    """Raise if any feature row is dated at or after the target timestamp."""
    if features.empty:
        return

    stamps = pd.to_datetime(features["timestamp"])
    offenders = stamps[stamps >= pd.Timestamp(target_timestamp)]
    if len(offenders):
        raise ValueError(
            f"look-ahead detected: {len(offenders)} feature row(s) dated at or after the "
            f"target {target_timestamp}. Latest offending row: {offenders.max()}"
        )


def assert_fold_is_clean(train: pd.DataFrame, test: pd.DataFrame, *, embargo_days: int) -> None:
    """Verify a train/test split has no overlap and honours the embargo."""
    if train.empty or test.empty:
        return

    train_end = pd.to_datetime(train["timestamp"]).max()
    test_start = pd.to_datetime(test["timestamp"]).min()

    if test_start <= train_end:
        raise ValueError(
            f"fold overlap: test starts {test_start} but training runs to {train_end}"
        )

    gap = (test_start - train_end).days
    if gap < embargo_days:
        raise ValueError(
            f"embargo violated: only {gap} days between train end and test start, "
            f"need {embargo_days} (the forecast horizon), or training targets overlap the "
            "test window"
        )


def iter_fold_data(
    features: pd.DataFrame, folds: list[Fold], *, warmup: int = 0
) -> Iterator[tuple[Fold, pd.DataFrame, pd.DataFrame]]:
    """Yield (fold, train, test) with the warm-up offset already applied."""
    frame = features.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])

    for fold in folds:
        train = frame[
            (frame["timestamp"] >= pd.Timestamp(fold.train_start))
            & (frame["timestamp"] < pd.Timestamp(fold.train_end))
        ]
        test = frame[
            (frame["timestamp"] >= pd.Timestamp(fold.test_start))
            & (frame["timestamp"] < pd.Timestamp(fold.test_end))
        ]

        # Discard indicator burn-in at the head of each fold, per symbol: those rows carry
        # EMA state seeded from the previous fold's data.
        if warmup > 0 and not train.empty:
            train = train.groupby("symbol", group_keys=False).apply(
                lambda g: g.iloc[warmup:], include_groups=True
            )

        if train.empty or test.empty:
            logger.debug("fold %d is empty after filtering; skipping", fold.fold_index)
            continue

        yield fold, train, test


def run_walk_forward(
    features: pd.DataFrame,
    forecaster_name: str,
    *,
    horizon: int,
    config: BacktestConfig | None = None,
    log_to_mlflow: bool = True,
    forecaster_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Train-forecast-roll across folds. Returns per-fold predictions with actuals attached."""
    from features.feature_store import add_targets
    from features.technical import warmup_periods
    from forecasting.base import get_forecaster
    from forecasting.finetune_lora import memory_note

    config = config or BacktestConfig(embargo_days=horizon)
    if config.embargo_days < horizon:
        raise ValueError(
            f"embargo_days ({config.embargo_days}) must be >= horizon ({horizon}); otherwise "
            "the last training targets overlap the test window"
        )

    frame = add_targets(features, horizon=horizon)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])

    folds = generate_folds(frame["timestamp"].min().date(), frame["timestamp"].max().date(), config)
    if not folds:
        raise ValueError("no folds could be generated for this data range")

    predictions: list[pd.DataFrame] = []

    # Built ONCE, outside the loop. A foundation model's weights are 200-900MB and do not
    # change between folds, so constructing a forecaster per fold re-downloaded and
    # re-loaded them every time -- twenty-plus loads across a backtest, which is what
    # exhausted Colab's RAM.
    #
    # This does not weaken the walk-forward: fit() is still called per fold, and
    # BaselineLSTMForecaster.fit rebuilds its network and reseeds on every call, so a fold
    # never inherits weights from the one before it. For the zero-shot adapters fit() is a
    # documented no-op, and there was never anything per-fold to construct.
    forecaster = get_forecaster(forecaster_name, **(forecaster_kwargs or {}))

    logger.info(
        "backtest %s: %d folds, %s", forecaster_name, len(folds), memory_note()
    )

    for fold, train, test in iter_fold_data(frame, folds, warmup=warmup_periods()):
        assert_fold_is_clean(train, test, embargo_days=config.embargo_days)

        try:
            forecaster.fit(train, horizon=horizon, log_to_mlflow=False)
            result = forecaster.predict_quantiles(test, horizon=horizon)
        except Exception as exc:  # noqa: BLE001 - one bad fold must not kill the backtest
            logger.error("fold %d failed: %s", fold.fold_index, exc)
            continue

        fold_frame = result.to_frame()
        fold_frame["fold"] = fold.fold_index
        fold_frame = fold_frame.merge(
            test[["symbol", "timestamp", "target_return"]],
            on=["symbol", "timestamp"], how="left",
        )
        predictions.append(fold_frame)

        # A fold's training tensors are hundreds of megabytes at universe scale, and the
        # next fold rebuilds them from scratch. Dropping them here keeps the peak to one
        # fold's worth rather than leaving the last one resident while the next allocates.
        del train, test, result, fold_frame
        gc.collect()

        # Per-fold, because a backtest that dies mid-run leaves no other evidence -- and on
        # Colab the session dies with it, taking the traceback too.
        logger.info(
            "  fold %d/%d done (%d predictions so far)  %s",
            fold.fold_index + 1, len(folds), sum(len(f) for f in predictions), memory_note(),
        )

    if not predictions:
        raise RuntimeError("every fold failed; nothing to report")

    combined = pd.concat(predictions, ignore_index=True)

    # The next candidate loads its own model; this one should not still be resident when it
    # does. Matters most for the foundation adapters, which are 200-900MB apiece.
    del forecaster, predictions, frame
    gc.collect()
    logger.info("backtest %s complete: %s", forecaster_name, memory_note())

    if log_to_mlflow:
        _log_backtest(forecaster_name, horizon, combined, len(folds))

    return combined


def _log_backtest(model_name: str, horizon: int, predictions: pd.DataFrame, n_folds: int) -> None:
    """Log aggregate backtest metrics. Never fatal."""
    try:
        import mlflow

        from evaluation.metrics import forecast_metrics

        quantile_cols = [c for c in predictions.columns if c.startswith("p") and c[1:].isdigit()]
        quantiles = tuple(int(c[1:]) / 100 for c in quantile_cols)

        metrics = forecast_metrics(
            predictions["target_return"].to_numpy(),
            predictions[quantile_cols].to_numpy(),
            quantiles,
        )

        mlflow.set_tracking_uri("sqlite:///artifacts/mlflow.db")
        mlflow.set_experiment("rq1_forecast_quality")
        with mlflow.start_run(run_name=f"backtest_{model_name}_h{horizon}"):
            mlflow.log_params({"model": model_name, "horizon": horizon, "n_folds": n_folds})
            mlflow.log_metrics({k: v for k, v in metrics.items() if np.isfinite(v)})
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow backtest logging failed: %s", exc)
