"""RQ1 baseline row on REAL market data.

Runs the full Phase 1 -> 2 -> 3 -> 7 chain: cached OHLCV, technical indicators, feature
store, baseline LSTM, walk-forward backtest. Uses baseline_lstm because it trains from
scratch locally and needs no downloaded weights.

COMPUTE BUDGET
--------------
Sized deliberately for a CPU-only laptop. The full 15-year universe at a 6-month step gives
~23 folds, and each fold trains an LSTM from scratch -- roughly a quarter of a million
gradient steps, which is hours. This uses a 5-year evaluation span with annual steps
(~5 folds) and a smaller network.

That is a smaller experiment, not a weaker one: folds stay non-overlapping and the embargo
still holds, so every reported number remains an honest out-of-sample estimate. Widen
EVAL_YEARS / raise the epoch count when running somewhere with a GPU.

Sentiment is omitted: scoring years of headlines through the local model is hours of serial
work and is not needed for RQ1's BASELINE row. The hybrid rows add it.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("rq1")

from data.ingestion import bars_to_frame, fetch_ohlcv
from data.schema import AssetClass
from evaluation.backtest import BacktestConfig, run_walk_forward
from evaluation.metrics import forecast_metrics
from features.feature_store import build_feature_table
from features.technical import compute_universe_indicators
from forecasting.baseline_lstm import LSTMConfig

HORIZON = 5
EVAL_YEARS = 5
CLASSES = {"equity": AssetClass.EQUITY, "etf": AssetClass.ETF, "forex": AssetClass.FOREX}

# Smaller than the library defaults, for the reasons in the module docstring.
FAST_LSTM = LSTMConfig(window=40, hidden_size=32, num_layers=1, epochs=12, patience=3, batch_size=128)


def main() -> int:
    started = time.time()
    resolved_path = Path("configs/resolved_universe.yaml")
    if not resolved_path.exists():
        logger.error("run experiments/resolve_universe.py first")
        return 1

    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))["symbols"]

    # Only the tail of each symbol's resolved window, to keep the fold count tractable.
    cutoff = date.today() - timedelta(days=int(EVAL_YEARS * 365.25))
    bars: dict[str, pd.DataFrame] = {}
    for symbol, entry in sorted(resolved.items()):
        rows = fetch_ohlcv(
            symbol, CLASSES[entry["asset_class"]],
            date.fromisoformat(entry["history_start"]), date.fromisoformat(entry["history_end"]),
        )
        if not rows:
            continue
        frame = bars_to_frame(rows)
        frame = frame[pd.to_datetime(frame["timestamp"]).dt.date >= cutoff]
        if len(frame) > 300:
            bars[symbol] = frame.reset_index(drop=True)

    logger.info("loaded %d symbols, %d bars", len(bars), sum(len(f) for f in bars.values()))

    table = build_feature_table(bars, compute_universe_indicators(bars))
    logger.info("feature table: %d rows x %d cols, %s -> %s",
                len(table), table.shape[1],
                table["timestamp"].min().date(), table["timestamp"].max().date())

    config = BacktestConfig(
        train_window_days=504, test_window_days=252, step_days=252, embargo_days=HORIZON
    )
    logger.info("starting walk-forward (this is the slow part)")
    preds = run_walk_forward(
        table, "baseline_lstm", horizon=HORIZON, config=config,
        forecaster_kwargs={"config": FAST_LSTM}, log_to_mlflow=True,
    )

    valid = preds.dropna(subset=["target_return"])
    metrics = forecast_metrics(
        valid["target_return"].to_numpy(), valid[["p10", "p50", "p90"]].to_numpy(), (0.1, 0.5, 0.9)
    )

    print()
    print("=== RQ1: baseline_lstm, horizon=5, walk-forward on real data ===")
    print(f"  folds        {preds['fold'].nunique()}")
    print(f"  predictions  {len(valid):,}")
    print(f"  symbols      {table['symbol'].nunique()}")
    for key, value in metrics.items():
        print(f"  {key:26} {value:.6f}")

    out = Path("artifacts/results")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "model": "baseline_lstm", "status": "ok", "horizon": HORIZON,
        "n_folds": int(preds["fold"].nunique()), "n_predictions": len(valid),
        "n_symbols": int(table["symbol"].nunique()), **metrics,
    }]).to_csv(out / "rq1_baseline.csv", index=False)

    print(f"\nwrote {out / 'rq1_baseline.csv'} in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
