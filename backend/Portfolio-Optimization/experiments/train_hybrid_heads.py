"""Colab Part A (updated): train the hybrid's LSTM residual heads over a frozen TimesFM base,
on CSE + Binance + Yahoo, and export them for `python -m forecasting.serving --install`.

    python -m experiments.train_hybrid_heads --out /content/drive/MyDrive/j26_hybrid \
        --backfill /content/drive/MyDrive/j26_se_325_backfill/fundamentals_backfill.parquet

TRAINING UNIVERSE (symbol-agnostic models, so it is broad on purpose)
  * every CSE voting/non-voting security with >= MIN_CSE_SESSIONS daily sessions in the
    public one-year window (thin names included: they are the regime liquidity matters in);
  * the top BINANCE_TOP USDT pairs by 24h quote volume (stablecoin pairs excluded);
  * the US equities and ETFs in configs/universe.yaml.
The head sees asset class as an embedding, never the symbol, so a symbol added to a
portfolio later is forecast without retraining.

SPLIT
Time-based: rows dated before the cutoff (the last HOLDOUT_FRACTION of dates) train the
head; the rest are scored for base vs hybrid pinball loss, overall and per asset class. The
metrics go into each head's meta.json and from there into the model registry.

RESUMABLE
Stages persist to --out (put it on Drive): bars.parquet, then features_h<H>.parquet, then
h<H>/ once that head is trained. Re-running skips finished stages.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HORIZONS = (5, 21)
MIN_CSE_SESSIONS = 200
BINANCE_TOP = 50
HOLDOUT_FRACTION = 0.2
YEARS_OF_HISTORY = 6
_STABLE = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "EUR", "USD1", "PYUSD"}


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------

def _binance_top(limit: int) -> list[str]:
    from data.http import request_json

    rows = request_json("binance", "GET", "https://data-api.binance.vision/api/v3/ticker/24hr")
    usdt = [r for r in rows if r["symbol"].endswith("USDT") and r["symbol"][:-4] not in _STABLE]
    usdt.sort(key=lambda r: float(r.get("quoteVolume") or 0), reverse=True)
    return [r["symbol"] for r in usdt[:limit]]


def collect_bars(out: Path, *, cse_limit: int | None = None) -> pd.DataFrame:
    path = out / "bars.parquet"
    if path.exists():
        return pd.read_parquet(path)

    import yaml

    from data.exchanges import binance, cse, yahoo
    from data.schema import AssetClass

    end = date.today()
    start = end - timedelta(days=365 * YEARS_OF_HISTORY)
    frames: list[pd.DataFrame] = []

    securities = [s for s in cse.list_securities() if ".N" in s["symbol"] or ".X" in s["symbol"]]
    for security in securities[:cse_limit] if cse_limit else securities:
        try:
            bars = cse.daily_bars(security["symbol"], security["provider_id"])
        except Exception as exc:  # noqa: BLE001
            logger.info("CSE %s skipped: %s", security["symbol"], exc)
            continue
        if len(bars) >= MIN_CSE_SESSIONS:
            frames.append(bars)
    logger.info("CSE: %d securities with >= %d sessions", len(frames), MIN_CSE_SESSIONS)

    for symbol in _binance_top(BINANCE_TOP):
        try:
            frames.append(binance.daily_bars(symbol, start, end))
        except Exception as exc:  # noqa: BLE001
            logger.info("Binance %s skipped: %s", symbol, exc)

    universe = yaml.safe_load(Path("configs/universe.yaml").read_text(encoding="utf-8"))
    for key, asset_class in (("equities", AssetClass.EQUITY), ("etfs", AssetClass.ETF)):
        for entry in universe.get(key, []):
            try:
                frames.append(yahoo.daily_bars(entry["symbol"], asset_class, start, end))
            except Exception as exc:  # noqa: BLE001
                logger.info("Yahoo %s skipped: %s", entry["symbol"], exc)

    table = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    table.to_parquet(path, index=False)
    logger.info("bars: %d rows, %d symbols", len(table), table["symbol"].nunique())
    return table


def build_features(bars: pd.DataFrame, horizon: int, out: Path, backfill: Path | None) -> pd.DataFrame:
    path = out / f"features_h{horizon}.parquet"
    if path.exists():
        return pd.read_parquet(path)

    from features.enrichment import add_enrichment, backfill_loader
    from features.feature_store import add_targets, build_feature_table
    from features.technical import compute_universe_indicators

    by_symbol = {s: g.drop(columns=["symbol"]).assign(symbol=s) for s, g in bars.groupby("symbol")}
    table = build_feature_table(by_symbol, compute_universe_indicators(by_symbol))
    # No tier-sentiment history exists for past dates; the sentinel columns stay zero with
    # has_tier_sentiment = 0, exactly as serving builds them for a newly tracked symbol.
    loader = backfill_loader(backfill) if backfill and backfill.exists() else (lambda symbol: [])
    table = add_enrichment(table, sentiment_dir=out / "no_sentiment_history", fundamentals_loader=loader)
    table = add_targets(table, horizon=horizon)
    table.to_parquet(path, index=False)
    return table


# --------------------------------------------------------------------------------------
# Train + evaluate
# --------------------------------------------------------------------------------------

def _pinball(actual: np.ndarray, preds: np.ndarray, quantiles: tuple[float, ...]) -> float:
    losses = [np.maximum(q * (actual - preds[:, i]), (q - 1) * (actual - preds[:, i])) for i, q in enumerate(quantiles)]
    return float(np.mean(losses))


def train_head(features: pd.DataFrame, horizon: int, out: Path, *, walk_forward_points: int, epochs: int) -> dict:
    target = out / f"h{horizon}"
    if (target / "head.pt").exists():
        logger.info("h=%d already trained; skipping", horizon)
        return json.loads((target / "meta.json").read_text(encoding="utf-8"))

    from forecasting.base import get_forecaster
    from forecasting.hybrid_model import HybridConfig, HybridForecaster
    from forecasting.residual_head import ResidualHeadConfig

    dates = np.sort(features["timestamp"].unique())
    cutoff = pd.Timestamp(dates[int(len(dates) * (1 - HOLDOUT_FRACTION))])
    train = features[features["timestamp"] < cutoff]

    model = HybridForecaster(get_forecaster("timesfm"), HybridConfig(
        head=ResidualHeadConfig(epochs=epochs), walk_forward_points=walk_forward_points, log_to_mlflow=False))
    model.fit(train, horizon=horizon)

    # Holdout: walk-forward base + head over the full history, scored only after the cutoff.
    metrics: dict[str, float] = {}
    rows = []
    groups = features.groupby("symbol", sort=False)
    for i, (symbol, group) in enumerate(groups, 1):
        if i % 25 == 0:  # the first full run went silent here for ~1h; show it is moving
            logger.info("h=%d holdout: %d/%d symbols", horizon, i, groups.ngroups)
        try:
            decomposed = model.decompose(group, horizon=horizon)
        except Exception as exc:  # noqa: BLE001
            logger.info("holdout skipped %s: %s", symbol, exc)
            continue
        decomposed["symbol"] = symbol
        decomposed["asset_class"] = group["asset_class"].iloc[0]
        rows.append(decomposed)
    if rows:
        scored = pd.concat(rows).merge(features[["symbol", "timestamp", "target_return"]], on=["symbol", "timestamp"])
        scored = scored[(scored["timestamp"] >= cutoff) & scored["target_return"].notna()]
        q = model.config.quantiles
        base_cols = [f"base_p{int(x * 100)}" for x in q]
        final_cols = [f"final_p{int(x * 100)}" for x in q]
        for name, part in [("all", scored), *scored.groupby("asset_class")]:
            if len(part) < 20:
                continue
            actual = part["target_return"].to_numpy()
            metrics[f"pinball_base_{name}"] = _pinball(actual, part[base_cols].to_numpy(), q)
            metrics[f"pinball_hybrid_{name}"] = _pinball(actual, np.sort(part[final_cols].to_numpy(), axis=1), q)
            metrics[f"holdout_rows_{name}"] = float(len(part))

    tmp = out / f"h{horizon}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save(tmp)
    meta = json.loads((tmp / "meta.json").read_text(encoding="utf-8"))
    meta.update({
        "train_window": {"start": str(pd.Timestamp(train["timestamp"].min()).date()),
                         "end": str(pd.Timestamp(train["timestamp"].max()).date())},
        "holdout_from": str(cutoff.date()),
        "metrics": metrics,
        "universe": {k: int(v) for k, v in features.groupby("asset_class")["symbol"].nunique().items()},
    })
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    tmp.rename(target)  # atomic "done" marker for resume
    logger.info("h=%d trained; holdout %s", horizon, {k: round(v, 5) for k, v in metrics.items() if "all" in k})
    return meta


def export(out: Path) -> Path:
    """Zip only the trained head directories (not the bar/feature caches beside them)."""
    import zipfile

    path = out / "hybrid_heads.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for head in sorted(out.glob("h*/head.pt")):
            for file in head.parent.glob("*"):
                zf.write(file, f"{head.parent.name}/{file.name}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("artifacts/hybrid_training"))
    parser.add_argument("--backfill", type=Path, default=None, help="fundamentals_backfill.parquet from Part D")
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    parser.add_argument("--cse-limit", type=int, default=None, help="smoke runs only")
    parser.add_argument("--walk-forward-points", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    args.out.mkdir(parents=True, exist_ok=True)
    bars = collect_bars(args.out, cse_limit=args.cse_limit)
    for horizon in args.horizons:
        features = build_features(bars, horizon, args.out, args.backfill)
        train_head(features, horizon, args.out, walk_forward_points=args.walk_forward_points, epochs=args.epochs)
    print("export:", export(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
