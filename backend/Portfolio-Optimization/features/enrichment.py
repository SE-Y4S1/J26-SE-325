"""Phase 3 model inputs beyond price: per-tier news sentiment and point-in-time fundamentals.

Both are joined onto the feature table (features/feature_store.py) so the residual head
sees them next to the technical indicators. The COLUMN SET IS FIXED: a symbol with no
fundamentals (crypto, a scanned-PDF company) or no sentiment history still gets every column,
zero-filled, with a `has_*` indicator. A head trained on Colab and served locally therefore
always sees the same schema, whatever data each environment happens to hold.

POINT IN TIME
Fundamentals enter on a row only when the report's `available_from` (CSE upload time, SEC
filing date) is on or before the row's date -- an as-of join, never "the latest report".
Valuation multiples are recomputed on every row from that day's close against the report's
per-share figures, so they move with the price between reports as they would for an analyst.

Sentiment tiers come from artifacts/sentiment/<symbol>.parquet (news/pipeline.py). That
history only accumulates from the day a symbol is first tracked, so historical training rows
carry zeros and `has_tier_sentiment = 0`; the head can learn to ignore the block where it is
absent rather than read zeros as "neutral news".
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SENTIMENT_TIER_COLUMNS = ["sent_company", "sent_sector", "sent_macro", "attention_company", "has_tier_sentiment"]
FUNDAMENTAL_COLUMNS = ["fund_roe", "fund_debt_to_equity", "fund_earnings_growth", "fund_f_score",
                       "fund_z_score", "fund_pb", "fund_earnings_yield", "fund_report_age",
                       "has_fundamentals"]
ENRICHMENT_COLUMNS = [*SENTIMENT_TIER_COLUMNS, *FUNDAMENTAL_COLUMNS]

SENTIMENT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "sentiment"
TIER_DECAY_DAYS = 3.0
# Winsorising bounds. A quarter's EPS of 0.004 gives P/E 1,150; left raw, one such row
# dominates a normalised input. Bounds are generous: they clip absurd values, not real ones.
_CLIP = {"fund_roe": (-1.0, 1.0), "fund_debt_to_equity": (0.0, 10.0), "fund_earnings_growth": (-3.0, 3.0),
         "fund_z_score": (-5.0, 15.0), "fund_pb": (0.0, 20.0), "fund_earnings_yield": (-0.5, 0.5),
         "fund_report_age": (0.0, 2.0)}

RecordsLoader = Callable[[str], list[dict[str, Any]]]


def _store_loader(symbol: str) -> list[dict[str, Any]]:
    from fundamentals.pipeline import load_records

    return list(load_records(symbol).values())


def backfill_loader(path: Path) -> RecordsLoader:
    """Records from the Colab Part D export, for training without a local store."""
    import json

    table = pd.read_parquet(path)
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in table.itertuples():
        record = json.loads(row.record)
        if row.ratios and isinstance(row.ratios, str):
            record["ratios"] = json.loads(row.ratios)
        by_symbol.setdefault(row.symbol, []).append(record)
    return lambda symbol: by_symbol.get(symbol, [])


# --------------------------------------------------------------------------------------
# Sentiment tiers
# --------------------------------------------------------------------------------------

def _tier_frame(symbol: str, sentiment_dir: Path) -> pd.DataFrame | None:
    from data.cache import _safe_symbol

    path = sentiment_dir / f"{_safe_symbol(symbol)}.parquet"
    if not path.exists():
        return None
    try:
        rows = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("unreadable sentiment history %s: %s", path, exc)
        return None
    if rows.empty:
        return None
    rows["timestamp"] = pd.to_datetime(rows["date"]).dt.normalize()
    pivot = rows.pivot_table(index="timestamp", columns="tier", values="mean_sentiment", aggfunc="mean")
    items = rows[rows["tier"] == "company"].groupby("timestamp")["items"].sum()
    out = pd.DataFrame(index=pivot.index.union(items.index))
    for tier in ("company", "sector", "macro"):
        out[f"sent_{tier}"] = pivot[tier] if tier in pivot else np.nan
    out["attention_company"] = np.log1p(items).reindex(out.index)
    return out


def _decay_join(spine: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """As-of join of daily observations onto the trading calendar, decaying toward 0 with
    trading days since the observation (same policy as feature_store.decay_fill_sentiment)."""
    stamps = pd.to_datetime(spine["timestamp"]).sort_values().reset_index(drop=True)
    result = pd.DataFrame({"timestamp": stamps})
    positions = np.arange(len(stamps))
    for column in observed.columns:
        series = observed[column].dropna()
        if series.empty:
            result[column] = 0.0
            continue
        obs = pd.DataFrame({"obs_ts": pd.to_datetime(series.index), "v": series.to_numpy()}).sort_values("obs_ts")
        joined = pd.merge_asof(result, obs, left_on="timestamp", right_on="obs_ts", direction="backward")
        # Spine position of the last trading day on or before the observation (weekend news
        # counts from Friday's row), so the decay runs in trading days.
        obs_pos = np.searchsorted(stamps.to_numpy(), joined["obs_ts"].to_numpy(), side="right") - 1
        days = np.clip(positions - obs_pos, 0, None).astype(float)
        result[column] = (joined["v"].to_numpy(dtype=float) * np.exp(-days / TIER_DECAY_DAYS))
        result[column] = result[column].fillna(0.0)
    return result


def add_tier_sentiment(table: pd.DataFrame, *, sentiment_dir: Path | None = None) -> pd.DataFrame:
    directory = sentiment_dir or SENTIMENT_DIR
    frames = []
    for symbol, group in table.groupby("symbol", sort=False):
        group = group.sort_values("timestamp").copy()
        observed = _tier_frame(str(symbol), directory)
        if observed is None or observed.dropna(how="all").empty:
            for column in SENTIMENT_TIER_COLUMNS:
                group[column] = 0.0
        else:
            first = observed.index.min()
            joined = _decay_join(group[["timestamp"]], observed)
            for column in ("sent_company", "sent_sector", "sent_macro", "attention_company"):
                group[column] = joined[column].to_numpy()
            group["has_tier_sentiment"] = (group["timestamp"] >= first).astype(float).to_numpy()
        frames.append(group)
    return pd.concat(frames).sort_values(["symbol", "timestamp"]).reset_index(drop=True) if frames else table


# --------------------------------------------------------------------------------------
# Fundamentals
# --------------------------------------------------------------------------------------

def _fundamental_rows(records: list[dict[str, Any]]) -> pd.DataFrame | None:
    rows = []
    for record in records:
        ratios = record.get("ratios") or {}
        current = record.get("current") or {}
        if not record.get("usable") or not ratios:
            continue
        f_max = ratios.get("f_score_max") or 0
        rows.append({
            "available": pd.Timestamp(record["available_from"]).tz_convert(None).normalize()
            if pd.Timestamp(record["available_from"]).tzinfo else pd.Timestamp(record["available_from"]).normalize(),
            "fund_roe": ratios.get("roe"),
            "fund_debt_to_equity": ratios.get("debt_to_equity"),
            "fund_earnings_growth": ratios.get("earnings_growth_yoy"),
            "fund_f_score": (ratios["f_score"] / f_max) if ratios.get("f_score") is not None and f_max else None,
            "fund_z_score": ratios.get("z_score"),
            "naps": current.get("naps"),
            "eps_annualised": ratios.get("eps_annualised"),
        })
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("available").drop_duplicates("available", keep="last")


def add_fundamentals(table: pd.DataFrame, *, loader: RecordsLoader | None = None) -> pd.DataFrame:
    load = loader or _store_loader
    frames = []
    for symbol, group in table.groupby("symbol", sort=False):
        group = group.sort_values("timestamp").copy()
        try:
            reports = _fundamental_rows(load(str(symbol)))
        except Exception as exc:  # noqa: BLE001 - fundamentals are optional inputs
            logger.warning("fundamentals unavailable for %s: %s", symbol, exc)
            reports = None
        if reports is None:
            for column in FUNDAMENTAL_COLUMNS:
                group[column] = 0.0
            frames.append(group)
            continue
        spine = group[["timestamp", "close"]].copy()
        spine["timestamp"] = pd.to_datetime(spine["timestamp"])
        joined = pd.merge_asof(spine, reports, left_on="timestamp", right_on="available", direction="backward")
        close = joined["close"].astype(float)
        joined["fund_pb"] = close / joined["naps"].where(joined["naps"] > 0)
        joined["fund_earnings_yield"] = joined["eps_annualised"] / close
        joined["fund_report_age"] = (joined["timestamp"] - joined["available"]).dt.days / 365.0
        joined["has_fundamentals"] = joined["available"].notna().astype(float)
        for column in FUNDAMENTAL_COLUMNS:
            values = joined[column].astype(float)
            if column in _CLIP:
                values = values.clip(*_CLIP[column])
            group[column] = values.fillna(0.0).to_numpy()
        frames.append(group)
    return pd.concat(frames).sort_values(["symbol", "timestamp"]).reset_index(drop=True) if frames else table


def add_enrichment(table: pd.DataFrame, *, sentiment_dir: Path | None = None,
                   fundamentals_loader: RecordsLoader | None = None) -> pd.DataFrame:
    """Both blocks. Always returns every ENRICHMENT_COLUMNS column."""
    if table.empty:
        return table
    out = add_tier_sentiment(table, sentiment_dir=sentiment_dir)
    return add_fundamentals(out, loader=fundamentals_loader)
