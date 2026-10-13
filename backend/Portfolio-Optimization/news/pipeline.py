"""News/sentiment preparation for one held symbol, and its active-set steps.

    keywords step    ensure a keyword profile exists (generated on first sight)
    sentiment step   collect from every available channel -> relevance -> score ->
                     aggregate; store the aggregate in the active set and append per-day
                     rows to artifacts/sentiment/<symbol>.parquet

A channel failing (an outlet down, GDELT rate-limiting) is recorded in the channel report
and the rest carry on. The step only fails when NO channel produced anything usable.

Host collection (for logged-in channels, which need a desktop Chrome):
    python -m news.pipeline --all
    python -m news.pipeline --symbols JKH.N0000 BTCUSDT
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from data.instruments import Instrument
from news.channels import build_channels
from news.channels.base import utcnow
from news.models import KeywordProfile, RawItem

logger = logging.getLogger(__name__)

SENTIMENT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "sentiment"


def collect(profile: KeywordProfile, *, window_days: int) -> tuple[list[RawItem], dict[str, Any]]:
    """Run every applicable, available channel. Returns items and a per-channel report."""
    since = utcnow() - pd.Timedelta(days=window_days).to_pytimedelta()
    items: list[RawItem] = []
    report: dict[str, Any] = {}
    for channel in build_channels():
        if not channel.applies_to(profile):
            continue
        status = channel.status()
        if not status.available:
            report[channel.name] = {"status": "unavailable", "reason": status.reason}
            continue
        try:
            found = channel.fetch(profile, since)
        except Exception as exc:  # noqa: BLE001 - one channel never sinks the rest
            logger.warning("channel %s failed for %s: %s", channel.name, profile.symbol, exc)
            report[channel.name] = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"[:300]}
            continue
        report[channel.name] = {"status": "ok", "items": len(found)}
        items.extend(found)
    return items, report


def analyse(profile: KeywordProfile, *, window_days: int | None = None) -> tuple[dict[str, Any], list[dict]]:
    """Collect -> relevance -> score -> aggregate. Returns (sentiment block, daily rows)."""
    from news.aggregate import aggregate, daily_rows
    from news.keywords import load_config
    from news.relevance import filter_relevant
    from news.scoring import score_items

    config = load_config()
    settings = config.get("aggregation", {})
    window = window_days or int(settings.get("window_days", 14))

    raw, report = collect(profile, window_days=window)
    relevant, method = filter_relevant(raw, profile)
    scored, backend = score_items(relevant)

    block = aggregate(
        scored, now=utcnow(), window_days=window,
        spike_window_days=int(settings.get("spike_window_days", 2)),
        tau_days=float(settings.get("recency_tau_days", 3.0)),
        source_weights=config.get("source_weights"),
    )
    block.update({
        "channels": report,
        "items_collected": len(raw),
        "relevance_method": method,
        "scoring_backend": backend,
        "keyword_profile": {"source": profile.source, "provenance": profile.provenance},
    })
    return block, daily_rows(scored, profile.symbol)


def store_daily(symbol: str, rows: list[dict]) -> None:
    """Merge per-day rows into the symbol's sentiment history (later runs replace a day)."""
    if not rows:
        return
    from data.cache import _safe_symbol

    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
    path = SENTIMENT_DIR / f"{_safe_symbol(symbol)}.parquet"
    fresh = pd.DataFrame(rows)
    if path.exists():
        try:
            fresh = pd.concat([pd.read_parquet(path), fresh], ignore_index=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("rebuilding corrupt sentiment history %s: %s", path, exc)
    merged = fresh.drop_duplicates(["date", "tier"], keep="last").sort_values(["date", "tier"])
    tmp = path.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, index=False)
    tmp.replace(path)


# --------------------------------------------------------------------------------------
# Active-set steps
# --------------------------------------------------------------------------------------

def keywords_step(instrument: Instrument) -> tuple[bool, dict[str, Any] | str]:
    from news.keywords import ensure_profile

    profile = ensure_profile(instrument)
    return True, {
        "source": profile.source,
        "provenance": profile.provenance,
        "sector": profile.sector,
        "terms": {
            "company": len(profile.company), "people_brands": len(profile.people_brands),
            "sector": len(profile.sector_terms), "macro": len(profile.macro_terms),
            "local_names": len(profile.local_names), "exclude": len(profile.exclude),
        },
    }


def sentiment_step(instrument: Instrument) -> tuple[bool, dict[str, Any] | str]:
    from news.keywords import ensure_profile

    profile = ensure_profile(instrument)
    block, rows = analyse(profile)
    store_daily(instrument.symbol, rows)
    if not any(entry.get("status") == "ok" for entry in block["channels"].values()):
        return False, "no channel returned results: " + "; ".join(
            f"{name}: {entry.get('reason')}" for name, entry in block["channels"].items()
        )
    return True, block


# --------------------------------------------------------------------------------------
# Host CLI
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    from data import active_set

    parser = argparse.ArgumentParser(description="Collect news/social sentiment for held symbols.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="every symbol in the active set")
    group.add_argument("--symbols", nargs="+", help="specific tracked symbols")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    symbols = [row["symbol"] for row in active_set.status()] if args.all else [s.upper() for s in args.symbols]
    for symbol in symbols:
        record = active_set.prepare(symbol)
        sentiment = (record.get("metrics") or {}).get("sentiment") or {}
        print(f"{symbol}: {record['status']}  items={sentiment.get('items_relevant')}  "
              f"overall={sentiment.get('overall_sentiment')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
