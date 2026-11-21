"""Accumulate CSE daily history beyond the one year the chart endpoint exposes.

The endpoint (`companyChartDataByStock`) returns about 250 sessions and nothing longer: every
period setting 1-10 was probed and 5 is the longest that returns daily bars (6+ collapse to a
handful of points). So history past one year only exists if each fetch is kept. This module
fetches every listed equity and merges the result into one parquet per security, so the store
grows by roughly a session per trading day it runs.

It complements data/history.py, which does the same merge but only for symbols a user has
tracked. This covers the whole exchange, which is what the forecasting heads and the Laya
decision model train on.

Run it daily (any cadence under a year works; a session is only lost if a window is missed):

    uv run python -m data.cse_history                        # artifacts/cse_history
    uv run python -m data.cse_history --out D:/cse_history --limit 20

Then give the trainers the directory: `train_hybrid_heads --cse-history <dir>`. On Colab, zip it
to Drive (`j26_cse_history/`); the notebooks pick it up when present.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from data.cache import CACHE_ROOT, _safe_symbol
from data.history import merge_bars
from data.ingestion import OHLCV_COLUMNS

logger = logging.getLogger(__name__)

DEFAULT_DIR = CACHE_ROOT.parent / "cse_history"


@dataclass
class AccumulationReport:
    updated: int = 0                     # securities whose store was written
    new_rows: int = 0                    # sessions added across all securities
    total_rows: int = 0                  # sessions held across all securities afterwards
    failed: dict[str, str] = field(default_factory=dict)
    skipped: int = 0                     # not an equity, or no security id


def is_equity(symbol: str) -> bool:
    """CSE equities carry a `.N` (voting) or `.X` (non-voting) series suffix; warrants, rights
    and debentures do not, and have no daily chart worth training on."""
    return ".N" in symbol or ".X" in symbol


def history_file(directory: Path, symbol: str) -> Path:
    return Path(directory) / f"{_safe_symbol(symbol)}.parquet"


def merge_history(stored: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Union on date; a refetched day replaces the stored one (late corrections win)."""
    return merge_bars(stored, fresh)


def load_accumulated(directory: Path | None, symbol: str) -> pd.DataFrame | None:
    """Stored bars for a symbol, or None when there is no store or it is unreadable."""
    if directory is None:
        return None
    path = history_file(Path(directory), symbol)
    if not path.exists():
        return None
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - a corrupt file is a miss; the next run rebuilds it
        logger.warning("unreadable CSE history %s (%s); ignoring it", path, exc)
        return None
    return frame if len(frame) else None


def _save(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)                                 # atomic: an interrupted run never leaves half a file


def accumulate_cse_history(
    directory: Path | str = DEFAULT_DIR,
    *,
    limit: int | None = None,
    pause_seconds: float = 0.3,
) -> AccumulationReport:
    """Fetch every listed CSE equity and merge it into its store. Never raises for one
    security's failure: that security keeps its old history and is named in the report."""
    from data.exchanges import cse

    directory = Path(directory)
    report = AccumulationReport()

    securities = cse.list_securities()
    eligible = []
    for security in securities:
        if is_equity(security["symbol"]) and security.get("provider_id"):
            eligible.append(security)
        else:
            report.skipped += 1
    if limit:
        eligible = eligible[:limit]

    for index, security in enumerate(eligible):
        symbol = security["symbol"]
        try:
            fresh = cse.daily_bars(symbol, security["provider_id"])
        except Exception as exc:  # noqa: BLE001 - one security must not stop the run
            report.failed[symbol] = f"{type(exc).__name__}: {exc}"
            logger.warning("CSE %s skipped: %s", symbol, exc)
            continue
        if fresh is None or fresh.empty:
            continue

        stored = load_accumulated(directory, symbol)
        before = 0 if stored is None else len(stored)
        merged = merge_history(stored if stored is not None else pd.DataFrame(columns=OHLCV_COLUMNS), fresh)
        _save(history_file(directory, symbol), merged)

        report.updated += 1
        report.new_rows += len(merged) - before
        report.total_rows += len(merged)
        if pause_seconds and index < len(eligible) - 1:
            time.sleep(pause_seconds)             # be polite to an unofficial endpoint

    logger.info(
        "CSE history: %d securities updated, %d new sessions, %d held, %d failed, %d skipped",
        report.updated, report.new_rows, report.total_rows, len(report.failed), report.skipped,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_DIR, help="directory for the per-security parquet files")
    parser.add_argument("--limit", type=int, default=None, help="first N equities only (smoke runs)")
    parser.add_argument("--pause", type=float, default=0.3, help="seconds between requests")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = accumulate_cse_history(args.out, limit=args.limit, pause_seconds=args.pause)
    print(f"updated {report.updated} | new sessions {report.new_rows} | held {report.total_rows} | "
          f"failed {len(report.failed)} | skipped {report.skipped} -> {args.out}")
    for symbol, reason in sorted(report.failed.items())[:10]:
        print(f"  failed {symbol}: {reason}")
    # Nonzero only when nothing could be fetched at all: a few failures are normal and the
    # report names them; a total failure means the endpoint changed or the network is down.
    return 0 if report.updated or not report.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
