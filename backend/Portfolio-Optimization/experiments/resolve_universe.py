"""Resolve per-symbol training windows against real market data (Phase 1 deliverable).

Writes `configs/resolved_universe.yaml`: for every symbol, the history window and forecast
horizons chosen for it, together with the criterion values that produced them. That file is
committed, because "why does AAPL train on 15 years and META on 13.6?" must be answerable
from the repo rather than from memory.

    uv run python experiments/resolve_universe.py
    uv run python experiments/resolve_universe.py --retries 3

RATE LIMITING
-------------
yfinance signals throttling exactly as it signals a delisting -- an empty frame. The fetch
layer retries with backoff and refuses to cache an empty OHLCV result, so re-running this
script picks up whatever was missed without re-downloading what succeeded. On a slow link a
couple of passes may be needed; symbols that never resolve are reported explicitly rather
than silently dropped.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from data.window_selector import resolve_universe

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("resolve_universe")

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = ROOT / "configs" / "universe.yaml"
RESOLVED = ROOT / "configs" / "resolved_universe.yaml"


def configured_symbols() -> set[str]:
    universe = yaml.safe_load(UNIVERSE.read_text(encoding="utf-8"))
    return {
        entry["symbol"]
        for group in ("equities", "etfs", "forex")
        for entry in universe.get(group, [])
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--as-of", default="2025-12-31", help="resolution date (YYYY-MM-DD)")
    parser.add_argument("--position-value", type=float, default=50_000.0,
                        help="typical position size, which drives the liquidity-based horizon")
    parser.add_argument("--retries", type=int, default=2,
                        help="extra passes to pick up rate-limited symbols")
    args = parser.parse_args()

    expected = configured_symbols()
    resolved: dict = {}

    for attempt in range(1, args.retries + 2):
        resolved = resolve_universe(
            UNIVERSE, RESOLVED,
            as_of=date.fromisoformat(args.as_of),
            typical_position_value=args.position_value,
        )
        missing = expected - set(resolved)
        if not missing:
            break
        if attempt <= args.retries:
            logger.warning("pass %d: %d symbol(s) unresolved %s; retrying",
                           attempt, len(missing), sorted(missing))

    missing = expected - set(resolved)
    print(f"\nresolved {len(resolved)}/{len(expected)} symbols -> {RESOLVED}")
    if missing:
        # Named, not swallowed: a quietly shorter universe would change every downstream
        # result without appearing anywhere in the output.
        print(f"UNRESOLVED (rate-limited or delisted): {sorted(missing)}")
        print("Re-run to retry; empty OHLCV responses are deliberately not cached.")

    _summarize(resolved)
    return 0


def _summarize(resolved: dict) -> None:
    """Print the criteria table -- the evidence that windows are per-symbol, not constant."""
    if not resolved:
        return

    print()
    print(f"{'symbol':10} {'start':12} {'horizons':10} {'regimes':>7} {'acf':>4} {'d2liq':>6} {'thin':>6}")
    for symbol, window in sorted(resolved.items()):
        criteria = window.criteria
        print(
            f"{symbol:10} {window.history_start.isoformat():12} "
            f"{str(list(window.horizons)):10} {criteria.regime_count:7} "
            f"{criteria.acf_decay_lag:4} {criteria.days_to_liquidate:6} "
            f"{str(criteria.below_liquidity_floor):>6}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
