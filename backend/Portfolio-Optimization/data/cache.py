"""Parquet cache for every raw pull.

GDELT and NewsAPI are rate-limited and yfinance throttles aggressively; repeated Phase 7
backtest runs must not re-hit them. Cache keys are content-addressed on
(source, symbol, start, end) so a widened date range misses rather than silently returning
a short series.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "cache"

# Symbols carry characters Windows forbids in filenames: 'EURUSD=X', '^VIX', 'BRK.B'.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_symbol(symbol: str) -> str:
    """Filesystem-safe rendering of a symbol.

    A short hash is appended because the substitution is lossy -- 'EUR/USD' and 'EUR=USD'
    would otherwise collide on the same path and silently serve each other's data.
    """
    cleaned = _UNSAFE.sub("_", symbol)
    digest = hashlib.sha1(symbol.encode("utf-8")).hexdigest()[:6]
    return f"{cleaned}-{digest}"


def cache_key(source: str, symbol: str, start: date, end: date) -> str:
    """Stable key for one (source, symbol, date-range) pull."""
    return f"{_safe_symbol(symbol)}__{start.isoformat()}__{end.isoformat()}"


def cache_path(source: str, symbol: str, start: date, end: date) -> Path:
    """Resolve the parquet path for a key, creating parent directories."""
    directory = CACHE_ROOT / source
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{cache_key(source, symbol, start, end)}.parquet"


def read_cached(source: str, symbol: str, start: date, end: date) -> pd.DataFrame | None:
    """Return the cached frame, or None on miss.

    A corrupt file (interrupted write, partial disk) is treated as a miss and deleted rather
    than raised: the caller can always re-fetch, and a poisoned cache entry that keeps
    raising is far more disruptive than one extra API call.
    """
    path = cache_path(source, symbol, start, end)
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - any read failure means "unusable cache entry"
        logger.warning("Corrupt cache entry %s (%s); discarding", path, exc)
        path.unlink(missing_ok=True)
        return None


def write_cache(source: str, symbol: str, start: date, end: date, frame: pd.DataFrame) -> Path:
    """Persist a frame atomically and return where it landed.

    Written to a temporary sibling then renamed, so an interrupted run cannot leave a
    half-written parquet that later reads as a valid short series.
    """
    path = cache_path(source, symbol, start, end)
    tmp = path.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def cached_fetch(
    source: str,
    symbol: str,
    start: date,
    end: date,
    fetch_fn: Callable[[], pd.DataFrame],
    *,
    force_refresh: bool = False,
    cache_empty: bool = True,
) -> pd.DataFrame:
    """Read-through cache wrapper. `fetch_fn` is only called on a miss.

    `cache_empty` decides whether an empty result is a fact or a failure, and the distinction
    is not cosmetic:

      * News: empty IS the answer. "No headlines for AAPL that week" is real, and caching it
        avoids re-hitting a rate-limited API on every backtest fold. cache_empty=True.
      * OHLCV: a listed instrument always has bars, so empty means the fetch FAILED --
        yfinance returns an empty frame for a transient rate-limit exactly as it does for a
        genuinely delisted ticker. Caching that poisons the entry permanently: the symbol is
        silently dropped from every subsequent run with no way to recover short of clearing
        the cache by hand. cache_empty=False.

    Observed for real: a universe resolution lost SPY, XOM, BA, TLT and XLE to rate-limiting,
    and without this they would have stayed lost.
    """
    if not force_refresh:
        cached = read_cached(source, symbol, start, end)
        if cached is not None:
            logger.debug("cache hit %s/%s %s..%s", source, symbol, start, end)
            return cached

    frame = fetch_fn()

    if frame.empty and not cache_empty:
        logger.warning(
            "%s/%s %s..%s returned no rows; NOT caching so a transient failure can be retried",
            source, symbol, start, end,
        )
        return frame

    write_cache(source, symbol, start, end, frame)
    return frame


def clear_cache(source: str | None = None) -> int:
    """Delete cached parquet files; returns how many were removed. Testing/maintenance only."""
    root = CACHE_ROOT / source if source else CACHE_ROOT
    if not root.exists():
        return 0
    files = list(root.rglob("*.parquet"))
    for path in files:
        path.unlink(missing_ok=True)
    return len(files)
