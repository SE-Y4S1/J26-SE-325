"""Accumulating daily-bar history for active-set instruments.

Different from data/cache.py's range-keyed cache on purpose. That cache answers "the bars for
exactly [start, end]" for reproducible backtests. This store answers "everything we have
ever seen for this symbol", merged across fetches, because:
  * CSE's chart endpoint only exposes about one year. Merging every fetch is the only way
    history longer than that ever exists locally.
  * An exchange outage then degrades to stale-but-real history (flagged `stale`) instead of
    no answer at all.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from data.cache import CACHE_ROOT, _safe_symbol
from data.http import UpstreamError
from data.ingestion import OHLCV_COLUMNS
from data.instruments import Exchange, Instrument

logger = logging.getLogger(__name__)

HISTORY_ROOT = CACHE_ROOT / "history"

# Daily bars change once a day; refetching more often only spends rate limit.
REFRESH_INTERVAL_SECONDS = 6 * 3600
DEFAULT_LOOKBACK_DAYS = 730


@dataclass(frozen=True)
class History:
    bars: pd.DataFrame
    # True when the refresh failed and these are the last stored bars.
    stale: bool
    error: str | None = None


def history_path(instrument: Instrument):
    directory = HISTORY_ROOT / instrument.exchange.value
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{_safe_symbol(instrument.symbol)}.parquet"


def load_stored(instrument: Instrument) -> pd.DataFrame:
    path = history_path(instrument)
    if not path.exists():
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - corrupt store is a miss, refetch rebuilds it
        logger.warning("corrupt history %s (%s); discarding", path, exc)
        path.unlink(missing_ok=True)
        return pd.DataFrame(columns=OHLCV_COLUMNS)


def merge_bars(stored: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Union on timestamp; a refetched day replaces the stored one (late corrections win)."""
    frames = [frame for frame in (stored, fresh) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    merged = pd.concat(frames, ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    merged = merged.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return merged[OHLCV_COLUMNS].reset_index(drop=True)


def _save(instrument: Instrument, frame: pd.DataFrame) -> None:
    path = history_path(instrument)
    tmp = path.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _fetch(instrument: Instrument, start: date, end: date) -> pd.DataFrame:
    if instrument.exchange is Exchange.CSE:
        from data.exchanges import cse

        if not instrument.provider_id:
            raise UpstreamError("cse", f"no security id for {instrument.symbol}")
        # The endpoint ignores dates and always returns its one-year window.
        return cse.daily_bars(instrument.symbol, instrument.provider_id)

    if instrument.exchange is Exchange.BINANCE:
        from data.exchanges import binance

        return binance.daily_bars(instrument.symbol, start, end)

    from data.exchanges import yahoo

    return yahoo.daily_bars(instrument.symbol, instrument.asset_class, start, end)


def get_history(
    instrument: Instrument,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    force_refresh: bool = False,
    today: date | None = None,
) -> History:
    """Stored bars, topped up from the exchange when they are older than the refresh interval.

    Raises UpstreamError only when the exchange is down AND nothing is stored.
    """
    today = today or date.today()
    stored = load_stored(instrument)
    path = history_path(instrument)

    fresh_enough = (
        not stored.empty
        and path.exists()
        and time.time() - path.stat().st_mtime < REFRESH_INTERVAL_SECONDS
    )
    if fresh_enough and not force_refresh:
        return History(bars=stored, stale=False)

    # Only ask for what is missing (plus a few days of overlap so late corrections land).
    if stored.empty:
        start = today - timedelta(days=lookback_days)
    else:
        start = pd.Timestamp(stored["timestamp"].max()).date() - timedelta(days=5)

    try:
        fresh = _fetch(instrument, start, today)
    except UpstreamError as exc:
        if stored.empty:
            raise
        logger.warning("history refresh for %s failed (%s); serving stored bars", instrument.symbol, exc)
        return History(bars=stored, stale=True, error=str(exc))

    merged = merge_bars(stored, fresh)
    if merged.empty:
        raise UpstreamError(instrument.exchange.value, f"no bars available for {instrument.symbol}")
    _save(instrument, merged)
    return History(bars=merged, stale=False)


# --------------------------------------------------------------------------------------
# Market benchmarks (relative strength)
# --------------------------------------------------------------------------------------

BENCHMARKS = {Exchange.YAHOO: "SPY", Exchange.BINANCE: "BTCUSDT"}  # CSE uses the ASPI index
_benchmark_cache: dict[Exchange, tuple[float, pd.Series]] = {}


def benchmark_close(instrument: Instrument) -> pd.Series | None:
    """Daily closes of the instrument's market benchmark, or None (the benchmark itself, or
    unreachable). Cached in memory for the refresh interval: every held symbol on an
    exchange shares one benchmark."""
    from data.instruments import resolve

    exchange = instrument.exchange
    if BENCHMARKS.get(exchange) == instrument.symbol:
        return None
    cached = _benchmark_cache.get(exchange)
    if cached and time.time() - cached[0] < REFRESH_INTERVAL_SECONDS:
        return cached[1]
    try:
        if exchange is Exchange.CSE:
            from data.exchanges import cse

            series = cse.aspi_daily()
        else:
            bars = get_history(resolve(BENCHMARKS[exchange], exchange)).bars
            series = pd.Series(bars["close"].to_numpy(), index=pd.to_datetime(bars["timestamp"]), name="close")
    except (UpstreamError, LookupError) as exc:
        logger.warning("benchmark for %s unavailable: %s", exchange.value, exc)
        return None
    _benchmark_cache[exchange] = (time.time(), series)
    return series
