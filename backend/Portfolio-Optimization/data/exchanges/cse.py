"""Colombo Stock Exchange adapter.

CSE publishes no official API. The cse.lk front end is a thin client over JSON endpoints at
https://www.cse.lk/api/, which answer unauthenticated form-encoded POSTs. They are
undocumented and could change without notice, so:
  * every response is normalised here and nowhere else, so a change breaks one module;
  * every bar fetched is merged into a local history store (data/history.py), so an outage
    serves stale history instead of nothing, and history accumulates past the one year the
    chart endpoint exposes.

ENDPOINTS USED (verified 2026-10-10)
  tradeSummary              all listed securities with last price -- the catalogue (282)
  companyInfoSummery        one security's quote; also resolves its numeric security id
  companyChartDataByStock   daily bars. `stockId` is the security id from tradeSummary
                            (NOT the beta-info securityId); `period=5` returns ~1 year.
  chartData                 index series. `chartId=1` is the All Share Price Index (ASPI),
                            `period=5` ~1 year of daily closes as {d: ms, v: level}.

DATA CAVEATS
  * No opening price (`o` is always null). Bars carry open=None rather than a fabricated one.
  * `q` is share volume. Only sessions with trades appear, so a thinly traded name's
    zero-volume days are absent rather than zero -- see features/liquidity.py.
  * Timestamps are milliseconds at Asia/Colombo midnight (18:30 UTC the previous day), so
    they must be converted in Colombo time or every bar lands on the wrong date.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from data.http import UpstreamError, request_json
from data.ingestion import OHLCV_COLUMNS
from data.schema import AssetClass

logger = logging.getLogger(__name__)

PROVIDER = "cse"
BASE_URL = "https://www.cse.lk/api/"
CURRENCY = "LKR"
EXCHANGE_TZ = "Asia/Colombo"

# companyChartDataByStock periods. 5 is the longest that returns daily bars; 6+ collapse to
# a handful of points.
DAILY_PERIOD_ONE_YEAR = 5


def _post(endpoint: str, **form: Any) -> Any:
    return request_json(PROVIDER, "POST", BASE_URL + endpoint, data=form or None)


def _first_price(row: dict[str, Any]) -> float | None:
    """Last trade, else today's close, else previous close: a security that has not traded
    today still has a meaningful price, and 0 would read as worthless."""
    for key in ("price", "lastTradedPrice", "closingPrice", "previousClose"):
        value = row.get(key)
        if value:
            return float(value)
    return None


def list_securities() -> list[dict[str, Any]]:
    """Every listed security, normalised to the catalogue's row shape."""
    payload = _post("tradeSummary")
    rows = payload.get("reqTradeSummery") if isinstance(payload, dict) else None
    if not rows:
        raise UpstreamError(PROVIDER, "tradeSummary returned no securities")

    securities = []
    for row in rows:
        symbol = (row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        securities.append(
            {
                "symbol": symbol,
                "name": (row.get("name") or symbol).strip(),
                "exchange": "cse",
                "asset_class": AssetClass.CSE_EQUITY.value,
                "currency": CURRENCY,
                "last_price": _first_price(row),
                "provider_id": str(row["id"]) if row.get("id") is not None else None,
            }
        )
    return securities


def company_info(symbol: str) -> dict[str, Any]:
    """One security's summary. Used to resolve a symbol missing from a stale catalogue."""
    payload = _post("companyInfoSummery", symbol=symbol)
    info = payload.get("reqSymbolInfo") if isinstance(payload, dict) else None
    if not info:
        raise UpstreamError(PROVIDER, f"unknown CSE symbol {symbol!r}")
    return {
        "symbol": str(info.get("symbol") or symbol).upper(),
        "name": (info.get("name") or symbol).strip(),
        "exchange": "cse",
        "asset_class": AssetClass.CSE_EQUITY.value,
        "currency": CURRENCY,
        "last_price": _first_price(info),
        "provider_id": str(info["id"]) if info.get("id") is not None else None,
    }


def daily_bars(symbol: str, security_id: str) -> pd.DataFrame:
    """About one year of daily bars, normalised to OHLCV_COLUMNS (open is NaN)."""
    payload = _post("companyChartDataByStock", stockId=security_id, period=DAILY_PERIOD_ONE_YEAR)
    points = payload.get("chartData") if isinstance(payload, dict) else None
    if not points:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    return chart_to_frame(symbol, points)


def chart_to_frame(symbol: str, points: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalise chart points. Separate from the fetch so tests can feed recorded payloads."""
    raw = pd.DataFrame(points)
    if raw.empty or "p" not in raw or "t" not in raw:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    close = pd.to_numeric(raw["p"], errors="coerce")
    # A session with a single trade can report high/low as null; the trade price is then
    # both, which keeps ATR defined instead of dropping the bar.
    high = pd.to_numeric(raw.get("h"), errors="coerce").fillna(close)
    low = pd.to_numeric(raw.get("l"), errors="coerce").fillna(close)

    stamps = (
        pd.to_datetime(raw["t"], unit="ms", utc=True)
        .dt.tz_convert(EXCHANGE_TZ)
        .dt.tz_localize(None)
        .dt.normalize()
    )

    frame = pd.DataFrame(
        {
            "timestamp": stamps,
            "symbol": symbol,
            "asset_class": AssetClass.CSE_EQUITY.value,
            "open": pd.to_numeric(raw.get("o"), errors="coerce") if "o" in raw else np.nan,
            "high": np.maximum(high, close),
            "low": np.minimum(low, close),
            "close": close,
            "volume": pd.to_numeric(raw.get("q"), errors="coerce").fillna(0.0),
        }
    )
    frame = frame[frame["close"] > 0]
    # Several intraday points can share a date when the endpoint mixes granularities; the
    # last one of the day is the close.
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return frame[OHLCV_COLUMNS].reset_index(drop=True)


ASPI_CHART_ID = 1


def aspi_daily() -> pd.Series:
    """About one year of daily ASPI closes, indexed by Colombo date. The benchmark for CSE
    relative strength."""
    points = _post("chartData", chartId=ASPI_CHART_ID, period=DAILY_PERIOD_ONE_YEAR)
    return index_points_to_series(points if isinstance(points, list) else [])


def index_points_to_series(points: list[dict[str, Any]]) -> pd.Series:
    raw = pd.DataFrame(points)
    if raw.empty or "d" not in raw or "v" not in raw:
        return pd.Series(dtype=float, name="close")
    stamps = (pd.to_datetime(raw["d"], unit="ms", utc=True).dt.tz_convert(EXCHANGE_TZ)
              .dt.tz_localize(None).dt.normalize())
    series = pd.Series(pd.to_numeric(raw["v"], errors="coerce").to_numpy(), index=stamps, name="close")
    series = series[series > 0]
    return series[~series.index.duplicated(keep="last")].sort_index()
