"""Yahoo Finance adapter: US equities, ETFs and FX via yfinance.

Bars reuse data.ingestion's existing fetch (backoff on yfinance's empty-frame rate limit),
so there is one Yahoo code path, not two. Yahoo has no full instrument list, so the
catalogue for Yahoo is a live search, cached per query.

Yahoo does NOT cover the Colombo Stock Exchange (`JKH.N0000.CM` returns no data), which is
why CSE has its own adapter.
"""

from __future__ import annotations

import logging
from datetime import date
from functools import lru_cache
from typing import Any

import pandas as pd

from data.http import UpstreamError
from data.schema import AssetClass

logger = logging.getLogger(__name__)

PROVIDER = "yahoo"

# Yahoo quoteType -> our asset class. Anything unlisted (mutual funds, indices, futures) is
# not tradeable on the simulated platform and is filtered out of search results.
_QUOTE_TYPES = {
    "EQUITY": AssetClass.EQUITY,
    "ETF": AssetClass.ETF,
    "CURRENCY": AssetClass.FOREX,
}


def asset_class_for(symbol: str, quote_type: str | None = None) -> AssetClass:
    """Asset class from Yahoo's quoteType, falling back to the ticker shape."""
    if quote_type and quote_type.upper() in _QUOTE_TYPES:
        return _QUOTE_TYPES[quote_type.upper()]
    if symbol.endswith("=X"):
        return AssetClass.FOREX
    return AssetClass.EQUITY


def search(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Yahoo's own search. Network failures yield [] -- search is best-effort."""
    try:
        return list(_search_cached(query.strip().lower(), limit))
    except Exception as exc:  # noqa: BLE001 - yfinance raises a zoo of exception types
        logger.info("yahoo search for %r failed: %s", query, exc)
        return []


@lru_cache(maxsize=512)
def _search_cached(query: str, limit: int) -> tuple[dict[str, Any], ...]:
    import yfinance as yf

    quotes = yf.Search(query, max_results=limit).quotes or []
    results = []
    for quote in quotes:
        quote_type = str(quote.get("quoteType", "")).upper()
        symbol = quote.get("symbol")
        if not symbol or quote_type not in _QUOTE_TYPES:
            continue
        results.append(
            {
                "symbol": str(symbol).upper(),
                "name": quote.get("longname") or quote.get("shortname") or symbol,
                "exchange": "yahoo",
                "asset_class": asset_class_for(symbol, quote_type).value,
                # Search does not return currency; resolved lazily by currency_for().
                "currency": None,
                "last_price": None,
                "provider_id": quote.get("exchange"),
            }
        )
    return tuple(results)


# Listing-venue suffix -> trading currency. Search results carry no currency, and a
# Frankfurt listing shown as USD would be off by the EUR/USD rate. Only a display hint:
# quotes use currency_for(), which asks Yahoo.
_SUFFIX_CURRENCY = {
    "DE": "EUR", "F": "EUR", "PA": "EUR", "AS": "EUR", "BR": "EUR", "MI": "EUR", "MC": "EUR",
    "VI": "EUR", "LS": "EUR", "HE": "EUR", "IR": "EUR", "L": "GBP", "SW": "CHF", "TO": "CAD",
    "V": "CAD", "NE": "CAD", "AX": "AUD", "NZ": "NZD", "T": "JPY", "HK": "HKD", "SS": "CNY",
    "SZ": "CNY", "NS": "INR", "BO": "INR", "KS": "KRW", "KQ": "KRW", "SI": "SGD", "TW": "TWD",
    "SA": "BRL", "MX": "MXN", "JO": "ZAR", "ST": "SEK", "OL": "NOK", "CO": "DKK",
}


def currency_from_suffix(symbol: str) -> str:
    """Best guess from the venue suffix ('NVD.DE' -> EUR); USD for bare US tickers."""
    if symbol.endswith("=X"):
        return currency_for(symbol)
    if "." in symbol:
        return _SUFFIX_CURRENCY.get(symbol.rsplit(".", 1)[1].upper(), "USD")
    return "USD"


@lru_cache(maxsize=1024)
def currency_for(symbol: str) -> str:
    """Trading currency of a Yahoo symbol. FX pairs are priced in their quote currency."""
    if symbol.endswith("=X"):
        # 'EURUSD=X' -> USD; the bare 'LKR=X' form is USD-based and priced in LKR.
        core = symbol[:-2]
        return core[3:] if len(core) == 6 else core
    try:
        import yfinance as yf

        currency = yf.Ticker(symbol).fast_info.get("currency")
        if currency:
            return str(currency).upper()
    except Exception as exc:  # noqa: BLE001
        logger.info("currency lookup for %s failed (%s); assuming USD", symbol, exc)
    return "USD"


def daily_bars(symbol: str, asset_class: AssetClass, start: date, end: date) -> pd.DataFrame:
    """Daily bars via the existing yfinance path (with its rate-limit backoff)."""
    from data.ingestion import _fetch_ohlcv_frame

    frame = _fetch_ohlcv_frame(symbol, asset_class, start, end)
    if frame.empty:
        raise UpstreamError(PROVIDER, f"no bars for {symbol} -- rate-limited or delisted")
    return frame
