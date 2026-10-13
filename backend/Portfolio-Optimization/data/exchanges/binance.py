"""Binance spot adapter (public market-data endpoints, no API key).

ENDPOINTS USED (verified 2026-10-10)
  ticker/price   every pair's last price -- the catalogue (3,727 pairs, 761 against USDT).
                 Chosen over exchangeInfo, which is >7 MB and timed out on the dev link.
  klines         daily candles, at most 1,000 per call, paged by startTime.

Pairs are quoted in another asset, not a fiat currency. USD stablecoins are treated as USD
(and flagged `pegged`), because converting a USDT price to LKR through a USDT/LKR rate that
nobody publishes would be invented precision. Pairs quoted in BTC/ETH/BNB keep that asset
as their currency and are converted through its USDT price in data/fx.py.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd

from data.http import UpstreamError, request_json
from data.ingestion import OHLCV_COLUMNS
from data.schema import AssetClass

logger = logging.getLogger(__name__)

PROVIDER = "binance"
BASE_URL = "https://api.binance.com/api/v3/"
KLINES_LIMIT = 1000

USD_STABLECOINS = ("USDT", "USDC", "FDUSD", "BUSD", "TUSD", "DAI", "USDP")

# Longest first, so 'FDUSD' is matched before 'USD'-anything and 'USDT' before 'USD'.
QUOTE_ASSETS = tuple(
    sorted(
        (*USD_STABLECOINS, "BTC", "ETH", "BNB", "EUR", "TRY", "BRL", "JPY", "GBP", "AUD",
         "MXN", "ARS", "ZAR", "PLN", "RON", "UAH", "IDR", "COP", "CZK", "SOL", "XRP", "DOGE",
         "TRX", "DOT"),
        key=len,
        reverse=True,
    )
)


def split_pair(symbol: str) -> tuple[str, str] | None:
    """'SOLUSDT' -> ('SOL', 'USDT'). None when no known quote asset matches."""
    symbol = symbol.upper()
    for quote in QUOTE_ASSETS:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return symbol[: -len(quote)], quote
    return None


def quote_currency(quote_asset: str) -> str:
    """The currency a pair's prices are denominated in, for FX conversion."""
    return "USD" if quote_asset in USD_STABLECOINS else quote_asset


def list_pairs() -> list[dict[str, Any]]:
    """Every spot pair with a last price, normalised to the catalogue's row shape."""
    payload = request_json(PROVIDER, "GET", BASE_URL + "ticker/price")
    if not isinstance(payload, list) or not payload:
        raise UpstreamError(PROVIDER, "ticker/price returned no pairs")

    pairs = []
    for row in payload:
        symbol = str(row.get("symbol", "")).upper()
        split = split_pair(symbol)
        if split is None:
            continue
        base, quote = split
        try:
            price = float(row.get("price"))
        except (TypeError, ValueError):
            price = None
        if not price:
            # Delisted pairs linger in ticker/price at 0.0.
            continue
        pairs.append(
            {
                "symbol": symbol,
                "name": f"{base} / {quote}",
                "exchange": "binance",
                "asset_class": AssetClass.CRYPTO.value,
                "currency": quote_currency(quote),
                "last_price": price,
                "provider_id": base,
            }
        )
    return pairs


def daily_bars(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Daily candles for [start, end], paged past the 1,000-candle limit."""
    start_ms = _to_ms(start)
    end_ms = _to_ms(end + timedelta(days=1)) - 1
    rows: list[list[Any]] = []

    while start_ms <= end_ms:
        batch = request_json(
            PROVIDER, "GET", BASE_URL + "klines",
            params={"symbol": symbol.upper(), "interval": "1d", "startTime": start_ms,
                    "endTime": end_ms, "limit": KLINES_LIMIT},
        )
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        next_start = int(batch[-1][0]) + 1
        # Stop on a short page, and also when the cursor fails to advance: a repeated page
        # would otherwise loop forever.
        if len(batch) < KLINES_LIMIT or next_start <= start_ms:
            break
        start_ms = next_start

    return klines_to_frame(symbol, rows)


def klines_to_frame(symbol: str, rows: list[list[Any]]) -> pd.DataFrame:
    """Normalise kline arrays. Separate from the fetch so tests can feed recorded payloads.

    Volume is BASE-asset volume (coins), the analogue of share volume, so ADV is computed the
    same way for every exchange: close x volume.
    """
    if not rows:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    raw = pd.DataFrame([row[:6] for row in rows], columns=["t", "o", "h", "l", "c", "v"])
    frame = pd.DataFrame(
        {
            # Binance days are UTC days; the open time is the session date.
            "timestamp": pd.to_datetime(raw["t"].astype("int64"), unit="ms").dt.normalize(),
            "symbol": symbol.upper(),
            "asset_class": AssetClass.CRYPTO.value,
            "open": pd.to_numeric(raw["o"], errors="coerce"),
            "high": pd.to_numeric(raw["h"], errors="coerce"),
            "low": pd.to_numeric(raw["l"], errors="coerce"),
            "close": pd.to_numeric(raw["c"], errors="coerce"),
            "volume": pd.to_numeric(raw["v"], errors="coerce").fillna(0.0),
        }
    )
    frame = frame.dropna(subset=["high", "low", "close"])
    frame = frame[frame["close"] > 0]
    return frame.drop_duplicates("timestamp", keep="last")[OHLCV_COLUMNS].reset_index(drop=True)


def last_price(symbol: str) -> float:
    """One pair's latest price. Used by data/fx.py to convert crypto-quoted pairs."""
    payload = request_json(PROVIDER, "GET", BASE_URL + "ticker/price", params={"symbol": symbol.upper()})
    try:
        return float(payload["price"])
    except (KeyError, TypeError, ValueError):
        raise UpstreamError(PROVIDER, f"no price for {symbol}") from None


def _to_ms(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)
