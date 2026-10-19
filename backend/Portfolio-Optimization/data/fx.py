"""Currency conversion into the platform's base currency (LKR by default).

Holdings stay in their NATIVE currency -- a CSE share is priced in LKR, a US ETF in USD, a
Binance pair in its quote asset. Conversion happens only where values are summed or
compared across holdings, and every rate carries its timestamp so a plan can say which rate
it used.

SOURCES
  fiat -> fiat     Yahoo FX ('USDLKR=X'), last daily close. Crosses through USD when a
                   direct pair is missing.
  crypto -> fiat   Binance '<ASSET>USDT' price, then USD -> target. USDT is treated as USD
                   (see data/exchanges/binance.py for why).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from data.http import UpstreamError

logger = logging.getLogger(__name__)

# Same env var as service.deps.Settings.base_currency; read directly so the data layer does
# not import the service layer.
BASE_CURRENCY = os.getenv("BASE_CURRENCY", "LKR").strip().upper() or "LKR"
RATE_TTL_SECONDS = 3600

# Quote assets that are crypto, not fiat; converted through their USDT price.
CRYPTO_ASSETS = {"BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "TRX", "DOT"}


@dataclass(frozen=True)
class FxRate:
    from_currency: str
    to_currency: str
    rate: float
    as_of: datetime
    source: str


_rates: dict[tuple[str, str], tuple[float, FxRate]] = {}


def normalise_currency(code: str) -> str:
    """'usdt' -> 'USD'. Stablecoins collapse to USD; everything else is upper-cased."""
    from data.exchanges.binance import USD_STABLECOINS

    code = code.strip().upper()
    return "USD" if code in USD_STABLECOINS else code


def _yahoo_last_close(ticker: str) -> float:
    """Most recent daily close of a Yahoo FX ticker. Isolated so tests can stub it."""
    import yfinance as yf

    history = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=True)
    if history is None or history.empty:
        raise UpstreamError("yahoo", f"no FX quote for {ticker}")
    value = float(history["Close"].dropna().iloc[-1])
    if value <= 0:
        raise UpstreamError("yahoo", f"non-positive FX quote for {ticker}")
    return value


def _crypto_usd(asset: str) -> float:
    from data.exchanges import binance

    return binance.last_price(f"{asset}USDT")


def _compute(from_ccy: str, to_ccy: str) -> tuple[float, str]:
    if from_ccy in CRYPTO_ASSETS:
        usd = _crypto_usd(from_ccy)
        return usd * (1.0 if to_ccy == "USD" else get_rate("USD", to_ccy).rate), "binance+yahoo"
    if to_ccy in CRYPTO_ASSETS:
        return 1.0 / _compute(to_ccy, from_ccy)[0], "binance+yahoo"

    try:
        return _yahoo_last_close(f"{from_ccy}{to_ccy}=X"), "yahoo"
    except UpstreamError:
        if "USD" in (from_ccy, to_ccy):
            # The inverse pair is sometimes the only one Yahoo lists.
            return 1.0 / _yahoo_last_close(f"{to_ccy}{from_ccy}=X"), "yahoo"
        via_usd = get_rate(from_ccy, "USD").rate * get_rate("USD", to_ccy).rate
        return via_usd, "yahoo-cross"


def get_rate(from_currency: str, to_currency: str = BASE_CURRENCY) -> FxRate:
    """Units of `to_currency` per one unit of `from_currency`, cached for an hour.

    Raises UpstreamError when no source can price the pair -- a silent 1.0 would make a USD
    holding look 300x smaller than it is in LKR.
    """
    from_ccy, to_ccy = normalise_currency(from_currency), normalise_currency(to_currency)
    now = time.time()

    if from_ccy == to_ccy:
        return FxRate(from_ccy, to_ccy, 1.0, datetime.now(timezone.utc), "identity")

    cached = _rates.get((from_ccy, to_ccy))
    if cached and now - cached[0] < RATE_TTL_SECONDS:
        return cached[1]

    rate, source = _compute(from_ccy, to_ccy)
    fx = FxRate(from_ccy, to_ccy, rate, datetime.now(timezone.utc), source)
    _rates[(from_ccy, to_ccy)] = (now, fx)
    return fx


def convert(amount: float, from_currency: str, to_currency: str = BASE_CURRENCY) -> float:
    return amount * get_rate(from_currency, to_currency).rate


def clear_rates() -> None:
    """Testing only."""
    _rates.clear()


# Conversion spread paid to bring proceeds into the base currency, as a fraction of the
# amount. Retail USD->LKR conversion runs ~0.5-1% over mid; stablecoin->fiat through a
# local P2P desk ~1%. Overridable because they are assumptions, not quotes.
FX_SPREAD_FIAT = float(os.getenv("FX_SPREAD_FIAT", "0.0075"))
FX_SPREAD_STABLECOIN = float(os.getenv("FX_SPREAD_STABLECOIN", "0.01"))


def conversion_spread(from_currency: str, to_currency: str = BASE_CURRENCY) -> float:
    """Spread for converting `from_currency` proceeds into `to_currency` (0 when equal)."""
    from data.exchanges.binance import USD_STABLECOINS

    raw = from_currency.strip().upper()
    if normalise_currency(raw) == normalise_currency(to_currency):
        return 0.0
    return FX_SPREAD_STABLECOIN if raw in USD_STABLECOINS else FX_SPREAD_FIAT
