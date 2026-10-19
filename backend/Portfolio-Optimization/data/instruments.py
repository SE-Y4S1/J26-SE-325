"""Instrument catalogue, search and symbol routing across CSE, Binance and Yahoo.

TWO TIERS, DELIBERATELY
  * The CATALOGUE is everything a user may pick: all ~282 CSE securities and all ~3,700
    Binance pairs (refreshed daily, cached to disk), plus Yahoo via live search. It is a
    lookup table only -- nothing heavier than a name and a last price is held for it.
  * The ACTIVE SET (data/active_set.py) is the symbols someone actually holds. Only those
    get history, liquidity metrics, news, forecasts and analysis. Analysis cost therefore
    scales with what users hold, not with the size of three exchanges.

ROUTING
  `JKH.N0000`-shaped symbols are CSE; symbols in the Binance catalogue (or ending in a USD
  stablecoin) are Binance; everything else is Yahoo. An explicit `exchange` always wins,
  because shapes can collide.
"""

from __future__ import annotations

import json
import logging
import re
import time
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from data.cache import CACHE_ROOT
from data.http import UpstreamError
from data.schema import AssetClass

logger = logging.getLogger(__name__)

CATALOGUE_DIR = CACHE_ROOT / "catalogue"
CATALOGUE_TTL_SECONDS = 24 * 3600

# <TICKER>.<issue type><4 digits>: N voting, X non-voting, R rights, W warrants, U units...
CSE_SYMBOL = re.compile(r"^[A-Z0-9&]{1,10}\.[A-Z]\d{4}$")


class Exchange(StrEnum):
    CSE = "cse"
    BINANCE = "binance"
    YAHOO = "yahoo"


class Instrument(BaseModel):
    """One tradeable instrument, as far as the catalogue knows it."""

    symbol: str
    name: str
    exchange: Exchange
    asset_class: AssetClass
    currency: str
    last_price: float | None = None
    # Exchange-internal handle: CSE security id (needed for its chart endpoint), Binance
    # base asset, Yahoo exchange code.
    provider_id: str | None = None


# --------------------------------------------------------------------------------------
# Catalogue (CSE + Binance), cached to disk for a day
# --------------------------------------------------------------------------------------

_memory: dict[str, tuple[float, list[Instrument]]] = {}


def _catalogue_path(exchange: Exchange) -> Path:
    CATALOGUE_DIR.mkdir(parents=True, exist_ok=True)
    return CATALOGUE_DIR / f"{exchange.value}.json"


def _fetch_catalogue(exchange: Exchange) -> list[dict[str, Any]]:
    if exchange is Exchange.CSE:
        from data.exchanges import cse

        return cse.list_securities()
    if exchange is Exchange.BINANCE:
        from data.exchanges import binance

        return binance.list_pairs()
    raise ValueError(f"{exchange} has no enumerable catalogue")


def catalogue(exchange: Exchange, *, refresh: bool = False) -> list[Instrument]:
    """All instruments on an enumerable exchange, from memory, disk, or the network.

    A failed refresh falls back to the last good copy on disk, however old: a day-stale
    price in a lookup list is harmless, an empty search box is not.
    """
    now = time.time()
    cached = _memory.get(exchange.value)
    if cached and not refresh and now - cached[0] < CATALOGUE_TTL_SECONDS:
        return cached[1]

    path = _catalogue_path(exchange)
    on_disk: tuple[float, list[Instrument]] | None = None
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            on_disk = (
                float(payload["fetched_at"]),
                [Instrument(**row) for row in payload["instruments"]],
            )
        except Exception as exc:  # noqa: BLE001 - a corrupt file is a cache miss
            logger.warning("discarding corrupt catalogue %s: %s", path, exc)

    if on_disk and not refresh and now - on_disk[0] < CATALOGUE_TTL_SECONDS:
        _memory[exchange.value] = on_disk
        return on_disk[1]

    try:
        instruments = [Instrument(**row) for row in _fetch_catalogue(exchange)]
    except (UpstreamError, ValueError) as exc:
        if on_disk:
            logger.warning("%s catalogue refresh failed (%s); serving the copy from disk", exchange, exc)
            _memory[exchange.value] = on_disk
            return on_disk[1]
        raise

    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({"fetched_at": now, "instruments": [i.model_dump(mode="json") for i in instruments]}),
        encoding="utf-8",
    )
    tmp.replace(path)
    _memory[exchange.value] = (now, instruments)
    logger.info("%s catalogue refreshed: %d instruments", exchange, len(instruments))
    return instruments


def _catalogue_or_empty(exchange: Exchange) -> list[Instrument]:
    try:
        return catalogue(exchange)
    except Exception as exc:  # noqa: BLE001 - one exchange down must not empty search
        logger.warning("%s catalogue unavailable: %s", exchange, exc)
        return []


# --------------------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------------------

def _score(instrument: Instrument, query: str) -> int:
    """Higher is better; 0 means no match.

    Ranks an exact ticker hit above a prefix above a name match, so 'COMB' finds
    COMB.N0000 first and 'sol' finds SOLUSDT before some token whose name merely contains it.
    """
    symbol = instrument.symbol.upper()
    q = query.upper()
    ticker = symbol.split(".")[0]
    base = (instrument.provider_id or "").upper() if instrument.exchange is Exchange.BINANCE else ticker

    if symbol == q or ticker == q:
        return 100
    if base == q:
        # A bare coin search wants the deepest market: USDT first, then other USD
        # stablecoins, then crypto/fiat-quoted pairs. 'SOL' -> SOLUSDT before SOLBTC.
        if symbol.endswith("USDT"):
            return 92
        return 90 if instrument.currency == "USD" else 80
    if symbol.startswith(q):
        return 70
    if q in instrument.name.upper():
        return 50 if instrument.name.upper().startswith(q) else 40
    return 0


def search(query: str, *, limit: int = 20, include_yahoo: bool = True) -> list[Instrument]:
    """Ranked matches across all three exchanges."""
    query = query.strip()
    if not query:
        return []

    scored: list[tuple[int, Instrument]] = []
    for exchange in (Exchange.CSE, Exchange.BINANCE):
        for instrument in _catalogue_or_empty(exchange):
            score = _score(instrument, query)
            if score:
                scored.append((score, instrument))

    if include_yahoo and len(query) >= 2:
        from data.exchanges import yahoo

        for row in yahoo.search(query, limit=10):
            instrument = Instrument(**{**row, "currency": row["currency"] or yahoo.currency_from_suffix(row["symbol"])})
            # Yahoo already ranked these; place them just under exact catalogue hits.
            scored.append((max(_score(instrument, query), 60), instrument))

    # Ties go to the local exchange first: on a Sri Lankan platform 'COMB' means Commercial
    # Bank of Ceylon, not the US ETF of the same ticker.
    priority = {Exchange.CSE: 0, Exchange.BINANCE: 1, Exchange.YAHOO: 2}
    seen: set[tuple[str, str]] = set()
    results: list[Instrument] = []
    ranked = sorted(scored, key=lambda pair: (-pair[0], priority[pair[1].exchange], pair[1].symbol))
    for _, instrument in ranked:
        key = (instrument.exchange.value, instrument.symbol)
        if key in seen:
            continue
        seen.add(key)
        results.append(instrument)
        if len(results) >= limit:
            break
    return results


# --------------------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------------------

def route(symbol: str) -> Exchange:
    """Which exchange a bare symbol belongs to. Cheap: never touches the network for CSE."""
    symbol = symbol.strip().upper()
    if CSE_SYMBOL.match(symbol):
        return Exchange.CSE

    from data.exchanges import binance

    if any(i.symbol == symbol for i in _memory.get(Exchange.BINANCE.value, (0, []))[1]):
        return Exchange.BINANCE
    split = binance.split_pair(symbol)
    if split and split[1] in binance.USD_STABLECOINS and "=" not in symbol and "." not in symbol:
        return Exchange.BINANCE
    return Exchange.YAHOO


def resolve(symbol: str, exchange: Exchange | str | None = None) -> Instrument:
    """Full Instrument for a symbol, consulting the catalogue and falling back to a lookup.

    Raises LookupError for a symbol no exchange recognises.
    """
    symbol = symbol.strip().upper()
    target = Exchange(exchange) if exchange else route(symbol)

    if target is Exchange.CSE:
        for instrument in _catalogue_or_empty(Exchange.CSE):
            if instrument.symbol == symbol:
                return instrument
        # Not in a (possibly stale) catalogue: ask CSE directly. A new listing must not be
        # unaddable for a day.
        from data.exchanges import cse

        try:
            return Instrument(**cse.company_info(symbol))
        except UpstreamError as exc:
            raise LookupError(f"{symbol} is not a CSE security ({exc})") from None

    if target is Exchange.BINANCE:
        from data.exchanges import binance

        for instrument in _catalogue_or_empty(Exchange.BINANCE):
            if instrument.symbol == symbol:
                return instrument
        split = binance.split_pair(symbol)
        if split is None:
            raise LookupError(f"{symbol} is not a Binance spot pair")
        base, quote = split
        return Instrument(
            symbol=symbol, name=f"{base} / {quote}", exchange=Exchange.BINANCE,
            asset_class=AssetClass.CRYPTO, currency=binance.quote_currency(quote),
            provider_id=base,
        )

    from data.exchanges import yahoo

    for row in yahoo.search(symbol, limit=5):
        if row["symbol"] == symbol:
            return Instrument(**{**row, "currency": yahoo.currency_for(symbol)})
    # Search does not index every ticker (FX pairs especially); the shape is enough to try.
    return Instrument(
        symbol=symbol, name=symbol, exchange=Exchange.YAHOO,
        asset_class=yahoo.asset_class_for(symbol), currency=yahoo.currency_for(symbol),
    )


def native_currency(symbol: str) -> str:
    """Offline guess of the currency a symbol trades in (no network for CSE/Binance/stocks):
    LKR for CSE shapes, the quote asset for Binance pairs (USDT stays USDT so its conversion
    spread is a stablecoin's), the venue suffix's currency for Yahoo."""
    from data.exchanges import binance, yahoo

    exchange = route(symbol)
    if exchange is Exchange.CSE:
        return "LKR"
    if exchange is Exchange.BINANCE:
        split = binance.split_pair(symbol.strip().upper())
        return split[1] if split else "USDT"
    return yahoo.currency_from_suffix(symbol.strip().upper())
