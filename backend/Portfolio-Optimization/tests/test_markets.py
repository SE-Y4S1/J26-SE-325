"""Phase 1 tests: exchange adapters, catalogue/routing, history, FX, liquidity, active set,
and the /market and /analysis endpoints.

Every exchange response is a RECORDED payload from tests/fixtures/markets/ (captured
2026-10-10 from cse.lk and api.binance.com), so the suite never touches the network and a
change in an upstream shape shows up as a fixture diff rather than a flaky test.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from data import active_set, fx, history, instruments
from data.exchanges import binance, cse
from data.http import UpstreamError
from data.instruments import Exchange, Instrument
from data.schema import AssetClass
from features.liquidity import FALLBACK_DAILY_VOLATILITY, compute_liquidity_metrics

FIXTURES = Path(__file__).parent / "fixtures" / "markets"


def load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def cse_bars() -> pd.DataFrame:
    return cse.chart_to_frame("JKH.N0000", load("cse_chart_jkh.json")["chartData"])


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point every on-disk store at a temp dir and clear in-memory caches."""
    monkeypatch.setattr(instruments, "CATALOGUE_DIR", tmp_path / "catalogue")
    monkeypatch.setattr(instruments, "_memory", {})
    monkeypatch.setattr(history, "HISTORY_ROOT", tmp_path / "history")
    monkeypatch.setattr(active_set, "REGISTRY_PATH", tmp_path / "active.sqlite")
    # Phase 1 tests cover the liquidity step only; the news steps (network) have their own
    # tests in test_news.py.
    monkeypatch.setattr(active_set, "PIPELINE", [("liquidity", active_set._history_step)])
    fx.clear_rates()
    yield tmp_path
    fx.clear_rates()


@pytest.fixture
def catalogues(monkeypatch):
    """Serve the recorded CSE and Binance catalogues without the network."""
    def fake_request(provider, method, url, *, params=None, data=None, timeout=30):
        if url.endswith("tradeSummary"):
            return load("cse_trade_summary.json")
        if url.endswith("ticker/price") and not params:
            return load("binance_ticker_price.json")
        raise AssertionError(f"unexpected request {url}")

    monkeypatch.setattr(cse, "request_json", fake_request)
    monkeypatch.setattr(binance, "request_json", fake_request)


# --------------------------------------------------------------------------------------
# CSE adapter
# --------------------------------------------------------------------------------------

def test_cse_catalogue_carries_the_chart_security_id(monkeypatch) -> None:
    monkeypatch.setattr(cse, "request_json", lambda *a, **k: load("cse_trade_summary.json"))
    securities = {s["symbol"]: s for s in cse.list_securities()}

    jkh = securities["JKH.N0000"]
    # tradeSummary's `id` is the stockId the chart endpoint wants (297 for JKH), NOT the
    # beta-info securityId (508). Mixing them up returns an empty chart.
    assert jkh["provider_id"] == "297"
    assert jkh["currency"] == "LKR"
    assert jkh["asset_class"] == "cse_equity"
    assert jkh["last_price"] > 0
    assert {"COMB.N0000", "COMB.X0000"} <= securities.keys()


def test_cse_chart_dates_are_colombo_trading_days(cse_bars) -> None:
    assert len(cse_bars) == 241
    assert cse_bars["timestamp"].is_monotonic_increasing
    assert cse_bars["timestamp"].is_unique
    # CSE stamps bars at Colombo midnight = 18:30 UTC the PREVIOUS day. Converting in UTC
    # would put every bar on a Sunday-to-Thursday calendar; in Colombo time they are all
    # weekdays. This is the test that catches the timezone bug.
    assert (cse_bars["timestamp"].dt.weekday < 5).all()


def test_cse_chart_has_no_fabricated_open(cse_bars) -> None:
    assert cse_bars["open"].isna().all()
    assert (cse_bars["high"] >= cse_bars["close"]).all()
    assert (cse_bars["low"] <= cse_bars["close"]).all()
    assert (cse_bars["volume"] >= 0).all()


def test_cse_bars_validate_as_market_bars(cse_bars) -> None:
    from data.ingestion import frame_to_bars

    bars = frame_to_bars(cse_bars.head(5))
    assert bars[0].open is None
    assert bars[0].asset_class is AssetClass.CSE_EQUITY


def test_cse_unknown_symbol_is_an_upstream_error(monkeypatch) -> None:
    monkeypatch.setattr(cse, "request_json", lambda *a, **k: {"reqSymbolInfo": None})
    with pytest.raises(UpstreamError):
        cse.company_info("NOPE.N0000")


# --------------------------------------------------------------------------------------
# Binance adapter
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("symbol", "expected"),
    [("SOLUSDT", ("SOL", "USDT")), ("BTCFDUSD", ("BTC", "FDUSD")), ("ETHBTC", ("ETH", "BTC")),
     ("BTCTRY", ("BTC", "TRY")), ("USDT", None)],
)
def test_binance_pair_split(symbol, expected) -> None:
    assert binance.split_pair(symbol) == expected


def test_binance_catalogue_currencies(monkeypatch) -> None:
    monkeypatch.setattr(binance, "request_json", lambda *a, **k: load("binance_ticker_price.json"))
    pairs = {p["symbol"]: p for p in binance.list_pairs()}

    assert pairs["SOLUSDT"]["currency"] == "USD"      # stablecoin -> USD
    assert pairs["BTCUSDC"]["currency"] == "USD"
    assert pairs["ETHBTC"]["currency"] == "BTC"       # crypto-quoted keeps its quote asset
    assert pairs["BTCTRY"]["currency"] == "TRY"
    assert pairs["SOLUSDT"]["provider_id"] == "SOL"
    assert all(p["asset_class"] == "crypto" for p in pairs.values())


def test_binance_klines_normalise() -> None:
    frame = binance.klines_to_frame("BTCUSDT", load("binance_klines_btcusdt.json"))
    assert len(frame) == 200
    assert frame["timestamp"].is_unique and frame["timestamp"].is_monotonic_increasing
    assert (frame["open"] > 0).all()
    assert frame["asset_class"].eq("crypto").all()


def test_binance_daily_bars_page_past_the_limit(monkeypatch) -> None:
    rows = load("binance_klines_btcusdt.json")
    calls = []

    def fake_request(provider, method, url, *, params=None, data=None, timeout=30):
        calls.append(params["startTime"])
        # First page is "full" (limit hit), second is a short remainder.
        return rows[:100] if len(calls) == 1 else rows[100:150]

    monkeypatch.setattr(binance, "KLINES_LIMIT", 100)
    monkeypatch.setattr(binance, "request_json", fake_request)
    frame = binance.daily_bars("BTCUSDT", date(2026, 1, 1), date(2026, 10, 10))

    assert len(calls) == 2
    assert calls[1] == rows[99][0] + 1
    assert len(frame) == 150


def test_binance_daily_bars_stop_on_a_repeated_page(monkeypatch) -> None:
    rows = load("binance_klines_btcusdt.json")[:100]
    calls = []

    def stuck(provider, method, url, *, params=None, data=None, timeout=30):
        calls.append(params["startTime"])
        return rows          # always the same full page: the cursor never advances

    monkeypatch.setattr(binance, "KLINES_LIMIT", 100)
    monkeypatch.setattr(binance, "request_json", stuck)
    binance.daily_bars("BTCUSDT", date(2026, 1, 1), date(2026, 10, 10))
    assert len(calls) == 2


# --------------------------------------------------------------------------------------
# Catalogue, search, routing
# --------------------------------------------------------------------------------------

def test_search_ranks_exact_ticker_first(isolated, catalogues) -> None:
    results = instruments.search("COMB", include_yahoo=False)
    assert results[0].symbol in {"COMB.N0000", "COMB.X0000"}
    assert {r.symbol for r in results[:2]} == {"COMB.N0000", "COMB.X0000"}


def test_search_by_company_name(isolated, catalogues) -> None:
    results = instruments.search("keells", include_yahoo=False)
    assert results[0].symbol == "JKH.N0000"


def test_search_prefers_usd_quote_for_bare_coin(isolated, catalogues) -> None:
    results = instruments.search("sol", include_yahoo=False)
    assert results[0].symbol == "SOLUSDT"
    assert {"SOLBTC", "SOLETH"} <= {r.symbol for r in results}


def test_search_prefers_cse_on_a_ticker_tie(isolated, catalogues, monkeypatch) -> None:
    us_comb = {"symbol": "COMB", "name": "GraniteShares ETF", "exchange": "yahoo",
               "asset_class": "etf", "currency": None, "last_price": None, "provider_id": "PCX"}
    monkeypatch.setattr("data.exchanges.yahoo.search", lambda query, limit=10: [us_comb])
    results = instruments.search("COMB")
    assert results[0].exchange is Exchange.CSE


@pytest.mark.parametrize(
    ("symbol", "currency"),
    [("NVDA", "USD"), ("NVD.DE", "EUR"), ("SHOP.TO", "CAD"), ("7203.T", "JPY"), ("BRK.B", "USD")],
)
def test_yahoo_currency_from_listing_suffix(symbol, currency) -> None:
    from data.exchanges.yahoo import currency_from_suffix

    assert currency_from_suffix(symbol) == currency


@pytest.mark.parametrize(
    ("symbol", "exchange"),
    [("JKH.N0000", Exchange.CSE), ("COMB.X0000", Exchange.CSE), ("BTCUSDT", Exchange.BINANCE),
     ("ETHFDUSD", Exchange.BINANCE), ("AAPL", Exchange.YAHOO), ("EURUSD=X", Exchange.YAHOO),
     ("BRK.B", Exchange.YAHOO)],
)
def test_routing_by_symbol_shape(isolated, symbol, exchange) -> None:
    assert instruments.route(symbol) is exchange


def test_resolve_cse_from_catalogue(isolated, catalogues) -> None:
    jkh = instruments.resolve("jkh.n0000")
    assert jkh.exchange is Exchange.CSE
    assert jkh.provider_id == "297"


def test_catalogue_survives_a_failed_refresh(isolated, catalogues, monkeypatch) -> None:
    first = instruments.catalogue(Exchange.CSE)

    def down(*args, **kwargs):
        raise UpstreamError("cse", "down")

    monkeypatch.setattr(cse, "request_json", down)
    again = instruments.catalogue(Exchange.CSE, refresh=True)
    assert [i.symbol for i in again] == [i.symbol for i in first]


# --------------------------------------------------------------------------------------
# History store
# --------------------------------------------------------------------------------------

JKH = Instrument(symbol="JKH.N0000", name="JOHN KEELLS HOLDINGS PLC", exchange=Exchange.CSE,
                 asset_class=AssetClass.CSE_EQUITY, currency="LKR", provider_id="297")


def test_history_accumulates_beyond_one_fetch(isolated, cse_bars, monkeypatch) -> None:
    older, newer = cse_bars.iloc[:150], cse_bars.iloc[100:]
    monkeypatch.setattr(history, "_fetch", lambda *a, **k: older)
    history.get_history(JKH, force_refresh=True)
    monkeypatch.setattr(history, "_fetch", lambda *a, **k: newer)
    result = history.get_history(JKH, force_refresh=True)

    # The second fetch alone covers rows 100+; the merge keeps rows 0-99 from the first.
    assert len(result.bars) == len(cse_bars)
    assert result.bars["timestamp"].is_unique
    assert not result.stale


def test_history_serves_stored_bars_when_exchange_is_down(isolated, cse_bars, monkeypatch) -> None:
    monkeypatch.setattr(history, "_fetch", lambda *a, **k: cse_bars)
    history.get_history(JKH, force_refresh=True)

    def down(*args, **kwargs):
        raise UpstreamError("cse", "maintenance")

    monkeypatch.setattr(history, "_fetch", down)
    result = history.get_history(JKH, force_refresh=True)
    assert result.stale
    assert len(result.bars) == len(cse_bars)


def test_history_raises_when_down_and_nothing_stored(isolated, monkeypatch) -> None:
    def down(*args, **kwargs):
        raise UpstreamError("cse", "maintenance")

    monkeypatch.setattr(history, "_fetch", down)
    with pytest.raises(UpstreamError):
        history.get_history(JKH)


def test_history_does_not_refetch_within_the_interval(isolated, cse_bars, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(history, "_fetch", lambda *a, **k: calls.append(1) or cse_bars)
    history.get_history(JKH)
    history.get_history(JKH)
    assert len(calls) == 1


# --------------------------------------------------------------------------------------
# FX
# --------------------------------------------------------------------------------------

@pytest.fixture
def rates(monkeypatch):
    table = {"USDLKR=X": 300.0, "EURUSD=X": 1.10}

    def fake_close(ticker: str) -> float:
        if ticker not in table:
            raise UpstreamError("yahoo", f"no {ticker}")
        return table[ticker]

    monkeypatch.setattr(fx, "_yahoo_last_close", fake_close)
    monkeypatch.setattr(fx, "_crypto_usd", lambda asset: {"BTC": 80_000.0}[asset])
    fx.clear_rates()
    return table


def test_fx_identity_and_direct(rates) -> None:
    assert fx.get_rate("LKR", "LKR").rate == 1.0
    assert fx.get_rate("USD", "LKR").rate == 300.0


def test_fx_stablecoins_are_usd(rates) -> None:
    assert fx.get_rate("USDT", "LKR").rate == 300.0


def test_fx_crypto_through_usd(rates) -> None:
    assert fx.get_rate("BTC", "LKR").rate == pytest.approx(80_000.0 * 300.0)


def test_fx_crosses_through_usd(rates) -> None:
    # No EURLKR=X in the table: EUR -> USD -> LKR.
    assert fx.get_rate("EUR", "LKR").rate == pytest.approx(1.10 * 300.0)


def test_fx_inverse_pair(rates) -> None:
    # Only USDLKR=X exists; LKR -> USD must invert it rather than fail.
    assert fx.get_rate("LKR", "USD").rate == pytest.approx(1 / 300.0)


# --------------------------------------------------------------------------------------
# Liquidity metrics
# --------------------------------------------------------------------------------------

def test_liquidity_metrics_are_estimated_from_real_bars(cse_bars) -> None:
    metrics = compute_liquidity_metrics(cse_bars)
    assert metrics.estimated
    assert metrics.adv_value > 0
    assert 0.0 <= metrics.volatility_pct <= 1.0
    assert 0.0 < metrics.daily_volatility < 0.2
    assert metrics.daily_volatility != FALLBACK_DAILY_VOLATILITY
    assert metrics.history_days == len(cse_bars)
    # JKH trades every session.
    assert metrics.sessions_per_week == pytest.approx(5.0, abs=0.6)


def test_liquidity_metrics_fall_back_on_short_history(cse_bars) -> None:
    metrics = compute_liquidity_metrics(cse_bars.tail(3))
    assert not metrics.estimated
    assert metrics.daily_volatility == FALLBACK_DAILY_VOLATILITY


def test_liquidity_metrics_reject_empty() -> None:
    with pytest.raises(ValueError):
        compute_liquidity_metrics(pd.DataFrame(columns=["timestamp", "close", "volume"]))


# --------------------------------------------------------------------------------------
# Active set
# --------------------------------------------------------------------------------------

def test_track_then_prepare(isolated, cse_bars, monkeypatch) -> None:
    monkeypatch.setattr(history, "_fetch", lambda *a, **k: cse_bars)

    assert active_set.track([JKH]) == ["JKH.N0000"]
    record = active_set.prepare("JKH.N0000")

    assert record["status"] == active_set.READY
    assert record["steps"] == {"liquidity": "ok"}
    metrics = active_set.metrics_for("JKH.N0000")
    assert metrics is not None and metrics["daily_volatility"] > 0
    # Prepared and fresh: tracking again needs no re-preparation.
    assert active_set.track([JKH]) == []


def test_failed_step_marks_failed_and_retracks(isolated, monkeypatch) -> None:
    def down(*args, **kwargs):
        raise UpstreamError("cse", "maintenance")

    monkeypatch.setattr(history, "_fetch", down)
    active_set.track([JKH])
    record = active_set.prepare("JKH.N0000")

    assert record["status"] == active_set.FAILED
    assert "maintenance" in record["error"]
    assert active_set.track([JKH]) == ["JKH.N0000"]


def test_one_failing_step_is_partial(isolated, cse_bars, monkeypatch) -> None:
    monkeypatch.setattr(history, "_fetch", lambda *a, **k: cse_bars)
    monkeypatch.setattr(
        active_set, "PIPELINE",
        [*active_set.PIPELINE, ("news", lambda instrument: (False, "no channels configured"))],
    )
    active_set.track([JKH])
    record = active_set.prepare("JKH.N0000")
    assert record["status"] == active_set.PARTIAL
    assert record["steps"] == {"liquidity": "ok", "news": "failed"}


def test_prune_drops_idle_symbols(isolated) -> None:
    active_set.track([JKH])
    assert active_set.prune(idle_days=-1) == 1
    assert active_set.status() == []


# --------------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------------

@pytest.fixture
def client(isolated, catalogues, monkeypatch):
    from service import events as events_mod
    from service.api import app
    from service.events import NullProducer

    monkeypatch.setattr(events_mod, "_producer_cache", {"producer": NullProducer()})
    monkeypatch.setattr("data.exchanges.yahoo.search", lambda query, limit=10: [])
    with TestClient(app) as test_client:
        yield test_client


def test_search_endpoint(client) -> None:
    response = client.get("/market/search", params={"q": "keells"})
    assert response.status_code == 200
    first = response.json()["results"][0]
    assert first["symbol"] == "JKH.N0000"
    assert first["exchange"] == "cse"
    assert first["currency"] == "LKR"


def test_quote_endpoint_converts_to_base(client, cse_bars, monkeypatch, rates) -> None:
    monkeypatch.setattr(history, "_fetch", lambda *a, **k: cse_bars)
    response = client.post("/market/quote", json={"symbol": "JKH.N0000", "quantity": 1000})
    assert response.status_code == 200

    body = response.json()
    assert body["currency"] == "LKR" and body["base_currency"] == "LKR"
    assert body["fx_rate_to_base"] == 1.0
    assert body["value_native"] == pytest.approx(1000 * body["price"])
    assert body["avg_daily_volume"] > 0
    assert body["estimated"] is True


def test_quote_endpoint_crypto_in_lkr(client, monkeypatch, rates) -> None:
    frame = binance.klines_to_frame("BTCUSDT", load("binance_klines_btcusdt.json"))
    monkeypatch.setattr(history, "_fetch", lambda *a, **k: frame)
    body = client.post("/market/quote", json={"symbol": "BTCUSDT", "quantity": 0.5}).json()

    assert body["currency"] == "USD"
    assert body["fx_rate_to_base"] == 300.0
    assert body["value_base"] == pytest.approx(body["value_native"] * 300.0)


def test_quote_exchange_down_is_503(client, monkeypatch) -> None:
    def down(*args, **kwargs):
        raise UpstreamError("cse", "maintenance")

    monkeypatch.setattr(history, "_fetch", down)
    response = client.post("/market/quote", json={"symbol": "JKH.N0000"})
    assert response.status_code == 503


def test_track_prepares_in_background_and_rejects_unknowns(client, cse_bars, monkeypatch) -> None:
    monkeypatch.setattr(history, "_fetch", lambda *a, **k: cse_bars)

    def unknown(symbol, exchange=None):
        for instrument in instruments.catalogue(Exchange.CSE):
            if instrument.symbol == symbol:
                return instrument
        raise LookupError(f"{symbol} not found")

    monkeypatch.setattr(instruments, "resolve", unknown)
    response = client.post("/analysis/track", json={"symbols": ["JKH.N0000", "TYPO.N0000"]})
    assert response.status_code == 200
    body = response.json()
    assert "TYPO.N0000" in body["rejected"]

    # TestClient runs background tasks before returning, so preparation has finished.
    status = client.get("/analysis/status", params={"symbols": "JKH.N0000"}).json()["symbols"]
    assert status[0]["status"] == "ready"


def test_withdraw_uses_prepared_volatility(isolated) -> None:
    from service.api import _to_portfolio_state
    from service.contracts import Holding

    holding = Holding(symbol="JKH.N0000", quantity=1000, current_price=18.4, avg_daily_volume=2e8)
    untracked = _to_portfolio_state([holding])["JKH.N0000"]
    assert untracked["daily_volatility"] == 0.0126       # no metrics yet -> old default

    active_set.track([JKH])
    with active_set._db() as connection:
        connection.execute(
            "UPDATE active_symbols SET status='ready', prepared_at=?, metrics=? WHERE symbol=?",
            (datetime.now(timezone.utc).isoformat(),
             json.dumps({"liquidity": {"daily_volatility": 0.021, "volatility_pct": 0.83}}),
             "JKH.N0000"),
        )
    tracked = _to_portfolio_state([holding])["JKH.N0000"]
    assert tracked["daily_volatility"] == 0.021
    assert tracked["volatility_pct"] == 0.83
