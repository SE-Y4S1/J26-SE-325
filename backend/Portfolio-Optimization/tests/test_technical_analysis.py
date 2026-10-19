"""Phase 2c tests: the technical snapshot, the ASPI benchmark parser and the step."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.exchanges import cse
from features.technical import technical_snapshot

MARKETS = Path(__file__).parent / "fixtures" / "markets"


def bars_from(close: np.ndarray, volume: np.ndarray | None = None, *, with_open: bool = False) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=len(close))
    return pd.DataFrame({
        "timestamp": dates, "symbol": "X", "asset_class": "cse_equity",
        "open": close if with_open else np.nan,
        "high": close * 1.01, "low": close * 0.99, "close": close,
        "volume": volume if volume is not None else np.full(len(close), 1000.0),
    })


def test_uptrend_reads_bullish_without_an_open_price() -> None:
    close = np.linspace(100, 200, 260)
    snap = technical_snapshot(bars_from(close))          # open is NaN throughout, as on CSE
    assert snap["trend"] == "bullish"
    assert snap["price_vs_sma_200"] > 0 and snap["drawdown_52w"] == pytest.approx(0.0)
    assert snap["rsi"] > 70
    assert snap["adx"] is not None and snap["adx"] > 25           # a steady trend is a strong one
    assert snap["dist_to_resistance"] == pytest.approx(0.01, abs=1e-3)


def test_downtrend_and_drawdown() -> None:
    close = np.linspace(200, 100, 260)
    snap = technical_snapshot(bars_from(close))
    assert snap["trend"] == "bearish"
    assert snap["drawdown_52w"] == pytest.approx(100 / close[-252:].max() - 1, abs=1e-4)


def test_golden_cross_detected_within_lookback() -> None:
    # Long decline then a sharp rally: the 50-day crosses above the 200-day near the end.
    close = np.concatenate([np.linspace(200, 100, 230), np.linspace(100, 260, 60)])
    crosses = [technical_snapshot(bars_from(close[:n]))["ma_cross_recent"] for n in range(240, 291)]
    assert "golden" in crosses


def test_relative_strength_against_benchmark() -> None:
    close = np.linspace(100, 130, 120)
    bars = bars_from(close)
    flat = pd.Series(1000.0, index=bars["timestamp"])
    snap = technical_snapshot(bars, flat)
    assert snap["benchmark_return_63d"] == pytest.approx(0.0)
    assert snap["relative_strength_63d"] == pytest.approx(snap["return_63d"])
    assert technical_snapshot(bars)["relative_strength_63d"] is None


def test_obv_accumulation_sign() -> None:
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0.2, 1, 80))
    volume = np.where(np.diff(close, prepend=close[0]) > 0, 5000.0, 500.0)   # up days heavier
    assert technical_snapshot(bars_from(close, volume))["obv_trend"] > 0


def test_short_history_gives_none_not_garbage() -> None:
    snap = technical_snapshot(bars_from(np.linspace(10, 11, 15)))
    assert snap["sma_50"] is None and snap["sma_200"] is None and snap["trend"] is None
    assert snap["return_63d"] is None and snap["adx"] is None
    assert technical_snapshot(bars_from(np.array([]))) == {"sessions": 0}


def test_aspi_points_parse_to_colombo_dates() -> None:
    series = cse.index_points_to_series(json.loads((MARKETS / "cse_aspi_chart.json").read_text()))
    assert len(series) > 200
    assert series.index.is_monotonic_increasing and not series.index.duplicated().any()
    assert (series.index.dayofweek < 5).all()          # Colombo dates, not UTC-shifted
    assert series.between(5_000, 50_000).all()


def test_technical_step_uses_stored_bars_and_benchmark(monkeypatch) -> None:
    from data import active_set, history
    from data.instruments import Exchange, Instrument
    from data.schema import AssetClass

    bars = bars_from(np.linspace(10, 20, 120))
    monkeypatch.setattr(history, "load_stored", lambda instrument: bars)
    monkeypatch.setattr(history, "benchmark_close",
                        lambda instrument: pd.Series(np.linspace(1000, 1100, 120), index=bars["timestamp"]))
    jkh = Instrument(symbol="JKH.N0000", name="John Keells", exchange=Exchange.CSE,
                     asset_class=AssetClass.CSE_EQUITY, currency="LKR", provider_id="508")
    ok, block = active_set._technical_step(jkh)
    assert ok and block["trend"] == "bullish"
    assert block["relative_strength_63d"] is not None

    monkeypatch.setattr(history, "load_stored", lambda instrument: bars.iloc[0:0])
    ok, reason = active_set._technical_step(jkh)
    assert not ok and "no stored history" in reason


def test_benchmark_is_none_for_the_benchmark_itself() -> None:
    from data.history import benchmark_close
    from data.instruments import Exchange, Instrument
    from data.schema import AssetClass

    spy = Instrument(symbol="SPY", name="SPDR", exchange=Exchange.YAHOO, asset_class=AssetClass.ETF, currency="USD")
    assert benchmark_close(spy) is None
