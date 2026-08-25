"""Phase 1 tests: schema conformance, cache behaviour, and window selection.

API clients are mocked throughout -- a test suite that hits yfinance or GDELT is slow,
flaky, and rate-limited. The window-selection tests use synthetic series with a KNOWN
property injected (a structural break, a single volatility regime, an illiquid position) so
each criterion is verified against a case where we know the right answer.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from data import cache as cache_mod
from data.ingestion import (
    NEWS_COLUMNS,
    OHLCV_COLUMNS,
    _dedupe_headlines,
    bars_to_frame,
    fetch_ohlcv,
    frame_to_bars,
)
from data.schema import AssetClass, MarketBar, NewsItem
from data.window_selector import (
    acf_decay_lag,
    completeness,
    count_volatility_regimes,
    days_to_liquidate,
    detect_structural_breaks,
    resolve_window,
)

WINDOW_CONFIG = {
    "min_history_years": 3,
    "max_history_years": 15,
    "max_gap_days": 10,
    "min_regimes": 2,
    "min_adv_usd": 5.0e7,
    "horizon_menu": [1, 5, 21],
    "participation_cap": 0.10,
}


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------

def _synthetic_bars(
    n: int = 1500,
    *,
    start: date = date(2018, 1, 1),
    seed: int = 7,
    drift: float = 0.0003,
    vol: float = 0.012,
    volume: float = 5.0e6,
    price: float = 100.0,
) -> pd.DataFrame:
    """A well-behaved daily OHLCV series on business days."""
    rng = np.random.default_rng(seed)
    stamps = pd.bdate_range(start=start, periods=n)
    returns = rng.normal(drift, vol, n)
    closes = price * np.exp(np.cumsum(returns))
    spread = np.abs(rng.normal(0, vol / 2, n)) * closes
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "symbol": "TEST",
            "asset_class": AssetClass.EQUITY.value,
            "open": closes - spread / 2,
            "high": closes + spread,
            "low": closes - spread,
            "close": closes,
            "volume": rng.normal(volume, volume * 0.1, n).clip(min=1.0),
        }
    )


# --------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------

def test_marketbar_rejects_nonpositive_prices() -> None:
    with pytest.raises(ValueError):
        MarketBar(
            timestamp=datetime(2024, 1, 2), symbol="AAPL", asset_class=AssetClass.EQUITY,
            open=0.0, high=1.0, low=1.0, close=1.0, volume=100.0,
        )


def test_marketbar_allows_zero_volume_for_forex() -> None:
    """yfinance reports FX volume as 0; that must validate, not raise."""
    bar = MarketBar(
        timestamp=datetime(2024, 1, 2), symbol="EURUSD=X", asset_class=AssetClass.FOREX,
        open=1.10, high=1.11, low=1.09, close=1.105, volume=0.0,
    )
    assert bar.volume == 0.0
    assert bar.asset_class is AssetClass.FOREX


def test_bars_roundtrip_through_frame() -> None:
    frame = _synthetic_bars(n=50)
    bars = frame_to_bars(frame)
    assert len(bars) == 50
    assert all(isinstance(b, MarketBar) for b in bars)
    assert list(bars_to_frame(bars).columns) == OHLCV_COLUMNS


def test_newsitem_accepts_missing_body() -> None:
    item = NewsItem(
        timestamp=datetime(2024, 5, 1), symbol="AAPL", source="gdelt",
        headline="Apple announces results", body=None,
    )
    assert item.body is None


# --------------------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------------------

@pytest.fixture
def temp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "CACHE_ROOT", tmp_path / "cache")
    return tmp_path / "cache"


def test_cache_key_is_collision_safe_for_unsafe_symbols(temp_cache) -> None:
    """'EUR/USD' and 'EUR=USD' both sanitize to 'EUR_USD'; the hash suffix must separate them."""
    a = cache_mod.cache_key("ohlcv", "EUR/USD", date(2020, 1, 1), date(2020, 2, 1))
    b = cache_mod.cache_key("ohlcv", "EUR=USD", date(2020, 1, 1), date(2020, 2, 1))
    assert a != b


def test_cached_fetch_calls_fetch_fn_once(temp_cache) -> None:
    """The whole point of the cache: a second call must not re-hit the API."""
    calls: list[int] = []

    def fetch_fn() -> pd.DataFrame:
        calls.append(1)
        return _synthetic_bars(n=10)

    args = ("ohlcv", "AAPL", date(2020, 1, 1), date(2020, 1, 15))
    first = cache_mod.cached_fetch(*args, fetch_fn)
    second = cache_mod.cached_fetch(*args, fetch_fn)

    assert len(calls) == 1, "cache miss on the second call"
    pd.testing.assert_frame_equal(first, second)


def test_cached_fetch_respects_force_refresh(temp_cache) -> None:
    calls: list[int] = []

    def fetch_fn() -> pd.DataFrame:
        calls.append(1)
        return _synthetic_bars(n=10)

    args = ("ohlcv", "AAPL", date(2020, 1, 1), date(2020, 1, 15))
    cache_mod.cached_fetch(*args, fetch_fn)
    cache_mod.cached_fetch(*args, fetch_fn, force_refresh=True)
    assert len(calls) == 2


def test_different_date_ranges_do_not_share_a_cache_entry(temp_cache) -> None:
    """A widened range must MISS, not silently return the shorter cached series."""
    def fetch_fn() -> pd.DataFrame:
        return _synthetic_bars(n=10)

    cache_mod.cached_fetch("ohlcv", "AAPL", date(2020, 1, 1), date(2020, 1, 15), fetch_fn)
    assert cache_mod.read_cached("ohlcv", "AAPL", date(2020, 1, 1), date(2020, 6, 30)) is None


def test_corrupt_cache_entry_is_treated_as_a_miss(temp_cache) -> None:
    path = cache_mod.cache_path("ohlcv", "AAPL", date(2020, 1, 1), date(2020, 1, 15))
    path.write_bytes(b"not a parquet file")
    assert cache_mod.read_cached("ohlcv", "AAPL", date(2020, 1, 1), date(2020, 1, 15)) is None
    assert not path.exists(), "poisoned cache entry should be removed"


# --------------------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------------------

def test_fetch_ohlcv_respects_inclusive_end_date(temp_cache) -> None:
    """Our API treats `end` as inclusive; yfinance's is exclusive, so we must pass end+1."""
    captured: dict[str, str] = {}

    def fake_history(**kwargs):
        captured.update(kwargs)
        stamps = pd.bdate_range("2024-01-02", periods=5)
        return pd.DataFrame(
            {
                "Open": np.linspace(100, 104, 5), "High": np.linspace(101, 105, 5),
                "Low": np.linspace(99, 103, 5), "Close": np.linspace(100, 104, 5),
                "Volume": np.full(5, 1e6),
            },
            index=pd.DatetimeIndex(stamps, name="Date"),
        )

    mock_ticker = MagicMock()
    mock_ticker.history.side_effect = fake_history

    with patch.dict("sys.modules", {"yfinance": MagicMock(Ticker=lambda s: mock_ticker)}):
        bars = fetch_ohlcv("AAPL", AssetClass.EQUITY, date(2024, 1, 2), date(2024, 1, 8))

    assert captured["end"] == "2024-01-09", "end date must be advanced by one day"
    assert captured["start"] == "2024-01-02"
    assert len(bars) == 5
    assert all(b.symbol == "AAPL" and b.asset_class is AssetClass.EQUITY for b in bars)


def test_fetch_ohlcv_returns_empty_on_no_data(temp_cache) -> None:
    """A delisted or bad ticker must return [] rather than raising mid-backtest."""
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()

    with patch.dict("sys.modules", {"yfinance": MagicMock(Ticker=lambda s: mock_ticker)}):
        assert fetch_ohlcv("BADTICKER", AssetClass.EQUITY, date(2024, 1, 1), date(2024, 2, 1)) == []


def test_dedupe_headlines_collapses_syndicated_wire_copy() -> None:
    """One press release syndicated verbatim must not inflate sentiment_volume."""
    frame = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 5, 1)] * 4,
            "symbol": ["AAPL"] * 4,
            "source": ["gdelt", "gdelt", "newsapi", "gdelt"],
            "headline": [
                "Apple beats Q2 estimates",
                "APPLE BEATS Q2 ESTIMATES!",   # same story, different casing/punctuation
                "Apple beats Q2 estimates.",
                "Apple announces new buyback",  # genuinely distinct
            ],
            "body": [None] * 4,
        }
    )
    assert len(_dedupe_headlines(frame)) == 2


# --------------------------------------------------------------------------------------
# Window selection
# --------------------------------------------------------------------------------------

def test_detect_structural_breaks_finds_an_injected_regime_shift() -> None:
    """Inject a hard mean shift; CUSUM must find it. A detector that never fires is useless."""
    rng = np.random.default_rng(3)
    n = 800
    stamps = pd.bdate_range("2019-01-01", periods=n)
    calm = rng.normal(0.0002, 0.008, n // 2)
    shifted = rng.normal(0.006, 0.008, n // 2)   # large, sustained mean shift
    returns = pd.Series(np.concatenate([calm, shifted]), index=stamps)

    breaks = detect_structural_breaks(returns)
    assert breaks, "CUSUM failed to detect an injected structural break"


def test_detect_structural_breaks_quiet_on_stationary_series() -> None:
    """No false positives on a clean stationary series."""
    rng = np.random.default_rng(11)
    stamps = pd.bdate_range("2019-01-01", periods=600)
    returns = pd.Series(rng.normal(0.0, 0.01, 600), index=stamps)
    assert len(detect_structural_breaks(returns)) <= 1


def test_count_volatility_regimes_distinguishes_calm_from_mixed() -> None:
    rng = np.random.default_rng(5)
    stamps = pd.bdate_range("2020-01-01", periods=900)

    calm = pd.Series(rng.normal(0, 0.004, 900), index=stamps)
    mixed = pd.Series(
        np.concatenate([rng.normal(0, 0.003, 300), rng.normal(0, 0.02, 300), rng.normal(0, 0.008, 300)]),
        index=stamps,
    )
    assert count_volatility_regimes(mixed) >= 2
    assert count_volatility_regimes(mixed) >= count_volatility_regimes(calm)


def test_acf_decay_lag_detects_strong_autocorrelation() -> None:
    """An AR(1) with high phi must report a lag > 1; white noise must report 1."""
    rng = np.random.default_rng(13)
    n = 1000
    stamps = pd.bdate_range("2020-01-01", periods=n)

    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.85 * ar[t - 1] + rng.normal(0, 0.01)

    assert acf_decay_lag(pd.Series(ar, index=stamps)) > 1
    assert acf_decay_lag(pd.Series(rng.normal(0, 0.01, n), index=stamps)) == 1


@pytest.mark.parametrize(
    "position,adv,cap,expected",
    [
        (100_000, 10_000_000, 0.10, 1),     # trivially liquid: one day
        (5_000_000, 10_000_000, 0.10, 5),   # 5m / (10m * 0.10) = 5 days
        (1_000_000, 0, 0.10, 0),            # no ADV -> undefined, reported as 0
    ],
)
def test_days_to_liquidate(position, adv, cap, expected) -> None:
    assert days_to_liquidate(position, adv, participation_cap=cap) == expected


def test_completeness_flags_a_large_gap() -> None:
    frame = _synthetic_bars(n=400)
    gapped = pd.concat([frame.iloc[:200], frame.iloc[260:]]).reset_index(drop=True)
    ratio, largest_gap = completeness(gapped)
    assert largest_gap > 10
    assert ratio < 1.0


def test_resolve_window_is_bounded_and_records_criteria() -> None:
    bars = _synthetic_bars(n=2000, start=date(2016, 1, 1))
    window = resolve_window(
        "TEST", AssetClass.EQUITY, bars, config=WINDOW_CONFIG, typical_position_value=50_000
    )

    assert window.history_start < window.history_end
    assert window.history_days <= int(WINDOW_CONFIG["max_history_years"] * 365.25) + 1
    assert set(window.horizons).issubset(set(WINDOW_CONFIG["horizon_menu"]))
    assert window.horizons == tuple(sorted(window.horizons))
    # The evidence must be recorded -- that is the whole point of the module.
    assert window.criteria.completeness_ratio > 0
    assert window.criteria.regime_count >= 1
    assert window.criteria.acf_decay_lag >= 1


def test_resolve_window_clamps_history_to_max_years() -> None:
    """A symbol with 25 years of data must be trimmed to max_history_years."""
    bars = _synthetic_bars(n=6300, start=date(2000, 1, 3))
    window = resolve_window(
        "OLD", AssetClass.EQUITY, bars, config=WINDOW_CONFIG, typical_position_value=50_000
    )
    max_days = int(WINDOW_CONFIG["max_history_years"] * 365.25)
    assert window.history_days <= max_days + 1


def test_resolve_window_flags_illiquid_symbol_without_dropping_it() -> None:
    """Illiquid holdings are exactly what the withdrawal module must handle (RQ3/RQ4), so
    they are flagged, never excluded."""
    bars = _synthetic_bars(n=1200, volume=1_000.0, price=5.0)
    window = resolve_window(
        "THIN", AssetClass.EQUITY, bars, config=WINDOW_CONFIG, typical_position_value=50_000
    )
    assert window.criteria.below_liquidity_floor is True
    assert window.symbol == "THIN"


def test_resolve_window_extends_horizon_for_illiquid_position() -> None:
    """The liquidity-driven horizon: a position needing many days to exit must not be
    evaluated only at a 1-day horizon."""
    bars = _synthetic_bars(n=1200, volume=20_000.0, price=10.0)
    window = resolve_window(
        "THIN", AssetClass.EQUITY, bars, config=WINDOW_CONFIG, typical_position_value=5_000_000
    )
    assert window.criteria.days_to_liquidate > 1
    assert max(window.horizons) > 1


def test_resolve_window_raises_on_empty_bars() -> None:
    with pytest.raises(ValueError, match="no bars"):
        resolve_window(
            "EMPTY", AssetClass.EQUITY, pd.DataFrame(columns=OHLCV_COLUMNS),
            config=WINDOW_CONFIG, typical_position_value=50_000,
        )


def test_resolve_universe_writes_criteria_to_disk(tmp_path) -> None:
    """The resolved config must record WHY each window was chosen, not just the dates."""
    import yaml

    from data.window_selector import resolve_universe

    universe = tmp_path / "universe.yaml"
    universe.write_text(
        yaml.safe_dump(
            {
                "equities": [{"symbol": "TEST", "name": "Test Corp"}],
                "etfs": [],
                "forex": [{"symbol": "EURUSD=X", "pair": "EUR/USD", "notional_adv_usd": 1.0e12}],
                "window_selection": WINDOW_CONFIG,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "resolved_universe.yaml"

    def fake_provider(symbol, asset_class, start, end):
        return _synthetic_bars(n=1200, start=date(2019, 1, 1))

    resolved = resolve_universe(
        universe, output, as_of=date(2024, 1, 1), bars_provider=fake_provider
    )

    assert set(resolved) == {"TEST", "EURUSD=X"}
    assert output.exists()

    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    entry = payload["symbols"]["TEST"]
    for field in ("completeness_ratio", "regime_count", "acf_decay_lag",
                  "days_to_liquidate", "median_adv_usd", "break_dates"):
        assert field in entry["criteria"], f"resolved config lost criterion: {field}"
    assert entry["horizons"]


def test_resolve_universe_survives_one_bad_symbol(tmp_path) -> None:
    """One unfetchable symbol must not abort resolution for the other 25."""
    import yaml

    from data.window_selector import resolve_universe

    universe = tmp_path / "universe.yaml"
    universe.write_text(
        yaml.safe_dump(
            {
                "equities": [{"symbol": "GOOD"}, {"symbol": "BAD"}],
                "etfs": [], "forex": [],
                "window_selection": WINDOW_CONFIG,
            }
        ),
        encoding="utf-8",
    )

    def flaky_provider(symbol, asset_class, start, end):
        if symbol == "BAD":
            raise RuntimeError("simulated API failure")
        return _synthetic_bars(n=1200, start=date(2019, 1, 1))

    resolved = resolve_universe(
        universe, tmp_path / "out.yaml", as_of=date(2024, 1, 1), bars_provider=flaky_provider
    )
    assert set(resolved) == {"GOOD"}


# --------------------------------------------------------------------------------------
# Empty results: a fact for news, a failure for prices
# --------------------------------------------------------------------------------------

def test_empty_ohlcv_is_not_cached_so_it_can_be_retried(temp_cache) -> None:
    """yfinance returns an empty frame for a transient rate-limit exactly as it does for a
    delisting. Caching that poisons the entry: the symbol silently vanishes from every later
    run. Observed for real -- a universe resolution lost SPY, XOM, BA, TLT and XLE this way.
    """
    calls: list[int] = []

    def flaky_fetch() -> pd.DataFrame:
        calls.append(1)
        # First call is throttled (empty); the retry succeeds.
        return pd.DataFrame(columns=OHLCV_COLUMNS) if len(calls) == 1 else _synthetic_bars(n=10)

    args = ("ohlcv", "SPY", date(2020, 1, 1), date(2020, 2, 1))

    first = cache_mod.cached_fetch(*args, flaky_fetch, cache_empty=False)
    assert first.empty
    assert cache_mod.read_cached(*args) is None, "an empty OHLCV result must not be cached"

    second = cache_mod.cached_fetch(*args, flaky_fetch, cache_empty=False)
    assert not second.empty, "the retry should have succeeded"
    assert len(calls) == 2
    # Now that real data arrived, it IS cached.
    assert cache_mod.read_cached(*args) is not None


def test_empty_news_is_cached_because_it_is_the_answer(temp_cache) -> None:
    """'No headlines for AAPL that week' is a real result; re-fetching it on every backtest
    fold would hammer a rate-limited API for nothing."""
    calls: list[int] = []

    def fetch_fn() -> pd.DataFrame:
        calls.append(1)
        return pd.DataFrame(columns=NEWS_COLUMNS)

    args = ("news", "AAPL", date(2020, 1, 1), date(2020, 1, 8))

    cache_mod.cached_fetch(*args, fetch_fn)            # cache_empty defaults to True
    cache_mod.cached_fetch(*args, fetch_fn)

    assert len(calls) == 1, "empty news should have been served from cache the second time"


def test_fetch_ohlcv_retries_before_giving_up(temp_cache) -> None:
    """Backoff is what distinguishes a throttle from a delisting."""
    from unittest.mock import MagicMock

    attempts: list[int] = []

    def fake_history(**_):
        attempts.append(1)
        if len(attempts) < 3:
            return pd.DataFrame()          # throttled
        stamps = pd.bdate_range("2024-01-02", periods=4)
        return pd.DataFrame(
            {"Open": [1.0] * 4, "High": [1.1] * 4, "Low": [0.9] * 4,
             "Close": [1.0] * 4, "Volume": [1e6] * 4},
            index=pd.DatetimeIndex(stamps, name="Date"),
        )

    mock_ticker = MagicMock()
    mock_ticker.history.side_effect = fake_history

    with patch.dict("sys.modules", {"yfinance": MagicMock(Ticker=lambda s: mock_ticker)}), \
         patch("time.sleep"):                          # no real backoff delay in tests
        bars = fetch_ohlcv("SPY", AssetClass.ETF, date(2024, 1, 2), date(2024, 1, 8))

    assert len(attempts) == 3, "should have retried through the throttled responses"
    assert len(bars) == 4
