"""Phase 2 tests: indicators, sentiment aggregation, and the feature-store join.

FinBERT is mocked everywhere -- loading a 420MB BERT in a unit test would make the suite
unusable. What is tested is the logic around it: the polarity collapse, confidence
weighting, the decay fill, and the join keys.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from data.schema import AssetClass, NewsItem
from features.feature_store import (
    SENTIMENT_DECAY_TAU_DAYS,
    add_targets,
    assert_no_leakage,
    build_feature_table,
    decay_fill_sentiment,
    feature_columns,
)
from features.sentiment import _to_polarity, aggregate_daily
from features.technical import (
    INDICATOR_COLUMNS,
    compute_indicators,
    compute_universe_indicators,
    warmup_periods,
)


def _bars(n: int = 300, *, seed: int = 3, volume: float = 2.0e6, symbol: str = "TEST") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stamps = pd.bdate_range("2022-01-03", periods=n)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    spread = np.abs(rng.normal(0, 0.006, n)) * closes
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "symbol": symbol,
            "asset_class": AssetClass.EQUITY.value,
            "open": closes - spread / 2,
            "high": closes + spread,
            "low": closes - spread,
            "close": closes,
            "volume": np.full(n, volume) if volume else np.zeros(n),
        }
    )


# --------------------------------------------------------------------------------------
# Technical indicators
# --------------------------------------------------------------------------------------

def test_all_taf_named_indicators_are_produced() -> None:
    """MACD, RSI, MFI and ATR are named in the TAF task list -- none may be missing."""
    result = compute_indicators(_bars())
    for required in ("macd", "rsi", "mfi", "atr"):
        assert required in result.columns, f"TAF-named indicator missing: {required}"
        assert result[required].notna().any(), f"{required} is entirely NaN"
    assert list(result.columns) == ["timestamp", *INDICATOR_COLUMNS]


def test_rsi_stays_within_bounds() -> None:
    rsi = compute_indicators(_bars())["rsi"].dropna()
    assert not rsi.empty
    assert rsi.between(0, 100).all()


def test_mfi_is_nan_for_zero_volume_forex_not_zero() -> None:
    """A fake 0 would read as an extreme oversold signal; NaN is the honest answer."""
    result = compute_indicators(_bars(volume=0.0))
    assert result["mfi"].isna().all()
    # Price-only indicators must still work for FX.
    assert result["rsi"].notna().any()
    assert result["atr"].notna().any()


def test_atr_pct_is_scale_invariant() -> None:
    """atr_pct must be comparable across a $5 and a $500 instrument; raw ATR is not."""
    cheap = _bars(seed=9).assign(**{c: lambda d, c=c: d[c] / 50 for c in ("open", "high", "low", "close")})
    expensive = _bars(seed=9)

    cheap_pct = compute_indicators(cheap)["atr_pct"].dropna().mean()
    expensive_pct = compute_indicators(expensive)["atr_pct"].dropna().mean()
    assert cheap_pct == pytest.approx(expensive_pct, rel=0.05)


def test_bb_width_is_positive() -> None:
    width = compute_indicators(_bars())["bb_width"].dropna()
    assert not width.empty
    assert (width > 0).all()


def test_empty_bars_return_empty_indicator_frame() -> None:
    result = compute_indicators(pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]))
    assert result.empty


def test_warmup_periods_covers_the_longest_lookback() -> None:
    """MACD slow(26) + signal(9) is the binding constraint."""
    assert warmup_periods() >= 35


def test_universe_indicators_are_keyed_by_symbol() -> None:
    result = compute_universe_indicators({"AAA": _bars(symbol="AAA"), "BBB": _bars(seed=8, symbol="BBB")})
    assert set(result["symbol"].unique()) == {"AAA", "BBB"}
    assert result.groupby("symbol")["timestamp"].is_monotonic_increasing.all()


# --------------------------------------------------------------------------------------
# Sentiment
# --------------------------------------------------------------------------------------

def test_polarity_collapse_signs_and_confidence() -> None:
    positive = [{"label": "positive", "score": 0.9}, {"label": "negative", "score": 0.05}, {"label": "neutral", "score": 0.05}]
    negative = [{"label": "positive", "score": 0.05}, {"label": "negative", "score": 0.9}, {"label": "neutral", "score": 0.05}]
    neutral = [{"label": "positive", "score": 0.05}, {"label": "negative", "score": 0.05}, {"label": "neutral", "score": 0.9}]

    pol_pos, conf_pos = _to_polarity(positive)
    pol_neg, _ = _to_polarity(negative)
    pol_neu, conf_neu = _to_polarity(neutral)

    assert pol_pos == pytest.approx(0.85)
    assert pol_neg == pytest.approx(-0.85)
    assert pol_neu == pytest.approx(0.0)
    # Confidence separates "decisively neutral" from "decisively positive".
    assert conf_pos > conf_neu


def test_aggregate_daily_weights_by_confidence() -> None:
    """A pile of hedged headlines must not drown out one decisive story."""
    scored = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 3, 1)] * 3,
            "symbol": ["AAPL"] * 3,
            "source": ["gdelt"] * 3,
            "polarity": [0.9, 0.0, 0.0],
            "confidence": [0.95, 0.05, 0.05],
        }
    )
    daily = aggregate_daily(scored)
    assert len(daily) == 1
    assert daily["sentiment_volume"].iloc[0] == 3
    # Unweighted mean would be 0.30; confidence weighting must pull it far higher.
    assert daily["mean_sentiment"].iloc[0] > 0.7


def test_aggregate_daily_computes_momentum_against_trailing_mean() -> None:
    stamps = pd.date_range("2024-03-01", periods=6, freq="D")
    scored = pd.DataFrame(
        {
            "timestamp": stamps,
            "symbol": ["AAPL"] * 6,
            "source": ["gdelt"] * 6,
            "polarity": [0.0, 0.0, 0.0, 0.0, 0.0, 0.9],   # sharp jump on the last day
            "confidence": [1.0] * 6,
        }
    )
    daily = aggregate_daily(scored)
    assert daily["sentiment_momentum"].iloc[-1] > 0.5
    assert daily["sentiment_momentum"].iloc[0] == pytest.approx(0.0)


def test_score_headlines_uses_cache_and_avoids_reloading_model(tmp_path, monkeypatch) -> None:
    """The expensive path must run once; a second call must hit the JSON cache.

    Backend is pinned to finbert (with the pipeline mocked) rather than left on "auto":
    auto-resolution would route to the local Ollama model and make this unit test issue real
    network calls, which is both slow and non-deterministic.
    """
    from features import sentiment as sent_mod

    monkeypatch.setattr(sent_mod, "SCORE_CACHE_PATH", tmp_path / "scores.json")
    monkeypatch.setattr(sent_mod, "finbert_weights_cached", lambda: True)
    sent_mod._model_cache.clear()

    calls: list[int] = []

    def fake_pipe(batch):
        calls.append(len(batch))
        return [
            [{"label": "positive", "score": 0.8}, {"label": "negative", "score": 0.1}, {"label": "neutral", "score": 0.1}]
            for _ in batch
        ]

    items = [
        NewsItem(timestamp=datetime(2024, 3, 1), symbol="AAPL", source="gdelt", headline=f"Headline {i}")
        for i in range(4)
    ]

    with patch.object(sent_mod, "_get_pipeline", return_value=fake_pipe):
        first = sent_mod.score_headlines(items, backend="finbert")
        second = sent_mod.score_headlines(items, backend="finbert")

    assert len(first) == 4
    assert sum(calls) == 4, "second call should have been served entirely from cache"
    pd.testing.assert_frame_equal(first, second)


# --------------------------------------------------------------------------------------
# Feature store
# --------------------------------------------------------------------------------------

def test_decay_fill_decays_toward_neutral_not_zero_fill() -> None:
    """The documented policy: influence fades geometrically, it does not vanish or persist."""
    stamps = pd.bdate_range("2024-01-01", periods=8)
    frame = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 8,
            "timestamp": stamps,
            "mean_sentiment": [0.8] + [np.nan] * 7,
            "sentiment_volume": [3.0] + [np.nan] * 7,
            "sentiment_momentum": [0.1] + [np.nan] * 7,
        }
    )
    filled = decay_fill_sentiment(frame)
    values = filled["mean_sentiment"].to_numpy()

    assert values[0] == pytest.approx(0.8)
    # Strictly decreasing, never negative, and matching exp(-dt/tau).
    assert all(values[i] > values[i + 1] for i in range(len(values) - 1))
    assert values[3] == pytest.approx(0.8 * np.exp(-3 / SENTIMENT_DECAY_TAU_DAYS), rel=1e-6)
    assert values[-1] < 0.1
    # Volume is genuinely zero on a no-news day.
    assert filled["sentiment_volume"].iloc[1:].eq(0.0).all()
    assert filled["days_since_news"].iloc[3] == 3


def test_decay_fill_handles_rows_before_first_ever_headline() -> None:
    """No prior score to decay from -- that is absence of information, not decayed signal."""
    stamps = pd.bdate_range("2024-01-01", periods=5)
    frame = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 5,
            "timestamp": stamps,
            "mean_sentiment": [np.nan, np.nan, 0.5, np.nan, np.nan],
            "sentiment_volume": [np.nan, np.nan, 2.0, np.nan, np.nan],
            "sentiment_momentum": [np.nan] * 5,
        }
    )
    filled = decay_fill_sentiment(frame)
    assert filled["mean_sentiment"].iloc[0] == 0.0
    assert filled["days_since_news"].iloc[0] == -1


def test_decay_fill_does_not_bleed_across_symbols() -> None:
    """AAPL's headline must never decay into MSFT's rows."""
    stamps = pd.bdate_range("2024-01-01", periods=3)
    frame = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 3 + ["MSFT"] * 3,
            "timestamp": list(stamps) * 2,
            "mean_sentiment": [0.9, np.nan, np.nan, np.nan, np.nan, np.nan],
            "sentiment_volume": [1.0, np.nan, np.nan, np.nan, np.nan, np.nan],
            "sentiment_momentum": [np.nan] * 6,
        }
    )
    filled = decay_fill_sentiment(frame)
    msft = filled[filled["symbol"] == "MSFT"]
    assert msft["mean_sentiment"].eq(0.0).all()


def test_build_feature_table_joins_on_symbol_and_timestamp() -> None:
    bars = {"AAA": _bars(symbol="AAA"), "BBB": _bars(seed=4, symbol="BBB")}
    indicators = compute_universe_indicators(bars)

    sentiment = pd.DataFrame(
        {
            "symbol": ["AAA"] * 3,
            "date": pd.bdate_range("2022-06-01", periods=3),
            "mean_sentiment": [0.4, -0.2, 0.1],
            "sentiment_volume": [5, 3, 2],
            "sentiment_momentum": [0.0, -0.3, 0.2],
        }
    )

    table = build_feature_table(bars, indicators, sentiment)

    assert not table.empty
    assert set(table["symbol"].unique()) == {"AAA", "BBB"}
    # Warm-up rows are dropped, not imputed.
    assert table[INDICATOR_COLUMNS].drop(columns=["mfi"], errors="ignore").notna().all().all()
    assert table["mean_sentiment"].notna().all()


def test_feature_columns_excludes_keys_raw_prices_and_targets() -> None:
    bars = {"AAA": _bars(symbol="AAA")}
    table = add_targets(build_feature_table(bars, compute_universe_indicators(bars)), horizon=5)
    columns = feature_columns(table)

    for leaked in ("close", "open", "high", "low", "volume", "symbol", "timestamp",
                   "target_return", "target_volatility"):
        assert leaked not in columns, f"{leaked} must not be a model input"
    assert "rsi" in columns and "macd" in columns


def test_add_targets_is_forward_looking_and_leaves_tail_nan() -> None:
    """The only intentional look-ahead: labels. The last `horizon` rows must be NaN."""
    bars = {"AAA": _bars(symbol="AAA")}
    table = add_targets(build_feature_table(bars, compute_universe_indicators(bars)), horizon=5)

    assert table["target_return"].iloc[-5:].isna().all()
    assert table["target_return"].iloc[:-5].notna().any()
    assert_no_leakage(table)


def test_assert_no_leakage_catches_a_backwards_shift() -> None:
    """The guard must actually fire -- a guard that never triggers proves nothing."""
    bars = {"AAA": _bars(symbol="AAA")}
    table = build_feature_table(bars, compute_universe_indicators(bars))
    # Shift the WRONG way: this makes the target a lagged (already-known) value.
    table["target_return"] = table.groupby("symbol")["close"].transform(lambda s: s.shift(5) / s - 1)

    with pytest.raises(ValueError, match="leaks future data"):
        assert_no_leakage(table)


# --------------------------------------------------------------------------------------
# Sentiment backend selection: local models first, never an implicit download
# --------------------------------------------------------------------------------------

def test_backend_auto_falls_back_to_ollama_when_finbert_is_absent(monkeypatch) -> None:
    """The whole point: an absent FinBERT must route to the model already installed
    locally, not start a ~420MB fetch."""
    from features import sentiment as sent

    monkeypatch.setattr(sent, "finbert_weights_cached", lambda: False)
    monkeypatch.setattr(sent, "ollama_available", lambda model=sent.OLLAMA_MODEL: True)
    assert sent.resolve_backend("auto") == "ollama"


def test_backend_auto_prefers_finbert_when_its_weights_are_present(monkeypatch) -> None:
    """FinBERT is the proposal-named method, so it wins whenever it costs nothing."""
    from features import sentiment as sent

    monkeypatch.setattr(sent, "finbert_weights_cached", lambda: True)
    monkeypatch.setattr(sent, "ollama_available", lambda model=sent.OLLAMA_MODEL: True)
    assert sent.resolve_backend("auto") == "finbert"


def test_backend_auto_raises_clearly_when_neither_is_available(monkeypatch) -> None:
    from features import sentiment as sent

    monkeypatch.setattr(sent, "finbert_weights_cached", lambda: False)
    monkeypatch.setattr(sent, "ollama_available", lambda model=sent.OLLAMA_MODEL: False)

    with pytest.raises(RuntimeError, match="no sentiment backend available"):
        sent.resolve_backend("auto")


def test_explicit_finbert_refuses_rather_than_downloading(monkeypatch) -> None:
    """Asking for FinBERT without its weights must fail fast with the fix, not stall."""
    from features import sentiment as sent

    monkeypatch.setattr(sent, "finbert_weights_cached", lambda: False)
    with pytest.raises(RuntimeError, match="huggingface-cli download"):
        sent.resolve_backend("finbert")


def test_ollama_polarity_matches_the_finbert_contract() -> None:
    """Both backends must emit the same (polarity, confidence) pair, or the feature store
    silently changes meaning depending on which one ran."""
    from unittest.mock import MagicMock

    from features import sentiment as sent

    def fake_client(**_):
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='{"positive": 0.9, "negative": 0.05, "neutral": 0.05}'))]
        )
        return client

    with patch("openai.OpenAI", side_effect=fake_client):
        polarity, confidence = sent._score_one_with_ollama("Great earnings", model="m", base_url="u")

    # Identical to what _to_polarity yields for the same three-way split.
    assert polarity == pytest.approx(0.85, abs=1e-6)
    assert confidence == pytest.approx(0.95, abs=1e-6)


def test_ollama_renormalizes_probabilities_that_do_not_sum_to_one() -> None:
    """Instruction models routinely return three numbers summing to 0.9 or 1.6."""
    from unittest.mock import MagicMock

    from features import sentiment as sent

    def fake_client(**_):
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='{"positive": 1.6, "negative": 0.2, "neutral": 0.2}'))]
        )
        return client

    with patch("openai.OpenAI", side_effect=fake_client):
        polarity, _ = sent._score_one_with_ollama("x", model="m", base_url="u")

    assert -1.0 <= polarity <= 1.0
    assert polarity == pytest.approx((1.6 - 0.2) / 2.0, abs=1e-6)


def test_ollama_malformed_reply_is_neutral_with_zero_confidence() -> None:
    """Zero confidence, not confident-neutral: aggregate_daily weights by confidence, so an
    unparseable score contributes nothing rather than dragging the daily mean toward zero."""
    from unittest.mock import MagicMock

    from features import sentiment as sent

    def fake_client(**_):
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="I think it is quite positive!"))]
        )
        return client

    with patch("openai.OpenAI", side_effect=fake_client):
        polarity, confidence = sent._score_one_with_ollama("x", model="m", base_url="u")

    assert polarity == 0.0
    assert confidence == 0.0
