"""The CSE chart endpoint returns about one year of daily bars and nothing longer (every period
setting 1-10 was probed: 5 is the longest daily one, 6+ collapse to a handful of points). The
only way to more history is to keep what each fetch returns. These tests pin that accumulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.ingestion import OHLCV_COLUMNS


def _bars(symbol: str, start: str, periods: int, price: float = 100.0) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=periods)
    return pd.DataFrame({
        "timestamp": dates, "symbol": symbol, "asset_class": "cse_equity",
        "open": np.nan, "high": price + 1, "low": price - 1,
        "close": price + np.arange(periods) * 0.1, "volume": 1000.0,
    })[OHLCV_COLUMNS]


@pytest.fixture
def fake_cse(monkeypatch):
    """A CSE whose one-year window can be moved between runs."""
    state = {
        "securities": [
            {"symbol": "AAA.N0000", "provider_id": "1"},
            {"symbol": "BBB.N0000", "provider_id": "2"},
            {"symbol": "WARR.W0001", "provider_id": "3"},       # a warrant: not an equity
            {"symbol": "NOID.N0000", "provider_id": None},      # no security id: cannot be fetched
        ],
        "start": "2025-01-01", "periods": 250, "fail": set(),
    }

    def list_securities():
        return state["securities"]

    def daily_bars(symbol, provider_id):  # noqa: ANN001
        if symbol in state["fail"]:
            raise RuntimeError("upstream timeout")
        return _bars(symbol, state["start"], state["periods"])

    import data.exchanges.cse as cse

    monkeypatch.setattr(cse, "list_securities", list_securities)
    monkeypatch.setattr(cse, "daily_bars", daily_bars)
    return state


def test_merge_keeps_the_union_and_lets_a_refetched_day_win() -> None:
    from data.cse_history import merge_history

    stored = _bars("AAA.N0000", "2025-01-01", 10, price=100.0)
    fresh = _bars("AAA.N0000", "2025-01-08", 10, price=200.0)         # overlaps the last 5 sessions
    merged = merge_history(stored, fresh)

    assert len(merged) == 15                                          # 10 + 10 - 5 overlap
    assert merged["timestamp"].is_monotonic_increasing
    overlap_day = pd.Timestamp("2025-01-08")
    assert merged.loc[merged["timestamp"] == overlap_day, "close"].iloc[0] == pytest.approx(200.0)


def test_one_file_per_equity_and_non_equities_are_skipped(tmp_path, fake_cse) -> None:
    from data.cse_history import accumulate_cse_history, load_accumulated

    report = accumulate_cse_history(tmp_path, pause_seconds=0)

    assert report.updated == 2 and not report.failed
    assert load_accumulated(tmp_path, "AAA.N0000") is not None
    assert load_accumulated(tmp_path, "BBB.N0000") is not None
    assert load_accumulated(tmp_path, "WARR.W0001") is None
    assert load_accumulated(tmp_path, "NOID.N0000") is None
    assert len(load_accumulated(tmp_path, "AAA.N0000")) == 250


def test_a_later_run_extends_history_past_the_one_year_window(tmp_path, fake_cse) -> None:
    from data.cse_history import accumulate_cse_history, load_accumulated

    accumulate_cse_history(tmp_path, pause_seconds=0)
    first = load_accumulated(tmp_path, "AAA.N0000")

    fake_cse["start"] = "2025-09-01"                                  # the window moved ~8 months on
    report = accumulate_cse_history(tmp_path, pause_seconds=0)
    second = load_accumulated(tmp_path, "AAA.N0000")

    assert len(second) > len(first) > 0
    assert len(second) > fake_cse["periods"], "history must exceed a single one-year window"
    assert second["timestamp"].min() == first["timestamp"].min()      # nothing earlier was lost
    assert report.new_rows > 0


def test_running_twice_on_the_same_window_adds_nothing(tmp_path, fake_cse) -> None:
    from data.cse_history import accumulate_cse_history

    accumulate_cse_history(tmp_path, pause_seconds=0)
    report = accumulate_cse_history(tmp_path, pause_seconds=0)
    assert report.new_rows == 0


def test_one_failing_security_does_not_stop_the_run_and_keeps_its_old_history(tmp_path, fake_cse) -> None:
    from data.cse_history import accumulate_cse_history, load_accumulated

    accumulate_cse_history(tmp_path, pause_seconds=0)
    before = len(load_accumulated(tmp_path, "AAA.N0000"))

    fake_cse["fail"] = {"AAA.N0000"}
    fake_cse["start"] = "2025-09-01"
    report = accumulate_cse_history(tmp_path, pause_seconds=0)

    assert "AAA.N0000" in report.failed and "upstream timeout" in report.failed["AAA.N0000"]
    assert report.updated == 1                                        # BBB still advanced
    assert len(load_accumulated(tmp_path, "AAA.N0000")) == before     # untouched, not truncated


def test_limit_caps_how_many_securities_are_fetched(tmp_path, fake_cse) -> None:
    from data.cse_history import accumulate_cse_history

    assert accumulate_cse_history(tmp_path, limit=1, pause_seconds=0).updated == 1


def test_an_empty_response_is_not_stored_as_history(tmp_path, fake_cse) -> None:
    from data.cse_history import accumulate_cse_history, load_accumulated

    fake_cse["periods"] = 0
    report = accumulate_cse_history(tmp_path, pause_seconds=0)
    assert report.updated == 0
    assert load_accumulated(tmp_path, "AAA.N0000") is None


def test_the_trainer_merges_accumulated_history_into_what_the_endpoint_returns(tmp_path, fake_cse) -> None:
    """collect_bars must use the accumulated store, or the whole exercise never reaches training."""
    from data.cse_history import accumulate_cse_history
    from experiments.train_hybrid_heads import _cse_bars

    accumulate_cse_history(tmp_path, pause_seconds=0)                 # window 1: 2025-01 onward
    fake_cse["start"] = "2025-09-01"                                  # the endpoint now only reaches back to Sept

    security = {"symbol": "AAA.N0000", "provider_id": "1"}
    with_store = _cse_bars(security, tmp_path)
    without_store = _cse_bars(security, None)

    assert len(without_store) == 250
    assert len(with_store) > len(without_store)
    assert with_store["timestamp"].min() < without_store["timestamp"].min()


def test_the_trainer_falls_back_to_stored_history_when_the_endpoint_is_down(tmp_path, fake_cse) -> None:
    from data.cse_history import accumulate_cse_history
    from experiments.train_hybrid_heads import _cse_bars

    accumulate_cse_history(tmp_path, pause_seconds=0)
    fake_cse["fail"] = {"AAA.N0000"}
    security = {"symbol": "AAA.N0000", "provider_id": "1"}

    assert len(_cse_bars(security, tmp_path)) == 250                  # served from the store
    with pytest.raises(RuntimeError):
        _cse_bars(security, None)                                     # no store: the error stands
