"""The hybrid heads' train/holdout split.

The first Colab run used ONE global cutoff (the 80th percentile of all dates). CSE bars only
exist for about the last year, which starts after that cutoff, so EVERY CSE row landed in the
holdout and none in training: the head never saw a CSE row, its CSE class embedding was never
trained, and it scored 21.7% WORSE than its base on CSE at horizon 5. `global` mode below
reproduces that on purpose; `per-class` is the fix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.train_hybrid_heads import (
    StaleCacheError,
    check_cache,
    holdout_mask,
    split_cutoffs,
    training_rows,
)


def _features() -> pd.DataFrame:
    equity_dates = pd.bdate_range("2020-01-01", "2025-12-31")
    cse_dates = pd.bdate_range("2025-06-02", "2025-12-31")          # a short, recent history
    frames = [
        pd.DataFrame({"timestamp": equity_dates, "symbol": "AAA", "asset_class": "equity"}),
        pd.DataFrame({"timestamp": cse_dates, "symbol": "JKH.N0000", "asset_class": "cse_equity"}),
    ]
    return pd.concat(frames, ignore_index=True)


def test_a_global_cutoff_puts_every_cse_row_in_the_holdout() -> None:
    """The defect, reproduced: this is why the CSE head never trained on CSE."""
    features = _features()
    cutoffs = split_cutoffs(features, mode="global")

    train = training_rows(features, cutoffs)
    assert (train["asset_class"] == "cse_equity").sum() == 0
    assert (train["asset_class"] == "equity").sum() > 0


def test_a_per_class_cutoff_gives_cse_its_own_training_and_holdout_rows() -> None:
    features = _features()
    cutoffs = split_cutoffs(features, mode="per-class")

    train = training_rows(features, cutoffs)
    held = features[holdout_mask(features, cutoffs)]
    cse = features["asset_class"] == "cse_equity"

    assert (train["asset_class"] == "cse_equity").sum() > 0
    assert (held["asset_class"] == "cse_equity").sum() > 0
    share = (held["asset_class"] == "cse_equity").sum() / cse.sum()
    assert 0.15 <= share <= 0.25, f"CSE holdout should be about the last 20% of ITS dates, got {share:.2f}"


def test_each_classs_cutoff_comes_from_its_own_dates() -> None:
    features = _features()
    cutoffs = split_cutoffs(features, mode="per-class")

    for asset_class in ("equity", "cse_equity"):
        own = np.sort(features.loc[features["asset_class"] == asset_class, "timestamp"].unique())
        assert cutoffs[asset_class] == pd.Timestamp(own[int(len(own) * 0.8)])


def test_training_and_holdout_never_overlap_within_a_class() -> None:
    features = _features()
    for mode in ("global", "per-class"):
        cutoffs = split_cutoffs(features, mode=mode)
        train = training_rows(features, cutoffs)
        held = features[holdout_mask(features, cutoffs)]
        assert len(train) + len(held) == len(features), mode
        for asset_class in ("equity", "cse_equity"):
            t = train.loc[train["asset_class"] == asset_class, "timestamp"]
            h = held.loc[held["asset_class"] == asset_class, "timestamp"]
            if len(t) and len(h):
                assert t.max() < h.min(), (mode, asset_class)


def test_the_holdout_mask_works_on_scored_rows_that_carry_only_a_class_and_a_date() -> None:
    cutoffs = {"equity": pd.Timestamp("2025-01-01"), "cse_equity": pd.Timestamp("2025-10-01")}
    scored = pd.DataFrame({
        "asset_class": ["equity", "equity", "cse_equity", "cse_equity"],
        "timestamp": pd.to_datetime(["2024-12-31", "2025-01-01", "2025-09-30", "2025-10-01"]),
    })
    assert holdout_mask(scored, cutoffs).tolist() == [False, True, False, True]


def test_an_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="mode"):
        split_cutoffs(_features(), mode="by-symbol")


def test_a_class_with_a_single_date_still_gets_a_cutoff() -> None:
    features = pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-01"]), "symbol": "X", "asset_class": "etf"})
    assert "etf" in split_cutoffs(features, mode="per-class")


# --------------------------------------------------------------------------------------
# Caches built under different settings must not be reused silently
# --------------------------------------------------------------------------------------

KEY = {"split": "per-class", "min_cse_sessions": 100, "cse_limit": None, "timesfm_generation": "3.0"}


def test_the_first_run_records_its_settings(tmp_path) -> None:
    check_cache(tmp_path, KEY)
    assert (tmp_path / "cache_key.json").exists()
    check_cache(tmp_path, KEY)                                         # the same settings: fine


def test_changed_settings_are_refused_and_the_message_names_what_changed(tmp_path) -> None:
    check_cache(tmp_path, KEY)
    with pytest.raises(StaleCacheError, match="min_cse_sessions"):
        check_cache(tmp_path, {**KEY, "min_cse_sessions": 200})


def test_caches_from_before_settings_were_recorded_are_refused(tmp_path) -> None:
    """The original j26_hybrid folder: bars and features exist, no record of how they were built."""
    (tmp_path / "bars.parquet").write_bytes(b"x")
    with pytest.raises(StaleCacheError, match="before settings were recorded"):
        check_cache(tmp_path, KEY)


def test_an_empty_directory_is_accepted(tmp_path) -> None:
    check_cache(tmp_path / "fresh", KEY)
