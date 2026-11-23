"""RQ1 compares models, so it must compare them on the SAME rows. The first Colab table did not:
the LSTM scored 15,488 predictions, the foundation models 30,008 and the hybrids 13,310, each over
a different set of (symbol, date) rows, so the gaps between them mixed model quality with which
rows each model happened to forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.alignment import aligned_metrics, align_predictions


def _preds(keys, *, seed: int, bias: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(keys)
    actual = rng.normal(0, 0.02, n)
    return pd.DataFrame({
        "symbol": [k[0] for k in keys], "timestamp": pd.to_datetime([k[1] for k in keys]),
        "target_return": actual,
        "p10": actual * 0 - 0.03 + bias, "p50": actual * 0 + bias, "p90": actual * 0 + 0.03 + bias,
    })


KEYS_A = [("AAA", f"2024-01-{d:02d}") for d in range(1, 21)]
KEYS_B = [("AAA", f"2024-01-{d:02d}") for d in range(5, 26)] + [("BBB", "2024-01-10")]
KEYS_C = [("AAA", f"2024-01-{d:02d}") for d in range(8, 18)]


def test_every_model_is_reduced_to_the_rows_all_of_them_forecast() -> None:
    aligned, n_common = align_predictions({
        "lstm": _preds(KEYS_A, seed=1), "timesfm": _preds(KEYS_B, seed=2), "hybrid": _preds(KEYS_C, seed=3),
    })

    assert n_common == 10                                       # AAA 8..17 is in all three
    assert {len(frame) for frame in aligned.values()} == {10}
    keys = {name: set(zip(f["symbol"], f["timestamp"], strict=True)) for name, f in aligned.items()}
    assert keys["lstm"] == keys["timesfm"] == keys["hybrid"]


def test_rows_without_a_target_or_a_forecast_never_count() -> None:
    a, b = _preds(KEYS_A, seed=1), _preds(KEYS_A, seed=2)
    a.loc[0, "target_return"] = np.nan                           # the final rows of a series have no realised target
    b.loc[1, "p50"] = np.nan
    aligned, n_common = align_predictions({"a": a, "b": b})

    assert n_common == len(KEYS_A) - 2
    assert all(frame["target_return"].notna().all() and frame[["p10", "p50", "p90"]].notna().all().all()
               for frame in aligned.values())


def test_duplicate_rows_are_collapsed_so_a_model_cannot_count_twice() -> None:
    a = _preds(KEYS_A, seed=1)
    doubled = pd.concat([a, a.iloc[:5]], ignore_index=True)
    aligned, n_common = align_predictions({"a": doubled, "b": _preds(KEYS_A, seed=2)})
    assert n_common == len(KEYS_A) and len(aligned["a"]) == len(KEYS_A)


def test_no_overlap_gives_empty_frames_and_zero_not_a_crash() -> None:
    aligned, n_common = align_predictions({
        "a": _preds([("AAA", "2024-01-01")], seed=1), "b": _preds([("BBB", "2024-02-01")], seed=2),
    })
    assert n_common == 0 and all(frame.empty for frame in aligned.values())


def test_metrics_are_computed_on_the_common_rows_and_report_both_counts() -> None:
    table = aligned_metrics({"lstm": _preds(KEYS_A, seed=1), "timesfm": _preds(KEYS_B, seed=2)}).set_index("model")

    assert table.loc["lstm", "n_original"] == len(KEYS_A)
    assert table.loc["timesfm", "n_original"] == len(KEYS_B)
    assert (table["n_common"] == 16).all()                       # AAA 5..20
    assert {"mae", "rmse", "pinball_loss", "coverage_p10", "coverage_p90"} <= set(table.columns)


def test_a_model_that_is_better_on_the_common_rows_ranks_better() -> None:
    """With the row sets equalised, a model closer to the truth must score lower pinball loss."""
    truth = _preds(KEYS_A, seed=1)
    good = truth.copy()
    good["p50"] = truth["target_return"]
    good["p10"], good["p90"] = truth["target_return"] - 0.01, truth["target_return"] + 0.01
    bad = _preds(KEYS_A, seed=1, bias=0.05)
    table = aligned_metrics({"good": good, "bad": bad}).set_index("model")
    assert table.loc["good", "pinball_loss"] < table.loc["bad", "pinball_loss"]


def test_a_single_model_is_returned_as_is() -> None:
    aligned, n_common = align_predictions({"only": _preds(KEYS_A, seed=1)})
    assert n_common == len(KEYS_A) and len(aligned["only"]) == len(KEYS_A)


def test_no_models_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least one"):
        align_predictions({})
