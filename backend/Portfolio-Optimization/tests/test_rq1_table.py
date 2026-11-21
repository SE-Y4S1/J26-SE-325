"""The hybrid heads' holdout table, one row per (horizon, asset class).

CSE names are their own class (`cse_equity`) and have at most about a year of daily history, so
their holdout is a few dozen sessions. The table must say so on the row, not leave a reader to
infer it from a bare number.
"""

from __future__ import annotations

import json

import pytest

from evaluation.rq1_table import holdout_table


def _head(root, horizon: int, metrics: dict, **meta) -> None:  # noqa: ANN001
    directory = root / f"h{horizon}"
    directory.mkdir(parents=True)
    (directory / "meta.json").write_text(json.dumps({
        "horizon": horizon, "base_model": "timesfm", "metrics": metrics, **meta,
    }), encoding="utf-8")


METRICS = {
    "pinball_base_all": 0.0112, "pinball_hybrid_all": 0.0094, "holdout_rows_all": 5000,
    "pinball_base_equity": 0.0121, "pinball_hybrid_equity": 0.0102, "holdout_rows_equity": 3000,
    "pinball_base_cse_equity": 0.0200, "pinball_hybrid_cse_equity": 0.0190, "holdout_rows_cse_equity": 48,
}


def test_one_row_per_asset_class_with_the_improvement_computed(tmp_path) -> None:
    _head(tmp_path, 5, METRICS, base_generation="3.0")
    table = holdout_table(tmp_path).set_index("asset_class")

    assert set(table.index) == {"all", "equity", "cse_equity"}
    assert table.loc["all", "improvement_%"] == pytest.approx(100 * (1 - 0.0094 / 0.0112))
    assert table.loc["equity", "rows"] == 3000
    assert (table["horizon"] == 5).all()
    assert (table["base_generation"] == "3.0").all()


def test_a_small_holdout_is_flagged_on_its_row(tmp_path) -> None:
    _head(tmp_path, 5, METRICS)
    table = holdout_table(tmp_path).set_index("asset_class")

    assert table.loc["cse_equity", "note"].startswith("indicative only")
    assert "48" in table.loc["cse_equity", "note"]
    assert table.loc["equity", "note"] == ""


def test_cse_rows_carry_the_one_year_history_caveat_even_when_the_holdout_is_large(tmp_path) -> None:
    big = {**METRICS, "holdout_rows_cse_equity": 5000}
    _head(tmp_path, 5, big)
    note = holdout_table(tmp_path).set_index("asset_class").loc["cse_equity", "note"]

    assert "one year" in note.lower() and not note.startswith("indicative only")


def test_a_head_without_a_recorded_generation_is_labelled_legacy_2_5(tmp_path) -> None:
    _head(tmp_path, 5, METRICS)
    assert (holdout_table(tmp_path)["base_generation"] == "2.5 (legacy, unrecorded)").all()


def test_both_horizons_and_the_smoke_directory_are_handled(tmp_path) -> None:
    _head(tmp_path, 5, METRICS)
    _head(tmp_path, 21, METRICS)
    smoke = tmp_path / "smoke" / "h5"
    smoke.mkdir(parents=True)
    (smoke / "meta.json").write_text(json.dumps({"horizon": 5, "metrics": METRICS}), encoding="utf-8")

    table = holdout_table(tmp_path)
    assert sorted(table["horizon"].unique()) == [5, 21]
    assert len(table) == 6                                  # smoke runs never leak into the table


def test_an_empty_directory_gives_an_empty_table_not_a_crash(tmp_path) -> None:
    assert holdout_table(tmp_path).empty


def test_a_missing_hybrid_metric_is_left_blank_rather_than_zero(tmp_path) -> None:
    metrics = {k: v for k, v in METRICS.items() if k != "pinball_hybrid_equity"}
    _head(tmp_path, 5, metrics)
    row = holdout_table(tmp_path).set_index("asset_class").loc["equity"]
    assert row["improvement_%"] != row["improvement_%"]     # NaN
