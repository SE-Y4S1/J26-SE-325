"""Phase 6 tests: the realised-outcome liquidation harness (RQ3/RQ5), extraction accuracy,
and sentiment-precision scoring."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from experiments import rq_liquidation as rq
from optimization.ga_withdrawal import GAConfig

FAST = GAConfig(population_size=30, n_generations=20, stagnation_patience=8)


def panel(n=200) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(2)
    out = {}
    for name, drift, volume in (("THIN.N0000", -0.004, 2e3), ("MID", 0.0, 5e4), ("LIQ", 0.003, 5e6)):
        close = 20 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))
        out[name] = pd.DataFrame({"timestamp": pd.bdate_range("2024-01-01", periods=n), "symbol": name,
                                  "asset_class": "cse_equity", "open": np.nan, "high": close * 1.01,
                                  "low": close * 0.99, "close": close, "volume": np.full(n, volume)})
    return out


def test_scenarios_are_point_in_time() -> None:
    scenarios = rq.sample_scenarios(panel(), 5, random.Random(1))
    assert scenarios
    for scenario in scenarios:
        for symbol, history in scenario["history"].items():
            # History ends at the scenario date; the realised path starts there.
            assert str(pd.Timestamp(history["timestamp"].iloc[-1]).date()) == scenario["date"]
            assert scenario["paths"][symbol][0] == pytest.approx(history["close"].iloc[-1])
            assert len(scenario["paths"][symbol]) == scenario["deadline"] + 1


def test_realised_scoring_uses_actual_prices() -> None:
    from optimization.ga_withdrawal import LiquidationStep, WithdrawalPlan

    scenario = {"paths": {"A": np.array([100.0, 90.0, 80.0])}}
    step = LiquidationStep("A", 1.0, 10.0, 100.0, 0.0, execution_day=2)
    plan = WithdrawalPlan((step,), 1000.0, 1000.0, 0.0, 0.0, {}, 3, True, 0.0, 0)
    assert rq.realised(plan, scenario)["implementation_shortfall"] == pytest.approx(0.2)


def test_ablation_runs_every_configuration() -> None:
    scenarios = rq.sample_scenarios(panel(), 3, random.Random(4))
    results = rq.run(scenarios, ga_config=FAST)
    assert set(results["config"]) == {"pro_rata", "largest_first", "most_liquid_first", "fuzzy_ga",
                                      "fuzzy_ga+forecast", "fuzzy_ga+oracle"}
    # No head installed in tests: the forecast configuration says it used the empirical band.
    assert set(results.loc[results["config"] == "fuzzy_ga+forecast", "forecast_source"]) <= {"empirical", "none",
                                                                                            "empirical+none"}
    summary = rq.summarise(results)
    assert {"all"} <= set(summary["subset"])
    assert summary["shortfall_bps"].notna().all()


def test_ablation_with_a_laya_engine() -> None:
    from decision.engine import LayaEngine
    from tests.test_decision import StubAgent

    scenarios = rq.sample_scenarios(panel(), 2, random.Random(4))
    results = rq.run(scenarios, laya_engine=LayaEngine(StubAgent(), "stub"), ga_config=FAST)
    assert (results["config"] == "laya").sum() == len(scenarios)


def test_extraction_accuracy_on_handchecked_reports() -> None:
    from evaluation.extraction_accuracy import ROOT, evaluate

    table = evaluate(ROOT / "evaluation" / "handchecked" / "extraction.yaml")
    assert (table["field_accuracy"] == 1.0).all()
    assert (table["template_expected"] == table["template_detected"]).all()


def test_precision_scoring(tmp_path) -> None:
    from evaluation.sentiment_precision import score

    path = tmp_path / "sample.csv"
    pd.DataFrame([
        {"symbol": "JKH.N0000", "tier": "company", "label": 1},
        {"symbol": "JKH.N0000", "tier": "sector", "label": 0},
        {"symbol": "COMB.N0000", "tier": "company", "label": 1},
        {"symbol": "COMB.N0000", "tier": "company", "label": ""},
    ]).to_csv(path, index=False)
    table = score(path).set_index("slice")
    assert table.loc["all", "labelled"] == 3
    assert table.loc["all", "precision"] == pytest.approx(2 / 3)
    assert table.loc["tier=company", "precision"] == 1.0
