"""Phase 0 structural checks: the shape of the system is visible and importable.

These pass against stubs. They assert the module tree exists and the documented contract
surface is present -- so a later rename or accidental deletion is caught immediately rather
than at the point some other phase tries to import it.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_MODULES = [
    "data.schema",
    "data.ingestion",
    "data.cache",
    "data.window_selector",
    "features.technical",
    "features.sentiment",
    "features.feature_store",
    "forecasting.base",
    "forecasting.baseline_lstm",
    "forecasting.timesfm_adapter",
    "forecasting.chronos_adapter",
    "forecasting.residual_head",
    "forecasting.hybrid_model",
    "forecasting.finetune_lora",
    "forecasting.model_registry",
    "optimization.objectives",
    "optimization.baseline_meanvariance",
    "optimization.moead_rebalance",
    "optimization.pareto_selection",
    "optimization.fuzzy_withdrawal",
    "optimization.ga_withdrawal",
    "optimization.naive_liquidation",
    "optimization.stress_scenarios",
    "service.contracts",
    "service.events",
    "service.deps",
    "agent.tool_schema",
    "agent.tools",
    "agent.reference_agent",
    "agent.trajectory_generation",
    "evaluation.backtest",
    "evaluation.metrics",
    "evaluation.stress_test_runner",
]


@pytest.mark.parametrize("module_name", EXPECTED_MODULES)
def test_module_imports(module_name: str) -> None:
    """Every planned module exists and imports cleanly."""
    assert importlib.import_module(module_name) is not None


def test_universe_config_is_valid() -> None:
    config = yaml.safe_load((ROOT / "configs" / "universe.yaml").read_text(encoding="utf-8"))

    assert 15 <= len(config["equities"]) <= 20, "universe should stay small; compute budget"
    assert 3 <= len(config["forex"]) <= 5
    assert 5 <= len(config["etfs"]) <= 8

    symbols = [row["symbol"] for group in ("equities", "etfs", "forex") for row in config[group]]
    assert len(symbols) == len(set(symbols)), "duplicate symbol in universe.yaml"

    window = config["window_selection"]
    assert window["min_history_years"] < window["max_history_years"]
    assert 0 < window["participation_cap"] <= 1
    assert window["horizon_menu"] == sorted(window["horizon_menu"])


def test_forex_entries_carry_notional_adv() -> None:
    """yfinance reports FX volume as 0, so FX liquidity must be configured explicitly --
    otherwise every FX position would look infinitely illiquid to the cost model."""
    config = yaml.safe_load((ROOT / "configs" / "universe.yaml").read_text(encoding="utf-8"))
    for pair in config["forex"]:
        value = pair.get("notional_adv_usd")
        # Type first: YAML 1.1 parses `1.0e12` as a str and only `1.0e+12` as a float, so a
        # missing `+` silently yields text that the cost model reads as untradeable.
        assert isinstance(value, (int, float)), (
            f"{pair['symbol']}: notional_adv_usd parsed as {type(value).__name__} "
            f"({value!r}) -- scientific notation in YAML needs an explicit sign, e.g. 1.0e+12"
        )
        assert value > 0, f"{pair['symbol']} missing notional_adv_usd"

    floor = config["window_selection"]["min_adv_usd"]
    assert isinstance(floor, (int, float)), f"min_adv_usd parsed as {type(floor).__name__}"


def test_withdrawal_contract_has_cross_component_fields() -> None:
    """These three fields are consumed by other components and must never be dropped:
    model_version (Component 3 provenance), fuzzy_rule_trace and agent_reasoning_trace
    (Component 4 explanation)."""
    from service.contracts import WithdrawalResponse

    fields = set(WithdrawalResponse.model_fields)
    for required in ("model_version", "fuzzy_rule_trace", "agent_reasoning_trace"):
        assert required in fields, f"WithdrawalResponse lost cross-component field: {required}"

    for required in ("assets_to_sell", "raised_amount", "expected_slippage_pct",
                     "residual_portfolio_weights"):
        assert required in fields, f"WithdrawalResponse missing brief-specified field: {required}"


def test_two_distinct_optimizers_exist() -> None:
    """Guard against the two optimizers being merged 'for simplicity'. The TAF names both
    MOEA/D and a fuzzy genetic algorithm as distinct methods."""
    from optimization import fuzzy_withdrawal, ga_withdrawal, moead_rebalance

    assert hasattr(moead_rebalance, "optimize_allocation")
    assert hasattr(ga_withdrawal, "optimize_withdrawal")
    assert hasattr(fuzzy_withdrawal, "compute_sell_priority")
    assert moead_rebalance.optimize_allocation is not ga_withdrawal.optimize_withdrawal


def test_grounding_tool_name_is_wired() -> None:
    """The tool whose output is the only legitimate source of withdrawal numbers."""
    from agent.tool_schema import GROUNDING_TOOL_NAME
    from agent.tools import TOOL_REGISTRY

    assert GROUNDING_TOOL_NAME == "run_fuzzy_ga_withdrawal"
    assert GROUNDING_TOOL_NAME in TOOL_REGISTRY


def test_artifacts_dir_is_gitignored() -> None:
    """Generated data must never reach the repo."""
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "artifacts/" in gitignore
    assert ".env" in gitignore
