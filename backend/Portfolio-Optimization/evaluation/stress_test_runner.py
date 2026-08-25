"""RQ4 driver: run the withdrawal module across the stress suite and measure degradation.

Runs the fuzzy GA and every naive baseline through the same scenario ladder, so RQ4 reports
not just "our method degrades" but whether it degrades more GRACEFULLY than the alternatives
-- which is the more interesting and more defensible claim. A method that starts better and
collapses faster is a worse product than one that starts level and holds.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from optimization.naive_liquidation import BASELINES
from optimization.stress_scenarios import (
    ScenarioType,
    StressScenario,
    apply_to_holdings,
    severity_sweep,
    standard_scenario_suite,
)

logger = logging.getLogger(__name__)

METHODS = ("fuzzy_ga", "pro_rata", "largest_first", "most_liquid_first")


def run_scenario(
    holdings: dict[str, dict[str, float]],
    target_amount: float,
    scenario: StressScenario,
    *,
    method: str = "fuzzy_ga",
    urgency: float = 0.5,
    deadline_days: int = 1,
    participation_cap: float = 0.10,
) -> dict[str, float]:
    """One (portfolio, scenario, method) cell of the RQ4 grid."""
    stressed = apply_to_holdings(holdings, scenario)

    if method == "fuzzy_ga":
        from optimization.ga_withdrawal import GAConfig, optimize_withdrawal

        plan = optimize_withdrawal(
            stressed, target_amount=target_amount, withdrawal_urgency=urgency,
            deadline_days=deadline_days, participation_cap=participation_cap,
            config=GAConfig(population_size=60, n_generations=60),
        )
    elif method in BASELINES:
        plan = BASELINES[method](
            stressed, target_amount,
            deadline_days=deadline_days, participation_cap=participation_cap,
        )
    else:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")

    return {
        "scenario": scenario.name,
        "scenario_type": scenario.scenario_type.value,
        "severity": scenario.severity,
        "method": method,
        "realized_loss": float(plan.expected_realized_loss),
        "slippage_pct": float(plan.expected_slippage_pct),
        "raised_amount": float(plan.raised_amount),
        "shortfall": float(plan.shortfall),
        "days_required": int(plan.days_required),
        "feasible": bool(plan.feasible),
        # Total cost of the outcome: what was lost to execution PLUS what was never raised.
        # Reporting loss alone would flatter a method that simply gives up early and
        # therefore incurs no slippage.
        "total_cost": float(plan.expected_realized_loss + plan.shortfall),
    }


def run_severity_sweep(
    holdings: dict[str, dict[str, float]],
    target_amount: float,
    scenario_type: ScenarioType,
    *,
    methods: tuple[str, ...] = METHODS,
    n_steps: int = 10,
    urgency: float = 0.5,
    deadline_days: int = 1,
) -> pd.DataFrame:
    """Sweep severity for one scenario family across all methods."""
    rows = []
    for scenario in severity_sweep(scenario_type, n_steps=n_steps):
        for method in methods:
            try:
                rows.append(
                    run_scenario(
                        holdings, target_amount, scenario,
                        method=method, urgency=urgency, deadline_days=deadline_days,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one cell must not kill the grid
                logger.error("%s at severity %.2f failed: %s", method, scenario.severity, exc)

    return pd.DataFrame(rows)


def run_full_suite(
    holdings: dict[str, dict[str, float]],
    target_amount: float,
    *,
    methods: tuple[str, ...] = METHODS,
    urgency: float = 0.5,
    deadline_days: int = 1,
    log_to_mlflow: bool = True,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Every scenario family x severity x method. The complete RQ4 result table."""
    rows = []
    for scenario in standard_scenario_suite():
        for method in methods:
            try:
                rows.append(
                    run_scenario(
                        holdings, target_amount, scenario,
                        method=method, urgency=urgency, deadline_days=deadline_days,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("%s on %s failed: %s", method, scenario.name, exc)

    frame = pd.DataFrame(rows)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        logger.info("wrote RQ4 results to %s", output_path)

    if log_to_mlflow and not frame.empty:
        _log_suite(frame, target_amount, urgency, deadline_days)

    return frame


def summarize_degradation(results: pd.DataFrame) -> pd.DataFrame:
    """Per-method degradation summary: baseline cost, worst-case cost, and breakdown point.

    The breakdown point -- the lowest severity at which a method first becomes infeasible --
    is the "where does it break down" half of RQ4. A method with a higher breakdown severity
    is more robust even if its unstressed cost is marginally worse.
    """
    if results.empty:
        return results

    summary = []
    for method, group in results.groupby("method", sort=False):
        baseline = group[group["severity"] == 0.0]
        worst = group.loc[group["severity"].idxmax()] if len(group) else None
        infeasible = group[~group["feasible"].astype(bool)]

        summary.append(
            {
                "method": method,
                "baseline_cost": float(baseline["total_cost"].mean()) if len(baseline) else float("nan"),
                "worst_case_cost": float(worst["total_cost"]) if worst is not None else float("nan"),
                "mean_cost": float(group["total_cost"].mean()),
                "breakdown_severity": float(infeasible["severity"].min()) if len(infeasible) else None,
                "n_infeasible": int(len(infeasible)),
                "n_scenarios": int(len(group)),
            }
        )

    frame = pd.DataFrame(summary)
    if "baseline_cost" in frame and "worst_case_cost" in frame:
        frame["degradation_ratio"] = frame["worst_case_cost"] / frame["baseline_cost"].replace(0, float("nan"))
    return frame.sort_values("mean_cost").reset_index(drop=True)


def _log_suite(results: pd.DataFrame, target: float, urgency: float, deadline_days: int) -> None:
    """Log the RQ4 grid to MLflow. Never fatal."""
    try:
        import mlflow

        mlflow.set_tracking_uri("sqlite:///artifacts/mlflow.db")
        mlflow.set_experiment("rq4_stress_robustness")

        with mlflow.start_run(run_name="stress_suite"):
            mlflow.log_params(
                {"target_amount": target, "urgency": urgency, "deadline_days": deadline_days,
                 "n_cells": len(results)}
            )
            summary = summarize_degradation(results)
            for row in summary.itertuples():
                mlflow.log_metric(f"{row.method}_mean_cost", row.mean_cost)
                mlflow.log_metric(f"{row.method}_worst_case_cost", row.worst_case_cost)
                if row.breakdown_severity is not None:
                    mlflow.log_metric(f"{row.method}_breakdown_severity", row.breakdown_severity)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow stress-suite logging failed: %s", exc)
