"""Deterministic ALLOW / STEP_UP / BLOCK rules.

A language model cannot change these decisions. Given the same score, thresholds,
amount, reputation, rate-limit count, and drift flag, the action is the same.
"""

from __future__ import annotations

from typing import Any

import numpy as np

SEVERITY = {"ALLOW": 0, "STEP_UP": 1, "BLOCK": 2}


def more_severe(left: str, right: str) -> str:
    return left if SEVERITY[left] >= SEVERITY[right] else right


def thresholds_from_legitimate_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    step_up_fpr: float,
    block_fpr: float,
) -> dict[str, float]:
    """Choose cut-points from legitimate validation scores.

    block_fpr is the share of legitimate validation scores allowed to reach BLOCK.
    step_up_fpr is the share allowed to reach STEP_UP or BLOCK. Both are selected
    before the test set is scored.
    """
    if not 0 < block_fpr < step_up_fpr < 1:
        raise ValueError("Require 0 < block_fpr < step_up_fpr < 1.")
    legitimate = np.asarray(scores, dtype=float)[np.asarray(labels) == 0]
    if len(legitimate) == 0:
        raise ValueError("Validation has no legitimate transactions, so thresholds cannot be selected.")
    step_up = float(np.quantile(legitimate, 1.0 - step_up_fpr))
    block = float(np.quantile(legitimate, 1.0 - block_fpr))
    if block < step_up:
        block = step_up
    return {
        "step_up": step_up,
        "block": block,
        "step_up_fpr_budget": float(step_up_fpr),
        "block_fpr_budget": float(block_fpr),
        "legitimate_validation_rows": int(len(legitimate)),
    }


def enforce(
    score: float,
    thresholds: dict[str, float],
    amount: float,
    hard_block_amount: float,
    reputation: float | None,
    reputation_step_up_below: float,
    reputation_block_below: float,
    rate_status: str,
    drift_flag: bool,
    drift_forces_step_up: bool,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if score >= thresholds["block"]:
        decision = "BLOCK"
        reasons.append("score_above_block_threshold")
    elif score >= thresholds["step_up"]:
        decision = "STEP_UP"
        reasons.append("score_above_step_up_threshold")
    else:
        decision = "ALLOW"
        reasons.append("score_below_step_up_threshold")

    if amount >= hard_block_amount:
        decision = more_severe(decision, "BLOCK")
        reasons.append("amount_hard_block")
    if reputation is not None:
        if reputation <= reputation_block_below:
            decision = more_severe(decision, "BLOCK")
            reasons.append("ip_reputation_block")
        elif reputation <= reputation_step_up_below:
            decision = more_severe(decision, "STEP_UP")
            reasons.append("ip_reputation_step_up")
    if rate_status == "block":
        decision = more_severe(decision, "BLOCK")
        reasons.append("rate_limit_block")
    elif rate_status == "step_up":
        decision = more_severe(decision, "STEP_UP")
        reasons.append("rate_limit_step_up")
    if drift_flag and drift_forces_step_up:
        decision = more_severe(decision, "STEP_UP")
        reasons.append("concept_drift_step_up")
    return decision, reasons


def decide_many(scores: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    """Score-only decisions for an offline test set. Other rules are counted separately."""
    decisions = np.full(len(scores), "ALLOW", dtype=object)
    decisions[scores >= thresholds["step_up"]] = "STEP_UP"
    decisions[scores >= thresholds["block"]] = "BLOCK"
    return decisions


def threshold_document(thresholds: dict[str, Any]) -> str:
    return (
        "BLOCK starts at the validation legitimate-score quantile "
        f"1 - {thresholds.get('block_fpr_budget')}. STEP_UP starts at "
        f"1 - {thresholds.get('step_up_fpr_budget')}. "
        f"Selected step_up={thresholds.get('step_up')}, block={thresholds.get('block')}."
    )
