"""Robustness checks on already labelled rows.

These functions copy a feature matrix and rescore it. They do not connect to a
payment network and they do not change training data.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from fraud_engine.evaluation.metrics import safe_auroc
from fraud_engine.gateway.policy import decide_many


def camouflage(features: np.ndarray, labels: np.ndarray, legitimate_mean: np.ndarray, alpha: float) -> np.ndarray:
    """Move fraud rows toward the legitimate training mean. Legitimate rows stay as they are."""
    attacked = features.copy()
    fraud = labels == 1
    attacked[fraud] = (1.0 - alpha) * features[fraud] + alpha * legitimate_mean
    return attacked


def slow_drift(features: np.ndarray, magnitude: float) -> np.ndarray:
    """Add a shift that grows from the first test row to the last."""
    if len(features) == 0:
        return features.copy()
    progress = np.linspace(0.0, 1.0, len(features), dtype=np.float32)[:, None]
    return features + magnitude * progress


def structure_amounts(amounts: np.ndarray, parts: int) -> np.ndarray:
    """Split each amount into equal parts. The caller scores every part."""
    if parts < 2:
        raise ValueError("Structuring needs at least 2 parts.")
    return np.repeat(amounts / parts, parts)


def evaluate_feature_attack(
    name: str,
    labels: np.ndarray,
    attacked: np.ndarray,
    score_fn: Callable[[np.ndarray], np.ndarray],
    thresholds: dict[str, float],
    clean_auroc: float | None,
) -> dict[str, Any]:
    scores = score_fn(attacked)
    auroc, note = safe_auroc(labels, scores)
    decisions = decide_many(scores, thresholds)
    fraud = labels == 1
    result = {
        "attack": name,
        "auroc": auroc,
        "clean_auroc": clean_auroc,
        "auroc_change": None if auroc is None or clean_auroc is None else float(auroc - clean_auroc),
        "fraud_still_blocked": int(((decisions == "BLOCK") & fraud).sum()),
        "fraud_rows": int(fraud.sum()),
        "notes": [note] if note else [],
    }
    return result


def evaluate_structuring(
    amounts: np.ndarray,
    labels: np.ndarray,
    features: np.ndarray,
    amount_index: int,
    parts: int,
    score_fn: Callable[[np.ndarray], np.ndarray],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Replace each fraud amount with several smaller copies and see whether every copy is ALLOW."""
    fraud_indexes = np.flatnonzero(labels == 1)
    if len(fraud_indexes) == 0 or amount_index < 0:
        return {
            "attack": "structuring",
            "cases": 0,
            "fully_allowed": None,
            "any_blocked": None,
            "notes": ["Structuring was not run because this split has no fraud rows or no amount feature."],
        }
    fully_allowed = 0
    any_blocked = 0
    # Cap the number of cases so the check stays small on a laptop.
    chosen = fraud_indexes[:100]
    for index in chosen:
        base = np.repeat(features[index : index + 1], parts, axis=0)
        base[:, amount_index] = features[index, amount_index] / parts
        scores = score_fn(base)
        decisions = decide_many(scores, thresholds)
        if np.all(decisions == "ALLOW"):
            fully_allowed += 1
        if np.any(decisions == "BLOCK"):
            any_blocked += 1
    return {
        "attack": "structuring",
        "cases": int(len(chosen)),
        "parts": int(parts),
        "fully_allowed": fully_allowed,
        "fully_allowed_rate": float(fully_allowed / len(chosen)),
        "any_part_blocked": any_blocked,
        "notes": [
            "Each fraud row is replaced by equal amount-parts in feature space. "
            "A case is fully allowed only when every part receives ALLOW."
        ],
    }
