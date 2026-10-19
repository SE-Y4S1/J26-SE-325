"""Metrics for imbalanced fraud scores and gateway actions."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> tuple[float | None, str | None]:
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return None, "AUROC needs both fraud and legitimate rows in this split."
    return float(roc_auc_score(labels, scores)), None


def safe_average_precision(labels: np.ndarray, scores: np.ndarray) -> tuple[float | None, str | None]:
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return None, "Average precision needs both classes."
    return float(average_precision_score(labels, scores)), None


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float | None, str | None]:
    """Pick a classification threshold on validation data by macro search of quantiles."""
    if len(labels) == 0 or int((labels == 1).sum()) == 0:
        return None, "F1 threshold needs fraud labels in the validation split."
    candidates = np.unique(np.quantile(scores, np.linspace(0.05, 0.95, 19)))
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)
        score = f1_score(labels, predictions, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold, None


def classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = (int(value) for value in matrix.ravel())
    notes = []
    if true_positive + false_positive == 0:
        notes.append("No rows were predicted as fraud at this threshold, so precision is reported as 0.")
    auroc, auroc_note = safe_auroc(labels, scores)
    average_precision, ap_note = safe_average_precision(labels, scores)
    if auroc_note:
        notes.append(auroc_note)
    if ap_note:
        notes.append(ap_note)
    return {
        "auroc": auroc,
        "average_precision": average_precision,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": true_negative,
            "fp": false_positive,
            "fn": false_negative,
            "tp": true_positive,
        },
        "notes": notes,
    }


def gateway_outcome(labels: np.ndarray, decisions: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels)
    decisions = np.asarray(decisions)
    fraud = labels == 1
    legitimate = labels == 0
    fraud_count = int(fraud.sum())
    legitimate_count = int(legitimate.sum())

    def _rate(mask: np.ndarray, base: int) -> float | None:
        if base == 0:
            return None
        return float(mask.sum() / base)

    notes = []
    if fraud_count == 0:
        notes.append("No fraud rows in this split, so the blocked-fraud rate cannot be calculated.")
    if legitimate_count == 0:
        notes.append("No legitimate rows in this split, so the false-challenge rate cannot be calculated.")
    return {
        "fraud_rows": fraud_count,
        "legitimate_rows": legitimate_count,
        "fraud_blocked": int(((decisions == "BLOCK") & fraud).sum()),
        "fraud_blocked_rate": _rate((decisions == "BLOCK") & fraud, fraud_count),
        "fraud_step_up": int(((decisions == "STEP_UP") & fraud).sum()),
        "legitimate_challenged": int((legitimate & np.isin(decisions, ["STEP_UP", "BLOCK"])).sum()),
        "legitimate_challenged_rate": _rate(legitimate & np.isin(decisions, ["STEP_UP", "BLOCK"]), legitimate_count),
        "legitimate_blocked": int(((decisions == "BLOCK") & legitimate).sum()),
        "legitimate_blocked_rate": _rate((decisions == "BLOCK") & legitimate, legitimate_count),
        "allow": int((decisions == "ALLOW").sum()),
        "step_up": int((decisions == "STEP_UP").sum()),
        "block": int((decisions == "BLOCK").sum()),
        "notes": notes,
    }


def time_scorer(score_fn: Callable[[np.ndarray], np.ndarray], features: np.ndarray, single_rows: int = 30) -> dict[str, Any]:
    """Batch throughput and a small single-row latency sample."""
    if len(features) == 0:
        return {"batch_throughput_rows_per_s": None, "single_row_latency_ms_p50": None, "notes": ["No rows to time."]}
    started = time.perf_counter()
    score_fn(features)
    elapsed = max(time.perf_counter() - started, 1e-9)
    sample = min(single_rows, len(features))
    latencies = []
    for index in range(sample):
        row_started = time.perf_counter()
        score_fn(features[index : index + 1])
        latencies.append((time.perf_counter() - row_started) * 1000.0)
    return {
        "batch_throughput_rows_per_s": float(len(features) / elapsed),
        "single_row_latency_ms_p50": float(np.percentile(latencies, 50)),
        "timed_rows": int(len(features)),
        "single_row_samples": int(sample),
    }


def slice_block_rates(labels: np.ndarray, decisions: np.ndarray, groups: np.ndarray, minimum: int = 20) -> dict[str, Any]:
    """False BLOCK rate by a device or product slice. This is a basic fairness check, not a full audit."""
    output = {}
    for name in sorted(set(str(item) for item in groups)):
        mask = np.array([str(item) == name for item in groups])
        legitimate = mask & (labels == 0)
        if int(legitimate.sum()) < minimum:
            output[name] = {"legitimate_rows": int(legitimate.sum()), "block_rate": None, "note": f"Fewer than {minimum} legitimate rows."}
            continue
        blocked = legitimate & (decisions == "BLOCK")
        output[name] = {
            "legitimate_rows": int(legitimate.sum()),
            "block_rate": float(blocked.sum() / legitimate.sum()),
        }
    return output
