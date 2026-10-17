"""Analyst labels can replace gateway thresholds. Neural weights stay fixed."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fraud_engine.gateway.policy import thresholds_from_legitimate_scores


@dataclass
class FeedbackStore:
    minimum_samples: int = 30
    rows: list[tuple[str, float, int]] = field(default_factory=list)

    def add(self, transaction_id: str, score: float, label: int) -> None:
        if label not in (0, 1):
            raise ValueError("Analyst label must be 0 (legitimate) or 1 (fraud).")
        self.rows.append((transaction_id, float(score), int(label)))

    def status(self) -> dict:
        labels = [label for _tx, _score, label in self.rows]
        return {
            "samples": len(self.rows),
            "fraud_labels": int(sum(labels)),
            "legitimate_labels": int(len(labels) - sum(labels)),
            "minimum_samples": self.minimum_samples,
            "ready": self._ready(),
        }

    def calibrate(self, step_up_fpr: float, block_fpr: float) -> dict[str, float]:
        if not self._ready():
            raise ValueError(
                "Not enough analyst labels. Need both classes and at least "
                f"{self.minimum_samples} labelled decisions."
            )
        scores = np.array([score for _tx, score, _label in self.rows], dtype=float)
        labels = np.array([label for _tx, _score, label in self.rows], dtype=int)
        thresholds = thresholds_from_legitimate_scores(scores, labels, step_up_fpr, block_fpr)
        thresholds["source"] = "analyst_feedback"
        return thresholds

    def _ready(self) -> bool:
        labels = {label for _tx, _score, label in self.rows}
        return len(self.rows) >= self.minimum_samples and labels == {0, 1}
