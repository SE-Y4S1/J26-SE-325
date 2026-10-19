"""Combine the LSTM and GNN scores without looking at the test split.

The mixture weight is chosen on the calibration slice. A stream whose calibration
AUROC is below 0.5 is reversed first, so a backwards score cannot dominate the mix.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


class ScoreFusion:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.alpha = 0.5
        self.flip_first = False
        self.flip_second = False
        self.fallback = "equal_mixture"
        self.weights: list[float] = [0.5, 0.5]

    def fit(self, first: np.ndarray, second: np.ndarray, labels: np.ndarray) -> "ScoreFusion":
        first = np.asarray(first, dtype=np.float64)
        second = np.asarray(second, dtype=np.float64)
        labels = np.asarray(labels)
        if len(np.unique(labels)) < 2:
            self.fallback = "mean_one_class_calibration_split"
            self.alpha = 0.5
            self.weights = [0.5, 0.5]
            return self
        oriented_first, self.flip_first = _orient(first, labels)
        oriented_second, self.flip_second = _orient(second, labels)
        best_alpha = 0.5
        best_auroc = -1.0
        for alpha in np.linspace(0.0, 1.0, 11):
            mixed = alpha * oriented_first + (1.0 - alpha) * oriented_second
            auroc = float(roc_auc_score(labels, mixed))
            if auroc > best_auroc:
                best_auroc = auroc
                best_alpha = float(alpha)
        self.alpha = best_alpha
        self.weights = [self.alpha, 1.0 - self.alpha]
        self.fallback = "validation_mixture"
        return self

    def score(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        oriented_first = _apply_flip(np.asarray(first, dtype=np.float64), self.flip_first)
        oriented_second = _apply_flip(np.asarray(second, dtype=np.float64), self.flip_second)
        mixed = self.alpha * oriented_first + (1.0 - self.alpha) * oriented_second
        return np.clip(mixed, 0.0, 1.0)

    def to_dict(self) -> dict:
        return {
            "fallback": self.fallback,
            "weights": self.weights,
            "alpha_on_lstm": self.alpha,
            "flip_lstm": self.flip_first,
            "flip_gnn": self.flip_second,
            "seed": self.seed,
        }


def _orient(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, bool]:
    auroc = float(roc_auc_score(labels, scores))
    if auroc < 0.5:
        return 1.0 - scores, True
    return scores, False


def _apply_flip(scores: np.ndarray, flip: bool) -> np.ndarray:
    return 1.0 - scores if flip else scores
