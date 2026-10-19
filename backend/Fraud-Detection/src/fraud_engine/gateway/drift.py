"""Detect a rise in recent fraud scores against a validation reference."""

from __future__ import annotations

from collections import deque

import numpy as np


class DriftMonitor:
    def __init__(self, reference_scores: np.ndarray | None, delta: float, window: int) -> None:
        self.delta = float(delta)
        self.window = int(window)
        self.recent: deque[float] = deque(maxlen=self.window)
        if reference_scores is None or len(reference_scores) == 0:
            self.reference_mean = None
        else:
            self.reference_mean = float(np.mean(reference_scores))

    def update(self, score: float) -> bool:
        self.recent.append(float(score))
        if self.reference_mean is None or len(self.recent) < max(5, self.window // 5):
            return False
        return float(np.mean(self.recent)) > self.reference_mean + self.delta

    def set_reference(self, scores: np.ndarray) -> None:
        if len(scores) == 0:
            self.reference_mean = None
        else:
            self.reference_mean = float(np.mean(scores))
