"""Deterministic gateway rules."""

from __future__ import annotations

import numpy as np
import pytest

from fraud_engine.gateway.drift import DriftMonitor
from fraud_engine.gateway.feedback import FeedbackStore
from fraud_engine.gateway.policy import decide_many, enforce, thresholds_from_legitimate_scores
from fraud_engine.gateway.ratelimit import RateLimiter


def _thresholds() -> dict:
    return {"step_up": 0.4, "block": 0.8}


def test_score_bands() -> None:
    allow, _ = enforce(0.1, _thresholds(), 10, 1_000_000, None, 0.4, 0.15, "ok", False, True)
    step, _ = enforce(0.5, _thresholds(), 10, 1_000_000, None, 0.4, 0.15, "ok", False, True)
    block, _ = enforce(0.95, _thresholds(), 10, 1_000_000, None, 0.4, 0.15, "ok", False, True)
    assert (allow, step, block) == ("ALLOW", "STEP_UP", "BLOCK")


def test_rules_can_raise_severity_but_not_lower_it() -> None:
    decision, reasons = enforce(0.1, _thresholds(), 2_000_000, 1_000_000, 0.05, 0.4, 0.15, "block", True, True)
    assert decision == "BLOCK"
    assert "amount_hard_block" in reasons
    assert "ip_reputation_block" in reasons
    assert "rate_limit_block" in reasons
    mild, mild_reasons = enforce(0.95, _thresholds(), 10, 1_000_000, 0.9, 0.4, 0.15, "ok", False, True)
    assert mild == "BLOCK"
    assert "score_above_block_threshold" in mild_reasons


def test_reputation_step_up_and_missing_reputation() -> None:
    stepped, reasons = enforce(0.1, _thresholds(), 10, 1_000_000, 0.2, 0.4, 0.15, "ok", False, True)
    assert stepped == "STEP_UP"
    assert "ip_reputation_step_up" in reasons
    allowed, _ = enforce(0.1, _thresholds(), 10, 1_000_000, None, 0.4, 0.15, "ok", False, True)
    assert allowed == "ALLOW"


def test_rate_limiter_uses_the_supplied_clock() -> None:
    limiter = RateLimiter(limit=2, window_seconds=60, block_multiplier=2)
    assert limiter.observe("acct", 100) == "ok"
    assert limiter.observe("acct", 110) == "ok"
    assert limiter.observe("acct", 120) == "step_up"
    assert limiter.observe("acct", 130) == "step_up"
    assert limiter.observe("acct", 140) == "block"
    assert limiter.observe("acct", 300) == "ok"


def test_thresholds_come_from_legitimate_validation_scores_only() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
    labels = np.array([0, 0, 0, 0, 1])
    chosen = thresholds_from_legitimate_scores(scores, labels, step_up_fpr=0.25, block_fpr=0.1)
    legitimate = scores[labels == 0]
    assert chosen["step_up"] == pytest.approx(float(np.quantile(legitimate, 0.75)))
    assert chosen["block"] >= chosen["step_up"]
    decisions = decide_many(scores, chosen)
    assert set(decisions).issubset({"ALLOW", "STEP_UP", "BLOCK"})


def test_drift_and_feedback_calibration() -> None:
    monitor = DriftMonitor(np.array([0.1, 0.1, 0.1]), delta=0.1, window=5)
    assert monitor.update(0.1) is False
    flagged = False
    for _ in range(5):
        flagged = monitor.update(0.9)
    assert flagged is True
    store = FeedbackStore(minimum_samples=4)
    for score, label in ((0.1, 0), (0.2, 0), (0.8, 1), (0.9, 1)):
        store.add(f"t{score}", score, label)
    updated = store.calibrate(0.5, 0.25)
    assert updated["block"] >= updated["step_up"]
    assert updated["source"] == "analyst_feedback"
