"""Streaming replay and adversarial simulations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_engine.data.synthetic import make_synthetic_transactions
from fraud_engine.evaluation.adversarial import camouflage, evaluate_structuring, slow_drift, structure_amounts
from fraud_engine.streaming.replay import LocalStream, replay_frame, score_messages, transaction_payload


def test_replay_keeps_labels_out_of_the_gateway_payload() -> None:
    frame = make_synthetic_transactions(30, seed=2)
    payload = transaction_payload(frame.iloc[0])
    assert "isFraud" not in payload
    assert "isFraud" not in payload["features"]
    stream = LocalStream()
    sent = replay_frame(frame, stream.publish, "finance.transactions.raw")
    messages = stream.consume("finance.transactions.raw")
    assert sent == 30
    assert len(messages) == 30
    assert messages[0].payload["transaction_id"] == "1"
    scored = score_messages(messages, lambda item: {"transaction_id": item["transaction_id"], "amount": item["amount"]})
    assert scored[0]["transaction_id"] == "1"


def test_adversarial_transforms_do_not_relabel_rows() -> None:
    features = np.array([[0.0, 1.0], [4.0, 4.0], [1.0, 1.0]], dtype=float)
    labels = np.array([0, 1, 0])
    hidden = camouflage(features, labels, legitimate_mean=np.zeros(2), alpha=0.5)
    assert np.allclose(hidden[0], features[0])
    assert np.allclose(hidden[1], np.array([2.0, 2.0]))
    drifted = slow_drift(features, magnitude=1.0)
    assert drifted.shape == features.shape
    assert drifted[0, 0] == features[0, 0]
    assert drifted[-1, 0] > features[-1, 0]
    parts = structure_amounts(np.array([100.0, 40.0]), 4)
    assert len(parts) == 8
    assert parts[0] == 25.0


def test_structuring_counts_a_full_bypass() -> None:
    features = np.array([[0.1], [0.2], [5.0]], dtype=float)
    labels = np.array([0, 0, 1])

    def score_fn(matrix: np.ndarray) -> np.ndarray:
        return matrix[:, 0]

    result = evaluate_structuring(
        features[:, 0],
        labels,
        features,
        amount_index=0,
        parts=4,
        score_fn=score_fn,
        thresholds={"step_up": 2.0, "block": 4.0},
    )
    # 5 / 4 = 1.25, which is below the ALLOW cut, so every part is allowed.
    assert result["cases"] == 1
    assert result["fully_allowed"] == 1
