"""Synthetic transactions for tests and the pipeline smoke check.

This is not IEEE-CIS, DGraphFin, or Elliptic data. Do not report these rows as a
research benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_transactions(n: int = 360, seed: int = 42) -> pd.DataFrame:
    """Build a small labelled stream with bursts, shared devices, and both classes."""
    if n < 30:
        raise ValueError("Use at least 30 synthetic rows so each time split can contain both classes.")
    rng = np.random.default_rng(seed)
    account = rng.integers(0, 40, size=n)
    transaction_dt = np.arange(n, dtype=np.int64)
    device_info = np.array([f"dev-{int(item) % 12}" for item in account])
    device_type = np.where(account % 2 == 0, "mobile", "desktop")
    product = np.where(rng.random(n) < 0.55, "W", "C")
    burst = np.zeros(n, dtype=np.float64)
    last_seen: dict[int, int] = {}
    for index, owner in enumerate(account.tolist()):
        previous = last_seen.get(owner)
        if previous is not None and index - previous <= 3:
            burst[index] = 1.0
        last_seen[owner] = index
    fraud_account = (account < 5).astype(np.float64)
    amount = rng.normal(80.0, 15.0, size=n)
    amount = np.clip(amount + burst * 45.0 + fraud_account * 25.0, 1.0, None)
    c1 = rng.poisson(2, size=n) + (burst * 3).astype(int)
    logit = -2.5 + 0.025 * (amount - 80.0) + 1.35 * burst + 1.05 * fraud_account + 0.12 * c1
    probability = 1.0 / (1.0 + np.exp(-logit))
    labels = (rng.random(n) < probability).astype(np.int64)
    _ensure_both_classes(labels)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(1, n + 1, dtype=np.int64),
            "isFraud": labels,
            "TransactionDT": transaction_dt,
            "TransactionAmt": amount.astype(np.float64),
            "card1": account.astype(np.int64),
            "ProductCD": product,
            "C1": c1.astype(np.int64),
            "DeviceType": device_type,
            "DeviceInfo": device_info,
            "dist1": rng.normal(10.0, 3.0, size=n),
        }
    )


def _ensure_both_classes(labels: np.ndarray) -> None:
    count = len(labels)
    for start in (0, count // 3, (2 * count) // 3):
        end = count if start == (2 * count) // 3 else start + count // 3
        window = labels[start:end]
        if window.sum() == 0:
            labels[start] = 1
        elif window.sum() == len(window):
            labels[start] = 0
