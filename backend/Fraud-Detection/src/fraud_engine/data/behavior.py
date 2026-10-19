"""Behavioural and device features.

IEEE-CIS identity fields are anonymised device attributes from the public
benchmark. Anything created by this module's simulator is marked simulated and
is independent of the fraud label, unless a separate labelled study calls
`label_correlated_signal` on a copy of the matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SIMULATED_COLUMNS = (
    "simulated_typing_interval_ms",
    "simulated_pointer_jitter",
    "simulated_session_duration_s",
)


def add_label_free_simulated_biometrics(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Add reproducible biometric-like columns that do not use isFraud."""
    if "TransactionID" not in frame.columns:
        raise ValueError("TransactionID is required to seed simulated signals.")
    output = frame.copy()
    identifiers = pd.to_numeric(output["TransactionID"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    # A per-row seed keeps the values reproducible without reading the label.
    row_seed = (identifiers * 1_000_003 + int(seed)) % (2**32)
    typing = np.empty(len(output), dtype=np.float64)
    jitter = np.empty(len(output), dtype=np.float64)
    duration = np.empty(len(output), dtype=np.float64)
    for index, item in enumerate(row_seed.tolist()):
        rng = np.random.default_rng(int(item))
        typing[index] = float(rng.lognormal(mean=5.2, sigma=0.25))
        jitter[index] = float(rng.uniform(0.0, 1.0))
        duration[index] = float(rng.lognormal(mean=3.4, sigma=0.4))
    output["simulated_typing_interval_ms"] = typing
    output["simulated_pointer_jitter"] = jitter
    output["simulated_session_duration_s"] = duration
    output["simulated_signal_origin"] = "simulated"
    return output


def label_correlated_signal(labels: np.ndarray, seed: int = 42, effect: float = 1.0) -> np.ndarray:
    """Synthetic study column. This is intentionally related to the label.

    Do not mix this column into the IEEE-CIS feature set. It only exists so the
    lift calculation can be tested when a behavioural signal is known to exist.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=len(labels))
    return effect * labels.astype(float) + noise
