"""The three label-free signals the assurance engine routes on, computed per transaction.

confidence    1 - distance of the calibrated ensemble probability from the decision threshold
              (the same formula as the JavaScript engine's margin uncertainty)
disagreement  spread (max - min) between the three calibrated members
context       mean of four contextual risks, each in [0, 1], fixed equal weights:
                amount        percentile of the amount among training transactions
                novelty       IsolationForest outlier score, model fitted on training rows only
                new_card      the card (card1) never appeared in any earlier transaction
                no_identity   no identity/device record for the transaction

Labels are never used to compute a signal. Limitations of each context signal are listed in
METHODS.md (e.g. new_card penalises new customers; novelty flags rare but legitimate behaviour).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from . import config
from .engine import margin_uncertainty

CONTEXT_PARTS = ["amount", "novelty", "new_card", "no_identity"]


def first_seen_flags(frame: pd.DataFrame) -> np.ndarray:
    """1 where the row's card1 has not appeared in any earlier row (frame is time-ordered).

    Strictly backward-looking: a row only ever looks at rows before it, so a flag in the test
    split never depends on the test split's future, and never on a label.
    """
    card = frame["card1"].astype("string")
    seen_before = card.duplicated(keep="first").to_numpy()
    flags = (~seen_before).astype(float)
    flags[card.isna().to_numpy()] = 1.0
    return flags


class ContextModel:
    """Statistics for the context signals, learned on the training split only."""

    def fit(self, train: pd.DataFrame, x_train: np.ndarray, seed: int = config.SEED) -> "ContextModel":
        self.amounts = np.sort(train["TransactionAmt"].to_numpy(dtype=float))
        rng = np.random.default_rng(seed)
        sample = x_train[rng.choice(len(x_train), size=min(len(x_train), 50_000), replace=False)]
        self.forest = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1).fit(sample)
        train_scores = -self.forest.score_samples(sample)
        self.novelty_lo, self.novelty_hi = np.quantile(train_scores, [0.01, 0.99])
        return self

    def amount_percentile(self, amounts: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.amounts, amounts, side="right") / len(self.amounts)

    def novelty(self, x: np.ndarray) -> np.ndarray:
        raw = -self.forest.score_samples(x)
        return np.clip((raw - self.novelty_lo) / max(self.novelty_hi - self.novelty_lo, 1e-9), 0.0, 1.0)


def compute(part: pd.DataFrame, x: np.ndarray, members: dict[str, np.ndarray], new_card: np.ndarray,
            context_model: ContextModel, threshold: float = config.DECISION_THRESHOLD) -> dict[str, np.ndarray]:
    stacked = np.vstack(list(members.values()))
    p = stacked.mean(axis=0)
    context = {
        "amount": context_model.amount_percentile(part["TransactionAmt"].to_numpy(dtype=float)),
        "novelty": context_model.novelty(x),
        "new_card": new_card,
        "no_identity": 1.0 - part["has_identity"].to_numpy(dtype=float),
    }
    return {
        "y": part["isFraud"].to_numpy(dtype=int),
        "amount_value": part["TransactionAmt"].to_numpy(dtype=float),
        "p": p,
        "decision": (p >= threshold).astype(int),
        "u_confidence": np.asarray(margin_uncertainty(p, threshold), dtype=float),
        "u_disagreement": stacked.max(axis=0) - stacked.min(axis=0),
        "u_context": np.mean([context[k] for k in CONTEXT_PARTS], axis=0),
        **{f"ctx_{k}": v for k, v in context.items()},
        **{f"member_{k}": v for k, v in members.items()},
    }
