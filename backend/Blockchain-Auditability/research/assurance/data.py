"""Loading IEEE-CIS and splitting it in time.

Only the labelled training file is used (the competition's test file has no public labels).
Rows are ordered by TransactionDT and cut into train / calibration / tuning / test; each cut
is moved forward past any run of equal timestamps, so no timestamp appears in two splits and
nothing from the future leaks into an earlier split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

ID_COLUMNS = ["TransactionID", "TransactionDT", "isFraud"]


@dataclass(frozen=True)
class Splits:
    frame: pd.DataFrame            # all rows, time-ordered, index 0..n-1
    index: dict[str, np.ndarray]   # split name -> row positions

    def part(self, name: str) -> pd.DataFrame:
        return self.frame.iloc[self.index[name]]


def transaction_columns() -> list[str]:
    return ID_COLUMNS + config.NUMERIC + config.BINARY + [c for c in config.CATEGORICAL]


def load_ieee_cis(data_dir: Path | None = None, max_rows: int | None = None) -> pd.DataFrame:
    """train_transaction.csv joined with train_identity.csv (left join: identity is optional)."""
    data_dir = Path(data_dir or config.DATA_DIR)
    transaction_file = data_dir / "train_transaction.csv"
    identity_file = data_dir / "train_identity.csv"
    if not transaction_file.exists():
        raise FileNotFoundError(
            f"{transaction_file} not found. Download the IEEE-CIS competition data (see research/README.md)."
        )
    wanted = set(transaction_columns())
    tx = pd.read_csv(transaction_file, usecols=lambda c: c in wanted, nrows=max_rows)
    if identity_file.exists():
        id_wanted = {"TransactionID", *config.IDENTITY_NUMERIC, *config.IDENTITY_CATEGORICAL}
        ident = pd.read_csv(identity_file, usecols=lambda c: c in id_wanted)
        ident["has_identity"] = 1
        tx = tx.merge(ident, on="TransactionID", how="left")
    else:
        for column in config.IDENTITY_NUMERIC + config.IDENTITY_CATEGORICAL:
            tx[column] = np.nan
        tx["has_identity"] = np.nan
    tx["has_identity"] = tx["has_identity"].fillna(0).astype(int)
    return tx


def time_split(frame: pd.DataFrame, shares: dict[str, float] | None = None) -> Splits:
    shares = shares or config.SPLITS
    ordered = frame.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True)
    times = ordered["TransactionDT"].to_numpy()
    n = len(ordered)
    names = list(shares)
    cuts = []
    running = 0.0
    for name in names[:-1]:
        running += shares[name]
        cut = int(round(n * running))
        # Move the cut past a run of equal timestamps so the boundary shares none.
        while 0 < cut < n and times[cut] == times[cut - 1]:
            cut += 1
        cuts.append(cut)
    bounds = [0, *cuts, n]
    index = {name: np.arange(bounds[i], bounds[i + 1]) for i, name in enumerate(names)}
    return Splits(frame=ordered, index=index)


def test_index_digest(splits: Splits) -> str:
    """Fingerprint of exactly which transactions form the test split (for the run log)."""
    import hashlib
    ids = splits.part("test")["TransactionID"].to_numpy(dtype=np.int64)
    return "sha256:" + hashlib.sha256(ids.tobytes()).hexdigest()
