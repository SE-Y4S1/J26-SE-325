"""A tiny synthetic stand-in for the IEEE-CIS files, with the same column names and quirks
(missing values, T/F flags, repeated timestamps, an optional identity file).

It exists only so the pipeline's code can be tested without the Kaggle data. Numbers produced
from it are meaningless and are never reported as results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from assurance import config


def make_ieee_like(folder: Path, n: int = 6000, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    dt = np.sort(rng.integers(86_400, 86_400 * 60, size=n))
    dt[100:110] = dt[100]                                    # a run of equal timestamps
    amount = np.round(np.exp(rng.normal(4, 1.2, size=n)), 2)
    latent = 0.004 * amount / amount.mean() + rng.normal(0, 1, size=n)
    fraud = (latent + rng.normal(0, 1, size=n) > 2.6).astype(int)
    frame = pd.DataFrame({"TransactionID": np.arange(3_000_000, 3_000_000 + n), "TransactionDT": dt,
                          "isFraud": fraud, "TransactionAmt": amount})
    for column in config.NUMERIC[1:]:
        values = rng.normal(latent * (1 if column.startswith("C") else 0.2), 1, size=n)
        values[rng.random(n) < 0.2] = np.nan
        frame[column] = values
    for column in config.BINARY:
        frame[column] = np.where(rng.random(n) < 0.5, "T", "F")
        frame.loc[rng.random(n) < 0.3, column] = np.nan
    frame["M4"] = rng.choice(["M0", "M1", "M2", None], size=n)
    frame["ProductCD"] = rng.choice(list("WCRHS"), size=n)
    frame["card4"] = rng.choice(["visa", "mastercard", None], size=n)
    frame["card6"] = rng.choice(["debit", "credit"], size=n)
    frame["P_emaildomain"] = rng.choice(["gmail.com", "yahoo.com", None], size=n)
    frame["R_emaildomain"] = rng.choice(["gmail.com", None], size=n)
    frame["card1"] = rng.integers(1000, 1600, size=n)
    frame.to_csv(folder / "train_transaction.csv", index=False)

    with_identity = frame.sample(frac=0.4, random_state=seed)["TransactionID"]
    identity = pd.DataFrame({"TransactionID": with_identity})
    for column in config.IDENTITY_NUMERIC:
        identity[column] = rng.normal(0, 1, size=len(identity))
    identity["DeviceType"] = rng.choice(["mobile", "desktop"], size=len(identity))
    identity["DeviceInfo"] = rng.choice(["Windows", "iOS Device", "MacOS"], size=len(identity))
    identity.to_csv(folder / "train_identity.csv", index=False)
    return folder
