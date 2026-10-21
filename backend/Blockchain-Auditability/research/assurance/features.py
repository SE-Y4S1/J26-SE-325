"""Feature matrix for the base models. Every statistic is learned from the training split
only and then applied unchanged to the later splits."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

BINARY_MAP = {"T": 1.0, "F": 0.0}


@dataclass
class Preprocessor:
    medians: dict[str, float] = field(default_factory=dict)
    frequencies: dict[str, dict] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)

    @property
    def numeric(self) -> list[str]:
        return config.NUMERIC + config.IDENTITY_NUMERIC

    @property
    def categorical(self) -> list[str]:
        return config.CATEGORICAL + config.IDENTITY_CATEGORICAL

    def fit(self, train: pd.DataFrame) -> "Preprocessor":
        for column in self.numeric:
            values = pd.to_numeric(train[column], errors="coerce")
            median = values.median()
            self.medians[column] = float(median) if pd.notna(median) else 0.0
        for column in config.BINARY:
            self.medians[column] = float(train[column].map(BINARY_MAP).median()) if train[column].notna().any() else 0.0
        for column in self.categorical:
            counts = train[column].astype("string").value_counts(normalize=True, dropna=True)
            self.frequencies[column] = counts.to_dict()
        self.columns = (
            self.numeric + config.BINARY
            + [f"freq__{c}" for c in self.categorical]
            + [f"missing__{c}" for c in ["dist1", "D1", "id_01"]]
            + ["has_identity"]
        )
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        out = pd.DataFrame(index=frame.index)
        for column in self.numeric:
            out[column] = pd.to_numeric(frame[column], errors="coerce").fillna(self.medians[column])
        for column in config.BINARY:
            out[column] = frame[column].map(BINARY_MAP).fillna(self.medians[column])
        for column in self.categorical:
            # Unseen or missing categories get frequency 0 -- never a value learned later.
            out[f"freq__{column}"] = frame[column].astype("string").map(self.frequencies[column]).astype(float).fillna(0.0)
        for column in ["dist1", "D1", "id_01"]:
            out[f"missing__{column}"] = frame[column].isna().astype(float)
        out["has_identity"] = frame["has_identity"].astype(float)
        return out[self.columns].to_numpy(dtype=np.float64)
