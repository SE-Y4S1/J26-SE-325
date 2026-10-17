"""Time-aware splits and train-only feature scaling.

TransactionDT is used to order rows. It is not copied into the model features,
because the raw clock would tell the model which part of the year it is looking at.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SPLIT_TRAIN = 0
SPLIT_VAL = 1
SPLIT_TEST = 2
SPLIT_NAME = {0: "train", 1: "validation", 2: "test"}

NUMERIC_EXACT = ["TransactionAmt", "dist1", "dist2"]
CATEGORICAL = [
    "ProductCD",
    "card4",
    "card6",
    "DeviceType",
    "DeviceInfo",
    "P_emaildomain",
    "R_emaildomain",
]
FREQUENCY_KEYS = ["card1"]
DEVICE_COLUMNS = ["DeviceType", "DeviceInfo"]
SKIP_COLUMNS = {"TransactionID", "isFraud", "TransactionDT", "event_time"}


def assign_time_splits(times: np.ndarray, train_ratio: float = 0.70, val_ratio: float = 0.15) -> np.ndarray:
    """Assign every row to one split so a timestamp never appears in two splits."""
    times = np.asarray(times)
    count = len(times)
    if count < 3:
        raise ValueError("At least 3 rows are required for train, validation, and test splits.")
    order = np.argsort(times, kind="mergesort")
    sorted_times = times[order]
    train_cut = int(round(count * train_ratio))
    val_cut = int(round(count * (train_ratio + val_ratio)))
    train_cut = min(max(train_cut, 1), count - 2)
    val_cut = min(max(val_cut, train_cut + 1), count - 1)

    def _extend(cut: int) -> int:
        if cut >= count:
            return count
        boundary = sorted_times[cut - 1]
        while cut < count and sorted_times[cut] == boundary:
            cut += 1
        return cut

    train_cut = _extend(train_cut)
    if train_cut >= count - 1:
        train_cut = count - 2
    val_cut = max(val_cut, train_cut + 1)
    val_cut = min(_extend(val_cut), count)
    if val_cut <= train_cut:
        val_cut = train_cut + 1
    if val_cut >= count:
        val_cut = count - 1
        if train_cut >= val_cut:
            train_cut = val_cut - 1

    split = np.full(count, SPLIT_TEST, dtype=np.int8)
    split[order[:train_cut]] = SPLIT_TRAIN
    split[order[train_cut:val_cut]] = SPLIT_VAL
    return split


def splits_are_time_ordered(times: np.ndarray, split: np.ndarray) -> bool:
    """True when every training time is at or before validation, then test."""
    bounds = []
    for code in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST):
        chosen = times[split == code]
        if len(chosen) == 0:
            return False
        bounds.append((float(np.min(chosen)), float(np.max(chosen))))
    return bounds[0][1] <= bounds[1][0] and bounds[1][1] <= bounds[2][0]


def _numbered(columns: list[str], prefix: str, limit: int) -> list[str]:
    found = []
    for name in columns:
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        if suffix.isdigit() and 1 <= int(suffix) <= limit:
            found.append(name)
    return found


def _binary_match(columns: list[str]) -> list[str]:
    return [name for name in columns if name.startswith("M") and name[1:].isdigit() and int(name[1:]) <= 9]


class TabularPreprocessor:
    """Median imputation, z-scoring, and frequency encoding fit on training rows only."""

    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self.medians: dict[str, float] = {}
        self.means: dict[str, float] = {}
        self.scales: dict[str, float] = {}
        self.frequencies: dict[str, dict[str, float]] = {}
        self.binary_columns: list[str] = []
        self.device_features: list[str] = []
        self.simulated_features: list[str] = []
        self.schema_version = 1

    def fit(self, frame: pd.DataFrame, train_mask: np.ndarray) -> "TabularPreprocessor":
        train = frame.loc[train_mask].copy()
        columns = [str(column) for column in frame.columns]
        numeric = [name for name in NUMERIC_EXACT if name in columns]
        numeric += _numbered(columns, "C", 14)
        numeric += _numbered(columns, "D", 15)
        numeric += _numbered(columns, "V", 20)
        numeric += _numbered(columns, "id_", 11)
        numeric += [name for name in columns if name.startswith("simulated_") and name not in SKIP_COLUMNS]
        binary = [name for name in _binary_match(columns) if name in frame.columns]
        categorical = [name for name in CATEGORICAL + FREQUENCY_KEYS if name in columns]

        names: list[str] = []
        for name in numeric:
            series = pd.to_numeric(train[name], errors="coerce")
            if series.notna().sum() == 0:
                continue
            median = float(series.median())
            filled = series.fillna(median).to_numpy(dtype=float)
            mean = float(filled.mean())
            scale = float(filled.std())
            if scale < 1e-8:
                scale = 1.0
            self.medians[name] = median
            self.means[name] = mean
            self.scales[name] = scale
            names.append(name)
            if name.startswith("id_") or name.startswith("simulated_"):
                if name.startswith("simulated_"):
                    self.simulated_features.append(name)
                else:
                    self.device_features.append(name)

        self.binary_columns = []
        for name in binary:
            mapped = _map_binary(train[name])
            if mapped.notna().sum() == 0:
                continue
            median = float(np.nanmedian(mapped.to_numpy(dtype=float)))
            filled = mapped.fillna(median).to_numpy(dtype=float)
            mean = float(filled.mean())
            scale = float(filled.std()) or 1.0
            key = f"binary__{name}"
            self.medians[key] = median
            self.means[key] = mean
            self.scales[key] = scale
            self.binary_columns.append(name)
            names.append(key)

        for name in categorical:
            counts = train[name].astype("string").fillna("__MISSING__").value_counts()
            mapping = {str(key): float(np.log1p(value)) for key, value in counts.items()}
            self.frequencies[name] = mapping
            names.append(f"freq__{name}")
            if name in DEVICE_COLUMNS or name.startswith("Device"):
                self.device_features.append(f"freq__{name}")

        self.feature_names = names
        if not names:
            raise ValueError("No usable features were found. Check the transaction columns.")
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        rows = []
        for name in self.feature_names:
            if name.startswith("freq__"):
                source = name[len("freq__") :]
                mapping = self.frequencies[source]
                values = frame[source].astype("string").fillna("__MISSING__") if source in frame.columns else pd.Series(["__MISSING__"] * len(frame))
                rows.append(values.map(lambda item: mapping.get(str(item), 0.0)).to_numpy(dtype=float))
            elif name.startswith("binary__"):
                source = name[len("binary__") :]
                mapped = _map_binary(frame[source]) if source in frame.columns else pd.Series(np.nan, index=frame.index)
                filled = mapped.fillna(self.medians[name]).to_numpy(dtype=float)
                rows.append(_scale(filled, self.means[name], self.scales[name]))
            else:
                if name in frame.columns:
                    series = pd.to_numeric(frame[name], errors="coerce")
                else:
                    series = pd.Series(np.nan, index=frame.index)
                filled = series.fillna(self.medians[name]).to_numpy(dtype=float)
                rows.append(_scale(filled, self.means[name], self.scales[name]))
        matrix = np.column_stack(rows).astype(np.float32)
        return np.nan_to_num(matrix, nan=0.0, posinf=8.0, neginf=-8.0)

    def transform_record(self, values: dict[str, Any]) -> np.ndarray:
        frame = pd.DataFrame([values])
        return self.transform(frame)[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_names": self.feature_names,
            "medians": self.medians,
            "means": self.means,
            "scales": self.scales,
            "frequencies": self.frequencies,
            "binary_columns": self.binary_columns,
            "device_features": self.device_features,
            "simulated_features": self.simulated_features,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TabularPreprocessor":
        model = cls()
        for key, value in payload.items():
            setattr(model, key, value)
        return model


def _scale(values: np.ndarray, mean: float, scale: float) -> np.ndarray:
    scaled = (values - mean) / scale
    return np.clip(scaled, -8.0, 8.0)


def _map_binary(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.upper()
    mapped = pd.Series(np.nan, index=series.index, dtype=float)
    mapped[text.isin(["T", "Y", "1", "TRUE"])] = 1.0
    mapped[text.isin(["F", "N", "0", "FALSE"])] = 0.0
    numeric = pd.to_numeric(series, errors="coerce")
    mapped = mapped.fillna(numeric)
    return mapped
