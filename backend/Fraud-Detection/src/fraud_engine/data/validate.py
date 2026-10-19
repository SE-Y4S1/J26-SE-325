"""Dataset checks: columns, imbalance, missing values, duplicates, checksums."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def profile_frame(frame: pd.DataFrame, label_column: str | None = None, id_column: str | None = None) -> dict[str, Any]:
    """Summarise a table without printing row values."""
    profile: dict[str, Any] = {
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_fraction": {
            str(column): float(frame[column].isna().mean()) for column in frame.columns
        },
    }
    if id_column and id_column in frame.columns:
        profile["duplicate_ids"] = int(frame[id_column].duplicated().sum())
    if label_column and label_column in frame.columns:
        labels = frame[label_column].dropna()
        counts = labels.value_counts(dropna=False).to_dict()
        profile["label_counts"] = {str(key): int(value) for key, value in counts.items()}
        if len(labels) > 0:
            positive = float((labels.astype(float) == 1).mean()) if _numeric_labels(labels) else None
            profile["positive_rate"] = positive
    return profile


def _numeric_labels(labels: pd.Series) -> bool:
    try:
        pd.to_numeric(labels)
        return True
    except (TypeError, ValueError):
        return False


def assert_columns(frame: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def class_balance_ready(labels: np.ndarray, minimum_positive: int = 1) -> str | None:
    """Return an explanation when a split cannot support supervised metrics."""
    unique = set(np.unique(labels).tolist())
    if len(unique) < 2:
        return "This split contains only one class, so discrimination metrics cannot be calculated."
    if int((labels == 1).sum()) < minimum_positive:
        return "This split has no positive (fraud) labels."
    if int((labels == 0).sum()) < 1:
        return "This split has no legitimate labels."
    return None
