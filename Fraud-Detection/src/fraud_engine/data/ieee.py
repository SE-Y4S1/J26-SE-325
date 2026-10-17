"""IEEE-CIS Fraud Detection loader.

The public test files have no isFraud labels. Metrics use a time-ordered holdout
from train_transaction.csv. train_identity.csv contributes anonymised device fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_engine.data.download import ieee_ready
from fraud_engine.data.validate import file_record, profile_frame
from fraud_engine.paths import METADATA_DIR, RAW_DIR

LABELLED_REQUIRED = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt"]
IDENTITY_OPTIONAL = ["TransactionID", "DeviceType", "DeviceInfo"]


def ieee_paths(root: Path | None = None) -> Path:
    return root or (RAW_DIR / "ieee-cis")


def _header(path: Path) -> list[str]:
    return [str(column) for column in pd.read_csv(path, nrows=0).columns]


def _selected_columns(header: list[str]) -> list[str]:
    wanted = set(LABELLED_REQUIRED)
    wanted.update(
        [
            "ProductCD",
            "card1",
            "card4",
            "card6",
            "addr1",
            "P_emaildomain",
            "R_emaildomain",
            "dist1",
            "dist2",
            "DeviceType",
            "DeviceInfo",
        ]
    )
    for prefix, limit in (("C", 14), ("D", 15), ("V", 20), ("M", 9), ("id_", 11)):
        for column in header:
            suffix = column[len(prefix) :] if column.startswith(prefix) else ""
            if suffix.isdigit() and 1 <= int(suffix) <= limit:
                wanted.add(column)
    return [column for column in header if column in wanted]


def load_ieee_frame(root: Path | None = None, max_rows: int | None = 20000) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the labelled IEEE-CIS training tables and an optional early-time subset."""
    folder = ieee_paths(root)
    transaction_path = folder / "train_transaction.csv"
    identity_path = folder / "train_identity.csv"
    if not transaction_path.is_file():
        raise FileNotFoundError(
            f"{transaction_path} is missing. Follow Fraud-Detection/docs/DATASETS.md to download IEEE-CIS."
        )
    columns = _selected_columns(_header(transaction_path))
    frame = pd.read_csv(transaction_path, usecols=columns)
    frame["TransactionDT"] = pd.to_numeric(frame["TransactionDT"], errors="coerce")
    frame = frame.dropna(subset=["TransactionDT", "TransactionID", "isFraud"]).copy()
    frame["isFraud"] = frame["isFraud"].astype(int)
    monotonic = bool(frame["TransactionDT"].is_monotonic_increasing)
    frame = frame.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True)
    subset_note = "all labelled training rows"
    if max_rows is not None and len(frame) > max_rows:
        frame = frame.iloc[: int(max_rows)].reset_index(drop=True)
        subset_note = (
            f"earliest {max_rows} labelled rows after sorting by TransactionDT; "
            "this development subset is not the full competition year"
        )
    identity_joined = False
    if identity_path.is_file():
        identity_header = _header(identity_path)
        identity_columns = [column for column in identity_header if column == "TransactionID" or column in columns or column.startswith("id_") or column in {"DeviceType", "DeviceInfo"}]
        identity_columns = list(dict.fromkeys(identity_columns))
        identity = pd.read_csv(identity_path, usecols=identity_columns)
        overlap = [column for column in identity.columns if column != "TransactionID" and column in frame.columns]
        identity = identity.drop(columns=overlap)
        frame = frame.merge(identity, on="TransactionID", how="left")
        identity_joined = True
    metadata = {
        "dataset": "ieee-cis",
        "source": "https://www.kaggle.com/competitions/ieee-fraud-detection/data",
        "rows_loaded": int(len(frame)),
        "subset": subset_note,
        "transaction_dt_was_sorted": monotonic,
        "identity_joined": identity_joined,
        "label_column": "isFraud",
        "time_column": "TransactionDT",
        "profile": profile_frame(frame, label_column="isFraud", id_column="TransactionID"),
        "files": [file_record(transaction_path)],
        "dropped_from_features": ["TransactionID", "isFraud", "TransactionDT", "card1", "addr1"],
        "notes": [
            "The official test CSV has no public labels, so evaluation uses a later time slice of the labelled file.",
            "card1 and addr1 are linkage keys. They are frequency-encoded or used as graph keys, not copied through as raw identifiers into every model explanation log.",
            "Identity fields are anonymised public benchmark fields, not newly collected customer biometrics.",
        ],
    }
    if identity_path.is_file():
        metadata["files"].append(file_record(identity_path))
    return frame, metadata


def write_ieee_metadata(metadata: dict[str, Any]) -> Path:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    target = METADATA_DIR / "ieee_cis.json"
    target.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return target


def ieee_available() -> bool:
    return ieee_ready(ieee_paths())
