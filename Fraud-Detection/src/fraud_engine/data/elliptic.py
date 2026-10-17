"""Elliptic Bitcoin transaction graph.

Official files are distributed by Elliptic through Kaggle under CC BY-NC-ND 4.0.
This loader keeps derived tables local and does not redistribute them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_engine.data.download import locate_file
from fraud_engine.data.validate import file_record, profile_frame
from fraud_engine.paths import RAW_DIR


def elliptic_dir(root: Path | None = None) -> Path:
    return root or (RAW_DIR / "elliptic")


def _read_table(path: Path) -> pd.DataFrame:
    sample = pd.read_csv(path, nrows=1, header=None)
    first = str(sample.iloc[0, 0])
    header = 0 if not _looks_numeric(first) else None
    return pd.read_csv(path, header=header)


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def load_elliptic(root: Path | None = None, max_nodes: int | None = 20000) -> dict[str, Any]:
    folder = elliptic_dir(root)
    feature_path = locate_file(folder, "elliptic_txs_features.csv")
    class_path = locate_file(folder, "elliptic_txs_classes.csv")
    edge_path = locate_file(folder, "elliptic_txs_edgelist.csv")
    missing = [
        name
        for name, path in (
            ("elliptic_txs_features.csv", feature_path),
            ("elliptic_txs_classes.csv", class_path),
            ("elliptic_txs_edgelist.csv", edge_path),
        )
        if path is None
    ]
    if missing or feature_path is None or class_path is None or edge_path is None:
        raise FileNotFoundError(
            f"Elliptic files missing ({missing}) in {folder}. See Fraud-Detection/docs/DATASETS.md."
        )
    features = _read_table(feature_path)
    classes = _read_table(class_path)
    edges = _read_table(edge_path)
    features = features.rename(columns={features.columns[0]: "txId", features.columns[1]: "time_step"})
    classes = classes.rename(columns={classes.columns[0]: "txId", classes.columns[1]: "class"})
    edges = edges.rename(columns={edges.columns[0]: "source", edges.columns[1]: "target"})
    features["txId"] = features["txId"].astype(str)
    classes["txId"] = classes["txId"].astype(str)
    merged = features.merge(classes, on="txId", how="left")
    class_text = merged["class"].astype(str).str.lower()
    labels = np.full(len(merged), -1, dtype=np.int64)
    labels[class_text.isin(["2", "licit"])] = 0
    labels[class_text.isin(["1", "illicit"])] = 1
    time_step = pd.to_numeric(merged["time_step"], errors="coerce").fillna(0).to_numpy()
    # Weber et al. test from time step 35. Validation is taken from the end of
    # their training period so the published test window is not used for tuning.
    split = np.full(len(merged), 2, dtype=np.int8)
    split[time_step <= 30] = 0
    split[(time_step >= 31) & (time_step <= 34)] = 1
    split[time_step >= 35] = 2
    keep = np.ones(len(merged), dtype=bool)
    subset = "all nodes"
    if max_nodes is not None and len(merged) > max_nodes:
        keep = _cap_nodes(split, labels, max_nodes)
        subset = f"induced subgraph of {int(keep.sum())} nodes"
    id_to_index = {tx_id: index for index, tx_id in enumerate(merged.loc[keep, "txId"].tolist())}
    sources = edges["source"].astype(str).map(id_to_index)
    targets = edges["target"].astype(str).map(id_to_index)
    valid = sources.notna() & targets.notna()
    edge_index = np.vstack(
        [sources[valid].to_numpy(dtype=np.int64), targets[valid].to_numpy(dtype=np.int64)]
    )
    feature_matrix = merged.loc[keep].drop(columns=["txId", "time_step", "class"]).apply(pd.to_numeric, errors="coerce")
    feature_matrix = feature_matrix.fillna(feature_matrix.median(numeric_only=True)).to_numpy(dtype=np.float32)
    return {
        "dataset": "elliptic",
        "license": "CC BY-NC-ND 4.0",
        "source": "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set",
        "x": np.nan_to_num(feature_matrix),
        "y": labels[keep],
        "split": split[keep],
        "edge_index": edge_index if edge_index.size else np.zeros((2, 0), dtype=np.int64),
        "subset": subset,
        "profile": profile_frame(merged.loc[keep, ["txId", "class", "time_step"]], id_column="txId"),
        "files": [file_record(path) for path in (feature_path, class_path, edge_path)],
        "notes": [
            "Positive class is illicit (original class 1). Licit is class 2. Unknown nodes stay in the graph with label -1 and are excluded from supervised metrics.",
            "Do not publish modified copies of this dataset. The licence is CC BY-NC-ND 4.0.",
        ],
    }


def _cap_nodes(split: np.ndarray, labels: np.ndarray, max_nodes: int) -> np.ndarray:
    """Keep a deterministic mix of train, validation, and test nodes."""
    keep = np.zeros(len(split), dtype=bool)
    quota = max(1, max_nodes // 3)
    for code in (0, 1, 2):
        candidates = np.flatnonzero((split == code) & (labels >= 0))
        keep[candidates[:quota]] = True
    return keep
