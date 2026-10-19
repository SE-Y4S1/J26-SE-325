"""DGraphFin loader.

The official site requires an account and acceptance of a non-commercial research
licence. This loader only reads a zip or npz that is already on disk. It samples
a subgraph when the full graph is larger than the configured node cap.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from fraud_engine.data.validate import file_record
from fraud_engine.paths import RAW_DIR


def dgraph_dir(root: Path | None = None) -> Path:
    return root or (RAW_DIR / "dgraphfin")


def resolve_npz(root: Path | None = None) -> Path:
    folder = dgraph_dir(root)
    npz = folder / "dgraphfin.npz"
    archive = folder / "DGraphFin.zip"
    if not npz.is_file() and archive.is_file():
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(folder)
        extracted = list(folder.rglob("dgraphfin.npz"))
        if extracted and extracted[0] != npz:
            extracted[0].replace(npz)
    if not npz.is_file():
        raise FileNotFoundError(
            "DGraphFin.npz is not available. Download it manually from https://dgraph.xinye.com/ "
            f"and place DGraphFin.zip in {folder}."
        )
    return npz


def load_dgraphfin(root: Path | None = None, max_nodes: int | None = 10000) -> dict[str, Any]:
    path = resolve_npz(root)
    with np.load(path, mmap_mode="r") as payload:
        features = payload["x"]
        labels = np.array(payload["y"]).reshape(-1)
        edge_index = np.array(payload["edge_index"])
        train_nodes = np.array(payload["train_mask"]).reshape(-1)
        val_nodes = np.array(payload["valid_mask"]).reshape(-1)
        test_nodes = np.array(payload["test_mask"]).reshape(-1)
        node_count = int(features.shape[0])
        split = np.full(node_count, -1, dtype=np.int8)
        split[_as_indices(train_nodes, node_count)] = 0
        split[_as_indices(val_nodes, node_count)] = 1
        split[_as_indices(test_nodes, node_count)] = 2
        supervised = (labels == 0) | (labels == 1)
        keep = (split >= 0) & supervised
        subset = "official labelled nodes"
        if max_nodes is not None and int(keep.sum()) > max_nodes:
            keep = _sample_labelled(split, supervised, max_nodes)
            subset = f"first {int(keep.sum())} labelled nodes spread across the official masks"
        chosen = np.flatnonzero(keep)
        remap = -np.ones(node_count, dtype=np.int64)
        remap[chosen] = np.arange(len(chosen))
        if edge_index.ndim != 2:
            raise ValueError("DGraphFin edge_index must be a 2-dimensional array.")
        if edge_index.shape[0] == 2 and edge_index.shape[1] != 2:
            sources, targets = edge_index[0], edge_index[1]
        else:
            sources, targets = edge_index[:, 0], edge_index[:, 1]
        sources = sources.astype(np.int64, copy=False)
        targets = targets.astype(np.int64, copy=False)
        valid = (remap[sources] >= 0) & (remap[targets] >= 0)
        mapped_edges = np.vstack([remap[sources[valid]], remap[targets[valid]]])
        matrix = np.array(features[chosen], dtype=np.float32)
    return {
        "dataset": "dgraphfin",
        "license": "Finvolution custom non-commercial research terms",
        "source": "https://dgraph.xinye.com/",
        "x": np.nan_to_num(matrix),
        "y": labels[chosen].astype(np.int64),
        "split": split[chosen],
        "edge_index": mapped_edges.astype(np.int64),
        "subset": subset,
        "files": [file_record(path)],
        "notes": [
            "Labels other than 0 (normal) and 1 (fraud) are excluded from supervised metrics.",
            "The official train, validation, and test node masks are preserved inside the sample.",
            "A sampled subgraph is not comparable to a published full-graph DGraphFin number.",
        ],
    }


def _as_indices(mask: np.ndarray, node_count: int) -> np.ndarray:
    mask = mask.astype(np.int64, copy=False)
    if mask.size == node_count and set(np.unique(mask[: min(1000, mask.size)]).tolist()).issubset({0, 1}):
        return np.flatnonzero(mask.astype(bool))
    return mask[(mask >= 0) & (mask < node_count)]


def _sample_labelled(split: np.ndarray, supervised: np.ndarray, max_nodes: int) -> np.ndarray:
    keep = np.zeros(len(split), dtype=bool)
    quota = max(1, max_nodes // 3)
    for code in (0, 1, 2):
        candidates = np.flatnonzero((split == code) & supervised)
        keep[candidates[:quota]] = True
    return keep
