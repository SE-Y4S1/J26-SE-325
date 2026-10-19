"""Public dataset loaders and the Kaggle / manual download workflow.

This module never reads a token from source code. Kaggle commands use the
official credential file or environment variables that the student creates.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

from fraud_engine.data.validate import file_record
from fraud_engine.paths import METADATA_DIR, RAW_DIR

IEEE_FILES = (
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
    "sample_submission.csv",
)
ELLIPTIC_FILES = (
    "elliptic_txs_features.csv",
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv",
)
DGRAPH_NPZ = "dgraphfin.npz"
DGRAPH_ZIP = "DGraphFin.zip"

REAL_DATASETS = {"ieee-cis", "dgraphfin", "elliptic"}


def kaggle_credentials_present() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return Path.home().joinpath(".kaggle", "kaggle.json").is_file()


def dataset_dir(name: str) -> Path:
    mapping = {
        "ieee-cis": RAW_DIR / "ieee-cis",
        "elliptic": RAW_DIR / "elliptic",
        "dgraphfin": RAW_DIR / "dgraphfin",
    }
    if name not in mapping:
        raise ValueError(f"Unknown dataset '{name}'.")
    return mapping[name]


def ieee_ready(root: Path | None = None) -> bool:
    folder = root or dataset_dir("ieee-cis")
    return (folder / "train_transaction.csv").is_file() and (folder / "train_identity.csv").is_file()


def locate_file(folder: Path, name: str) -> Path | None:
    """Find a dataset file in the folder or one nested archive directory."""
    direct = folder / name
    if direct.is_file():
        return direct
    matches = [path for path in folder.rglob(name) if path.is_file()]
    if not matches:
        return None
    return min(matches, key=lambda path: (len(path.parts), str(path).lower()))


def elliptic_ready(root: Path | None = None) -> bool:
    folder = root or dataset_dir("elliptic")
    return all(locate_file(folder, name) is not None for name in ELLIPTIC_FILES)


def dgraph_ready(root: Path | None = None) -> bool:
    folder = root or dataset_dir("dgraphfin")
    return (folder / DGRAPH_NPZ).is_file() or (folder / DGRAPH_ZIP).is_file()


def _run_kaggle(arguments: list[str], destination: Path) -> None:
    if not kaggle_credentials_present():
        raise RuntimeError(
            "Kaggle credentials were not found. Read Fraud-Detection/docs/DATASETS.md. "
            "Place kaggle.json in %USERPROFILE%\\.kaggle\\ or set KAGGLE_API_TOKEN. "
            "Do not paste the token into source code or chat."
        )
    executable = shutil.which("kaggle")
    if executable is None:
        command = [sys.executable, "-m", "kaggle", *arguments, "-p", str(destination)]
    else:
        command = [executable, *arguments, "-p", str(destination)]
    destination.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"Kaggle download failed (exit {completed.returncode}): {detail[:800]}")


def download_ieee(destination: Path | None = None) -> dict[str, Any]:
    folder = destination or dataset_dir("ieee-cis")
    _run_kaggle(["competitions", "download", "-c", "ieee-fraud-detection"], folder)
    _extract_zips(folder)
    return describe_folder(folder, IEEE_FILES)


def download_elliptic(destination: Path | None = None) -> dict[str, Any]:
    folder = destination or dataset_dir("elliptic")
    _run_kaggle(["datasets", "download", "-d", "ellipticco/elliptic-data-set"], folder)
    _extract_zips(folder)
    return describe_folder(folder, ELLIPTIC_FILES)


def prepare_dgraph_manual(destination: Path | None = None) -> dict[str, Any]:
    """Unzip a manually downloaded DGraphFin archive. Never downloads it."""
    folder = destination or dataset_dir("dgraphfin")
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / DGRAPH_ZIP
    if archive.is_file() and not (folder / DGRAPH_NPZ).is_file():
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(folder)
    if not (folder / DGRAPH_NPZ).is_file():
        raise RuntimeError(
            "DGraphFin was not found. Register at https://dgraph.xinye.com/, accept the dataset terms, "
            f"download {DGRAPH_ZIP}, and place it in {folder}. Automatic download is not available."
        )
    return describe_folder(folder, (DGRAPH_NPZ,))


def _extract_zips(folder: Path) -> None:
    for archive in folder.glob("*.zip"):
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(folder)


def describe_folder(folder: Path, expected: tuple[str, ...]) -> dict[str, Any]:
    present = []
    missing = []
    for name in expected:
        path = locate_file(folder, name)
        if path is not None:
            present.append(file_record(path))
        else:
            missing.append(name)
    record = {"folder": str(folder), "present": present, "missing": missing}
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    return record
