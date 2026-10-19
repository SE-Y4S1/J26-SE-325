"""Filesystem locations for Component 02."""

from pathlib import Path

COMPONENT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = COMPONENT_ROOT / "config"
DATA_DIR = COMPONENT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
ARTIFACT_DIR = COMPONENT_ROOT / "artifacts"
EXPERIMENT_DIR = COMPONENT_ROOT / "experiments"
RESULT_DIR = EXPERIMENT_DIR / "results"
LOG_DIR = EXPERIMENT_DIR / "logs"
