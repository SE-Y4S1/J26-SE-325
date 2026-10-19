"""Load YAML profiles. A profile overrides only the keys it sets."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from fraud_engine.paths import CONFIG_DIR


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_settings(profile: str = "default", config_path: str | None = None) -> dict[str, Any]:
    """Load default.yaml, then overlay a named profile or an explicit file."""
    default_path = CONFIG_DIR / "default.yaml"
    with default_path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle) or {}

    chosen = config_path or os.environ.get("FRAUD_CONFIG")
    if chosen:
        path = Path(chosen)
        with path.open("r", encoding="utf-8") as handle:
            overlay = yaml.safe_load(handle) or {}
        settings = _deep_merge(settings, overlay)
    elif profile and profile != "default":
        path = CONFIG_DIR / f"{profile}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Unknown profile '{profile}'. Expected {path}.")
        with path.open("r", encoding="utf-8") as handle:
            overlay = yaml.safe_load(handle) or {}
        settings = _deep_merge(settings, overlay)

    if os.environ.get("FRAUD_ENABLE_KAFKA"):
        settings.setdefault("kafka", {})
        settings["kafka"]["enabled"] = os.environ["FRAUD_ENABLE_KAFKA"].lower() in {"1", "true", "yes"}
    if os.environ.get("FRAUD_KAFKA_BOOTSTRAP"):
        settings.setdefault("kafka", {})
        settings["kafka"]["bootstrap_servers"] = os.environ["FRAUD_KAFKA_BOOTSTRAP"]
    if os.environ.get("FRAUD_API_PORT"):
        settings.setdefault("api", {})
        settings["api"]["port"] = int(os.environ["FRAUD_API_PORT"])
    return settings
