"""Save and load the trained preprocessor, neural weights, and thresholds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import torch

from fraud_engine.models.fusion import ScoreFusion
from fraud_engine.models.gnn import MeanAggregatorNet
from fraud_engine.models.lstm_ae import LSTMAutoencoder


def save_bundle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = dict(payload)
    if "lstm_state" in stored:
        stored["lstm_state"] = {key: value.detach().cpu() for key, value in stored["lstm_state"].items()}
    if "gnn_state" in stored:
        stored["gnn_state"] = {key: value.detach().cpu() for key, value in stored["gnn_state"].items()}
    joblib.dump(stored, path)


def load_bundle(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return joblib.load(path)


def load_lstm(bundle: dict[str, Any]) -> LSTMAutoencoder:
    kwargs = bundle["lstm_kwargs"]
    model = LSTMAutoencoder(n_features=kwargs["n_features"], hidden=kwargs["hidden"])
    model.load_state_dict(bundle["lstm_state"])
    model.eval()
    return model


def load_gnn(bundle: dict[str, Any]) -> MeanAggregatorNet:
    kwargs = bundle["gnn_kwargs"]
    model = MeanAggregatorNet(n_features=kwargs["n_features"], hidden=kwargs["hidden"])
    model.load_state_dict(bundle["gnn_state"])
    model.eval()
    return model


def fusion_from_bundle(bundle: dict[str, Any]) -> ScoreFusion:
    fusion = ScoreFusion(seed=bundle.get("seed", 42))
    fusion.alpha = float(bundle.get("fusion_alpha", 0.5))
    fusion.flip_first = bool(bundle.get("fusion_flip_first", False))
    fusion.flip_second = bool(bundle.get("fusion_flip_second", False))
    fusion.fallback = bundle.get("fusion_fallback", "equal_mixture")
    fusion.weights = list(bundle.get("fusion_weights", [fusion.alpha, 1.0 - fusion.alpha]))
    return fusion


def torch_available() -> bool:
    return torch.__version__ is not None
