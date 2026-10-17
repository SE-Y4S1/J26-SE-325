"""Tabular baselines: logistic regression, linear SVM, and random forest.

These models see one transaction at a time. They do not see the sequence or the graph.
Class weights are computed from the training split only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def _both_classes(y: np.ndarray) -> None:
    if len(np.unique(y)) < 2:
        raise ValueError("The training split needs both classes.")


def fit_logistic(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> LogisticRegression:
    _both_classes(y_train)
    model = LogisticRegression(
        max_iter=300,
        class_weight="balanced",
        solver="liblinear",
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model


def fit_svm(x_train: np.ndarray, y_train: np.ndarray, seed: int, max_iter: int) -> LinearSVC:
    _both_classes(y_train)
    model = LinearSVC(
        class_weight="balanced",
        max_iter=max_iter,
        random_state=seed,
        dual="auto",
    )
    model.fit(x_train, y_train)
    return model


def fit_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    trees: int,
    depth: int,
) -> RandomForestClassifier:
    _both_classes(y_train)
    model = RandomForestClassifier(
        n_estimators=trees,
        max_depth=depth,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model


def positive_scores(model: Any, features: np.ndarray) -> np.ndarray:
    """Higher means more fraud-like. Probabilities are used when the model has them."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1].astype(np.float64)
    return model.decision_function(features).astype(np.float64)


def model_parameters(model: Any) -> dict[str, Any]:
    if hasattr(model, "get_params"):
        raw = model.get_params(deep=False)
        return {key: _plain(value) for key, value in raw.items() if _plain(value) is not None}
    return {}


def _plain(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None
