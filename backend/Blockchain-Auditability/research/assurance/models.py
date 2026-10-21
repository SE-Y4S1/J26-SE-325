"""The AI whose decisions are being assured: three different fraud models, each fitted on the
training split and calibrated (isotonic) on the calibration split.

Three model families are used on purpose -- a linear model, a bagged tree ensemble and a
boosted tree ensemble -- because disagreement is only informative between models that make
different kinds of mistakes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config


def base_models(seed: int = config.SEED, fast: bool = False) -> dict:
    trees = 40 if fast else 300
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "forest": RandomForestClassifier(
            n_estimators=trees, max_depth=14, min_samples_leaf=20, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed,
        ),
        "boosting": HistGradientBoostingClassifier(
            max_iter=60 if fast else 400, learning_rate=0.06, max_leaf_nodes=48, class_weight="balanced",
            random_state=seed,
        ),
    }


@dataclass
class CalibratedEnsemble:
    models: dict
    calibrators: dict

    @classmethod
    def fit(cls, x_train, y_train, x_cal, y_cal, seed: int = config.SEED, fast: bool = False) -> "CalibratedEnsemble":
        models = base_models(seed, fast)
        calibrators = {}
        for name, model in models.items():
            model.fit(x_train, y_train)
            raw = model.predict_proba(x_cal)[:, 1]
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(raw, y_cal)
            calibrators[name] = iso
        return cls(models=models, calibrators=calibrators)

    def member_probabilities(self, x) -> dict[str, np.ndarray]:
        return {name: self.calibrators[name].predict(model.predict_proba(x)[:, 1]) for name, model in self.models.items()}
