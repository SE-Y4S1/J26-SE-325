"""The routing methods compared. Each is fitted on the policy-tuning split only and returns a
function that routes any split.

Routes are coded HUMAN=0, VERIFY=1, AUTO=2. Two-tier baselines never use VERIFY.

  auto_all          M0  automate everything
  risk_band         M1  review the highest fraud probabilities (fraud-gateway style, risk only)
  confidence_only   M2  review the least confident decisions (selective prediction)
  p001_deployed     M3  the legacy P001 rule exactly as deployed (risk >= 80 and confidence < 70)
  p001_retuned      M3' P001's two thresholds re-tuned to the budget, so it competes fairly
  random_review     M4  review a random share
  assurance         M5  the assurance engine: weighted confidence/disagreement/context + veto
  assurance_no_*    M5 ablations: one signal's weight forced to 0
  learned_deferral  M6  logistic regression predicting "this decision is wrong" from the same
                        signals -- a strong, less interpretable baseline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import config

HUMAN, VERIFY, AUTO = 0, 1, 2
Signals = dict[str, np.ndarray]


def errors(s: Signals) -> np.ndarray:
    return (s["decision"] != s["y"]).astype(int)


def incorrect_automatic(routes: np.ndarray, s: Signals) -> float:
    """Primary metric: share of ALL decisions that were processed without a human and were
    wrong. VERIFY counts as automatic here -- the engine gets no credit for step-up."""
    return float(np.mean(errors(s) * (routes != HUMAN)))


@dataclass
class Fitted:
    name: str
    route: Callable[[Signals], np.ndarray]
    params: dict = field(default_factory=dict)
    rank: Callable[[Signals], np.ndarray] | None = None   # higher = defer first (for curves)


def _top_share(score: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(score >= threshold, HUMAN, AUTO)


def _quantile_threshold(score: np.ndarray, share: float) -> float:
    return float(np.quantile(score, 1 - share)) if share > 0 else np.inf


def fit_auto_all(_: Signals, budget: float) -> Fitted:
    return Fitted("auto_all", lambda s: np.full(len(s["p"]), AUTO), {})


def fit_random(_: Signals, budget: float, seed: int = config.SEED) -> Fitted:
    def route(s: Signals) -> np.ndarray:
        # Exactly the budgeted share, chosen at random (seeded), so its workload matches.
        n = len(s["p"])
        routes = np.full(n, AUTO)
        chosen = np.random.default_rng(seed + n).permutation(n)[: int(round(budget * n))]
        routes[chosen] = HUMAN
        return routes
    return Fitted("random_review", route, {"share": budget})


def fit_risk_band(tune: Signals, budget: float) -> Fitted:
    t = _quantile_threshold(tune["p"], budget)
    return Fitted("risk_band", lambda s: _top_share(s["p"], t), {"pThreshold": t}, rank=lambda s: s["p"])


def fit_confidence_only(tune: Signals, budget: float) -> Fitted:
    t = _quantile_threshold(tune["u_confidence"], budget)
    return Fitted("confidence_only", lambda s: _top_share(s["u_confidence"], t), {"uThreshold": t},
                  rank=lambda s: s["u_confidence"])


def _p001(s: Signals, risk_threshold: float, confidence_threshold: float) -> np.ndarray:
    risk = 100 * s["p"]
    confidence = 100 * (1 - s["u_confidence"])
    return np.where((risk >= risk_threshold) & (confidence < confidence_threshold), HUMAN, AUTO)


def fit_p001_deployed(_: Signals, budget: float) -> Fitted:
    return Fitted("p001_deployed", lambda s: _p001(s, 80, 70), {"riskThreshold": 80, "confidenceThreshold": 70})


def fit_p001_retuned(tune: Signals, budget: float) -> Fitted:
    best = None
    for risk_t in np.unique(np.quantile(100 * tune["p"], np.linspace(0.0, 0.99, 34))):
        for conf_t in np.linspace(5, 100, 20):
            routes = _p001(tune, risk_t, conf_t)
            if np.mean(routes == HUMAN) > budget:
                continue
            score = incorrect_automatic(routes, tune)
            if best is None or score < best[0]:
                best = (score, float(risk_t), float(conf_t))
    _, risk_t, conf_t = best
    return Fitted("p001_retuned", lambda s: _p001(s, risk_t, conf_t), {"riskThreshold": risk_t, "confidenceThreshold": conf_t})


# --- the assurance engine ----------------------------------------------------------------------

def assurance_score(s: Signals, w: tuple[float, float, float]) -> np.ndarray:
    return np.clip(1 - (w[0] * s["u_confidence"] + w[1] * s["u_disagreement"] + w[2] * s["u_context"]), 0, 1)


def _assurance_routes(s: Signals, w, max_disagreement: float, verify_min: float, auto_min: float) -> np.ndarray:
    score = assurance_score(s, w)
    routes = np.where(score >= auto_min, AUTO, np.where(score >= verify_min, VERIFY, HUMAN))
    routes[s["u_disagreement"] > max_disagreement] = HUMAN      # veto
    return routes


def _thresholds_for(score: np.ndarray, vetoed: np.ndarray, budget: float, verify_budget: float):
    """verifyMin so that vetoes + low scores fill the human budget; autoMin so that the next
    band fills the verify budget. None when the vetoes alone exceed the budget."""
    veto_share = vetoed.mean()
    if veto_share > budget:
        return None
    free = score[~vetoed]
    extra = (budget - veto_share) / max(1 - veto_share, 1e-12)
    verify_min = float(np.quantile(free, extra)) if extra > 0 else 0.0
    extra_verify = min(1.0, extra + verify_budget / max(1 - veto_share, 1e-12))
    auto_min = float(np.quantile(free, extra_verify))
    return verify_min, max(auto_min, verify_min)


def weight_grid(step: float = 0.05, fixed_zero: str | None = None):
    names = ["confidence", "disagreement", "context"]
    k = int(round(1 / step))
    for a in range(k + 1):
        for b in range(k + 1 - a):
            w = (a * step, b * step, 1 - (a + b) * step)
            if fixed_zero is not None and w[names.index(fixed_zero)] > 1e-9:
                continue
            yield tuple(round(x, 4) for x in w)


VETO_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 1.0]   # 1.0 = no disagreement veto


def fit_assurance(tune: Signals, budget: float, fixed_zero: str | None = None,
                  verify_budget: float = config.VERIFY_BUDGET) -> Fitted:
    best = None
    for w in weight_grid(fixed_zero=fixed_zero):
        score = assurance_score(tune, w)
        for max_d in VETO_GRID if fixed_zero != "disagreement" else [1.0]:
            vetoed = tune["u_disagreement"] > max_d
            thresholds = _thresholds_for(score, vetoed, budget, verify_budget)
            if thresholds is None:
                continue
            routes = _assurance_routes(tune, w, max_d, *thresholds)
            objective = incorrect_automatic(routes, tune)
            if best is None or objective < best[0] - 1e-12:
                best = (objective, w, max_d, thresholds)
    _, w, max_d, (verify_min, auto_min) = best
    name = "assurance" if fixed_zero is None else f"assurance_no_{fixed_zero}"
    params = {"weights": {"confidence": w[0], "disagreement": w[1], "context": w[2]},
              "maxDisagreement": max_d, "verifyMin": verify_min, "autoMin": auto_min}
    return Fitted(name, lambda s: _assurance_routes(s, w, max_d, verify_min, auto_min), params,
                  rank=lambda s: 1 - assurance_score(s, w) + (s["u_disagreement"] > max_d))


# --- learned deferral (M6) -------------------------------------------------------------------

LEARNED_FEATURES = ["p", "u_confidence", "u_disagreement", "ctx_amount", "ctx_novelty", "ctx_new_card", "ctx_no_identity"]


def _features(s: Signals) -> np.ndarray:
    return np.column_stack([s[k] for k in LEARNED_FEATURES])


def fit_learned_deferral(tune: Signals, budget: float) -> Fitted:
    x, err = _features(tune), errors(tune)
    if err.min() == err.max():  # degenerate (tiny fixtures): fall back to confidence ranking
        fallback = fit_confidence_only(tune, budget)
        return Fitted("learned_deferral", fallback.route, {"fallback": "confidence_only"}, rank=fallback.rank)
    # Fitted on the tuning split's own error labels (the test split is never seen). The budget
    # threshold is a quantile of the model's scores -- it uses no labels -- so the workload on
    # the tuning split is exactly the budget, as for every other method.
    model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(x, err)
    risk = lambda s: model.predict_proba(_features(s))[:, 1]
    t = _quantile_threshold(risk(tune), budget)
    return Fitted("learned_deferral", lambda s: _top_share(risk(s), t), {"threshold": t}, rank=risk)


def fit_all(tune: Signals, budget: float) -> dict[str, Fitted]:
    fitted = [
        fit_auto_all(tune, budget),
        fit_risk_band(tune, budget),
        fit_confidence_only(tune, budget),
        fit_p001_deployed(tune, budget),
        fit_p001_retuned(tune, budget),
        fit_random(tune, budget),
        fit_assurance(tune, budget),
        fit_assurance(tune, budget, fixed_zero="confidence"),
        fit_assurance(tune, budget, fixed_zero="disagreement"),
        fit_assurance(tune, budget, fixed_zero="context"),
        fit_learned_deferral(tune, budget),
    ]
    return {f.name: f for f in fitted}
