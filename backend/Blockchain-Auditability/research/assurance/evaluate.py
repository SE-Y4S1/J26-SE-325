"""Metrics, confidence intervals and the paired comparison, identical for every method."""

from __future__ import annotations

import numpy as np

from . import config
from .routers import AUTO, HUMAN, VERIFY, Fitted, Signals, errors


def point_metrics(routes: np.ndarray, s: Signals) -> dict:
    err = errors(s)
    human, verify, auto = routes == HUMAN, routes == VERIFY, routes == AUTO
    unattended = ~human
    fraud_missed = (s["y"] == 1) & (s["decision"] == 0) & unattended
    legit_rejected = (s["y"] == 0) & (s["decision"] == 1) & unattended
    out = {
        "n": int(len(err)),
        "incorrect_automatic_rate": float(np.mean(err & unattended)),
        "human_review_rate": float(np.mean(human)),
        "verify_rate": float(np.mean(verify)),
        "automatic_rate": float(np.mean(auto)),
        "error_rate_within_unattended": float(err[unattended].mean()) if unattended.any() else 0.0,
        "fraud_auto_approved": int(fraud_missed.sum()),
        "fraud_auto_approved_amount": float(s["amount_value"][fraud_missed].sum()),
        "legit_auto_rejected": int(legit_rejected.sum()),
        "base_error_rate": float(err.mean()),
    }
    # Sensitivity to assumptions that are not measured.
    out["verify_effectiveness_sweep"] = {
        str(e): float(np.mean(err & auto) + (1 - e) * np.mean(err & verify)) for e in config.VERIFY_EFFECTIVENESS
    }
    out["analyst_accuracy_sweep"] = {
        str(a): float(np.mean(err & unattended) + (1 - a) * np.mean(human)) for a in config.ANALYST_ACCURACY
    }
    return out


def _resample_index(n: int, b: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(b, n))


def bootstrap(routes_by_method: dict[str, np.ndarray], s: Signals, reference: str, comparators: list[str],
              b: int = config.BOOTSTRAP_RESAMPLES, seed: int = config.SEED) -> dict:
    """95% percentile intervals for the two headline rates of every method, and the paired
    difference reference - comparator (negative = the reference automates fewer errors)."""
    err = errors(s).astype(bool)
    n = len(err)
    wrong_unattended = {m: (err & (r != HUMAN)).astype(np.float32) for m, r in routes_by_method.items()}
    reviewed = {m: (r == HUMAN).astype(np.float32) for m, r in routes_by_method.items()}
    stats = {m: {"ia": [], "hr": []} for m in routes_by_method}
    diffs = {c: [] for c in comparators}
    chunk = 50
    for start in range(0, b, chunk):
        idx = _resample_index(n, min(chunk, b - start), seed + start)
        for m in routes_by_method:
            stats[m]["ia"].append(wrong_unattended[m][idx].mean(axis=1))
            stats[m]["hr"].append(reviewed[m][idx].mean(axis=1))
        ref = wrong_unattended[reference][idx].mean(axis=1)
        for c in comparators:
            diffs[c].append(ref - wrong_unattended[c][idx].mean(axis=1))
    ci = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
    out = {"intervals": {}, "paired": {}}
    for m, v in stats.items():
        ia, hr = np.concatenate(v["ia"]), np.concatenate(v["hr"])
        out["intervals"][m] = {"incorrect_automatic_rate": ci(ia), "human_review_rate": ci(hr)}
    for c, v in diffs.items():
        d = np.concatenate(v)
        out["paired"][c] = {
            "difference": float(d.mean()),
            "ci95": ci(d),
            # one-sided: share of resamples where the reference is not better
            "p_reference_not_better": float(np.mean(d >= 0)),
        }
    return out


def deferral_curve(fitted: Fitted, s: Signals, max_share: float = 0.30, points: int = 31) -> list[dict]:
    """Incorrect-automatic rate against human-review rate when the method's own ranking decides
    what to defer (descriptive: shows ranking quality across budgets)."""
    if fitted.rank is None:
        return []
    err = errors(s)
    order = np.argsort(-fitted.rank(s), kind="mergesort")
    n = len(err)
    cumulative = np.concatenate([[0], np.cumsum(err[order])])
    total = err.sum()
    out = []
    for share in np.linspace(0, max_share, points):
        k = int(round(share * n))
        out.append({"human_review_rate": k / n, "incorrect_automatic_rate": float((total - cumulative[k]) / n)})
    return out


def aurc(fitted: Fitted, s: Signals) -> float | None:
    """Area under the risk-coverage curve (selective risk vs coverage), lower is better."""
    if fitted.rank is None:
        return None
    err = errors(s)
    order = np.argsort(fitted.rank(s), kind="mergesort")          # most certain first
    accepted_errors = np.cumsum(err[order])
    coverage_counts = np.arange(1, len(err) + 1)
    return float(np.mean(accepted_errors / coverage_counts))
