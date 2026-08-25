"""Choosing one allocation from the Pareto front.

Any front-to-point rule embeds a value judgement, so rather than hard-coding one we
implement all three and report the sensitivity in Phase 7. That converts an arbitrary
methodological choice into a small empirical result -- and pre-empts the obvious examiner
question, "why that point?"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

logger = logging.getLogger(__name__)


class SelectionRule(StrEnum):
    KNEE = "knee"
    MAX_SHARPE = "max_sharpe"
    SCALARIZED = "scalarized"


@dataclass(frozen=True)
class SelectedPoint:
    index: int
    weights: np.ndarray
    objectives: np.ndarray
    rule: SelectionRule
    rationale: str      # human-readable justification, surfaced in the API response


def _normalize_objectives(objectives: np.ndarray) -> np.ndarray:
    """Min-max each objective to [0, 1].

    Required before any geometric reasoning: return is ~0.001, CVaR ~0.02 and liquidity cost
    ~0.0005, so an un-normalized distance would be dominated entirely by whichever objective
    happens to have the largest units.
    """
    obj = np.atleast_2d(np.asarray(objectives, dtype=float))
    span = obj.max(axis=0) - obj.min(axis=0)
    span = np.where(span > 0, span, 1.0)     # a degenerate objective contributes nothing
    return (obj - obj.min(axis=0)) / span


def knee_point(objectives: np.ndarray) -> int:
    """Maximum-curvature point on the normalized front.

    Implemented as maximum distance from the chord joining the front's two extreme points
    (the "Menger curvature" / utopia-line construction). Parameter-free and preference-free:
    the knee is where giving up more return stops buying a proportionate reduction in risk.
    Standard in the MOO literature (Das 1999; Branke et al. 2004, "Finding knees in
    multi-objective optimization"). Our default.
    """
    normalized = _normalize_objectives(objectives)
    if normalized.shape[0] < 3:
        return 0

    # The chord runs between the two most distant solutions on the front.
    distances = np.linalg.norm(normalized[:, None, :] - normalized[None, :, :], axis=-1)
    end_a, end_b = np.unravel_index(np.argmax(distances), distances.shape)
    start, finish = normalized[end_a], normalized[end_b]

    chord = finish - start
    chord_length = np.linalg.norm(chord)
    if chord_length == 0:
        return 0
    chord_unit = chord / chord_length

    offsets = normalized - start
    projections = np.outer(offsets @ chord_unit, chord_unit)
    perpendicular = np.linalg.norm(offsets - projections, axis=1)

    return int(np.argmax(perpendicular))


def max_sharpe_point(objectives: np.ndarray, risk_free_rate: float = 0.0) -> int:
    """Best return-per-unit-risk on the front.

    Directly comparable to the mean-variance baseline's tangency portfolio, but note the
    circularity: RQ2 also evaluates on Sharpe, so this rule optimizes the metric it is
    scored on. Reported for completeness, not used as the headline.

    Objective column 0 is NEGATIVE expected return (pymoo minimizes), column 1 is CVaR.
    """
    obj = np.atleast_2d(np.asarray(objectives, dtype=float))
    returns = -obj[:, 0]
    risk = obj[:, 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(risk > 0, (returns - risk_free_rate) / risk, -np.inf)

    if not np.isfinite(ratio).any():
        logger.warning("no finite risk-adjusted ratio on the front; falling back to max return")
        return int(np.argmax(returns))
    return int(np.nanargmax(ratio))


def scalarized_point(objectives: np.ndarray, weights: tuple[float, float, float]) -> int:
    """Best weighted-sum point under explicit user preferences.

    Maps to a real risk-profile slider in the product, and is how a per-user allocation would
    actually be produced. The preference weights are an input, not a discovery -- which is
    both its strength (it encodes a real user) and its weakness (someone must justify them).
    """
    normalized = _normalize_objectives(objectives)
    preference = np.asarray(weights, dtype=float)

    if preference.size != normalized.shape[1]:
        raise ValueError(f"expected {normalized.shape[1]} preference weights, got {preference.size}")
    total = preference.sum()
    if total <= 0:
        raise ValueError("preference weights must sum to a positive value")

    return int(np.argmin(normalized @ (preference / total)))


def select(
    objectives: np.ndarray,
    portfolio_weights: np.ndarray,
    *,
    rule: SelectionRule = SelectionRule.KNEE,
    preference_weights: tuple[float, float, float] | None = None,
) -> SelectedPoint:
    """Dispatch to a rule and return the chosen point with its rationale."""
    obj = np.atleast_2d(np.asarray(objectives, dtype=float))
    weights = np.atleast_2d(np.asarray(portfolio_weights, dtype=float))

    if obj.shape[0] != weights.shape[0]:
        raise ValueError(f"{obj.shape[0]} objective rows but {weights.shape[0]} weight rows")
    if obj.shape[0] == 0:
        raise ValueError("cannot select from an empty Pareto front")

    rule = SelectionRule(rule)

    if rule is SelectionRule.KNEE:
        index = knee_point(obj)
        rationale = (
            "Knee of the Pareto front (maximum trade-off curvature): the point beyond which "
            "further risk reduction costs disproportionate return. Parameter-free, so no "
            "preference weights had to be assumed."
        )
    elif rule is SelectionRule.MAX_SHARPE:
        index = max_sharpe_point(obj)
        rationale = (
            "Highest return-per-unit-CVaR on the front -- directly comparable to the "
            "mean-variance tangency portfolio."
        )
    else:
        preference = preference_weights or (1 / 3, 1 / 3, 1 / 3)
        index = scalarized_point(obj, preference)
        rationale = (
            f"Best weighted-sum score under preference weights {tuple(round(p, 3) for p in preference)} "
            "over (return, CVaR, liquidity cost)."
        )

    return SelectedPoint(
        index=index,
        weights=weights[index],
        objectives=obj[index],
        rule=rule,
        rationale=rationale,
    )


def compare_rules(
    objectives: np.ndarray,
    portfolio_weights: np.ndarray,
    *,
    preference_weights: tuple[float, float, float] | None = None,
) -> dict[str, SelectedPoint]:
    """All three selections side by side -- the RQ2 sensitivity analysis.

    Phase 7 reports how much the headline result actually moves with the rule. If the three
    agree, the choice was immaterial and the result is robust; if they diverge sharply, that
    divergence is itself a finding worth reporting.
    """
    return {
        rule.value: select(
            objectives, portfolio_weights, rule=rule, preference_weights=preference_weights
        )
        for rule in SelectionRule
    }


def is_non_dominated(objectives: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows (all objectives minimized).

    Used to verify MOEA/D actually returned a front rather than an arbitrary population.
    """
    obj = np.atleast_2d(np.asarray(objectives, dtype=float))
    n = obj.shape[0]
    mask = np.ones(n, dtype=bool)

    for i in range(n):
        if not mask[i]:
            continue
        # j dominates i if it is <= on every objective and strictly < on at least one.
        dominates = np.all(obj <= obj[i], axis=1) & np.any(obj < obj[i], axis=1)
        if dominates.any():
            mask[i] = False

    return mask
