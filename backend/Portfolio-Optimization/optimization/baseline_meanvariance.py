"""Classic Markowitz mean-variance optimizer -- the RQ2 comparison point.

NOT a throwaway. The TAF's literature review states that commercial robo-advisors
"automate long-term portfolio allocation using mean-variance optimization but offer no
instant, loss-minimized withdrawal mechanism". This module IS that incumbent. The entire
novelty claim of Component 1 rests on beating it, so it must be implemented properly and
given a fair shot -- a strawman baseline would invalidate the result rather than support it.

Two objectives only (return, variance). No liquidity term, by definition: its absence is
precisely the gap the MOEA/D optimizer is meant to close.

Solved with scipy SLSQP rather than a dedicated QP solver, to avoid adding cvxpy to an
already tight dependency set. For a 26-asset universe the difference is immaterial.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from optimization.objectives import portfolio_volatility

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MeanVarianceResult:
    weights: np.ndarray
    expected_return: float
    variance: float
    sharpe: float


def _bounds(n: int, allow_shorting: bool, max_weight: float | None) -> list[tuple[float, float]]:
    lower = -1.0 if allow_shorting else 0.0
    upper = max_weight if max_weight is not None else 1.0
    return [(lower, upper)] * n


def _solve(
    objective,
    n: int,
    *,
    allow_shorting: bool,
    max_weight: float | None,
    extra_constraints: list[dict] | None = None,
) -> np.ndarray | None:
    """SLSQP from an equal-weight start, with the sum-to-1 budget constraint."""
    constraints: list[dict] = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if extra_constraints:
        constraints.extend(extra_constraints)

    start = np.full(n, 1.0 / n)
    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=_bounds(n, allow_shorting, max_weight),
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        logger.debug("SLSQP did not converge: %s", result.message)
        return None
    return np.asarray(result.x, dtype=float)


def _result(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, risk_free_rate: float) -> MeanVarianceResult:
    expected = float(np.dot(weights, mu))
    vol = portfolio_volatility(weights, cov)
    return MeanVarianceResult(
        weights=weights,
        expected_return=expected,
        variance=float(vol**2),
        sharpe=float((expected - risk_free_rate) / vol) if vol > 0 else 0.0,
    )


def efficient_frontier(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    *,
    n_points: int = 50,
    allow_shorting: bool = False,
    max_weight: float | None = None,
    risk_free_rate: float = 0.0,
) -> list[MeanVarianceResult]:
    """Trace the frontier: min w'Sw subject to w'mu = target, for a ladder of targets."""
    mu = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    n = mu.size

    # Target range is bounded by what the weight caps actually permit -- with a 25% cap you
    # cannot reach the best single asset's return, and asking for it yields no solution.
    min_target = float(_result(min_variance_portfolio(cov, allow_shorting=allow_shorting,
                                                     max_weight=max_weight).weights, mu, cov, 0.0).expected_return)
    cap = max_weight if max_weight is not None else 1.0
    top_k = max(1, int(np.ceil(1.0 / cap)))
    max_target = float(np.sort(mu)[-top_k:].mean())

    if max_target <= min_target:
        max_target = min_target + abs(min_target) * 0.5 + 1e-6

    frontier: list[MeanVarianceResult] = []
    for target in np.linspace(min_target, max_target, n_points):
        weights = _solve(
            lambda w: float(w @ cov @ w),
            n,
            allow_shorting=allow_shorting,
            max_weight=max_weight,
            extra_constraints=[{"type": "eq", "fun": lambda w, t=target: float(np.dot(w, mu)) - t}],
        )
        if weights is not None:
            frontier.append(_result(weights, mu, cov, risk_free_rate))

    if not frontier:
        logger.warning("efficient frontier is empty; falling back to min-variance only")
        frontier = [min_variance_portfolio(cov, allow_shorting=allow_shorting, max_weight=max_weight)]

    return frontier


def max_sharpe_portfolio(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    *,
    risk_free_rate: float = 0.0,
    allow_shorting: bool = False,
    max_weight: float | None = None,
) -> MeanVarianceResult:
    """The tangency portfolio -- the single allocation a robo-advisor would hand a user.

    This is the RQ2 head-to-head opponent for MOEA/D's recommended point.
    """
    mu = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(covariance, dtype=float)

    def negative_sharpe(w: np.ndarray) -> float:
        vol = portfolio_volatility(w, cov)
        if vol <= 0:
            return 1e6
        return -float((np.dot(w, mu) - risk_free_rate) / vol)

    weights = _solve(negative_sharpe, mu.size, allow_shorting=allow_shorting, max_weight=max_weight)
    if weights is None:
        logger.warning("max-Sharpe solve failed; returning min-variance portfolio")
        return min_variance_portfolio(cov, allow_shorting=allow_shorting, max_weight=max_weight)
    return _result(weights, mu, cov, risk_free_rate)


def min_variance_portfolio(
    covariance: np.ndarray,
    *,
    allow_shorting: bool = False,
    max_weight: float | None = None,
) -> MeanVarianceResult:
    """Global minimum-variance portfolio; the conservative end of the frontier."""
    cov = np.asarray(covariance, dtype=float)
    n = cov.shape[0]

    weights = _solve(lambda w: float(w @ cov @ w), n,
                     allow_shorting=allow_shorting, max_weight=max_weight)
    if weights is None:
        logger.warning("min-variance solve failed; returning equal weights")
        weights = np.full(n, 1.0 / n)

    return _result(weights, np.zeros(n), cov, 0.0)
