"""Exact solve of the allocation problem for one preference vector: the check on MOEA/D.

With scenario CVaR (Rockafellar & Uryasev 2000) and total slippage, the allocation problem

    minimize  a * (-w.mu)  +  b * CVaR_alpha(w)  +  c * slippage(w) / V

is CONVEX: the return term is linear, CVaR is a linear program over auxiliary variables, and
slippage is a sum of |w_i - w0_i|^1.5, which is convex. A convex problem has no business
being solved by an evolutionary algorithm unless the evolutionary front is checked against
the exact answer, and that is the job of this module: for any (a, b, c) the exact optimum
bounds what MOEA/D can reach, so a front that is worse is an optimizer gap, not a trade-off.

It does NOT replace MOEA/D (a TAF-named method that returns a whole front in one run). Each
call here yields one point; sweep (a, b, c) to trace a front.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize

from optimization.objectives import DEFAULT_DAILY_VOLATILITY, IMPACT_COEFFICIENT

logger = logging.getLogger(__name__)


def _slippage_coefficients(
    avg_daily_volume: np.ndarray, volatility: np.ndarray | None, portfolio_value: float
) -> np.ndarray:
    """k_i such that slippage_i / V = k_i * |delta w_i| ** 1.5 (square-root impact law)."""
    adv = np.asarray(avg_daily_volume, dtype=float)
    sigma = np.full_like(adv, DEFAULT_DAILY_VOLATILITY) if volatility is None else np.asarray(volatility, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = IMPACT_COEFFICIENT * sigma * np.sqrt(portfolio_value / adv)
    return np.where(adv > 0, k, 1e6)


def solve_weighted_sum(
    expected_returns: np.ndarray,
    scenarios: np.ndarray,
    avg_daily_volume: np.ndarray,
    current_weights: np.ndarray,
    portfolio_value: float,
    *,
    coefficients: tuple[float, float, float] = (1.0, 1.0, 1.0),
    volatility: np.ndarray | None = None,
    max_weight: float = 0.25,
    alpha: float = 0.95,
) -> np.ndarray:
    """Weights minimizing  a*(-w.mu) + b*CVaR + c*slippage/V  over the capped long-only simplex.

    `coefficients` are RAW multipliers on the three objectives in their natural units (the
    same units MOEA/D reports), not normalized preferences.
    """
    a, b, c = coefficients
    mu = np.asarray(expected_returns, dtype=float)
    returns = np.asarray(scenarios, dtype=float)
    current = np.asarray(current_weights, dtype=float)
    n = mu.size
    n_scen = returns.shape[0]
    if returns.shape[1] != n:
        raise ValueError(f"scenarios {returns.shape} do not match {n} assets")

    cap = max(max_weight, 1.0 / n)
    k = _slippage_coefficients(avg_daily_volume, volatility, float(portfolio_value))
    tail = 1.0 / ((1.0 - alpha) * n_scen)

    # Variables: [w (n), t (1), u (n_scen)] -- u_s is scenario s's excess loss over t.
    def split(x):
        return x[:n], x[n], x[n + 1:]

    def objective(x):
        w, t, u = split(x)
        delta = np.abs(w - current)
        return float(-a * (w @ mu) + b * (t + tail * u.sum()) + c * (k * delta ** 1.5).sum())

    def gradient(x):
        w, _, _ = split(x)
        grad = np.zeros_like(x)
        delta = w - current
        grad[:n] = -a * mu + c * k * 1.5 * np.sqrt(np.abs(delta)) * np.sign(delta)
        grad[n] = b
        grad[n + 1:] = b * tail
        return grad

    def excess_loss(x):
        w, t, u = split(x)
        return u + returns @ w + t        # u_s >= -r_s.w - t

    excess_jacobian = np.hstack([returns, np.ones((n_scen, 1)), np.eye(n_scen)])

    start_w = np.clip(current, 0.0, cap)
    start_w = start_w / start_w.sum() if start_w.sum() > 0 else np.full(n, 1.0 / n)
    losses = -(returns @ start_w)
    start_t = float(np.quantile(losses, alpha))
    x0 = np.concatenate([start_w, [start_t], np.maximum(losses - start_t, 0.0)])

    result = minimize(
        objective,
        x0,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, cap)] * n + [(None, None)] + [(0.0, None)] * n_scen,
        constraints=[
            {"type": "eq", "fun": lambda x: split(x)[0].sum() - 1.0,
             "jac": lambda x: np.concatenate([np.ones(n), np.zeros(1 + n_scen)])},
            {"type": "ineq", "fun": excess_loss, "jac": lambda x: excess_jacobian},
        ],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        logger.warning("convex reference solve did not converge: %s", result.message)

    weights = np.clip(result.x[:n], 0.0, cap)
    return weights / weights.sum()
