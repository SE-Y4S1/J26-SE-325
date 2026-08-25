"""Long-term allocation via MOEA/D (Phase 5a).

Uses pymoo's built-in MOEA/D -- the decomposition algorithm is a solved problem and
hand-rolling it would add risk without adding contribution.

Three objectives, which is exactly the point: a mean-variance baseline optimizes two
(return, risk) and is structurally blind to the third. Decomposition-based MOEA/D suits
3-objective problems better than dominance-only methods like NSGA-II, whose selection
pressure thins as objective count grows.

    maximize  expected_return      (minimized internally as -return)
    minimize  CVaR
    minimize  liquidity_cost
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from optimization.objectives import expected_return, liquidity_cost, risk_cvar

logger = logging.getLogger(__name__)

OBJECTIVE_NAMES = ("neg_expected_return", "cvar", "liquidity_cost")


@dataclass
class MOEADConfig:
    n_partitions: int = 12      # ref-direction density; 12 -> 91 directions for 3 objectives
    n_neighbors: int = 15
    prob_neighbor_mating: float = 0.9
    n_generations: int = 200
    seed: int = 42              # fixed so dissertation results reproduce exactly
    max_weight: float = 0.25    # per-asset cap; prevents degenerate single-name solutions
    min_weight: float = 0.0     # no shorting by default
    allow_shorting: bool = False


@dataclass(frozen=True)
class ParetoFront:
    weights: np.ndarray          # (n_solutions, n_assets)
    objectives: np.ndarray       # (n_solutions, 3): [-return, cvar, liquidity_cost]
    symbols: tuple[str, ...]
    config: MOEADConfig = field(repr=False)

    def __len__(self) -> int:
        return int(self.weights.shape[0])


def _normalize(weights: np.ndarray, min_weight: float, max_weight: float) -> np.ndarray:
    """Project a raw genome onto the feasible weight simplex (sum to 1, each asset <= cap).

    Handled by construction rather than by a penalty term: an equality constraint like
    sum(w)==1 is very hard for an evolutionary algorithm to satisfy exactly, and a penalty
    would leave most of the final population marginally infeasible.

    Water-filling, not clip-and-renormalize. Naive clip-then-renormalize does NOT converge:
    for a genome like [10, 0, 0, 0] the clip zeroes every other asset, renormalizing sends
    the survivor straight back to 1.0, and the loop oscillates forever -- returning a vector
    that violates the very cap it was asked to enforce. Instead we cap the offenders and
    redistribute their excess into the REMAINING HEADROOM of the others, which terminates
    and provably satisfies the cap whenever n * max_weight >= 1.
    """
    w = np.asarray(weights, dtype=float).copy()

    if min_weight < 0:
        # Shorting enabled: the feasible set is no longer the simplex, so we clip to the box
        # and renormalize. Documented as approximate; no-shorting is the default and the
        # case the dissertation reports.
        w = np.clip(w, min_weight, max_weight)
        total = w.sum()
        return w / total if abs(total) > 1e-12 else np.full_like(w, 1.0 / w.size)

    n = w.size
    if n * max_weight < 1.0 - 1e-9:
        # No allocation can sum to 1 under this cap; the cap is the binding answer.
        return np.full(n, 1.0 / n)

    w = np.clip(w, 0.0, None)
    total = w.sum()
    w = np.full(n, 1.0 / n) if total <= 1e-12 else w / total

    for _ in range(100):
        over = w > max_weight + 1e-12
        if not over.any():
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight

        headroom = np.where(~over, max_weight - w, 0.0)
        available = float(headroom.sum())
        if available <= 1e-12:
            break
        # Proportional to headroom, so no recipient can be pushed above the cap.
        w = w + excess * headroom / available

    return np.clip(w, 0.0, max_weight)


def optimize_allocation(
    expected_returns: np.ndarray,
    quantile_forecasts: np.ndarray,
    avg_daily_volume: np.ndarray,
    current_weights: np.ndarray,
    portfolio_value: float,
    symbols: tuple[str, ...],
    *,
    volatility: np.ndarray | None = None,
    config: MOEADConfig | None = None,
) -> ParetoFront:
    """Solve the 3-objective allocation problem and return the whole Pareto front.

    `current_weights` matters because liquidity cost depends on the TRADE (the delta), not
    the target -- rebalancing INTO an illiquid name is what costs money, and an optimizer
    that scored the target holding instead would happily churn the book for free.
    """
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.util.ref_dirs import get_reference_directions

    config = config or MOEADConfig()
    mu = np.asarray(expected_returns, dtype=float)
    quantiles = np.asarray(quantile_forecasts, dtype=float)
    adv = np.asarray(avg_daily_volume, dtype=float)
    current = np.asarray(current_weights, dtype=float)
    n_assets = mu.size

    if len(symbols) != n_assets:
        raise ValueError(f"{len(symbols)} symbols but {n_assets} expected returns")

    min_w = -config.max_weight if config.allow_shorting else config.min_weight
    max_w = config.max_weight

    class AllocationProblem(Problem):
        def __init__(self) -> None:
            super().__init__(n_var=n_assets, n_obj=3, n_constr=0, xl=0.0, xu=1.0)

        def _evaluate(self, X, out, *args, **kwargs):  # noqa: N803 - pymoo's signature
            results = np.empty((X.shape[0], 3))
            for i, genome in enumerate(X):
                w = _normalize(genome, min_w, max_w)
                trade_notional = np.abs(w - current) * portfolio_value
                results[i, 0] = -expected_return(w, mu)
                results[i, 1] = risk_cvar(w, quantiles)
                results[i, 2] = liquidity_cost(w, trade_notional, adv, volatility)
            out["F"] = results

    ref_dirs = get_reference_directions("uniform", 3, n_partitions=config.n_partitions)
    algorithm = MOEAD(
        ref_dirs=ref_dirs,
        n_neighbors=config.n_neighbors,
        prob_neighbor_mating=config.prob_neighbor_mating,
    )

    result = pymoo_minimize(
        AllocationProblem(),
        algorithm,
        ("n_gen", config.n_generations),
        seed=config.seed,
        verbose=False,
    )

    raw = np.atleast_2d(result.X)
    objectives = np.atleast_2d(result.F)

    # Return the PROJECTED weights, not the raw genomes -- the caller must receive weights
    # that actually satisfy the constraints they were promised.
    weights = np.vstack([_normalize(row, min_w, max_w) for row in raw])

    logger.info("MOEA/D produced %d solutions over %d generations", len(weights), config.n_generations)

    return ParetoFront(
        weights=weights,
        objectives=objectives,
        symbols=tuple(symbols),
        config=config,
    )
