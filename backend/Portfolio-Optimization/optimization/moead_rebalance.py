"""Long-term allocation via MOEA/D (Phase 5a).

Uses pymoo's built-in MOEA/D -- the decomposition algorithm is a solved problem and
hand-rolling it would add risk without adding contribution.

Three objectives, which is exactly the point: a mean-variance baseline optimizes two
(return, risk) and is structurally blind to the third. Decomposition-based MOEA/D suits
3-objective problems better than dominance-only methods like NSGA-II, whose selection
pressure thins as objective count grows.

    maximize  expected_return      (minimized internally as -return)
    minimize  CVaR
    minimize  liquidity_cost       (total slippage / portfolio value)

The solver works on objectives rescaled to a common range (`_objective_scales`); everything
it returns is in real units.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from optimization.objectives import expected_return, liquidity_cost_total, risk_cvar

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
    # Add the exact corner portfolios (do nothing, equal weight, greedy max-return, min-variance,
    # min-CVaR) to the final front as an elitist archive. At 100 generations the evolved front
    # otherwise stops well short of the low-risk corner (measured: min CVaR 0.0220 against an
    # exact 0.0157), and PBI decomposition would discard them again if used as a starting
    # population (see _merge_corner_solutions).
    seed_corners: bool = True


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


def _objective_scales(
    raw_objectives,
    n_assets: int,
    min_w: float,
    max_w: float,
    current: np.ndarray,
    seed: int,
    n_samples: int = 64,
) -> np.ndarray:
    """Per-objective range over a seeded sample of feasible portfolios.

    The sample includes the current book (zero trading cost) and equal weights, so the cost
    objective's range always spans "do nothing" to "rebuild everything". A degenerate
    objective (e.g. every return forecast is zero) gets scale 1 instead of dividing by ~0.
    """
    rng = np.random.default_rng(seed)
    genomes = [current, np.full(n_assets, 1.0 / n_assets)]
    genomes += [rng.random(n_assets) for _ in range(n_samples)]
    sample = np.vstack([raw_objectives(_normalize(g, min_w, max_w)) for g in genomes])
    span = sample.max(axis=0) - sample.min(axis=0)
    return np.where(span > 1e-15, span, 1.0)


def _min_cvar_weights(returns: np.ndarray, max_w: float, alpha: float = 0.95) -> np.ndarray | None:
    """Exact minimum-CVaR long-only weights: Rockafellar-Uryasev as a linear program.

    With no trading-cost term the CVaR corner of the front is a pure LP, which HiGHS solves in
    milliseconds even at 500 scenarios; the cost-aware version needs the ~300-variable
    nonlinear solve in convex_reference.py, which is too slow for the interactive endpoint.
    """
    from scipy.optimize import linprog
    from scipy.sparse import bmat, csr_matrix, identity

    n_scen, n = returns.shape
    tail = 1.0 / ((1.0 - alpha) * n_scen)
    cost = np.concatenate([np.zeros(n), [1.0], np.full(n_scen, tail)])        # [w, t, u]
    # u_s + r_s.w + t >= 0   <=>   -r_s.w - t - u_s <= 0
    a_ub = bmat([[csr_matrix(-returns), csr_matrix(-np.ones((n_scen, 1))), -identity(n_scen)]], format="csr")
    a_eq = csr_matrix(np.concatenate([np.ones(n), np.zeros(1 + n_scen)])[None, :])
    result = linprog(
        cost, A_ub=a_ub, b_ub=np.zeros(n_scen), A_eq=a_eq, b_eq=[1.0],
        bounds=[(0.0, max_w)] * n + [(None, None)] + [(0.0, None)] * n_scen, method="highs",
    )
    if not result.success:
        return None
    weights = np.clip(result.x[:n], 0.0, max_w)
    return weights / weights.sum()


def _corner_seeds(
    mu: np.ndarray,
    scenarios: np.ndarray,
    current: np.ndarray,
    min_w: float,
    max_w: float,
) -> list[np.ndarray]:
    """The corner portfolios alone (see _corner_seeds_tagged)."""
    return [seed for seed, _ in _corner_seeds_tagged(mu, scenarios, current, min_w, max_w)]


def _corner_seeds_tagged(
    mu: np.ndarray,
    scenarios: np.ndarray,
    current: np.ndarray,
    min_w: float,
    max_w: float,
) -> list[tuple[np.ndarray, int | None]]:
    """Feasible portfolios at the front's corners, kept as an elitist archive.

    * the current book (zero trading cost: the low-cost corner)
    * equal weights
    * greedy max-return: fill the best-forecast assets to the cap (the high-return corner;
      no feasible vector earns more)
    * minimum variance on the scenarios
    * exact minimum CVaR on the scenarios (an LP; the low-risk corner itself)
    """
    from scipy.optimize import minimize

    n = mu.size
    # (portfolio, objective whose corner it is): 0 return, 1 CVaR, 2 cost; None = no corner.
    seeds: list[tuple[np.ndarray, int | None]] = [
        (_normalize(current, min_w, max_w), 2), (np.full(n, 1.0 / n), None),
    ]

    greedy = np.zeros(n)
    remaining = 1.0
    for index in np.argsort(-mu):
        take = min(max_w, remaining)
        greedy[index] = take
        remaining -= take
        if remaining <= 1e-12:
            break
    seeds.append((_normalize(greedy, min_w, max_w), 0))

    returns = np.asarray(scenarios, dtype=float)
    if returns.shape[1] != n and returns.shape[0] == n:
        returns = returns.T
    if returns.shape[1] == n and returns.shape[0] > n:
        cov = np.cov(returns, rowvar=False)
        result = minimize(
            lambda w: float(w @ cov @ w), np.full(n, 1.0 / n), jac=lambda w: 2.0 * cov @ w,
            method="SLSQP", bounds=[(0.0, max_w)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
            options={"maxiter": 200, "ftol": 1e-12},
        )
        if result.success and np.isfinite(result.x).all():
            seeds.append((_normalize(result.x, min_w, max_w), None))

        exact = _min_cvar_weights(returns, max_w)
        if exact is not None and np.isfinite(exact).all():
            seeds.append((_normalize(exact, min_w, max_w), 1))

    return seeds


def _merge_corner_solutions(weights, objectives, corners, raw_objectives):
    """Add the exact corner portfolios to a front and keep only the non-dominated rows.

    Seeding the starting population does not work here: for three objectives pymoo's MOEA/D
    decomposes with PBI, which prefers balanced points and replaces an extreme one within a
    generation or two (measured: the exact min-CVaR seed was present after generation 1 and
    gone after generation 2). So the corners are kept as an elitist archive instead. They are
    real Pareto-optimal points, and including them gives the selection rules the front's true
    extent to normalize against.
    """
    from optimization.pareto_selection import is_non_dominated

    corner_w = np.vstack(corners)
    corner_f = np.vstack([raw_objectives(w) for w in corner_w])
    all_w = np.vstack([weights, corner_w])
    all_f = np.vstack([objectives, corner_f])

    _, unique = np.unique(np.round(all_f, 12), axis=0, return_index=True)
    unique = np.sort(unique)
    all_w, all_f = all_w[unique], all_f[unique]
    keep = is_non_dominated(all_f)
    return all_w[keep], all_f[keep]


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

    def raw_objectives(w: np.ndarray) -> np.ndarray:
        trade_notional = np.abs(w - current) * portfolio_value
        return np.array([
            -expected_return(w, mu),
            risk_cvar(w, quantiles),
            # Total slippage as a fraction of the book. NOT the per-dollar average: that is
            # lowered by padding a liquid trade, so an optimizer would learn to pad.
            liquidity_cost_total(trade_notional, adv, volatility) / max(portfolio_value, 1e-12),
        ])

    scales = _objective_scales(raw_objectives, n_assets, min_w, max_w, current, config.seed)

    class AllocationProblem(Problem):
        def __init__(self) -> None:
            super().__init__(n_var=n_assets, n_obj=3, n_constr=0, xl=0.0, xu=1.0)

        def _evaluate(self, X, out, *args, **kwargs):  # noqa: N803 - pymoo's signature
            # The solver sees objectives on a common scale; return is ~1e-3, CVaR ~1e-2 and
            # cost ~1e-5, and Tchebycheff decomposition does not normalize, so unscaled it
            # chases whichever objective has the largest units.
            out["F"] = np.vstack(
                [raw_objectives(_normalize(genome, min_w, max_w)) / scales for genome in X]
            )

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
    # Back to real units: callers and the API report return/CVaR/cost, not solver scale.
    objectives = np.atleast_2d(result.F) * scales

    # Return the PROJECTED weights, not the raw genomes -- the caller must receive weights
    # that actually satisfy the constraints they were promised.
    weights = np.vstack([_normalize(row, min_w, max_w) for row in raw])

    if config.seed_corners and min_w >= 0:
        weights, objectives = _merge_corner_solutions(
            weights, objectives, _corner_seeds(mu, quantiles, current, min_w, max_w), raw_objectives
        )

    logger.info("MOEA/D produced %d solutions over %d generations", len(weights), config.n_generations)

    return ParetoFront(
        weights=weights,
        objectives=objectives,
        symbols=tuple(symbols),
        config=config,
    )
