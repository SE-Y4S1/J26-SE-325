"""Genetic algorithm over liquidation sequences (Phase 5b).

A SEPARATE optimizer from MOEA/D, per the TAF. MOEA/D searches the weight simplex for a
long-term allocation; this searches an ordered execution schedule under a hard cash
deadline. Different decision variables, different constraint, different algorithm.

CHROMOSOME
----------
    (pi, f)  where  pi = permutation of the N holdings  (the sell ORDER)
                    f  = vector in [0,1]^N              (the sell FRACTION of each)

Order matters independently of size for two reasons. First, execution stops once the target
is raised, so the order decides which holdings are touched at all. Second, market impact is
path-dependent: starting an illiquid name early spreads its participation across more days
and costs less than dumping it on the final day.

Operators:  ordered crossover (OX) on pi -- preserves relative order and always yields a
            valid permutation, unlike one-point crossover
            blend crossover on f, Gaussian mutation on f, swap mutation on pi

THE COST MODEL (Almgren-Chriss in miniature)
--------------------------------------------
Every schedule trades off two costs that move in opposite directions:

  * Sell fast  -> high daily participation -> high market impact (square-root law)
  * Sell slow  -> longer exposure to price risk -> expected adverse drift ~ sigma * sqrt(days)

A schedule that ignores either one is trivially beatable, which is why the naive baselines
in naive_liquidation.py lose: pro-rata ignores both, largest-first ignores liquidity, and
most-liquid-first ignores the deadline.

FITNESS (minimize)
------------------
    slippage + timing_risk + expected_drift + fx_conversion
  + realized_loss_weight x crystallized_loss   (selling below cost basis)
  + shortfall_penalty            (hard constraint: raise $X within N days)
  + priority_violation_penalty   (deviation from the fuzzy sell ordering)

FORECAST- AND MARKET-AWARE TERMS (all optional per holding; neutral when absent)
  drift_daily            expected daily log-drift from the forecaster. A slice sold on day d
                         forgoes ~drift x (deadline - d) of forecast return: selling a holding
                         forecast to rise costs that rise, selling one forecast to fall avoids
                         the fall. So falling names go first and rising ones are kept.
  forecast_sigma_daily   forecast-implied daily volatility (from the p10-p90 band). Replaces
                         the historical sigma in the timing-risk term when present, so an
                         asset the model is unsure about carries more risk per day held.
  fx_cost_pct            conversion spread paid to bring proceeds into the base currency.
                         Selling a USD asset to raise LKR pays it; an LKR holding does not.
  cost_basis             per-unit cost in the base currency. Proceeds below it crystallise a
                         loss, which is weighted into the fitness (and reported), so the GA
                         prefers raising cash from winners when the costs are otherwise close.

The fuzzy layer enters twice -- seeding the initial population AND penalizing deviation --
so the GA cannot quietly ignore it and converge to a purely cost-driven schedule. The
penalty is soft: the GA may override the fuzzy ordering when the cost saving justifies it,
but it pays for doing so.

Constraint handling is by penalty rather than repair: a repair operator would mask
infeasibility, whereas RQ4 specifically needs to observe WHERE the module starts failing as
liquidity worsens.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field

import numpy as np

from optimization.objectives import (
    DEFAULT_DAILY_VOLATILITY,
    IMPACT_COEFFICIENT,
    IMPACT_EXPONENT,
)

logger = logging.getLogger(__name__)

# Coefficient on holding risk. 1.0 means a risk-neutral trade-off between expected impact
# saved and expected adverse drift incurred; higher values make the schedule more urgent.
TIMING_RISK_AVERSION = 1.0


def risk_aversion_for(risk_tolerance: float) -> float:
    """Timing-risk aversion from the request's risk tolerance in [0, 1]: 1.5 at 0 (sell
    sooner, accept more impact), 1.0 at the neutral 0.5, 0.5 at 1 (wait for cheaper fills)."""
    return 1.5 - min(max(float(risk_tolerance), 0.0), 1.0)


@dataclass
class GAConfig:
    population_size: int = 120
    n_generations: int = 200
    crossover_prob: float = 0.7
    mutation_prob: float = 0.2
    tournament_size: int = 3
    seed: int = 42                       # fixed for reproducible dissertation results
    elitism: int = 2
    # Penalty weights, in units of the TARGET AMOUNT so they scale with the money the way
    # the costs do. (They were absolute -- 1e6 x shortfall_fraction^2 -- which at LKR amounts,
    # ~300x the USD figures, let execution cost outweigh the shortfall: the GA "chose" to
    # under-raise, and feasible plans fell to 6% of realistic scenarios.)
    # Shortfall dominates: missing 0.1% of the target costs ~1% of the target, far above any
    # execution cost, because failing to raise the cash is a constraint violation.
    shortfall_penalty_weight: float = 10.0
    # A fully inverted order vs the fuzzy priorities costs ~7 bps of the target -- the same
    # weight the absolute 1e2 carried on the 150k reference withdrawal it was tuned on.
    priority_penalty_weight: float = 7e-4
    # Fraction of the initial population seeded from the fuzzy ordering; the rest is random
    # so the GA can still discover schedules the fuzzy rules would not have proposed.
    fuzzy_seed_fraction: float = 0.3
    # Stop early once the best fitness has not improved for this many generations.
    stagnation_patience: int = 30
    # Weight on crystallised losses (below cost basis) relative to execution costs. Below 1:
    # a realised loss is an accounting/tax event, not cash lost to the market, so it should
    # steer ties rather than dominate execution quality.
    realized_loss_weight: float = 0.25


@dataclass(frozen=True)
class LiquidationStep:
    """One instruction in the plan."""

    symbol: str
    sell_fraction: float
    quantity: float
    expected_price: float
    expected_slippage_pct: float
    execution_day: int


@dataclass(frozen=True)
class WithdrawalPlan:
    """The GA's output -- the authoritative numbers for a withdrawal.

    The agent (Phase 5c) may decide HOW to invoke the optimizer, but every figure a user or
    downstream component ever sees originates here. See agent/tools.py.
    """

    steps: tuple[LiquidationStep, ...]
    raised_amount: float
    target_amount: float
    expected_slippage_pct: float
    expected_realized_loss: float
    residual_weights: dict[str, float]
    days_required: int
    feasible: bool
    fitness: float
    generations_run: int
    method: str = "fuzzy_ga"
    fuzzy_rule_trace: tuple[dict[str, object], ...] = field(default=())
    # Cost breakdown in the base currency. expected_realized_loss is the sum of the first
    # four; the P&L pair is reported beside it, not inside it.
    slippage_cost: float = 0.0
    timing_cost: float = 0.0
    drift_cost: float = 0.0
    fx_cost: float = 0.0
    realized_pnl: float = 0.0          # proceeds minus cost basis of what was sold
    crystallized_loss: float = 0.0     # the losing lots' part of that, as a positive number

    @property
    def shortfall(self) -> float:
        return max(0.0, self.target_amount - self.raised_amount)


# --------------------------------------------------------------------------------------
# Schedule simulation
# --------------------------------------------------------------------------------------

def simulate_schedule(
    chromosome: tuple[np.ndarray, np.ndarray],
    holdings: dict[str, dict[str, float]],
    *,
    target_amount: float,
    deadline_days: int,
    participation_cap: float,
    timing_risk_aversion: float = TIMING_RISK_AVERSION,
) -> WithdrawalPlan:
    """Walk a chromosome forward day by day, applying the ADV participation cap.

    The cap is what makes the deadline bite: a position larger than
    (cap x ADV x deadline_days) simply cannot be fully exited in time, and the plan comes
    back infeasible rather than pretending otherwise.
    """
    order, fractions = chromosome
    symbols = list(holdings.keys())

    # `max_sell_fraction` caps a holding the decision layer chose to keep (Laya `exclude`).
    remaining_notional = {
        symbols[i]: float(holdings[symbols[i]].get("value", 0.0))
        * float(np.clip(fractions[i], 0.0, float(holdings[symbols[i]].get("max_sell_fraction", 1.0))))
        for i in range(len(symbols))
    }

    steps: list[LiquidationStep] = []
    raised = 0.0
    total_slippage_cost = 0.0
    total_timing_cost = 0.0
    total_drift_cost = 0.0
    total_fx_cost = 0.0
    realized_pnl = 0.0
    crystallized_loss = 0.0
    days_used = 0

    for day in range(deadline_days):
        if raised >= target_amount:
            break
        day_had_activity = False

        for idx in order:
            symbol = symbols[int(idx)]
            if raised >= target_amount:
                break

            holding = holdings[symbol]
            outstanding = remaining_notional[symbol]
            if outstanding <= 1e-9:
                continue

            adv = float(holding.get("adv_usd", 0.0))
            if adv <= 0:
                continue  # untradeable; contributes to shortfall

            capacity = adv * participation_cap
            still_needed = target_amount - raised
            notional = min(outstanding, capacity, still_needed)
            if notional <= 1e-9:
                continue

            price = float(holding.get("price", 1.0))
            sigma = float(holding.get("daily_volatility", DEFAULT_DAILY_VOLATILITY))

            # Market impact on this slice (square-root law on daily participation).
            participation = notional / adv
            slippage_pct = IMPACT_COEFFICIENT * sigma * (participation ** IMPACT_EXPONENT)
            total_slippage_cost += slippage_pct * notional

            # Timing risk: this slice sat unsold for `day` days before execution. A forecast
            # band, when there is one, says how uncertain the coming days are better than the
            # historical sigma does.
            timing_sigma = float(holding.get("forecast_sigma_daily") or sigma)
            total_timing_cost += timing_risk_aversion * timing_sigma * math.sqrt(day) * notional
            # Forecast return forgone on this slice from its sale day to the deadline: selling
            # an asset forecast to rise gives that rise up (+cost); selling one forecast to
            # fall avoids the fall (-cost). This is what makes "sell what is forecast to fall,
            # keep what is forecast to rise" the cheaper schedule.
            total_drift_cost += float(holding.get("drift_daily", 0.0)) * (deadline_days - day) * notional
            total_fx_cost += float(holding.get("fx_cost_pct", 0.0)) * notional

            basis = holding.get("cost_basis")
            if basis is not None and price > 0:
                pnl = (price * (1.0 - slippage_pct) - float(basis)) * (notional / price)
                realized_pnl += pnl
                crystallized_loss += max(0.0, -pnl)

            raised += notional
            remaining_notional[symbol] = outstanding - notional
            day_had_activity = True
            days_used = day + 1

            steps.append(
                LiquidationStep(
                    symbol=symbol,
                    sell_fraction=notional / max(float(holding.get("value", 0.0)), 1e-9),
                    quantity=notional / max(price, 1e-9),
                    expected_price=price * (1.0 - slippage_pct),
                    expected_slippage_pct=slippage_pct,
                    execution_day=day,
                )
            )

        if not day_had_activity:
            # Nothing tradeable left; further days cannot help.
            break

    # Residual portfolio weights after the plan executes.
    residual: dict[str, float] = {}
    for symbol in symbols:
        original = float(holdings[symbol].get("value", 0.0))
        sold = sum(step.quantity * float(holdings[symbol].get("price", 1.0))
                   for step in steps if step.symbol == symbol)
        residual[symbol] = max(0.0, original - sold)

    residual_total = sum(residual.values())
    residual_weights = (
        {s: v / residual_total for s, v in residual.items()} if residual_total > 0
        else {s: 0.0 for s in symbols}
    )

    slippage_pct_overall = (total_slippage_cost / raised) if raised > 0 else 0.0

    return WithdrawalPlan(
        steps=tuple(steps),
        raised_amount=raised,
        target_amount=target_amount,
        expected_slippage_pct=slippage_pct_overall,
        expected_realized_loss=total_slippage_cost + total_timing_cost + total_drift_cost + total_fx_cost,
        residual_weights=residual_weights,
        days_required=days_used,
        feasible=raised >= target_amount - 1e-6,
        fitness=float("nan"),
        generations_run=0,
        slippage_cost=total_slippage_cost,
        timing_cost=total_timing_cost,
        drift_cost=total_drift_cost,
        fx_cost=total_fx_cost,
        realized_pnl=realized_pnl,
        crystallized_loss=crystallized_loss,
    )


def evaluate_fitness(
    chromosome: tuple[np.ndarray, np.ndarray],
    holdings: dict[str, dict[str, float]],
    fuzzy_priorities: dict[str, float],
    *,
    target_amount: float,
    deadline_days: int,
    participation_cap: float,
    config: GAConfig,
    timing_risk_aversion: float = TIMING_RISK_AVERSION,
) -> tuple[float]:
    """Scalar fitness (DEAP expects a tuple). Lower is better."""
    plan = simulate_schedule(
        chromosome,
        holdings,
        target_amount=target_amount,
        deadline_days=deadline_days,
        participation_cap=participation_cap,
        timing_risk_aversion=timing_risk_aversion,
    )

    cost = plan.expected_realized_loss + config.realized_loss_weight * plan.crystallized_loss
    shortfall = max(0.0, target_amount - plan.raised_amount)
    # Linear term so even a tiny miss outweighs execution cost; quadratic so large misses are
    # rejected decisively. Both scaled by the target.
    fraction = shortfall / max(target_amount, 1.0)
    penalty = config.shortfall_penalty_weight * (fraction + fraction ** 2) * target_amount

    priority_penalty = _priority_violation(chromosome[0], list(holdings.keys()), fuzzy_priorities)
    penalty += config.priority_penalty_weight * priority_penalty * target_amount

    return (float(cost + penalty),)


def _priority_violation(order: np.ndarray, symbols: list[str], priorities: dict[str, float]) -> float:
    """Normalized disagreement between the GA's order and the fuzzy priority ranking.

    Counts inverted pairs (a Kendall-tau distance), normalized to [0, 1]. Pairwise rather
    than positional, so swapping two adjacent near-equal-priority holdings costs almost
    nothing while moving a very-high-priority name to the back is expensive.
    """
    n = len(symbols)
    if n < 2:
        return 0.0

    ranked = [priorities.get(symbols[int(i)], 50.0) for i in order]
    inversions = sum(
        1
        for i in range(n)
        for j in range(i + 1, n)
        if ranked[i] < ranked[j] - 1e-9      # a lower-priority name placed earlier
    )
    return inversions / (n * (n - 1) / 2)


# --------------------------------------------------------------------------------------
# GA driver
# --------------------------------------------------------------------------------------

def _seed_population(
    n_symbols: int,
    symbols: list[str],
    priorities: dict[str, float],
    config: GAConfig,
    rng: random.Random,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Initial population: part fuzzy-guided, part random."""
    population: list[tuple[np.ndarray, np.ndarray]] = []
    n_seeded = int(config.population_size * config.fuzzy_seed_fraction)

    fuzzy_order = np.argsort([-priorities.get(s, 50.0) for s in symbols])

    for _ in range(n_seeded):
        order = fuzzy_order.copy()
        # Light perturbation so the seeded block is not all identical.
        if n_symbols > 1:
            i, j = rng.randrange(n_symbols), rng.randrange(n_symbols)
            order[i], order[j] = order[j], order[i]
        fractions = np.clip(np.array([rng.uniform(0.4, 1.0) for _ in range(n_symbols)]), 0, 1)
        population.append((order, fractions))

    for _ in range(config.population_size - n_seeded):
        order = np.array(rng.sample(range(n_symbols), n_symbols))
        fractions = np.array([rng.uniform(0.0, 1.0) for _ in range(n_symbols)])
        population.append((order, fractions))

    return population


def _ordered_crossover(parent_a: np.ndarray, parent_b: np.ndarray, rng: random.Random) -> np.ndarray:
    """OX: keep a slice of parent_a, fill the rest in parent_b's relative order."""
    n = len(parent_a)
    if n < 2:
        return parent_a.copy()

    start, end = sorted(rng.sample(range(n), 2))
    child = np.full(n, -1, dtype=int)
    child[start : end + 1] = parent_a[start : end + 1]

    taken = set(child[start : end + 1].tolist())
    fill = [gene for gene in parent_b if gene not in taken]

    pos = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[pos]
            pos += 1
    return child


def _blend_crossover(a: np.ndarray, b: np.ndarray, rng: random.Random) -> np.ndarray:
    """BLX-style blend on the fraction vector."""
    alpha = rng.random()
    return np.clip(alpha * a + (1 - alpha) * b, 0.0, 1.0)


def optimize_withdrawal(
    holdings: dict[str, dict[str, float]],
    target_amount: float,
    *,
    withdrawal_urgency: float = 0.5,
    deadline_days: int = 1,
    risk_tolerance: float = 0.5,
    participation_cap: float = 0.10,
    config: GAConfig | None = None,
) -> WithdrawalPlan:
    """Full Phase 5b pipeline: fuzzy priorities -> seeded GA -> best feasible plan.

    THE tool the agent calls. Returns the plan with its fuzzy_rule_trace attached.
    """
    from optimization.fuzzy_withdrawal import compute_portfolio_priorities, rule_trace_to_dict

    config = config or GAConfig()
    symbols = list(holdings.keys())
    n = len(symbols)
    aversion = risk_aversion_for(risk_tolerance)

    if n == 0:
        raise ValueError("cannot plan a withdrawal from an empty portfolio")
    if target_amount <= 0:
        raise ValueError(f"target_amount must be positive, got {target_amount}")

    # --- Fuzzy layer: per-holding sell priority + audit trace -------------------------
    fuzzy_results = compute_portfolio_priorities(holdings, withdrawal_urgency)
    priorities = {s: r.sell_priority for s, r in fuzzy_results.items()}
    trace = tuple(rule_trace_to_dict(fuzzy_results))

    # --- GA search --------------------------------------------------------------------
    rng = random.Random(config.seed)
    np_rng = np.random.default_rng(config.seed)

    population = _seed_population(n, symbols, priorities, config, rng)

    def fitness_of(individual: tuple[np.ndarray, np.ndarray]) -> float:
        return evaluate_fitness(
            individual, holdings, priorities,
            target_amount=target_amount,
            deadline_days=deadline_days,
            participation_cap=participation_cap,
            config=config,
            timing_risk_aversion=aversion,
        )[0]

    scores = [fitness_of(ind) for ind in population]
    best_idx = int(np.argmin(scores))
    best, best_score = population[best_idx], scores[best_idx]

    generations_run = 0
    stagnant = 0

    for generation in range(config.n_generations):
        generations_run = generation + 1

        ranked = sorted(zip(scores, range(len(population)), strict=True), key=lambda t: t[0])
        next_population = [population[i] for _, i in ranked[: config.elitism]]

        while len(next_population) < config.population_size:
            def tournament() -> tuple[np.ndarray, np.ndarray]:
                contenders = rng.sample(range(len(population)), min(config.tournament_size, len(population)))
                winner = min(contenders, key=lambda i: scores[i])
                return population[winner]

            parent_a, parent_b = tournament(), tournament()

            if rng.random() < config.crossover_prob:
                order = _ordered_crossover(parent_a[0], parent_b[0], rng)
                fractions = _blend_crossover(parent_a[1], parent_b[1], rng)
            else:
                order, fractions = parent_a[0].copy(), parent_a[1].copy()

            if rng.random() < config.mutation_prob:
                if n > 1:
                    i, j = rng.randrange(n), rng.randrange(n)
                    order[i], order[j] = order[j], order[i]
                fractions = np.clip(fractions + np_rng.normal(0, 0.1, n), 0.0, 1.0)

            next_population.append((order, fractions))

        population = next_population
        scores = [fitness_of(ind) for ind in population]

        current_best_idx = int(np.argmin(scores))
        if scores[current_best_idx] < best_score - 1e-9:
            best, best_score = population[current_best_idx], scores[current_best_idx]
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= config.stagnation_patience:
                logger.debug("GA converged after %d generations", generations_run)
                break

    plan = simulate_schedule(
        best, holdings,
        target_amount=target_amount,
        deadline_days=deadline_days,
        participation_cap=participation_cap,
        timing_risk_aversion=aversion,
    )

    if not plan.feasible:
        logger.info(
            "withdrawal infeasible: raised %.2f of %.2f in %d days (cap %.0f%% of ADV)",
            plan.raised_amount, target_amount, deadline_days, participation_cap * 100,
        )

    # Rebuild as a frozen record carrying the search metadata and the fuzzy trace.
    return WithdrawalPlan(
        steps=plan.steps,
        raised_amount=plan.raised_amount,
        target_amount=plan.target_amount,
        expected_slippage_pct=plan.expected_slippage_pct,
        expected_realized_loss=plan.expected_realized_loss,
        residual_weights=plan.residual_weights,
        days_required=plan.days_required,
        feasible=plan.feasible,
        fitness=best_score,
        generations_run=generations_run,
        method="fuzzy_ga",
        fuzzy_rule_trace=trace,
        slippage_cost=plan.slippage_cost,
        timing_cost=plan.timing_cost,
        drift_cost=plan.drift_cost,
        fx_cost=plan.fx_cost,
        realized_pnl=plan.realized_pnl,
        crystallized_loss=plan.crystallized_loss,
    )
