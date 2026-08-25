"""Naive liquidation baselines -- the RQ3 comparison points.

RQ3 asks whether the fuzzy-GA module reduces realized slippage "versus a naive liquidation
baseline". These are those baselines, and like the mean-variance optimizer they must be
implemented honestly: three plausible heuristics a real platform might actually ship, not
strawmen chosen to lose.

All three run through the SAME simulate_schedule() the GA uses, so the comparison isolates
the choice of schedule and nothing else. If the baselines were costed with a different model
the RQ3 result would be an artefact of the accounting rather than of the algorithm.
"""

from __future__ import annotations

import numpy as np

from optimization.ga_withdrawal import WithdrawalPlan, simulate_schedule


def _plan_from_order(
    order: np.ndarray,
    fractions: np.ndarray,
    holdings: dict[str, dict[str, float]],
    target_amount: float,
    *,
    deadline_days: int,
    participation_cap: float,
    method: str,
) -> WithdrawalPlan:
    """Cost a fixed order/fraction pair through the shared simulator."""
    plan = simulate_schedule(
        (order, fractions),
        holdings,
        target_amount=target_amount,
        deadline_days=deadline_days,
        participation_cap=participation_cap,
    )
    return WithdrawalPlan(
        steps=plan.steps,
        raised_amount=plan.raised_amount,
        target_amount=plan.target_amount,
        expected_slippage_pct=plan.expected_slippage_pct,
        expected_realized_loss=plan.expected_realized_loss,
        residual_weights=plan.residual_weights,
        days_required=plan.days_required,
        feasible=plan.feasible,
        fitness=float("nan"),      # baselines do not search, so there is no fitness
        generations_run=0,
        method=method,
    )


def pro_rata(
    holdings: dict[str, dict[str, float]],
    target_amount: float,
    *,
    deadline_days: int = 1,
    participation_cap: float = 0.10,
    **_: object,
) -> WithdrawalPlan:
    """Sell the same fraction of every holding.

    Preserves the existing allocation exactly and is trivially explainable, which is why it
    is the most common default. It ignores liquidity entirely: the illiquid 2% of the book
    gets sold at the same rate as the liquid 40%, incurring impact out of all proportion to
    the cash it raises.
    """
    symbols = list(holdings.keys())
    total_value = sum(float(h.get("value", 0.0)) for h in holdings.values())
    fraction = min(1.0, target_amount / total_value) if total_value > 0 else 1.0

    return _plan_from_order(
        np.arange(len(symbols)),
        np.full(len(symbols), fraction),
        holdings, target_amount,
        deadline_days=deadline_days, participation_cap=participation_cap,
        method="pro_rata",
    )


def largest_position_first(
    holdings: dict[str, dict[str, float]],
    target_amount: float,
    *,
    deadline_days: int = 1,
    participation_cap: float = 0.10,
    **_: object,
) -> WithdrawalPlan:
    """Drain the biggest holdings first.

    Minimizes the number of trades and the operational overhead. Its failure mode is that
    position size and liquidity are only loosely correlated, so it happily concentrates
    impact in a large-but-thin name.
    """
    symbols = list(holdings.keys())
    order = np.argsort([-float(holdings[s].get("value", 0.0)) for s in symbols])

    return _plan_from_order(
        order, np.ones(len(symbols)),
        holdings, target_amount,
        deadline_days=deadline_days, participation_cap=participation_cap,
        method="largest_first",
    )


def most_liquid_first(
    holdings: dict[str, dict[str, float]],
    target_amount: float,
    *,
    deadline_days: int = 1,
    participation_cap: float = 0.10,
    **_: object,
) -> WithdrawalPlan:
    """Sell the easiest-to-exit holdings first, ranked by ADV relative to position size.

    The STRONGEST baseline, and the fair one to beat: it is genuinely liquidity aware, just
    myopically so. It minimizes cost for THIS withdrawal while leaving the illiquid tail
    behind, which is fine once and progressively worse every subsequent time. It also has no
    notion of the deadline, so under time pressure it can exhaust the liquid names and then
    discover the remainder cannot be moved in time.
    """
    symbols = list(holdings.keys())

    def liquidity_rank(symbol: str) -> float:
        holding = holdings[symbol]
        value = float(holding.get("value", 0.0))
        adv = float(holding.get("adv_usd", 0.0))
        return -(adv / value) if value > 0 else 0.0   # higher ADV/value = easier to exit

    order = np.argsort([liquidity_rank(s) for s in symbols])

    return _plan_from_order(
        order, np.ones(len(symbols)),
        holdings, target_amount,
        deadline_days=deadline_days, participation_cap=participation_cap,
        method="most_liquid_first",
    )


BASELINES = {
    "pro_rata": pro_rata,
    "largest_first": largest_position_first,
    "most_liquid_first": most_liquid_first,
}
