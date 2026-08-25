"""Phase 5b tests: fuzzy inference, the GA, and the RQ3/RQ4 machinery.

The two properties that matter most, and that a supervisor is likely to probe:

  1. MONOTONICITY -- raising withdrawal_urgency must never LOWER sell_priority. Swept across
     the whole input grid, not spot-checked, because a fuzzy rule base is easy to write with
     an accidental non-monotonic hole in it.
  2. CONSTRAINT SATISFACTION -- the GA either raises the cash or reports infeasible. It must
     never claim success while short.
"""

from __future__ import annotations

import numpy as np
import pytest

from optimization.fuzzy_withdrawal import (
    RULE_BASE,
    LiquidityTerm,
    UrgencyTerm,
    compute_portfolio_priorities,
    compute_sell_priority,
    position_liquidity_score,
    rule_trace_to_dict,
)
from optimization.ga_withdrawal import (
    GAConfig,
    optimize_withdrawal,
    simulate_schedule,
)
from optimization.naive_liquidation import BASELINES, pro_rata
from optimization.stress_scenarios import (
    ScenarioType,
    apply_to_holdings,
    make_scenario,
    severity_sweep,
    standard_scenario_suite,
)

# A small portfolio with a deliberate liquidity spread: LIQ is trivially exitable, THIN is
# not. Any competent liquidity-aware method should treat them differently.
PORTFOLIO = {
    "LIQ": {"value": 400_000, "price": 200.0, "adv_usd": 5.0e8, "daily_volatility": 0.012, "volatility_pct": 0.4},
    "MID": {"value": 300_000, "price": 50.0, "adv_usd": 2.0e7, "daily_volatility": 0.018, "volatility_pct": 0.5},
    "THIN": {"value": 300_000, "price": 15.0, "adv_usd": 8.0e5, "daily_volatility": 0.030, "volatility_pct": 0.8},
}

FAST_GA = GAConfig(population_size=30, n_generations=25, seed=42, stagnation_patience=10)


# --------------------------------------------------------------------------------------
# Fuzzy inference system
# --------------------------------------------------------------------------------------

def test_rule_base_covers_the_full_grid() -> None:
    """27 rules = 3 antecedents x 3 terms. A missing combination is an input for which no
    rule fires, which skfuzzy raises on."""
    assert len(RULE_BASE) == 27
    assert len({(u, v, liq) for u, v, liq, _ in RULE_BASE}) == 27


@pytest.mark.parametrize("volatility", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("liquidity", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_sell_priority_is_monotonic_in_urgency(volatility: float, liquidity: float) -> None:
    """THE core property: more urgency never means less priority, anywhere in the grid."""
    priorities = [
        compute_sell_priority("TEST", urgency, volatility, liquidity).sell_priority
        for urgency in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    ]
    for lower, higher in zip(priorities, priorities[1:]):
        assert higher >= lower - 1e-6, (
            f"non-monotonic at volatility={volatility}, liquidity={liquidity}: {priorities}"
        )


def test_sell_priority_is_bounded() -> None:
    for urgency in (0.0, 0.5, 1.0):
        for volatility in (0.0, 0.5, 1.0):
            for liquidity in (0.0, 0.5, 1.0):
                value = compute_sell_priority("T", urgency, volatility, liquidity).sell_priority
                assert 0.0 <= value <= 100.0


def test_high_urgency_prioritizes_illiquid_positions_first() -> None:
    """The non-obvious rule the design turns on: under time pressure the illiquid name needs
    the most lead time, so it must start moving first."""
    illiquid = compute_sell_priority("T", 0.95, 0.5, 0.05).sell_priority
    liquid = compute_sell_priority("T", 0.95, 0.5, 0.95).sell_priority
    assert illiquid > liquid


def test_low_urgency_reverses_the_liquidity_preference() -> None:
    """With time to work the order, the cheap exits go first and illiquid names are spared."""
    illiquid = compute_sell_priority("T", 0.05, 0.3, 0.05).sell_priority
    liquid = compute_sell_priority("T", 0.05, 0.3, 0.95).sell_priority
    assert liquid > illiquid


def test_every_evaluation_produces_a_nonempty_rule_trace() -> None:
    """The trace is a contractual output consumed by Component 4 -- it can never be empty."""
    result = compute_sell_priority("AAPL", 0.7, 0.6, 0.3)
    assert result.fired_rules
    assert all(0.0 < rule.firing_strength <= 1.0 for rule in result.fired_rules)
    assert all(rule.rule_id.startswith("R") for rule in result.fired_rules)


def test_rule_trace_serializes_for_the_api_contract() -> None:
    results = compute_portfolio_priorities(PORTFOLIO, withdrawal_urgency=0.8)
    trace = rule_trace_to_dict(results)

    assert {entry["symbol"] for entry in trace} == set(PORTFOLIO)
    for entry in trace:
        assert entry["rules"], f"{entry['symbol']} has no fired rules"
        for rule in entry["rules"]:
            assert {"rule_id", "if", "then", "strength"} <= set(rule)


@pytest.mark.parametrize(
    "value,adv,expected",
    [
        (1_000, 1.0e9, 1.0),      # 1e-6 of ADV -> below the floor, maximally liquid
        (1.0e9, 1.0e9, 0.0),      # a full day's volume -> maximally illiquid
        (1.0e4, 1.0e9, 1.0),      # 1e-5 of ADV -> exactly at the floor
        (3.162e6, 1.0e9, 0.5),    # ~10^-2.5 of ADV -> midpoint of the 5-decade log span
        (1_000, 0.0, 0.0),        # no ADV at all -> untradeable
    ],
)
def test_position_liquidity_score(value: float, adv: float, expected: float) -> None:
    assert position_liquidity_score(value, adv) == pytest.approx(expected, abs=1e-3)


def test_position_liquidity_score_keeps_discriminating_after_an_adv_collapse() -> None:
    """THE RQ4 fix. A linear score saturating at 20% of ADV floors every holding to 0 once a
    stress scenario cuts ADV by 95%, at which point the fuzzy layer returns an identical
    priority for every asset and stops discriminating exactly when it matters most.

    Log scaling makes a common ADV shock a constant OFFSET, so the ordering survives.
    """
    liquid, illiquid = 100_000.0, 100_000.0
    liquid_adv, illiquid_adv = 4.0e10, 8.0e5

    before = (position_liquidity_score(liquid, liquid_adv),
              position_liquidity_score(illiquid, illiquid_adv))
    shock = 0.05                                   # severity-1.0 ADV collapse
    after = (position_liquidity_score(liquid, liquid_adv * shock),
             position_liquidity_score(illiquid, illiquid_adv * shock))

    assert before[0] > before[1], "should rank liquid above illiquid before the shock"
    assert after[0] > after[1], "ordering collapsed under stress -- the RQ4 saturation bug"
    assert after[0] - after[1] > 0.1, "scores converged too far to drive the rule base"


# --------------------------------------------------------------------------------------
# Schedule simulation
# --------------------------------------------------------------------------------------

def test_simulate_schedule_respects_the_adv_participation_cap() -> None:
    """No single day may sell more than cap x ADV of any one symbol."""
    order = np.arange(len(PORTFOLIO))
    fractions = np.ones(len(PORTFOLIO))
    plan = simulate_schedule(
        (order, fractions), PORTFOLIO,
        target_amount=900_000, deadline_days=5, participation_cap=0.10,
    )

    symbols = list(PORTFOLIO)
    for day in range(5):
        for symbol in symbols:
            sold = sum(
                step.quantity * PORTFOLIO[symbol]["price"]
                for step in plan.steps
                if step.symbol == symbol and step.execution_day == day
            )
            cap = PORTFOLIO[symbol]["adv_usd"] * 0.10
            assert sold <= cap + 1e-6, f"{symbol} day {day}: sold {sold} > cap {cap}"


def test_simulate_schedule_stops_once_the_target_is_raised() -> None:
    """Over-selling would leave the user with unwanted cash and needless realized loss."""
    plan = simulate_schedule(
        (np.arange(3), np.ones(3)), PORTFOLIO,
        target_amount=100_000, deadline_days=5, participation_cap=0.10,
    )
    assert plan.raised_amount == pytest.approx(100_000, rel=1e-6)


def test_simulate_schedule_reports_infeasible_rather_than_pretending() -> None:
    """A target beyond what the cap allows in the deadline must come back infeasible."""
    thin_only = {"THIN": PORTFOLIO["THIN"]}
    plan = simulate_schedule(
        (np.array([0]), np.array([1.0])), thin_only,
        target_amount=300_000, deadline_days=1, participation_cap=0.10,
    )
    # One day at 10% of an 800k ADV can raise at most 80k of the 300k position.
    assert plan.feasible is False
    assert plan.shortfall > 0
    assert plan.raised_amount == pytest.approx(80_000, rel=1e-6)


def test_longer_deadline_reduces_slippage() -> None:
    """The central economic claim: spreading execution lowers impact."""
    args = dict(target_amount=500_000, participation_cap=0.10)
    fast = simulate_schedule((np.arange(3), np.ones(3)), PORTFOLIO, deadline_days=1, **args)
    slow = simulate_schedule((np.arange(3), np.ones(3)), PORTFOLIO, deadline_days=10, **args)
    assert slow.expected_slippage_pct <= fast.expected_slippage_pct + 1e-9


# --------------------------------------------------------------------------------------
# The GA
# --------------------------------------------------------------------------------------

def test_ga_meets_the_raise_constraint_when_feasible() -> None:
    plan = optimize_withdrawal(
        PORTFOLIO, target_amount=200_000,
        withdrawal_urgency=0.5, deadline_days=5, config=FAST_GA,
    )
    assert plan.feasible
    assert plan.raised_amount >= 200_000 - 1e-6
    assert plan.shortfall == pytest.approx(0.0, abs=1e-6)


def test_ga_attaches_the_fuzzy_rule_trace() -> None:
    """Proves the fuzzy layer actually ran -- the guard against it being quietly bypassed."""
    plan = optimize_withdrawal(PORTFOLIO, target_amount=150_000, config=FAST_GA)
    assert plan.fuzzy_rule_trace
    assert {entry["symbol"] for entry in plan.fuzzy_rule_trace} == set(PORTFOLIO)
    assert plan.method == "fuzzy_ga"


def test_ga_is_deterministic_under_a_fixed_seed() -> None:
    """Dissertation results must reproduce exactly."""
    kwargs = dict(target_amount=250_000, withdrawal_urgency=0.6, deadline_days=3, config=FAST_GA)
    first = optimize_withdrawal(PORTFOLIO, **kwargs)
    second = optimize_withdrawal(PORTFOLIO, **kwargs)

    assert first.raised_amount == pytest.approx(second.raised_amount)
    assert first.expected_realized_loss == pytest.approx(second.expected_realized_loss)
    assert [s.symbol for s in first.steps] == [s.symbol for s in second.steps]


def test_ga_beats_pro_rata_on_a_portfolio_with_an_illiquid_holding() -> None:
    """RQ3 in miniature. Pro-rata sells 33% of THIN regardless of the impact that causes;
    the GA should route around it."""
    target = 250_000
    ga = optimize_withdrawal(
        PORTFOLIO, target_amount=target, withdrawal_urgency=0.5,
        deadline_days=3, config=FAST_GA,
    )
    naive = pro_rata(PORTFOLIO, target, deadline_days=3, participation_cap=0.10)

    assert ga.feasible
    assert ga.expected_realized_loss <= naive.expected_realized_loss, (
        f"GA {ga.expected_realized_loss:.2f} did not beat pro-rata {naive.expected_realized_loss:.2f}"
    )


def test_ga_rejects_empty_portfolio_and_bad_target() -> None:
    with pytest.raises(ValueError, match="empty portfolio"):
        optimize_withdrawal({}, target_amount=1000, config=FAST_GA)
    with pytest.raises(ValueError, match="must be positive"):
        optimize_withdrawal(PORTFOLIO, target_amount=-5, config=FAST_GA)


def test_ga_reports_infeasible_without_claiming_success() -> None:
    """Asking for more than the whole portfolio must fail honestly."""
    plan = optimize_withdrawal(
        PORTFOLIO, target_amount=5_000_000, deadline_days=1, config=FAST_GA
    )
    assert plan.feasible is False
    assert plan.shortfall > 0


# --------------------------------------------------------------------------------------
# Baselines and stress scenarios
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(BASELINES))
def test_every_baseline_produces_a_costed_plan(name: str) -> None:
    plan = BASELINES[name](PORTFOLIO, 200_000, deadline_days=3, participation_cap=0.10)
    assert plan.method == name
    assert plan.raised_amount > 0
    assert plan.expected_realized_loss >= 0
    assert plan.steps


def test_baselines_differ_from_one_another() -> None:
    """If they all produced identical plans, RQ3 would be comparing nothing."""
    plans = {n: fn(PORTFOLIO, 250_000, deadline_days=3, participation_cap=0.10)
             for n, fn in BASELINES.items()}
    orders = {n: tuple(s.symbol for s in p.steps) for n, p in plans.items()}
    assert len(set(orders.values())) > 1


def test_stress_scenario_adv_decay_is_convex() -> None:
    """A linear map would understate how sharply real depth collapses."""
    sweep = severity_sweep(ScenarioType.ADV_COLLAPSE, n_steps=5)
    multipliers = [s.adv_multiplier for s in sweep]

    assert multipliers[0] == pytest.approx(1.0)
    assert multipliers[-1] == pytest.approx(0.05)
    assert all(a > b for a, b in zip(multipliers, multipliers[1:]))
    # Convexity: the drop accelerates.
    first_half = multipliers[0] - multipliers[2]
    second_half = multipliers[2] - multipliers[4]
    assert second_half > first_half


def test_apply_to_holdings_does_not_mutate_the_original() -> None:
    """A sweep that mutated in place would compound shocks and silently distort RQ4."""
    original_adv = PORTFOLIO["THIN"]["adv_usd"]
    scenario = make_scenario(ScenarioType.COMPOUND, 1.0)

    stressed = apply_to_holdings(PORTFOLIO, scenario)

    assert PORTFOLIO["THIN"]["adv_usd"] == original_adv, "source portfolio was mutated"
    assert stressed["THIN"]["adv_usd"] < original_adv
    assert stressed["THIN"]["daily_volatility"] > PORTFOLIO["THIN"]["daily_volatility"]


def test_compound_scenario_is_worse_than_either_component() -> None:
    """The realistic joint shock must dominate the isolated ones."""
    target, kwargs = 200_000, dict(deadline_days=3, participation_cap=0.10)

    losses = {}
    for scenario_type in (ScenarioType.ADV_COLLAPSE, ScenarioType.VOLATILITY_SPIKE, ScenarioType.COMPOUND):
        stressed = apply_to_holdings(PORTFOLIO, make_scenario(scenario_type, 0.8))
        losses[scenario_type] = pro_rata(stressed, target, **kwargs).expected_realized_loss

    assert losses[ScenarioType.COMPOUND] >= losses[ScenarioType.ADV_COLLAPSE]
    assert losses[ScenarioType.COMPOUND] >= losses[ScenarioType.VOLATILITY_SPIKE]


def test_standard_suite_includes_a_control() -> None:
    suite = standard_scenario_suite()
    assert any(s.scenario_type is ScenarioType.BASELINE for s in suite)
    assert len(suite) == 10   # 1 control + 3 families x 3 severities


def test_withdrawal_quality_degrades_as_liquidity_worsens() -> None:
    """RQ4's headline shape: monotonically worse plans as severity rises."""
    target = 200_000
    losses = []
    for scenario in severity_sweep(ScenarioType.COMPOUND, n_steps=4):
        stressed = apply_to_holdings(PORTFOLIO, scenario)
        plan = optimize_withdrawal(
            stressed, target_amount=target, deadline_days=3, config=FAST_GA
        )
        losses.append(plan.expected_realized_loss + plan.shortfall)

    assert losses[-1] > losses[0], f"no degradation observed across severity: {losses}"
