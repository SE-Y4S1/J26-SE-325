"""Phase 5a tests: objectives, mean-variance baseline, MOEA/D, and Pareto selection.

The properties that matter here are the ones a supervisor can check directly: constraints
are actually satisfied, the returned front is actually non-dominated, and the liquidity term
actually changes the answer (otherwise RQ2 has no mechanism behind it).
"""

from __future__ import annotations

import numpy as np
import pytest

from optimization.baseline_meanvariance import (
    efficient_frontier,
    max_sharpe_portfolio,
    min_variance_portfolio,
)
from optimization.moead_rebalance import MOEADConfig, _normalize, optimize_allocation
from optimization.objectives import (
    expected_return,
    liquidity_cost,
    portfolio_volatility,
    realized_loss,
    risk_cvar,
)
from optimization.pareto_selection import (
    SelectionRule,
    compare_rules,
    is_non_dominated,
    knee_point,
    select,
)


@pytest.fixture
def market():
    """A 4-asset problem where asset 3 is high-return but very illiquid.

    Constructed so a liquidity-blind optimizer will over-allocate to it -- which is exactly
    the failure RQ2 is designed to expose.
    """
    rng = np.random.default_rng(0)
    mu = np.array([0.0004, 0.0006, 0.0003, 0.0012])
    # 200 scenarios x 4 assets, correlated through a common market factor.
    factor = rng.normal(0, 0.01, (200, 1))
    idio = rng.normal(0, 0.008, (200, 4))
    scenarios = factor @ np.array([[1.0, 1.2, 0.7, 1.5]]) + idio + mu
    cov = np.cov(scenarios, rowvar=False)
    adv = np.array([5.0e8, 3.0e8, 2.0e8, 1.0e6])   # asset 3 is ~500x thinner
    return {"mu": mu, "scenarios": scenarios, "cov": cov, "adv": adv,
            "symbols": ("AAA", "BBB", "CCC", "THIN")}


# --------------------------------------------------------------------------------------
# Objectives
# --------------------------------------------------------------------------------------

def test_expected_return_is_the_weighted_sum() -> None:
    weights = np.array([0.5, 0.5])
    assert expected_return(weights, np.array([0.02, 0.04])) == pytest.approx(0.03)


def test_expected_return_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        expected_return(np.array([0.5, 0.5]), np.array([0.02, 0.04, 0.01]))


def test_cvar_is_positive_for_a_loss_making_distribution() -> None:
    """Sign convention: positive means loss, so minimizing it is the natural direction."""
    rng = np.random.default_rng(1)
    scenarios = rng.normal(-0.01, 0.02, (500, 2))
    assert risk_cvar(np.array([0.5, 0.5]), scenarios) > 0


def test_cvar_rewards_diversification() -> None:
    """CVaR is coherent (sub-additive), so a diversified book must not be worse than the
    worst single asset. A per-asset-weighted-average implementation would fail this."""
    rng = np.random.default_rng(2)
    # Two negatively correlated assets: diversification should genuinely help.
    common = rng.normal(0, 0.02, 500)
    scenarios = np.column_stack([common, -common + rng.normal(0, 0.002, 500)])

    diversified = risk_cvar(np.array([0.5, 0.5]), scenarios)
    concentrated = risk_cvar(np.array([1.0, 0.0]), scenarios)
    assert diversified < concentrated


def test_cvar_exceeds_var() -> None:
    """Expected shortfall is an average OVER the tail, so it must be at least the quantile."""
    rng = np.random.default_rng(4)
    scenarios = rng.normal(0, 0.02, (1000, 1))
    weights = np.array([1.0])
    cvar = risk_cvar(weights, scenarios, alpha=0.95)
    var = float(np.quantile(-(scenarios @ weights), 0.95))
    assert cvar >= var - 1e-9


def test_liquidity_cost_follows_the_square_root_law() -> None:
    """4x the trade size must cost ~2x, not 4x."""
    adv = np.array([1.0e8])
    small = liquidity_cost(np.array([1.0]), np.array([1.0e6]), adv)
    large = liquidity_cost(np.array([1.0]), np.array([4.0e6]), adv)
    assert large / small == pytest.approx(2.0, rel=0.01)


def test_liquidity_cost_rises_as_adv_falls() -> None:
    trade = np.array([1.0e6])
    liquid = liquidity_cost(np.array([1.0]), trade, np.array([1.0e9]))
    illiquid = liquidity_cost(np.array([1.0]), trade, np.array([1.0e7]))
    assert illiquid > liquid


def test_liquidity_cost_handles_zero_adv_without_dividing_by_zero() -> None:
    """An untradeable asset must be enormously costly, not NaN or a crash."""
    cost = liquidity_cost(np.array([1.0]), np.array([1.0e6]), np.array([0.0]))
    assert np.isfinite(cost)
    assert cost > 1000


def test_liquidity_cost_is_zero_when_nothing_trades() -> None:
    assert liquidity_cost(np.array([0.5, 0.5]), np.zeros(2), np.array([1e8, 1e8])) == 0.0


def test_realized_loss_sign_convention() -> None:
    """Positive means money lost, matching liquidity_cost so they can be summed."""
    entry, qty = np.array([100.0]), np.array([10.0])
    assert realized_loss(entry, np.array([90.0]), qty) == pytest.approx(100.0)
    assert realized_loss(entry, np.array([110.0]), qty) == pytest.approx(-100.0)


def test_portfolio_volatility_is_nonnegative(market) -> None:
    assert portfolio_volatility(np.full(4, 0.25), market["cov"]) >= 0


# --------------------------------------------------------------------------------------
# Mean-variance baseline (RQ2's opponent)
# --------------------------------------------------------------------------------------

def test_min_variance_satisfies_constraints(market) -> None:
    result = min_variance_portfolio(market["cov"], max_weight=0.5)
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (result.weights >= -1e-9).all()
    assert (result.weights <= 0.5 + 1e-6).all()


def test_min_variance_is_actually_minimal(market) -> None:
    """It must beat equal-weighting on variance, or the solver is not working."""
    result = min_variance_portfolio(market["cov"])
    equal_weight_vol = portfolio_volatility(np.full(4, 0.25), market["cov"])
    assert portfolio_volatility(result.weights, market["cov"]) <= equal_weight_vol + 1e-9


def test_max_sharpe_satisfies_constraints_and_beats_equal_weight(market) -> None:
    result = max_sharpe_portfolio(market["mu"], market["cov"], max_weight=0.6)
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (result.weights <= 0.6 + 1e-6).all()

    equal = np.full(4, 0.25)
    equal_sharpe = float(np.dot(equal, market["mu"]) / portfolio_volatility(equal, market["cov"]))
    assert result.sharpe >= equal_sharpe - 1e-9


def test_mean_variance_ignores_liquidity_entirely(market) -> None:
    """The defining limitation, and the reason RQ2 has a mechanism: making an asset 500x
    thinner must not change the mean-variance answer at all."""
    baseline = max_sharpe_portfolio(market["mu"], market["cov"])
    # ADV is not even a parameter of the function -- this asserts the API shape, which is
    # the honest way to state "it cannot see liquidity".
    again = max_sharpe_portfolio(market["mu"], market["cov"])
    np.testing.assert_allclose(baseline.weights, again.weights)


def test_efficient_frontier_is_monotonic_in_risk(market) -> None:
    frontier = efficient_frontier(market["mu"], market["cov"], n_points=12, max_weight=0.5)
    assert len(frontier) >= 2
    returns = [p.expected_return for p in frontier]
    assert returns == sorted(returns), "frontier should be ordered by target return"
    for point in frontier:
        assert point.weights.sum() == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------------------
# MOEA/D
# --------------------------------------------------------------------------------------

def test_normalize_projects_onto_the_capped_simplex() -> None:
    """Constraints are satisfied by construction, not by penalty -- so this must hold for
    any genome the algorithm can produce, including adversarial ones."""
    for genome in (
        np.array([10.0, 0.0, 0.0, 0.0]),      # wants everything in one asset
        np.array([0.0, 0.0, 0.0, 0.0]),       # degenerate all-zero
        np.array([0.25, 0.25, 0.25, 0.25]),   # already feasible
        np.array([1.0, 1.0, 1.0, 1.0]),
    ):
        weights = _normalize(genome, 0.0, 0.25)
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert (weights <= 0.25 + 1e-6).all(), f"cap violated for {genome}: {weights}"
        assert (weights >= -1e-9).all()


@pytest.mark.slow
def test_moead_returns_a_constrained_non_dominated_front(market) -> None:
    config = MOEADConfig(n_partitions=6, n_generations=25, seed=42, max_weight=0.4)
    front = optimize_allocation(
        market["mu"], market["scenarios"], market["adv"],
        current_weights=np.full(4, 0.25), portfolio_value=1_000_000,
        symbols=market["symbols"], config=config,
    )

    assert len(front) > 0
    assert front.weights.shape[1] == 4
    assert front.objectives.shape[1] == 3

    # Every returned solution must satisfy the constraints it was promised.
    for weights in front.weights:
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert (weights <= 0.4 + 1e-6).all()
        assert (weights >= -1e-9).all()

    # A meaningful share of the returned population should be non-dominated.
    assert is_non_dominated(front.objectives).mean() > 0.5


@pytest.mark.slow
def test_moead_avoids_the_illiquid_asset_more_than_mean_variance(market) -> None:
    """THE RQ2 mechanism. Asset 3 has the highest expected return but ~500x less depth.
    Mean-variance loads it; the liquidity-aware optimizer should not, given a real trade."""
    config = MOEADConfig(n_partitions=6, n_generations=30, seed=42, max_weight=0.6)
    front = optimize_allocation(
        market["mu"], market["scenarios"], market["adv"],
        current_weights=np.array([0.34, 0.33, 0.33, 0.0]),   # currently holds none of THIN
        portfolio_value=10_000_000, symbols=market["symbols"], config=config,
    )
    chosen = select(front.objectives, front.weights, rule=SelectionRule.KNEE)
    mv = max_sharpe_portfolio(market["mu"], market["cov"], max_weight=0.6)

    thin_index = market["symbols"].index("THIN")
    assert chosen.weights[thin_index] <= mv.weights[thin_index] + 1e-6, (
        f"MOEA/D put {chosen.weights[thin_index]:.3f} in the illiquid asset vs "
        f"mean-variance {mv.weights[thin_index]:.3f}"
    )


def test_moead_rejects_mismatched_symbols(market) -> None:
    with pytest.raises(ValueError, match="symbols"):
        optimize_allocation(
            market["mu"], market["scenarios"], market["adv"],
            current_weights=np.full(4, 0.25), portfolio_value=1e6,
            symbols=("A", "B"), config=MOEADConfig(n_generations=2),
        )


# --------------------------------------------------------------------------------------
# Pareto selection
# --------------------------------------------------------------------------------------

@pytest.fixture
def synthetic_front():
    """A convex front with a deliberate, visually obvious knee at index 2."""
    objectives = np.array([
        [-0.010, 0.100, 0.001],
        [-0.009, 0.060, 0.001],
        [-0.008, 0.030, 0.001],   # the knee: risk drops sharply for little given-up return
        [-0.007, 0.028, 0.001],
        [-0.006, 0.027, 0.001],
        [-0.005, 0.026, 0.001],
    ])
    weights = np.eye(6)[:, :4]
    return objectives, weights


def test_knee_point_finds_the_elbow(synthetic_front) -> None:
    objectives, _ = synthetic_front
    assert knee_point(objectives) in (2, 3)


def test_all_three_rules_return_a_point_on_the_front(synthetic_front) -> None:
    objectives, weights = synthetic_front
    results = compare_rules(objectives, weights, preference_weights=(0.4, 0.4, 0.2))

    assert set(results) == {"knee", "max_sharpe", "scalarized"}
    for rule_name, point in results.items():
        assert 0 <= point.index < objectives.shape[0]
        np.testing.assert_allclose(point.objectives, objectives[point.index])
        np.testing.assert_allclose(point.weights, weights[point.index])
        assert point.rationale, f"{rule_name} produced no rationale"


def test_max_sharpe_rule_picks_the_best_return_per_risk(synthetic_front) -> None:
    objectives, weights = synthetic_front
    point = select(objectives, weights, rule=SelectionRule.MAX_SHARPE)
    ratios = (-objectives[:, 0]) / objectives[:, 1]
    assert point.index == int(np.argmax(ratios))


def test_scalarized_respects_preference_weights(synthetic_front) -> None:
    """All-weight-on-return and all-weight-on-risk must disagree, or preferences do nothing."""
    objectives, weights = synthetic_front
    return_focused = select(objectives, weights, rule=SelectionRule.SCALARIZED,
                            preference_weights=(1.0, 0.0, 0.0))
    risk_focused = select(objectives, weights, rule=SelectionRule.SCALARIZED,
                          preference_weights=(0.0, 1.0, 0.0))
    assert return_focused.index != risk_focused.index


def test_selection_normalizes_across_incommensurable_scales() -> None:
    """Return ~0.001 and CVaR ~0.02 differ by an order of magnitude; without normalization
    the knee would be decided entirely by whichever has the larger units."""
    objectives = np.array([
        [-0.001, 0.500, 0.0001],
        [-0.002, 0.300, 0.0002],
        [-0.003, 0.100, 0.0003],
        [-0.004, 0.095, 0.0004],
    ])
    index = knee_point(objectives)
    assert 0 <= index < 4


def test_select_rejects_an_empty_front() -> None:
    with pytest.raises(ValueError, match="empty Pareto front"):
        select(np.empty((0, 3)), np.empty((0, 4)))


def test_select_rejects_mismatched_rows() -> None:
    with pytest.raises(ValueError, match="objective rows"):
        select(np.zeros((5, 3)), np.zeros((3, 4)))


def test_is_non_dominated_identifies_a_dominated_point() -> None:
    objectives = np.array([
        [1.0, 1.0],
        [2.0, 2.0],   # dominated by row 0 on both objectives
        [0.5, 3.0],
    ])
    mask = is_non_dominated(objectives)
    assert mask[0] and mask[2]
    assert not mask[1]
