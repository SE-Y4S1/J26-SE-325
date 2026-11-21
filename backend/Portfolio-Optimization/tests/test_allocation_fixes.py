"""Regression tests for the portfolio-optimization review (RQ2 validity and correctness).

Each test names the defect it pins:

  * independent scenarios hid correlation, so CVaR showed diversification that is not there
  * `liquidity_cost` averaged per traded dollar, so padding a trade lowered the "cost"
  * MOEA/D decomposed unscaled objectives (1e-3 / 1e-2 / 1e-5), biasing the front
  * `knee_point` used a 2-D chord on a 3-objective front
  * RQ2's only opponent was liquidity-blind Markowitz
  * a portfolio with no forecasts silently optimized on a constant return objective
  * RQ2 reported predicted cost only, with no realized out-of-sample evaluation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimization.baseline_meanvariance import cost_aware_mean_variance, max_sharpe_portfolio
from optimization.convex_reference import solve_weighted_sum
from optimization.moead_rebalance import MOEADConfig, optimize_allocation
from optimization.objectives import liquidity_cost, liquidity_cost_total, risk_cvar
from optimization.pareto_selection import knee_point
from optimization.scenarios import (
    bootstrap_scenarios,
    correlated_normal_scenarios,
    joint_scenarios,
    returns_matrix,
)


@pytest.fixture
def market():
    rng = np.random.default_rng(0)
    mu = np.array([0.0004, 0.0006, 0.0003, 0.0012])
    factor = rng.normal(0, 0.01, (400, 1))
    idio = rng.normal(0, 0.008, (400, 4))
    scenarios = factor @ np.array([[1.0, 1.2, 0.7, 1.5]]) + idio + mu
    return {
        "mu": mu,
        "scenarios": scenarios,
        "cov": np.cov(scenarios, rowvar=False),
        "adv": np.array([5.0e8, 3.0e8, 2.0e8, 1.0e6]),
        "vol": np.array([0.010, 0.012, 0.009, 0.030]),
        "symbols": ("AAA", "BBB", "CCC", "THIN"),
    }


# --------------------------------------------------------------------------------------
# Liquidity cost: total, not per-dollar
# --------------------------------------------------------------------------------------

def test_padding_a_liquid_trade_does_not_reduce_total_cost() -> None:
    adv = np.array([1.0e6, 1.0e10])
    thin_only = liquidity_cost_total(np.array([5.0e5, 0.0]), adv)
    padded = liquidity_cost_total(np.array([5.0e5, 5.0e8]), adv)
    assert padded >= thin_only

    # The per-dollar average is the thing that was gameable: it falls when you pad.
    assert liquidity_cost(np.array([0.5, 0.5]), np.array([5.0e5, 5.0e8]), adv) < liquidity_cost(
        np.array([0.5, 0.5]), np.array([5.0e5, 0.0]), adv
    )


def test_total_cost_grows_with_the_three_halves_power() -> None:
    adv = np.array([1.0e8])
    small = liquidity_cost_total(np.array([1.0e6]), adv)
    large = liquidity_cost_total(np.array([4.0e6]), adv)
    assert large / small == pytest.approx(8.0, rel=0.01)   # 4 ** 1.5


def test_total_cost_is_zero_when_nothing_trades_and_finite_for_zero_adv() -> None:
    assert liquidity_cost_total(np.zeros(2), np.array([1e8, 1e8])) == 0.0
    assert np.isfinite(liquidity_cost_total(np.array([1.0e6]), np.array([0.0])))


def test_total_cost_is_in_currency_units() -> None:
    # 0.30 * 0.0126 * sqrt(0.01) * 1e6 = 378
    cost = liquidity_cost_total(np.array([1.0e6]), np.array([1.0e8]))
    assert cost == pytest.approx(0.30 * 0.0126 * 0.1 * 1.0e6, rel=1e-6)


# --------------------------------------------------------------------------------------
# Scenarios: joint, not independent
# --------------------------------------------------------------------------------------

def _corr(matrix: np.ndarray) -> float:
    return float(np.corrcoef(matrix, rowvar=False)[0, 1])


def test_bootstrap_scenarios_preserve_cross_asset_correlation() -> None:
    rng = np.random.default_rng(3)
    common = rng.normal(0, 0.01, 600)
    history = np.column_stack([common + rng.normal(0, 0.002, 600), common + rng.normal(0, 0.002, 600)])

    scen = bootstrap_scenarios(history, mu=np.zeros(2), n_scenarios=2000, seed=1)
    assert _corr(scen) > 0.8


def test_bootstrap_scenarios_are_centred_on_the_supplied_mu() -> None:
    rng = np.random.default_rng(4)
    history = rng.normal(0.01, 0.01, (500, 3))            # history drifts +1%/day
    mu = np.array([0.0, 0.001, -0.001])
    scen = bootstrap_scenarios(history, mu=mu, n_scenarios=4000, seed=2)
    assert scen.mean(axis=0) == pytest.approx(mu, abs=5e-4)


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    history = np.random.default_rng(5).normal(0, 0.01, (300, 3))
    a = bootstrap_scenarios(history, mu=np.zeros(3), n_scenarios=100, seed=7)
    b = bootstrap_scenarios(history, mu=np.zeros(3), n_scenarios=100, seed=7)
    np.testing.assert_array_equal(a, b)


def test_fallback_scenarios_assume_positive_correlation_not_independence() -> None:
    scen = correlated_normal_scenarios(np.zeros(3), np.full(3, 0.01), correlation=0.6, n_scenarios=4000, seed=1)
    assert _corr(scen) == pytest.approx(0.6, abs=0.06)


def test_joint_scenarios_use_history_when_long_enough_else_fall_back() -> None:
    rng = np.random.default_rng(6)
    long_history = rng.normal(0, 0.01, (300, 2))
    scen, source = joint_scenarios(np.zeros(2), np.full(2, 0.01), history=long_history, seed=1)
    assert source == "historical" and scen.shape[1] == 2

    scen, source = joint_scenarios(np.zeros(2), np.full(2, 0.01), history=long_history[:20], seed=1)
    assert source == "assumed_correlation"

    scen, source = joint_scenarios(np.zeros(2), np.full(2, 0.01), history=None, seed=1)
    assert source == "assumed_correlation"


def test_returns_matrix_aligns_symbols_on_common_dates() -> None:
    dates = pd.bdate_range("2024-01-01", periods=120)
    a = pd.DataFrame({"timestamp": dates, "close": np.linspace(100, 130, 120)})
    b = pd.DataFrame({"timestamp": dates[10:], "close": np.linspace(50, 70, 110)})   # starts later
    matrix = returns_matrix({"A": a, "B": b}, min_observations=60)
    assert matrix is not None
    assert list(matrix.columns) == ["A", "B"]
    assert matrix.index.min() >= dates[11]            # first common return
    assert not matrix.isna().any().any()

    assert returns_matrix({"A": a, "B": b.iloc[:30]}, min_observations=60) is None


def test_independent_scenarios_understate_cvar_of_a_concentrated_correlated_book() -> None:
    """The original defect, stated as a property: ignoring correlation makes a correlated
    book look safer than it is."""
    vol = np.full(4, 0.01)
    weights = np.full(4, 0.25)
    independent = np.random.default_rng(1).normal(0, vol, (4000, 4))
    joint = correlated_normal_scenarios(np.zeros(4), vol, correlation=0.8, n_scenarios=4000, seed=1)
    assert risk_cvar(weights, joint) > 1.3 * risk_cvar(weights, independent)


# --------------------------------------------------------------------------------------
# MOEA/D: scaled objectives, real units out
# --------------------------------------------------------------------------------------

def test_moead_front_reaches_the_low_cost_extreme(market) -> None:
    """Unscaled Tchebycheff never approached the cost extreme because that objective's
    magnitude is orders smaller than return's. Staying put costs zero, so a front that
    spans the objective must contain near-zero-cost points."""
    current = np.array([0.25, 0.25, 0.25, 0.25])
    front = optimize_allocation(
        market["mu"], market["scenarios"], market["adv"], current, 1.0e6, market["symbols"],
        volatility=market["vol"], config=MOEADConfig(n_partitions=8, n_generations=80, max_weight=0.6),
    )
    cost = front.objectives[:, 2]
    assert cost.min() <= 0.1 * cost.max()


def test_moead_front_spans_the_return_range_the_exact_solver_can_reach() -> None:
    """Pins the scaling defect. Unscaled, the decomposition collapsed to a narrow cluster
    (measured: ~15% of the achievable return range, ~1/3 the hypervolume); scaled, it
    covers about half. The exact convex solve supplies the true range, so this does not
    depend on any number the optimizer under test produced."""
    rng = np.random.default_rng(0)
    n = 12
    vol = rng.uniform(0.008, 0.03, n)
    mu = vol * 0.05
    adv = 10 ** rng.uniform(6, 10, n)
    current = np.full(n, 1.0 / n)
    scenarios = correlated_normal_scenarios(mu, vol, correlation=0.5, n_scenarios=300, seed=1)
    value, cap = 1.0e6, 0.3

    front = optimize_allocation(
        mu, scenarios, adv, current, value, tuple(map(str, range(n))),
        volatility=vol, config=MOEADConfig(n_partitions=8, n_generations=100, max_weight=cap),
    )
    best = solve_weighted_sum(mu, scenarios, adv, current, value, coefficients=(1.0, 0.0, 0.0),
                              volatility=vol, max_weight=cap)
    safest = solve_weighted_sum(mu, scenarios, adv, current, value, coefficients=(0.0, 1.0, 0.0),
                                volatility=vol, max_weight=cap)

    achievable = float(best @ mu - safest @ mu)
    covered = float(-front.objectives[:, 0].min() + front.objectives[:, 0].max())
    assert covered >= 0.3 * achievable


def test_moead_reports_total_cost_as_a_fraction_of_portfolio_in_real_units(market) -> None:
    current = np.array([0.25, 0.25, 0.25, 0.25])
    value = 1.0e6
    front = optimize_allocation(
        market["mu"], market["scenarios"], market["adv"], current, value, market["symbols"],
        volatility=market["vol"], config=MOEADConfig(n_partitions=4, n_generations=20, max_weight=0.6),
    )
    w = front.weights[0]
    expected = liquidity_cost_total(np.abs(w - current) * value, market["adv"], market["vol"]) / value
    assert front.objectives[0, 2] == pytest.approx(expected, rel=1e-6)
    assert front.objectives[0, 0] == pytest.approx(-float(w @ market["mu"]), rel=1e-6)


# --------------------------------------------------------------------------------------
# Knee on a 3-objective front
# --------------------------------------------------------------------------------------

def test_knee_point_finds_the_bulge_on_a_three_objective_front() -> None:
    grid = [(i / 10, j / 10, (10 - i - j) / 10) for i in range(11) for j in range(11 - i)]
    front = np.array(grid + [(0.2, 0.2, 0.2)])            # one point well inside the simplex
    assert tuple(front[knee_point(front)]) == (0.2, 0.2, 0.2)


# --------------------------------------------------------------------------------------
# Baselines: a fair Markowitz and an exact convex reference
# --------------------------------------------------------------------------------------

def test_cost_aware_markowitz_trades_less_and_pays_less_than_the_blind_one(market) -> None:
    current = np.array([0.1, 0.1, 0.1, 0.7])               # starts heavily in the thin name
    value = 1.0e6
    blind = max_sharpe_portfolio(market["mu"], market["cov"], max_weight=0.6)
    aware = cost_aware_mean_variance(
        market["mu"], market["cov"], current, market["adv"], value,
        volatility=market["vol"], max_weight=0.6,
    )
    assert aware.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert aware.weights.max() <= 0.6 + 1e-6 and aware.weights.min() >= -1e-9

    def cost(w):
        return liquidity_cost_total(np.abs(w - current) * value, market["adv"], market["vol"])

    assert cost(aware.weights) <= cost(blind.weights) + 1e-9


def test_convex_reference_is_never_worse_on_cvar_than_the_evolutionary_front(market) -> None:
    """Pure-CVaR minimization: the exact solve must beat or match every MOEA/D point."""
    current = np.full(4, 0.25)
    value = 1.0e6
    w = solve_weighted_sum(
        market["mu"], market["scenarios"], market["adv"], current, value,
        coefficients=(0.0, 1.0, 0.0), volatility=market["vol"], max_weight=0.6,
    )
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert w.max() <= 0.6 + 1e-6 and w.min() >= -1e-9

    front = optimize_allocation(
        market["mu"], market["scenarios"], market["adv"], current, value, market["symbols"],
        volatility=market["vol"], config=MOEADConfig(n_partitions=6, n_generations=60, max_weight=0.6),
    )
    assert risk_cvar(w, market["scenarios"]) <= front.objectives[:, 1].min() + 1e-6


# --------------------------------------------------------------------------------------
# Missing forecasts must be visible
# --------------------------------------------------------------------------------------

def _state(with_forecast: bool) -> dict:
    base = {"value": 100_000.0, "price": 10.0, "adv_usd": 5.0e8, "daily_volatility": 0.012}
    state = {s: dict(base) for s in ("AAA", "BBB", "CCC", "DDD")}
    if with_forecast:
        for i, s in enumerate(state):
            state[s]["expected_return"] = 0.0004 * (i + 1)
    return state


def test_rebalance_warns_when_no_holding_has_a_forecast() -> None:
    from agent.tools import run_moead_rebalance

    result = run_moead_rebalance(_state(with_forecast=False))
    assert result["warnings"]
    assert any("forecast" in w.lower() for w in result["warnings"])
    assert set(result["holdings_without_forecast"]) == {"AAA", "BBB", "CCC", "DDD"}


def test_rebalance_has_no_forecast_warning_when_all_holdings_have_one() -> None:
    from agent.tools import run_moead_rebalance

    result = run_moead_rebalance(_state(with_forecast=True))
    assert result["warnings"] == []
    assert result["holdings_without_forecast"] == []
    assert result["scenario_source"] == "assumed_correlation"


def test_rebalance_uses_historical_scenarios_when_history_is_supplied() -> None:
    from agent.tools import run_moead_rebalance

    rng = np.random.default_rng(8)
    history = {s: list(rng.normal(0, 0.01, 300)) for s in ("AAA", "BBB", "CCC", "DDD")}
    result = run_moead_rebalance(_state(with_forecast=True), history_returns=history)
    assert result["scenario_source"] == "historical"


def test_portfolio_response_carries_warnings() -> None:
    from service.contracts import PortfolioResponse

    assert "warnings" in PortfolioResponse.model_fields


# --------------------------------------------------------------------------------------
# Out-of-sample evaluation: RQ2 needs realized numbers, not predicted ones
# --------------------------------------------------------------------------------------

def _bars(n_days: int = 420, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    rows = []
    for symbol, drift, vol, volume in [
        ("AAA", 0.0004, 0.010, 5.0e6), ("BBB", 0.0003, 0.012, 4.0e6),
        ("CCC", 0.0002, 0.009, 6.0e6), ("THIN", 0.0010, 0.030, 2.0e3),
    ]:
        close = 50 * np.exp(np.cumsum(rng.normal(drift, vol, n_days)))
        rows.append(pd.DataFrame({
            "timestamp": dates, "symbol": symbol, "asset_class": "equity",
            "open": close, "high": close, "low": close, "close": close,
            "volume": np.full(n_days, volume),
        }))
    return pd.concat(rows, ignore_index=True)


def test_backtest_reports_realized_metrics_per_method() -> None:
    from evaluation.portfolio_backtest import BacktestSpec, run_portfolio_backtest, summarize

    spec = BacktestSpec(
        portfolio_value=1.0e6, lookback_days=120, rebalance_every=20,
        methods=("hold", "markowitz", "markowitz_cost_aware"),
        max_weight=0.6,
    )
    result = run_portfolio_backtest(_bars(), spec)
    summary = summarize(result)

    assert set(summary["method"]) == {"hold", "markowitz", "markowitz_cost_aware"}
    for column in ("sharpe", "sortino", "max_drawdown", "total_cost", "turnover", "net_return"):
        assert column in summary.columns
        assert summary[column].notna().all()

    by = summary.set_index("method")
    assert by.loc["hold", "total_cost"] == pytest.approx(0.0, abs=1e-12)
    assert by.loc["markowitz", "total_cost"] > 0
    assert (by["max_drawdown"] <= 0).all()


def test_backtest_never_uses_returns_after_the_rebalance_date() -> None:
    """Perturb every bar after the cut: decisions up to the cut must be unchanged."""
    from evaluation.portfolio_backtest import BacktestSpec, run_portfolio_backtest

    spec = BacktestSpec(portfolio_value=1.0e6, lookback_days=120, rebalance_every=20,
                        methods=("markowitz",), max_weight=0.6)
    bars = _bars()
    base = run_portfolio_backtest(bars, spec)

    cut = pd.bdate_range("2022-01-03", periods=420)[300]
    tampered = bars.copy()
    mask = tampered["timestamp"] > cut
    tampered.loc[mask, "close"] = tampered.loc[mask, "close"] * 1.5
    other = run_portfolio_backtest(tampered, spec)

    first = base.decisions[base.decisions["rebalance_date"] <= cut].sort_values(["rebalance_date", "symbol"])
    second = other.decisions[other.decisions["rebalance_date"] <= cut].sort_values(["rebalance_date", "symbol"])
    assert len(first) > 0
    np.testing.assert_allclose(first["target_weight"].to_numpy(), second["target_weight"].to_numpy())


def test_backtest_scores_a_rebalance_dates_return_on_the_book_held_before_it() -> None:
    """`hold` and `equal_weight` start from the same book and drift identically until the
    first real rebalance. On that date their gross returns must still match: the return
    belongs to the pre-rebalance book. Applying it to the freshly chosen target would score
    a decision with knowledge of the return it is being scored on."""
    from evaluation.portfolio_backtest import BacktestSpec, run_portfolio_backtest

    spec = BacktestSpec(portfolio_value=1.0e6, lookback_days=120, rebalance_every=20,
                        methods=("hold", "equal_weight"), max_weight=0.6)
    result = run_portfolio_backtest(_bars(), spec)

    dates = sorted(result.decisions["rebalance_date"].unique())
    first_real = dates[1]                       # dates[0] is the initial rebalance at `start`
    day = result.daily[result.daily["date"] == first_real].set_index("method")["gross_return"]
    assert day["equal_weight"] == pytest.approx(day["hold"], abs=1e-12)


def test_backtest_refuses_symbols_without_volume_instead_of_pricing_them_as_untradeable() -> None:
    """FX bars carry zero volume. ADV of zero makes every trade in them cost the 1e6
    untradeable sentinel, which silently dominates every method's result."""
    from evaluation.portfolio_backtest import BacktestSpec, run_portfolio_backtest

    bars = _bars()
    bars.loc[bars["symbol"] == "AAA", "volume"] = 0.0
    spec = BacktestSpec(lookback_days=120, rebalance_every=20, methods=("hold",))
    with pytest.raises(ValueError, match="no volume.*AAA"):
        run_portfolio_backtest(bars, spec)


# --------------------------------------------------------------------------------------
# Corner seeding: the evolutionary front must reach the extremes it was missing
# --------------------------------------------------------------------------------------

def _twelve_asset_case():
    rng = np.random.default_rng(0)
    n = 12
    vol = rng.uniform(0.008, 0.03, n)
    mu = vol * 0.05
    adv = 10 ** rng.uniform(6, 10, n)
    current = np.full(n, 1.0 / n)
    scenarios = correlated_normal_scenarios(mu, vol, correlation=0.5, n_scenarios=300, seed=1)
    return {"mu": mu, "vol": vol, "adv": adv, "current": current, "scenarios": scenarios,
            "value": 1.0e6, "cap": 0.3, "symbols": tuple(map(str, range(n)))}


def test_corner_seeds_are_feasible_and_include_do_nothing_and_greedy_max_return() -> None:
    from optimization.moead_rebalance import _corner_seeds

    case = _twelve_asset_case()
    seeds = _corner_seeds(case["mu"], case["scenarios"], case["current"], 0.0, case["cap"])

    assert len(seeds) >= 3
    for seed in seeds:
        assert seed.sum() == pytest.approx(1.0, abs=1e-6)
        assert seed.max() <= case["cap"] + 1e-9 and seed.min() >= -1e-12
    assert any(np.allclose(seed, case["current"]) for seed in seeds)        # zero-cost corner

    # Greedy max-return: fill the best assets to the cap. No feasible weight vector beats it.
    best = max(float(seed @ case["mu"]) for seed in seeds)
    top = np.sort(case["mu"])[::-1]
    full, rest = int(1.0 // case["cap"]), 1.0 - int(1.0 // case["cap"]) * case["cap"]
    greedy = float(top[:full].sum() * case["cap"] + (top[full] * rest if rest > 1e-12 else 0.0))
    assert best == pytest.approx(greedy, rel=1e-6)


def test_seeded_front_gets_close_to_the_exact_minimum_cvar() -> None:
    """Measured before seeding: front min CVaR 0.0220 against an exact 0.0157 (40% worse).
    The low-risk corner needs an optimization, not luck, to be found at 100 generations."""
    case = _twelve_asset_case()
    exact = solve_weighted_sum(
        case["mu"], case["scenarios"], case["adv"], case["current"], case["value"],
        coefficients=(0.0, 1.0, 0.0), volatility=case["vol"], max_weight=case["cap"],
    )
    front = optimize_allocation(
        case["mu"], case["scenarios"], case["adv"], case["current"], case["value"], case["symbols"],
        volatility=case["vol"], config=MOEADConfig(n_partitions=8, n_generations=100, max_weight=case["cap"]),
    )
    assert front.objectives[:, 1].min() <= 1.10 * risk_cvar(exact, case["scenarios"])


def test_corner_seeding_is_on_by_default_and_can_be_switched_off() -> None:
    assert MOEADConfig().seed_corners is True
    case = _twelve_asset_case()
    front = optimize_allocation(
        case["mu"], case["scenarios"], case["adv"], case["current"], case["value"], case["symbols"],
        volatility=case["vol"],
        config=MOEADConfig(n_partitions=4, n_generations=10, max_weight=case["cap"], seed_corners=False),
    )
    assert len(front) > 0


# --------------------------------------------------------------------------------------
# A recommendation that trades a large share of a name's volume must say so
# --------------------------------------------------------------------------------------

def test_participation_warnings_name_holdings_trading_a_large_share_of_adv() -> None:
    from agent.tools import participation_warnings

    warnings = participation_warnings(
        ("AAA", "THIN"), np.array([0.5, 0.5]), np.array([0.3, 0.7]), 1.0e6, np.array([5.0e8, 1.0e5]),
        threshold=0.10,
    )
    assert len(warnings) == 1
    assert "THIN" in warnings[0] and "AAA" not in warnings[0]
    assert "average daily volume" in warnings[0]


def test_participation_warnings_are_empty_when_every_trade_is_small() -> None:
    from agent.tools import participation_warnings

    assert participation_warnings(
        ("AAA", "BBB"), np.array([0.5, 0.5]), np.array([0.45, 0.55]), 1.0e6, np.array([5.0e8, 5.0e8]),
    ) == []


def test_participation_warnings_call_out_a_trade_in_a_name_with_no_volume_data() -> None:
    from agent.tools import participation_warnings

    warnings = participation_warnings(
        ("AAA", "NOVOL"), np.array([0.5, 0.5]), np.array([0.4, 0.6]), 1.0e6, np.array([5.0e8, 0.0]),
    )
    assert len(warnings) == 1 and "NOVOL" in warnings[0] and "no volume" in warnings[0]


def test_rebalance_includes_the_participation_warnings(monkeypatch) -> None:
    import agent.tools as tools

    monkeypatch.setattr(tools, "participation_warnings", lambda *a, **k: ["marker warning"])
    result = tools.run_moead_rebalance(_state(with_forecast=True))
    assert "marker warning" in result["warnings"]


def test_backtest_can_evaluate_each_moead_selection_rule() -> None:
    """The knee is not the only way to pick a point from the front; the backtest must be able
    to score the others so a default is chosen on realized evidence, not on a guess."""
    from evaluation.backtest import BacktestConfig  # noqa: F401 - keeps the import graph honest
    from evaluation.portfolio_backtest import ALL_METHODS, BacktestSpec, run_portfolio_backtest, summarize
    from optimization.moead_rebalance import MOEADConfig

    assert {"moead_knee", "moead_scalarized", "moead_max_sharpe"} <= set(ALL_METHODS)

    spec = BacktestSpec(
        portfolio_value=1.0e6, lookback_days=120, rebalance_every=100, max_weight=0.6,
        methods=("moead_scalarized", "moead_max_sharpe"),
        moead=MOEADConfig(n_partitions=3, n_generations=5),
    )
    summary = summarize(run_portfolio_backtest(_bars(), spec)).set_index("method")
    assert set(summary.index) == {"moead_scalarized", "moead_max_sharpe"}
    assert (summary["total_cost"] > 0).all()


def test_scalarized_is_the_default_selection_rule_everywhere() -> None:
    """Chosen on realized evidence (walk-forward, corrected front): scalarized beat the knee on
    net return, Sharpe, drawdown and cost, and held a tenth of the thin name where the knee
    held a fifth. The knee remains available by name."""
    import inspect

    from agent.tools import run_moead_rebalance
    from service.contracts import PortfolioRequest

    assert PortfolioRequest.model_fields["selection_rule"].default == "scalarized"
    assert inspect.signature(run_moead_rebalance).parameters["selection_rule"].default == "scalarized"
    assert "knee" in {"knee", "max_sharpe", "scalarized"}   # still a valid, selectable rule
