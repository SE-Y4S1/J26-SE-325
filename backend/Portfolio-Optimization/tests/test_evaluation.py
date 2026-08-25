"""Phase 7 tests: metrics correctness and the no-look-ahead guarantee.

The look-ahead tests matter most. Look-ahead bias is invisible in output -- results just
look good -- so the guard must be proven to FIRE, not merely to exist. A guard that never
triggers proves nothing.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from evaluation.backtest import (
    BacktestConfig,
    assert_fold_is_clean,
    assert_no_lookahead,
    generate_folds,
)
from evaluation.metrics import (
    allocation_metrics,
    degradation_curve,
    forecast_metrics,
    mae,
    max_drawdown,
    pinball_loss,
    quantile_coverage,
    realized_transaction_cost,
    rmse,
    sharpe_ratio,
    slippage_vs_baseline,
    sortino_ratio,
)


# --------------------------------------------------------------------------------------
# RQ1 metrics
# --------------------------------------------------------------------------------------

def test_mae_and_rmse_on_known_values() -> None:
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.5, 2.0, 2.0])
    assert mae(actual, predicted) == pytest.approx(0.5)
    assert rmse(actual, predicted) == pytest.approx(np.sqrt((0.25 + 0 + 1) / 3))


def test_mae_ignores_nans_rather_than_propagating() -> None:
    """A single NaN fold must not wipe out a whole metric table."""
    assert mae(np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, 3.0])) == pytest.approx(0.0)


def test_pinball_loss_is_asymmetric_per_quantile() -> None:
    """The defining property: under-prediction at p90 is penalized 0.9, over-prediction 0.1.
    A symmetric loss would produce a conditional mean, not a quantile."""
    actual = np.array([10.0])

    under = pinball_loss(actual, np.array([[5.0]]), (0.9,))    # predicted too low
    over = pinball_loss(actual, np.array([[15.0]]), (0.9,))    # predicted too high by the same
    assert under == pytest.approx(0.9 * 5)
    assert over == pytest.approx(0.1 * 5)
    assert under > over


def test_pinball_loss_is_zero_for_a_perfect_forecast() -> None:
    actual = np.array([1.0, 2.0])
    assert pinball_loss(actual, np.array([[1.0], [2.0]]), (0.5,)) == pytest.approx(0.0)


def test_pinball_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="prediction columns"):
        pinball_loss(np.array([1.0]), np.array([[1.0, 2.0]]), (0.5,))


def test_quantile_coverage_detects_a_well_calibrated_model() -> None:
    """A correctly calibrated p10 is exceeded ~10% of the time."""
    rng = np.random.default_rng(0)
    actual = rng.normal(0, 1, 20_000)
    preds = np.column_stack([
        np.full(20_000, np.quantile(actual, 0.1)),
        np.full(20_000, np.quantile(actual, 0.5)),
        np.full(20_000, np.quantile(actual, 0.9)),
    ])
    coverage = quantile_coverage(actual, preds, (0.1, 0.5, 0.9))
    assert coverage["coverage_p10"] == pytest.approx(0.1, abs=0.02)
    assert coverage["coverage_p90"] == pytest.approx(0.9, abs=0.02)


def test_quantile_coverage_flags_an_overconfident_model() -> None:
    """An over-tight interval must show a large calibration error -- this is the diagnostic
    that stops an over-confident forecaster from silently making CVaR optimistic."""
    rng = np.random.default_rng(1)
    actual = rng.normal(0, 1, 5_000)
    # Interval 10x too narrow.
    preds = np.column_stack([np.full(5_000, -0.12), np.zeros(5_000), np.full(5_000, 0.12)])
    coverage = quantile_coverage(actual, preds, (0.1, 0.5, 0.9))
    assert coverage["calibration_error_p90"] > 0.2


def test_forecast_metrics_bundle_has_every_rq1_field() -> None:
    actual = np.array([0.01, -0.02, 0.03, 0.0])
    preds = np.column_stack([actual - 0.01, actual, actual + 0.01])
    metrics = forecast_metrics(actual, preds)
    for key in ("mae", "rmse", "pinball_loss", "coverage_p10", "coverage_p90"):
        assert key in metrics


# --------------------------------------------------------------------------------------
# RQ2 metrics
# --------------------------------------------------------------------------------------

def test_sharpe_is_positive_for_a_profitable_series() -> None:
    rng = np.random.default_rng(2)
    assert sharpe_ratio(pd.Series(rng.normal(0.001, 0.01, 500))) > 0


def test_sortino_exceeds_sharpe_when_volatility_is_upside() -> None:
    """The reason RQ2 reports both: Sharpe punishes upside variance, Sortino does not."""
    returns = pd.Series([0.05, 0.06, -0.005, 0.07, -0.004, 0.08] * 40)
    assert sortino_ratio(returns) > sharpe_ratio(returns)


def test_max_drawdown_on_a_known_curve() -> None:
    equity = pd.Series([100.0, 120.0, 90.0, 110.0])   # peak 120 -> trough 90 = 25%
    assert max_drawdown(equity) == pytest.approx(0.25)


def test_max_drawdown_is_zero_for_a_monotonic_curve() -> None:
    assert max_drawdown(pd.Series([100.0, 110.0, 120.0])) == pytest.approx(0.0)


def test_realized_transaction_cost_from_notional_and_slippage() -> None:
    trades = pd.DataFrame({"notional": [100_000.0, 50_000.0], "slippage_pct": [0.001, 0.002]})
    assert realized_transaction_cost(trades) == pytest.approx(100 + 100)


def test_realized_transaction_cost_rejects_an_unusable_frame() -> None:
    with pytest.raises(ValueError, match="cost"):
        realized_transaction_cost(pd.DataFrame({"symbol": ["AAPL"]}))


def test_allocation_metrics_bundle() -> None:
    rng = np.random.default_rng(3)
    metrics = allocation_metrics(pd.Series(rng.normal(0.0005, 0.01, 300)))
    for key in ("sharpe", "sortino", "max_drawdown", "total_return"):
        assert key in metrics


# --------------------------------------------------------------------------------------
# RQ3 / RQ4 metrics
# --------------------------------------------------------------------------------------

class _Plan:
    """Minimal stand-in for a WithdrawalPlan."""

    def __init__(self, loss: float, slippage: float, feasible: bool = True) -> None:
        self.expected_realized_loss = loss
        self.expected_slippage_pct = slippage
        self.feasible = feasible
        self.steps = ()


def test_slippage_vs_baseline_reports_the_improvement() -> None:
    comparison = slippage_vs_baseline(_Plan(800.0, 0.0016), _Plan(1000.0, 0.0020))
    assert comparison["absolute_improvement"] == pytest.approx(200.0)
    assert comparison["relative_improvement_pct"] == pytest.approx(20.0)


def test_slippage_vs_baseline_survives_a_zero_cost_baseline() -> None:
    """Guarded division -- a nothing-to-sell baseline must not blow up the RQ3 table."""
    comparison = slippage_vs_baseline(_Plan(0.0, 0.0), _Plan(0.0, 0.0))
    assert comparison["relative_improvement_pct"] == 0.0


def test_degradation_curve_identifies_the_breakdown_point() -> None:
    """The 'where does it break down' half of RQ4."""
    curve = degradation_curve(
        {
            0.0: {"realized_loss": 100.0, "feasible": True},
            0.5: {"realized_loss": 250.0, "feasible": True},
            1.0: {"realized_loss": 900.0, "feasible": False},
        }
    )
    assert list(curve["severity"]) == [0.0, 0.5, 1.0]
    assert curve.attrs["breakdown_severity"] == pytest.approx(1.0)
    assert curve["loss_vs_baseline_pct"].iloc[-1] == pytest.approx(800.0)


# --------------------------------------------------------------------------------------
# Walk-forward integrity
# --------------------------------------------------------------------------------------

def test_folds_do_not_overlap_and_honour_the_embargo() -> None:
    config = BacktestConfig(train_window_days=365, test_window_days=90, step_days=90, embargo_days=5)
    folds = generate_folds(date(2018, 1, 1), date(2024, 1, 1), config)

    assert len(folds) > 1
    for fold in folds:
        assert fold.train_end < fold.test_start
        assert (fold.test_start - fold.train_end).days >= config.embargo_days
    for earlier, later in zip(folds, folds[1:]):
        assert later.test_start >= earlier.test_end, "test windows overlap"


def test_generate_folds_returns_empty_when_the_range_is_too_short() -> None:
    config = BacktestConfig(train_window_days=756, test_window_days=63)
    assert generate_folds(date(2023, 1, 1), date(2023, 6, 1), config) == []


def test_assert_no_lookahead_passes_on_clean_features() -> None:
    features = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=5)})
    assert_no_lookahead(features, pd.Timestamp("2024-01-10"))


def test_assert_no_lookahead_FIRES_on_contaminated_features() -> None:
    """The guard must actually trigger. This is the test that makes the guarantee real."""
    features = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=10)})
    with pytest.raises(ValueError, match="look-ahead detected"):
        assert_no_lookahead(features, pd.Timestamp("2024-01-05"))


def test_assert_fold_is_clean_catches_overlap() -> None:
    train = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=30)})
    test = pd.DataFrame({"timestamp": pd.date_range("2024-01-20", periods=30)})   # overlaps
    with pytest.raises(ValueError, match="fold overlap"):
        assert_fold_is_clean(train, test, embargo_days=5)


def test_assert_fold_is_clean_catches_embargo_violation() -> None:
    """A gap shorter than the horizon means training targets reach into the test window."""
    train = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=30)})
    test = pd.DataFrame({"timestamp": pd.date_range("2024-01-31", periods=30)})   # 1-day gap
    with pytest.raises(ValueError, match="embargo violated"):
        assert_fold_is_clean(train, test, embargo_days=5)


def test_assert_fold_is_clean_passes_with_a_sufficient_gap() -> None:
    train = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=30)})
    test = pd.DataFrame({"timestamp": pd.date_range("2024-02-10", periods=30)})
    assert_fold_is_clean(train, test, embargo_days=5)
