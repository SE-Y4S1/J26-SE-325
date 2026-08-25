"""Metrics for RQ1-RQ4.

Each research question gets a concrete number here -- this is what turns "a pipeline that
runs" into a defensible result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# --- RQ1: forecast quality -------------------------------------------------------------

def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    return float(np.mean(np.abs(actual[mask] - predicted[mask]))) if mask.any() else float("nan")


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    return float(np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2))) if mask.any() else float("nan")


def pinball_loss(
    actual: np.ndarray, quantile_preds: np.ndarray, quantiles: tuple[float, ...]
) -> float:
    """Mean pinball loss -- the proper scoring rule for quantile forecasts.

    MAE/RMSE only score the median; pinball is what tells us whether the p10/p90 band is
    honest, which is what Phase 5a's CVaR objective actually consumes. A model can win on
    MAE while producing a useless interval.
    """
    actual = np.asarray(actual, dtype=float).reshape(-1, 1)
    preds = np.asarray(quantile_preds, dtype=float)

    if preds.shape[1] != len(quantiles):
        raise ValueError(f"{preds.shape[1]} prediction columns but {len(quantiles)} quantiles")

    errors = actual - preds
    q = np.asarray(quantiles, dtype=float)
    losses = np.maximum(q * errors, (q - 1.0) * errors)
    return float(np.nanmean(losses))


def quantile_coverage(
    actual: np.ndarray, quantile_preds: np.ndarray, quantiles: tuple[float, ...]
) -> dict[str, float]:
    """Empirical vs nominal coverage -- is the p10 really exceeded 10% of the time?

    Calibration diagnostic. A model can win on pinball while being systematically
    over-confident, which would make CVaR optimistic and understate withdrawal risk -- the
    exact failure that would make RQ3's headline number look better than reality.
    """
    actual = np.asarray(actual, dtype=float)
    preds = np.asarray(quantile_preds, dtype=float)

    coverage: dict[str, float] = {}
    for i, q in enumerate(quantiles):
        mask = ~(np.isnan(actual) | np.isnan(preds[:, i]))
        if not mask.any():
            coverage[f"coverage_p{int(q * 100)}"] = float("nan")
            continue
        empirical = float(np.mean(actual[mask] <= preds[mask, i]))
        coverage[f"coverage_p{int(q * 100)}"] = empirical
        coverage[f"calibration_error_p{int(q * 100)}"] = abs(empirical - q)
    return coverage


def forecast_metrics(
    actual: np.ndarray, quantile_preds: np.ndarray, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
) -> dict[str, float]:
    """The full RQ1 metric bundle for one model/asset-class/horizon cell."""
    preds = np.asarray(quantile_preds, dtype=float)
    median_idx = int(np.argmin(np.abs(np.asarray(quantiles) - 0.5)))
    point = preds[:, median_idx]

    return {
        "mae": mae(actual, point),
        "rmse": rmse(actual, point),
        "pinball_loss": pinball_loss(actual, preds, quantiles),
        **quantile_coverage(actual, preds, quantiles),
    }


# --- RQ2: allocation quality -----------------------------------------------------------

def sharpe_ratio(returns: pd.Series, *, risk_free_rate: float = 0.0, annualize: bool = True) -> float:
    clean = pd.Series(returns).dropna()
    if len(clean) < 2:
        return float("nan")

    excess = clean - risk_free_rate / TRADING_DAYS_PER_YEAR
    std = excess.std(ddof=1)
    if std == 0:
        return float("nan")

    ratio = float(excess.mean() / std)
    return ratio * np.sqrt(TRADING_DAYS_PER_YEAR) if annualize else ratio


def sortino_ratio(returns: pd.Series, *, risk_free_rate: float = 0.0, annualize: bool = True) -> float:
    """Like Sharpe but penalizing only downside deviation.

    The right lens for RQ2: the liquidity-aware optimizer deliberately accepts upside
    variance to cut tail risk, and Sharpe would punish it for exactly the behaviour we want.
    """
    clean = pd.Series(returns).dropna()
    if len(clean) < 2:
        return float("nan")

    excess = clean - risk_free_rate / TRADING_DAYS_PER_YEAR
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")      # no losing periods at all

    downside_std = float(np.sqrt(np.mean(downside**2)))
    if downside_std == 0:
        return float("nan")

    ratio = float(excess.mean() / downside_std)
    return ratio * np.sqrt(TRADING_DAYS_PER_YEAR) if annualize else ratio


def max_drawdown(equity_curve: pd.Series) -> float:
    """Largest peak-to-trough decline, returned POSITIVE (0.2 == a 20% drawdown)."""
    clean = pd.Series(equity_curve).dropna()
    if clean.empty:
        return float("nan")
    running_peak = clean.cummax()
    return float((-(clean / running_peak - 1.0)).max())


def realized_transaction_cost(trades: pd.DataFrame) -> float:
    """Total realized cost.

    RQ2 requires MOEA/D to win at EQUAL OR LOWER cost -- beating the baseline by simply
    trading more would not support the claim, so this must be reported alongside Sharpe
    rather than left implicit.
    """
    if trades.empty:
        return 0.0
    if "cost" in trades.columns:
        return float(trades["cost"].sum())
    if {"notional", "slippage_pct"} <= set(trades.columns):
        return float((trades["notional"] * trades["slippage_pct"]).sum())
    raise ValueError("trades needs either a 'cost' column or 'notional' + 'slippage_pct'")


def allocation_metrics(returns: pd.Series, trades: pd.DataFrame | None = None) -> dict[str, float]:
    """The full RQ2 metric bundle for one strategy."""
    equity = (1 + pd.Series(returns).dropna()).cumprod()
    return {
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "total_return": float(equity.iloc[-1] - 1) if len(equity) else float("nan"),
        "realized_transaction_cost": realized_transaction_cost(trades) if trades is not None else 0.0,
    }


# --- RQ3 / RQ4: withdrawal quality ------------------------------------------------------

def realized_slippage_pct(plan, executed_prices: dict[str, float] | None = None) -> float:
    """Realized slippage as a fraction of notional raised.

    With `executed_prices` supplied the slippage is measured against them (a backtest with
    real fills); without, the plan's own expectation is reported.
    """
    if executed_prices is None:
        return float(plan.expected_slippage_pct)

    total_notional = 0.0
    total_cost = 0.0
    for step in plan.steps:
        actual = executed_prices.get(step.symbol)
        if actual is None:
            continue
        notional = step.quantity * actual
        # step.expected_price already has expected slippage applied; the difference against
        # the true fill is the realized error.
        total_cost += abs(step.expected_price - actual) * step.quantity
        total_notional += notional

    return float(total_cost / total_notional) if total_notional > 0 else 0.0


def slippage_vs_baseline(fuzzy_ga_result, baseline_result) -> dict[str, float]:
    """Head-to-head slippage and realized loss, absolute and relative. The RQ3 headline."""
    ga_loss = float(fuzzy_ga_result.expected_realized_loss)
    base_loss = float(baseline_result.expected_realized_loss)

    return {
        "fuzzy_ga_realized_loss": ga_loss,
        "baseline_realized_loss": base_loss,
        "absolute_improvement": base_loss - ga_loss,
        # Guarded: a zero-cost baseline (nothing to sell) would otherwise divide by zero.
        "relative_improvement_pct": ((base_loss - ga_loss) / base_loss * 100) if base_loss > 0 else 0.0,
        "fuzzy_ga_slippage_pct": float(fuzzy_ga_result.expected_slippage_pct),
        "baseline_slippage_pct": float(baseline_result.expected_slippage_pct),
        "fuzzy_ga_feasible": bool(fuzzy_ga_result.feasible),
        "baseline_feasible": bool(baseline_result.feasible),
    }


def degradation_curve(results_by_severity: dict[float, dict[str, float]]) -> pd.DataFrame:
    """Plan quality as a function of stress severity, plus the breakdown point.

    The breakdown point is the first severity at which the plan becomes infeasible -- the
    "where does it break" half of RQ4, which a curve alone does not answer.
    """
    rows = []
    for severity in sorted(results_by_severity):
        row = {"severity": severity, **results_by_severity[severity]}
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    if "feasible" in frame.columns:
        infeasible = frame.loc[~frame["feasible"].astype(bool), "severity"]
        frame.attrs["breakdown_severity"] = float(infeasible.min()) if len(infeasible) else None

    if "realized_loss" in frame.columns:
        baseline = frame["realized_loss"].iloc[0]
        frame["loss_vs_baseline_pct"] = (
            (frame["realized_loss"] - baseline) / baseline * 100 if baseline > 0 else np.nan
        )

    return frame
