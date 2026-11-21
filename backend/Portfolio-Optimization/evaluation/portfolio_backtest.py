"""Rolling out-of-sample evaluation of the allocation methods (RQ2).

RQ2 asks whether the liquidity-aware optimizer beats mean-variance on Sharpe, Sortino,
drawdown and realized cost. The predicted objectives an optimizer reports about its own
answer cannot settle that: they come from the model that chose the answer. This replays the
decisions through time instead. At each rebalance date every method sees ONLY bars up to that
date, picks target weights, pays the modelled trading cost on the trade from its CURRENT
(drifted) book, and is then scored on the returns that actually followed.

What "realized cost" means here, honestly: there are no fills, so execution cost is the
square-root impact model evaluated on the trade actually made, with ADV and volatility
measured from trailing bars at that date. It is a modelled cost on realized trades, applied
identically to every method -- not a measurement of real slippage. The comparison is fair;
the absolute level is only as good as the impact coefficient.

Inputs shared by every method at a rebalance date (so a difference is the method, not the
inputs): the expected-return vector, the shrunk covariance, the joint scenarios, ADV, vol.
Expected returns are the trailing mean shrunk halfway to the cross-sectional mean -- a
forecast-free stand-in; swap in forecaster output to evaluate the full system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from optimization.baseline_meanvariance import cost_aware_mean_variance, max_sharpe_portfolio
from optimization.moead_rebalance import MOEADConfig, optimize_allocation
from optimization.objectives import liquidity_cost_total
from optimization.pareto_selection import SelectionRule, select
from optimization.scenarios import joint_scenarios

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
ADV_WINDOW = 20
METHODS = ("hold", "equal_weight", "markowitz", "markowitz_cost_aware", "moead_knee")
# Every selection rule MOEA/D can be scored with; the default run only reports the knee.
MOEAD_RULES = {
    "moead_knee": SelectionRule.KNEE,
    "moead_scalarized": SelectionRule.SCALARIZED,
    "moead_max_sharpe": SelectionRule.MAX_SHARPE,
}
ALL_METHODS = METHODS + tuple(m for m in MOEAD_RULES if m not in METHODS)


@dataclass(frozen=True)
class BacktestSpec:
    portfolio_value: float = 1.0e6
    lookback_days: int = 252
    rebalance_every: int = 21
    methods: tuple[str, ...] = METHODS
    max_weight: float = 0.25
    n_scenarios: int = 500
    mu_shrinkage: float = 0.5          # weight on the cross-sectional mean
    cov_shrinkage: float = 0.2         # weight on the diagonal
    seed: int = 42
    moead: MOEADConfig = field(default_factory=lambda: MOEADConfig(n_partitions=8, n_generations=100))


@dataclass
class BacktestResult:
    decisions: pd.DataFrame   # method, rebalance_date, symbol, prior_weight, target_weight, cost_fraction
    daily: pd.DataFrame       # method, date, gross_return, cost_fraction, net_return


def _panel(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """close and volume panels on the dates every symbol traded."""
    bars = bars.drop_duplicates(["timestamp", "symbol"])
    close = bars.pivot(index="timestamp", columns="symbol", values="close").sort_index().dropna()
    volume = bars.pivot(index="timestamp", columns="symbol", values="volume").reindex(close.index)
    return close, volume.fillna(0.0)


def _target_weights(
    method: str,
    spec: BacktestSpec,
    *,
    current: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    scenarios: np.ndarray,
    adv: np.ndarray,
    vol: np.ndarray,
    symbols: tuple[str, ...],
) -> np.ndarray:
    n = mu.size
    cap = max(spec.max_weight, 1.0 / n)
    if method == "hold":
        return current
    if method == "equal_weight":
        return np.full(n, 1.0 / n)
    if method == "markowitz":
        return max_sharpe_portfolio(mu, cov, max_weight=cap).weights
    if method == "markowitz_cost_aware":
        return cost_aware_mean_variance(
            mu, cov, current, adv, spec.portfolio_value, volatility=vol, max_weight=cap
        ).weights
    if method in MOEAD_RULES:
        config = MOEADConfig(**{**spec.moead.__dict__, "max_weight": cap, "seed": spec.seed})
        front = optimize_allocation(
            mu, scenarios, adv, current, spec.portfolio_value, symbols, volatility=vol, config=config
        )
        return select(front.objectives, front.weights, rule=MOEAD_RULES[method]).weights
    raise ValueError(f"unknown method {method!r}; choose from {ALL_METHODS}")


def run_portfolio_backtest(bars: pd.DataFrame, spec: BacktestSpec | None = None) -> BacktestResult:
    """Walk forward from the first date with `lookback_days` of history.

    Every method starts from the same equal-weight book (no cost to establish it), then
    drifts with returns between rebalances and pays to trade back to its targets.
    """
    spec = spec or BacktestSpec()
    close, volume = _panel(bars)
    symbols = tuple(close.columns)
    n = len(symbols)
    no_volume = [s for s in symbols if not (volume[s] > 0).any()]
    if no_volume:
        raise ValueError(
            f"no volume for {', '.join(no_volume)}: ADV is zero, so every trade in them would "
            "cost the untradeable sentinel and swamp the comparison. Drop them (FX bars have none)."
        )
    if len(close) <= spec.lookback_days + spec.rebalance_every:
        raise ValueError(f"need more than {spec.lookback_days + spec.rebalance_every} common dates, got {len(close)}")

    returns = close.pct_change().fillna(0.0)
    dollar_volume = (close * volume).rolling(ADV_WINDOW, min_periods=1).mean()
    dates = close.index
    rebalance_idx = set(range(spec.lookback_days, len(dates) - 1, spec.rebalance_every))
    start = spec.lookback_days

    decisions: list[dict] = []
    daily: list[dict] = []

    for method in spec.methods:
        weights = np.full(n, 1.0 / n)      # drifted book, as fractions of portfolio value
        for t in range(start, len(dates)):
            # Day t's return accrues to the book held BEFORE any rebalance at the close of t;
            # applying it after would score weights chosen with knowledge of that return.
            gross = 0.0
            if t > start:
                day = returns.iloc[t].to_numpy()
                gross = float(weights @ day)
                book = weights * (1.0 + day)
                weights = book / book.sum()

            cost_fraction = 0.0
            if t in rebalance_idx:
                window = returns.iloc[t - spec.lookback_days + 1: t + 1]
                history = window.to_numpy()
                vol = history.std(axis=0, ddof=1)
                trailing = history.mean(axis=0)
                mu = (1 - spec.mu_shrinkage) * trailing + spec.mu_shrinkage * trailing.mean()
                sample = np.cov(history, rowvar=False)
                cov = (1 - spec.cov_shrinkage) * sample + spec.cov_shrinkage * np.diag(np.diag(sample))
                scenarios, _ = joint_scenarios(
                    mu, vol, history=history, n_scenarios=spec.n_scenarios, seed=spec.seed
                )
                adv = dollar_volume.iloc[t].to_numpy()

                target = _target_weights(
                    method, spec, current=weights, mu=mu, cov=cov, scenarios=scenarios,
                    adv=adv, vol=vol, symbols=symbols,
                )
                trade = np.abs(target - weights) * spec.portfolio_value
                cost_fraction = liquidity_cost_total(trade, adv, vol) / spec.portfolio_value
                for symbol, before, after in zip(symbols, weights, target, strict=True):
                    decisions.append({
                        "method": method, "rebalance_date": dates[t], "symbol": symbol,
                        "prior_weight": float(before), "target_weight": float(after),
                        "cost_fraction": cost_fraction,
                    })
                weights = target

            daily.append({
                "method": method, "date": dates[t], "gross_return": gross,
                "cost_fraction": cost_fraction, "net_return": gross - cost_fraction,
            })

    return BacktestResult(decisions=pd.DataFrame(decisions), daily=pd.DataFrame(daily))


def _metrics(frame: pd.DataFrame, risk_free_daily: float = 0.0) -> dict[str, float]:
    net = frame["net_return"].to_numpy()
    excess = net - risk_free_daily
    std = excess.std(ddof=1) if len(excess) > 1 else 0.0
    downside = np.minimum(excess, 0.0)
    downside_dev = float(np.sqrt((downside ** 2).mean()))
    equity = np.cumprod(1.0 + net)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    return {
        "net_return": float(equity[-1] - 1.0),
        "ann_return": float(equity[-1] ** (TRADING_DAYS / len(net)) - 1.0),
        "sharpe": float(excess.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0,
        "sortino": float(excess.mean() / downside_dev * np.sqrt(TRADING_DAYS)) if downside_dev > 0 else float("inf"),
        "max_drawdown": float(drawdown.min()),
        "total_cost": float(frame["cost_fraction"].sum()),
    }


def summarize(result: BacktestResult) -> pd.DataFrame:
    """One row per method: realized net performance, cost, and turnover."""
    rows = []
    for method, frame in result.daily.groupby("method", sort=False):
        trades = result.decisions[result.decisions["method"] == method]
        turnover = 0.0
        if not trades.empty:
            per_date = (trades["target_weight"] - trades["prior_weight"]).abs().groupby(trades["rebalance_date"]).sum()
            turnover = float((per_date / 2).sum())   # one-way turnover, summed over rebalances
        rows.append({
            "method": method, **_metrics(frame), "turnover": turnover,
            "n_rebalances": int(trades["rebalance_date"].nunique()) if not trades.empty else 0,
        })
    return pd.DataFrame(rows)
