"""The three objective functions, shared by MOEA/D (5a) and the fuzzy GA (5b).

Both optimizers score candidates with the SAME liquidity cost model, which is what makes
RQ2 and RQ3 comparable: any difference between them is the algorithm, not the accounting.
"""

from __future__ import annotations

import numpy as np

# Market-impact exponent. 0.5 is the square-root law: impact grows with the SQUARE ROOT of
# participation, not linearly -- selling 4x as much costs ~2x as much in impact, not 4x.
# Standard in the market-impact literature (Almgren, Thum, Hauptmann & Li 2005, "Direct
# estimates of equity market impact"; the Barra/Kyle family of models). Empirically robust
# across venues and the conventional starting point for a transaction-cost model.
IMPACT_EXPONENT = 0.5

# Impact coefficient (dimensionless), calibrated so a 10% ADV trade in a 20%-annual-vol name
# costs roughly 30bp -- the right order of magnitude for US large-cap. Phase 7 treats this as
# a sensitivity parameter, not a constant of nature.
IMPACT_COEFFICIENT = 0.30

# Fallback daily volatility when none is supplied (~20% annualized).
DEFAULT_DAILY_VOLATILITY = 0.0126


def expected_return(weights: np.ndarray, forecasts: np.ndarray) -> float:
    """Portfolio expected return: w . mu, using the p50 forecast as mu."""
    weights = np.asarray(weights, dtype=float)
    forecasts = np.asarray(forecasts, dtype=float)
    if weights.shape != forecasts.shape:
        raise ValueError(f"shape mismatch: weights {weights.shape} vs forecasts {forecasts.shape}")
    return float(np.dot(weights, forecasts))


def risk_cvar(weights: np.ndarray, quantile_forecasts: np.ndarray, alpha: float = 0.95) -> float:
    """Conditional Value-at-Risk (expected shortfall) at level `alpha`.

    CVaR rather than variance because variance penalizes upside and downside equally, while
    a withdrawing user cares only about the downside tail. CVaR is also coherent
    (sub-additive), so it rewards diversification rather than fighting it.

    `quantile_forecasts` is (n_assets, n_quantiles) or (n_scenarios, n_assets):
      * A scenario matrix is used directly -- the empirical CVaR of portfolio returns.
      * A quantile matrix is expanded into scenarios first, since CVaR of a portfolio is not
        the weighted sum of per-asset CVaRs (that would ignore diversification entirely and
        systematically overstate risk).

    Returned POSITIVE for a loss, so minimizing it is the natural objective direction.

    NOTE FOR THE WRITE-UP: CVaR is not named in the TAF, which commits only to
    "liquidity-aware" and "loss-minimized". It is a deliberate strengthening -- be ready to
    justify it. See README "Open items".
    """
    weights = np.asarray(weights, dtype=float)
    matrix = np.asarray(quantile_forecasts, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("quantile_forecasts must be 2-D")

    # Orient so columns are assets.
    if matrix.shape[0] == weights.size and matrix.shape[1] != weights.size:
        scenarios = matrix.T          # (n_assets, n_quantiles) -> (n_quantiles, n_assets)
    elif matrix.shape[1] == weights.size:
        scenarios = matrix            # already (n_scenarios, n_assets)
    else:
        raise ValueError(
            f"cannot align weights ({weights.size}) with forecasts {matrix.shape}"
        )

    portfolio_scenarios = scenarios @ weights
    if portfolio_scenarios.size == 0:
        return 0.0

    # Losses are negative returns; VaR is the alpha-quantile of the loss distribution.
    losses = -portfolio_scenarios
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    # With few quantile points the tail can be a single value; fall back to VaR itself.
    return float(tail.mean()) if tail.size else var


def liquidity_cost(
    weights: np.ndarray,
    trade_sizes: np.ndarray,
    avg_daily_volume: np.ndarray,
    volatility: np.ndarray | None = None,
) -> float:
    """Expected slippage from executing `trade_sizes`, via the square-root impact law:

        cost_i = IMPACT_COEFFICIENT * sigma_i * (|Q_i| / ADV_i) ** IMPACT_EXPONENT

    where Q_i and ADV_i are both in currency units. The per-asset costs are weighted by the
    notional actually traded, so the result is a portfolio-level fractional cost.

    This is the term that makes the optimizer "liquidity-aware" -- it is what a mean-variance
    baseline structurally cannot see, and therefore the source of the RQ2 advantage.
    """
    trade_sizes = np.abs(np.asarray(trade_sizes, dtype=float))
    adv = np.asarray(avg_daily_volume, dtype=float)

    if trade_sizes.shape != adv.shape:
        raise ValueError(f"shape mismatch: trades {trade_sizes.shape} vs ADV {adv.shape}")

    sigma = (
        np.full_like(trade_sizes, DEFAULT_DAILY_VOLATILITY)
        if volatility is None
        else np.asarray(volatility, dtype=float)
    )

    # Zero ADV means the asset cannot be traded at any price. Treating it as infinitely
    # costly (rather than dividing by zero) lets the optimizer route around it naturally.
    with np.errstate(divide="ignore", invalid="ignore"):
        participation = np.where(adv > 0, trade_sizes / adv, np.inf)

    per_asset = IMPACT_COEFFICIENT * sigma * np.power(participation, IMPACT_EXPONENT)
    per_asset = np.where(trade_sizes > 0, per_asset, 0.0)

    total_notional = trade_sizes.sum()
    if total_notional <= 0:
        return 0.0

    cost = float(np.dot(per_asset, trade_sizes) / total_notional)
    return cost if np.isfinite(cost) else float(1e6)


def realized_loss(
    entry_prices: np.ndarray,
    exit_prices: np.ndarray,
    quantities: np.ndarray,
) -> float:
    """Realized loss on a liquidation, POSITIVE when money was lost.

    Used as the fuzzy GA's primary fitness term (RQ3). Sign convention matches
    liquidity_cost so both can be summed directly into a minimization objective.
    """
    entry = np.asarray(entry_prices, dtype=float)
    exit_ = np.asarray(exit_prices, dtype=float)
    qty = np.asarray(quantities, dtype=float)

    if not (entry.shape == exit_.shape == qty.shape):
        raise ValueError("entry_prices, exit_prices and quantities must share a shape")

    pnl = float(np.dot(exit_ - entry, qty))
    return -pnl


def portfolio_volatility(weights: np.ndarray, covariance: np.ndarray) -> float:
    """sqrt(w' S w). Used by the mean-variance baseline and by Sharpe reporting."""
    weights = np.asarray(weights, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    variance = float(weights @ cov @ weights)
    return float(np.sqrt(max(variance, 0.0)))
