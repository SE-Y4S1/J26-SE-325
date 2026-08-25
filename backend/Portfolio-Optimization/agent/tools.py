"""Callable implementations of the agent tools.

THIN WRAPPERS ONLY. Every function here delegates to an already-tested Phase 2-5b module.
No business logic lives in this file -- if a calculation appears here, it belongs upstream
where it has its own tests. Keeping this layer trivial is what makes the agent auditable:
the tools cannot introduce behaviour that the deterministic pipeline has not already
validated.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from agent.tool_schema import validate_tool_arguments

logger = logging.getLogger(__name__)


def get_technical_signals(symbol: str) -> dict[str, float | None]:
    """Latest MACD, RSI, MFI and ATR. Wraps features.technical."""
    from data.ingestion import bars_to_frame, fetch_ohlcv
    from data.schema import AssetClass
    from features.technical import compute_indicators
    from datetime import date, timedelta

    end = date.today()
    start = end - timedelta(days=180)          # enough for a 26-period MACD to warm up
    bars = bars_to_frame(fetch_ohlcv(symbol, _infer_asset_class(symbol), start, end))

    if bars.empty:
        return {"symbol": symbol, "error": "no price data available"}

    latest = compute_indicators(bars).iloc[-1]
    return {
        "symbol": symbol,
        "macd": _clean(latest.get("macd")),
        "macd_signal": _clean(latest.get("macd_signal")),
        "rsi": _clean(latest.get("rsi")),
        "mfi": _clean(latest.get("mfi")),          # null for forex, by design
        "atr": _clean(latest.get("atr")),
        "atr_pct": _clean(latest.get("atr_pct")),
    }


def get_sentiment(symbol: str) -> dict[str, float | None]:
    """Latest mean_sentiment, sentiment_volume, sentiment_momentum. Wraps features.sentiment."""
    from datetime import date, timedelta

    from features.sentiment import build_sentiment_features

    end = date.today()
    start = end - timedelta(days=14)
    frame = build_sentiment_features(symbol, start, end)

    if frame.empty:
        # An honest "no news" rather than a fabricated neutral score -- the distinction
        # matters, and the model should be able to see it.
        return {
            "symbol": symbol, "mean_sentiment": None,
            "sentiment_volume": 0, "sentiment_momentum": None,
            "note": "no headlines found in the last 14 days",
        }

    latest = frame.iloc[-1]
    return {
        "symbol": symbol,
        "mean_sentiment": _clean(latest["mean_sentiment"]),
        "sentiment_volume": int(latest["sentiment_volume"]),
        "sentiment_momentum": _clean(latest["sentiment_momentum"]),
    }


def get_forecast(symbol: str, horizon: int) -> dict[str, float | str | None]:
    """p10/p50/p90 from the active hybrid model. Wraps forecasting.hybrid_model."""
    from forecasting.model_registry import get_active_version

    try:
        from forecasting.base import get_forecaster

        forecaster = get_forecaster("hybrid")
    except Exception as exc:  # noqa: BLE001 - no trained model yet is a normal early state
        logger.info("no hybrid forecaster available: %s", exc)
        return {
            "symbol": symbol, "horizon": horizon,
            "error": "no trained forecaster is currently registered",
        }

    result = forecaster.predict_quantiles(_features_for(symbol), horizon=horizon)
    frame = result.to_frame().iloc[-1]

    return {
        "symbol": symbol,
        "horizon": horizon,
        "p10": _clean(frame["p10"]),
        "p50": _clean(frame["p50"]),
        "p90": _clean(frame["p90"]),
        "model_version": get_active_version(),
    }


def run_fuzzy_ga_withdrawal(
    urgency: float,
    risk_tolerance: float,
    liquidity_target: float,
    portfolio_state: dict[str, Any],
    deadline_days: int = 1,
) -> dict[str, Any]:
    """Produce the withdrawal plan. Wraps optimization.ga_withdrawal.optimize_withdrawal.

    THE grounded tool: this is the only source of sell amounts and percentages anywhere in
    the agent path. The model chooses the inputs; the numbers come from here.
    """
    from optimization.ga_withdrawal import optimize_withdrawal

    plan = optimize_withdrawal(
        portfolio_state,
        target_amount=liquidity_target,
        withdrawal_urgency=urgency,
        risk_tolerance=risk_tolerance,
        deadline_days=deadline_days,
    )

    return {
        "assets_to_sell": [
            {
                "symbol": step.symbol,
                "sell_fraction": round(step.sell_fraction, 6),
                "quantity": round(step.quantity, 6),
                "expected_price": round(step.expected_price, 4),
                "expected_slippage_pct": round(step.expected_slippage_pct, 6),
                "execution_day": step.execution_day,
            }
            for step in plan.steps
        ],
        "raised_amount": round(plan.raised_amount, 2),
        "target_amount": round(plan.target_amount, 2),
        "shortfall": round(plan.shortfall, 2),
        "expected_slippage_pct": round(plan.expected_slippage_pct, 6),
        "expected_realized_loss": round(plan.expected_realized_loss, 2),
        "residual_portfolio_weights": {k: round(v, 6) for k, v in plan.residual_weights.items()},
        "days_required": plan.days_required,
        "feasible": plan.feasible,
        "fuzzy_rule_trace": list(plan.fuzzy_rule_trace),
    }


def run_moead_rebalance(
    portfolio_state: dict[str, Any],
    risk_preference: float = 0.5,
    selection_rule: str = "knee",
) -> dict[str, Any]:
    """Long-term allocation. Wraps optimization.moead_rebalance + pareto_selection."""
    import numpy as np

    from optimization.moead_rebalance import MOEADConfig, optimize_allocation
    from optimization.pareto_selection import SelectionRule, select

    symbols = tuple(portfolio_state)
    if not symbols:
        return {"error": "portfolio_state is empty"}

    values = np.array([float(portfolio_state[s].get("value", 0.0)) for s in symbols])
    total = values.sum()
    current_weights = values / total if total > 0 else np.full(len(symbols), 1.0 / len(symbols))

    expected_returns = np.array([float(portfolio_state[s].get("expected_return", 0.0)) for s in symbols])
    adv = np.array([float(portfolio_state[s].get("adv_usd", 0.0)) for s in symbols])
    volatility = np.array([float(portfolio_state[s].get("daily_volatility", 0.0126)) for s in symbols])

    # Scenario matrix from a normal approximation around each asset's forecast. A crude but
    # transparent stand-in when no full quantile forecast was supplied by the caller.
    rng = np.random.default_rng(42)
    scenarios = rng.normal(expected_returns, volatility, size=(200, len(symbols)))

    # Lighter than MOEADConfig()'s research defaults (12 partitions x 200 generations,
    # ~18k evaluations, ~30s). This is an INTERACTIVE endpoint and the TAF's claim for it is
    # a "real-time, user-facing service" -- 30 seconds of a spinner is not that. 8 x 100
    # matches what experiments/run_rq_analysis.py actually used to produce the RQ2 result, so
    # this is the measured configuration rather than a degraded one.
    #
    # Offline analysis should pass its own MOEADConfig if it wants a denser front.
    front = optimize_allocation(
        expected_returns, scenarios, adv, current_weights, float(total), symbols,
        volatility=volatility,
        config=MOEADConfig(n_partitions=8, n_generations=100),
    )
    chosen = select(
        front.objectives, front.weights,
        rule=SelectionRule(selection_rule),
        preference_weights=(risk_preference, 1.0 - risk_preference, 0.5),
    )

    return {
        "recommended_weights": {s: round(float(w), 6) for s, w in zip(symbols, chosen.weights, strict=True)},
        "expected_return": round(float(-chosen.objectives[0]), 6),
        "expected_cvar": round(float(chosen.objectives[1]), 6),
        "expected_liquidity_cost": round(float(chosen.objectives[2]), 6),
        "pareto_front_size": len(front),
        "selection_rule": chosen.rule.value,
        "selection_rationale": chosen.rationale,
    }


TOOL_REGISTRY = {
    "get_technical_signals": get_technical_signals,
    "get_sentiment": get_sentiment,
    "get_forecast": get_forecast,
    "run_fuzzy_ga_withdrawal": run_fuzzy_ga_withdrawal,
    "run_moead_rebalance": run_moead_rebalance,
}


def dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate arguments, invoke the tool, return a JSON-serializable observation.

    Tool errors are returned as observations rather than raised: a small model should get
    the chance to correct a bad call, and a crashed run produces no trajectory at all. The
    error text is written for the model to act on, not for a human log reader.
    """
    if name not in TOOL_REGISTRY:
        return {"error": f"unknown tool {name!r}. Available: {', '.join(sorted(TOOL_REGISTRY))}"}

    try:
        validated = validate_tool_arguments(name, arguments)
    except ValueError as exc:
        return {"error": str(exc)}

    try:
        return TOOL_REGISTRY[name](**validated)
    except Exception as exc:  # noqa: BLE001 - surfaced to the model, not swallowed silently
        logger.warning("tool %s failed: %s", name, exc)
        return {"error": f"{name} failed: {exc}"}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _infer_asset_class(symbol: str):
    """Best-effort asset-class inference from the ticker shape."""
    from data.schema import AssetClass

    if symbol.endswith("=X"):
        return AssetClass.FOREX
    if symbol in {"SPY", "QQQ", "IWM", "GLD", "TLT", "XLE"}:
        return AssetClass.ETF
    return AssetClass.EQUITY


def _clean(value: Any) -> float | None:
    """NaN -> None, so the JSON handed to the model is valid and unambiguous.

    `NaN` is not legal JSON and some clients silently coerce it; null is explicit.
    """
    import math

    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(as_float) else round(as_float, 6)


def _features_for(symbol: str):
    """Assemble the recent feature window a forecaster needs for one symbol."""
    from datetime import date, timedelta

    from data.ingestion import bars_to_frame, fetch_ohlcv
    from features.feature_store import build_feature_table
    from features.technical import compute_universe_indicators

    end = date.today()
    start = end - timedelta(days=365)
    bars = {symbol: bars_to_frame(fetch_ohlcv(symbol, _infer_asset_class(symbol), start, end))}
    return build_feature_table(bars, compute_universe_indicators(bars))
