"""FastAPI service -- Component 1's interface to the rest of the platform.

The TAF's novelty claim for this component is that liquidity-aware withdrawal planning is
"operationalized as a real-time, user-facing microservice". /portfolio/withdraw IS that
claim; the rest of the codebase exists to make it correct.

Responses carry no natural-language explanation by design. This service returns structured
decisions plus traces; Component 4 renders them for users.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException

from service.contracts import (
    AssetSale,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    Holding,
    PortfolioRequest,
    PortfolioResponse,
    RebalanceOrder,
    SymbolForecast,
    WithdrawalRequest,
    WithdrawalResponse,
)
from data.http import UpstreamError
from service.auth import CallerIdentity, require_user
from service.events import publish_decision

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Component 1 - Liquidity-Aware Forecasting & Portfolio Optimization",
    version="0.1.0",
    description=(
        "Liquidity-aware forecasting and multi-objective portfolio optimization with "
        "instant, loss-minimized withdrawal planning (J26-SE-325, Component 1)."
    ),
)


def _install_cors(application: FastAPI) -> None:
    """Enable CORS so the browser may call this service directly.

    The frontend talks to each backend from the browser rather than proxying through
    Next.js, so without these headers the browser rejects every response before any handler
    runs. Origins come from ALLOWED_ORIGINS; see service/cors.py.
    """
    try:
        from service.cors import install_cors

        install_cors(application)
    except ImportError:  # pragma: no cover - cors.py is part of the package
        logger.warning("service/cors.py missing; browser calls will be blocked")


_install_cors(app)


def _install_metrics(application: FastAPI) -> None:
    """Expose /metrics for Prometheus (Phase 8).

    Optional at import time: the instrumentator is only needed for the containerized
    deployment, and a missing monitoring dependency must never stop the service that carries
    this component's novelty claim from starting.
    """
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator().instrument(application).expose(application, include_in_schema=False)
        logger.info("Prometheus metrics available at /metrics")
    except ImportError:
        logger.info("prometheus-fastapi-instrumentator not installed; /metrics disabled")


_install_metrics(app)


from service.market import router as market_router  # noqa: E402 - after app construction

app.include_router(market_router)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _model_version() -> str:
    """Active model version, or a sentinel when nothing is registered yet.

    Never raises: the withdrawal path does not depend on a trained forecaster, and a missing
    registry must not take down the endpoint that carries this component's novelty claim.
    """
    try:
        from forecasting.model_registry import get_active_version

        return get_active_version()
    except Exception as exc:  # noqa: BLE001
        logger.debug("no active model version: %s", exc)
        return "unregistered"


# Used only for holdings with no prepared metrics (untracked, or preparation pending).
DEFAULT_DAILY_VOLATILITY = 0.0126
DEFAULT_VOLATILITY_PCT = 0.5


def _build_portfolio(
    holdings: list[Holding], *, deadline_days: int = 1, base_currency: str | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, dict], dict]:
    """Contract models -> (optimizer state, per-holding market context, currency block).

    Request path: reads the active set's SQLite snapshot only, never a model or an exchange
    -- except an FX rate for a currency not seen in the last hour (data/fx.py caches them).

    * Every amount is converted into the base currency, so a CSE holding in LKR and a US
      holding in USD are planned against one LKR target. The conversion spread becomes the
      holding's fx_cost_pct, which the GA charges on proceeds.
    * Volatility, forecast drift/spread, news sentiment and attention come from the prepared
      snapshot when the symbol has one; a symbol not yet prepared falls back to the old
      constants and no signals (its market_context says `untracked` or `preparing`).
    """
    from data import active_set
    from data.fx import BASE_CURRENCY, conversion_spread, get_rate
    from optimization.signals import holding_signals, market_context

    from data.instruments import native_currency

    base = (base_currency or BASE_CURRENCY).strip().upper()
    # Legacy single-currency request (no base named, no holding currency): every amount is
    # already in one currency and nothing is converted -- exactly the old behaviour.
    legacy = base_currency is None and all(h.currency is None for h in holdings)
    active_set.touch([holding.symbol for holding in holdings])

    state: dict[str, dict[str, float]] = {}
    contexts: dict[str, dict] = {}
    rates: dict[str, dict] = {}
    for holding in holdings:
        if legacy:
            currency = base
        else:
            # Stored portfolios carry no currency: take the tracked instrument's, else route
            # the symbol offline (CSE -> LKR, Binance -> quote asset, Yahoo -> suffix).
            tracked = active_set.get_instrument(holding.symbol)
            currency = (holding.currency or (tracked.currency if tracked else None)
                        or native_currency(holding.symbol)).strip().upper()
        rate = get_rate(currency, base)
        spread = conversion_spread(currency, base)
        if currency != base:
            rates[currency] = {"rate": rate.rate, "as_of": rate.as_of.isoformat(), "source": rate.source,
                               "spread": spread}
        price = holding.current_price * rate.rate
        metrics = active_set.metrics_for(holding.symbol) or {}
        snapshot = active_set.snapshot_for(holding.symbol)
        signals = holding_signals(snapshot, deadline_days)
        state[holding.symbol] = {
            "value": holding.quantity * price,
            "price": price,
            "quantity": holding.quantity,
            "adv_usd": holding.avg_daily_volume * rate.rate,
            "daily_volatility": float(metrics.get("daily_volatility", DEFAULT_DAILY_VOLATILITY)),
            "volatility_pct": float(metrics.get("volatility_pct", DEFAULT_VOLATILITY_PCT)),
            "cost_basis": (holding.cost_basis if holding.cost_basis is not None else holding.current_price) * rate.rate,
            "fx_cost_pct": spread,
            **signals,
        }
        if "drift_daily" in signals:
            state[holding.symbol]["expected_return"] = signals["drift_daily"]
        context = market_context(snapshot, signals, deadline_days)
        context["currency"] = currency
        contexts[holding.symbol] = context
    return state, contexts, {"base": base, "rates": rates}


def _to_portfolio_state(holdings: list[Holding], **kwargs: object) -> dict[str, dict[str, float]]:
    """The optimizer state alone (see _build_portfolio)."""
    return _build_portfolio(holdings, **kwargs)[0]  # type: ignore[arg-type]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness plus the active model version and which forecasters loaded."""
    from service.events import get_producer

    try:
        from forecasting.base import registered_forecasters

        # require_weights: report what can actually run now, not what is merely installed.
        available = registered_forecasters(require_weights=True)
    except Exception:  # noqa: BLE001
        available = []

    return HealthResponse(
        status="ok",
        model_version=_model_version(),
        kafka_connected=get_producer().connected,
        available_forecasters=available,
    )


@app.post("/forecast", response_model=ForecastResponse)
def forecast(
    request: ForecastRequest,
    caller: CallerIdentity | None = Depends(require_user),
) -> ForecastResponse:
    """Per-symbol quantile forecast from the active hybrid model (TimesFM + LSTM head).

    Strictly the model: when no trained head (or its base) can run for this horizon the
    answer is 503 with the reason, never the empirical band the analysis pipeline falls back
    to -- that one is labelled as such on /analysis/forecast/{symbol}.
    """
    from service.deps import forecaster_unavailable_reason, get_active_forecaster

    service = get_active_forecaster()
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="No trained forecaster is registered or runnable here. " + forecaster_unavailable_reason(),
        )

    from data import active_set
    from data.history import get_history
    from data.http import UpstreamError
    from data.instruments import resolve

    forecasts: list[SymbolForecast] = []
    versions: set[str] = set()
    active_set.touch(request.symbols)
    for symbol in request.symbols:
        try:
            instrument = active_set.get_instrument(symbol.upper()) or resolve(symbol)
            bars = get_history(instrument).bars
        except LookupError as exc:
            raise HTTPException(status_code=422, detail=f"{symbol}: {exc}") from None
        except UpstreamError as exc:
            raise HTTPException(status_code=503, detail=f"{symbol}: {exc}") from None
        result = service.forecast(instrument.symbol, bars, request.horizon)
        if result["method"] != "hybrid":
            raise HTTPException(status_code=503, detail=f"{symbol}: {result.get('reason') or 'model unavailable'}")
        versions.add(result["model_version"])
        forecasts.append(SymbolForecast(symbol=symbol, horizon=request.horizon,
                                        p10=result["p10"], p50=result["p50"], p90=result["p90"]))

    return ForecastResponse(
        forecasts=forecasts,
        model_version=versions.pop() if len(versions) == 1 else _model_version(),
        generated_at=_now(),
    )


# Below this a rebalance leg is rounding noise rather than a trade.
MIN_REBALANCE_NOTIONAL = 0.01


def _rebalance_orders(state: dict[str, dict[str, float]], weights: dict[str, float]) -> list[RebalanceOrder]:
    """Trades taking each holding from its current to its recommended weight.

    Values and prices in `state` are already in the base currency, so a weight is a fraction
    of one converted total; dividing the value change by the converted unit price gives the
    change in the holding's own units.
    """
    total = sum(h["value"] for h in state.values())
    orders = []
    for symbol, holding in sorted(state.items()):
        price = holding["price"]
        if price <= 0:
            continue
        change = weights.get(symbol, 0.0) * total - holding["value"]
        if abs(change) < MIN_REBALANCE_NOTIONAL:
            continue
        orders.append(RebalanceOrder(symbol=symbol, side="BUY" if change > 0 else "SELL",
                                     quantity=abs(change) / price, price=price))
    return orders


@app.post("/portfolio/optimize", response_model=PortfolioResponse)
def optimize_portfolio(
    request: PortfolioRequest,
    caller: CallerIdentity | None = Depends(require_user),
) -> PortfolioResponse:
    """Long-term allocation via MOEA/D. Publishes a decision event."""
    from agent.tools import run_moead_rebalance

    if not request.holdings:
        raise HTTPException(status_code=422, detail="holdings must not be empty")

    try:
        state, _contexts, currency = _build_portfolio(request.holdings, base_currency=request.base_currency)
    except UpstreamError as exc:
        raise HTTPException(status_code=503, detail=f"FX rate unavailable: {exc}") from None
    result = run_moead_rebalance(
        state,
        risk_preference=request.risk_preference,
        selection_rule=request.selection_rule,
        max_weight=request.max_weight,
    )
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    response = PortfolioResponse(
        recommended_weights=result["recommended_weights"],
        expected_return=result["expected_return"],
        expected_cvar=result["expected_cvar"],
        expected_liquidity_cost=result["expected_liquidity_cost"],
        pareto_front_size=result["pareto_front_size"],
        selection_rule=result["selection_rule"],
        selection_rationale=result["selection_rationale"],
        model_version=_model_version(),
        generated_at=_now(),
        rebalance_orders=_rebalance_orders(state, result["recommended_weights"]),
        currency=currency,
    )

    publish_decision("optimize", request.model_dump(mode="json"),
                     response.model_dump(mode="json"), response.model_version)
    return response


@app.post("/portfolio/withdraw", response_model=WithdrawalResponse)
def withdraw(
    request: WithdrawalRequest,
    caller: CallerIdentity | None = Depends(require_user),
) -> WithdrawalResponse:
    """Instant, loss-minimized withdrawal plan via the fuzzy GA. Publishes a decision event.

    With use_agent=True the Phase 5c agent chooses the optimizer inputs; the numbers still
    come from the GA either way.

    An infeasible plan is returned with HTTP 200 and `feasible=false`, not an error status.
    "You can raise 80k of the 300k you asked for, and here is the cheapest way to do it" is
    a successful, actionable answer -- and RQ4 depends on infeasible plans being observable
    rather than thrown away.
    """
    from agent.tools import run_fuzzy_ga_withdrawal

    if not request.holdings:
        raise HTTPException(status_code=422, detail="holdings must not be empty")

    try:
        portfolio_state, market_context, currency = _build_portfolio(
            request.holdings, deadline_days=request.deadline_days, base_currency=request.base_currency)
    except UpstreamError as exc:
        raise HTTPException(status_code=503, detail=f"FX rate unavailable: {exc}") from None
    total_value = sum(h["value"] for h in portfolio_state.values())
    if request.target_amount > total_value:
        raise HTTPException(
            status_code=422,
            detail=(
                f"target_amount {request.target_amount:,.2f} exceeds total portfolio value "
                f"{total_value:,.2f} {currency['base']}"
            ),
        )

    inputs = {"urgency": request.urgency, "risk_tolerance": request.risk_tolerance,
              "liquidity_target": request.target_amount, "deadline_days": request.deadline_days}

    # Laya Stage 2: per-holding action and portfolio urgency/risk, folded into the GA inputs
    # when confident. Without Laya (or below threshold) nothing changes; the record says why.
    from data import active_set
    from decision.engine import apply as apply_decisions

    snapshots = {symbol: active_set.snapshot_for(symbol) for symbol in portfolio_state}
    portfolio_state, inputs, decisions, decision_meta = apply_decisions(portfolio_state, snapshots, inputs)
    for symbol, record in decisions.items():
        market_context.setdefault(symbol, {})["decision"] = record

    result: dict | None = None
    agent_trace: list[dict] = []
    if request.use_agent:
        # The agent chooses the optimizer inputs AND its tool call is the plan returned, so
        # the trace shown is exactly the reasoning behind these numbers (previously the
        # trace came from a second, separate GA run with different inputs).
        result, agent_trace, agent_inputs = _run_agent(request, portfolio_state)
        if result is not None:
            inputs.update(agent_inputs)
    if result is None:
        result = run_fuzzy_ga_withdrawal(portfolio_state=portfolio_state, **inputs)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    response = WithdrawalResponse(
        assets_to_sell=[AssetSale(**sale) for sale in result["assets_to_sell"]],
        raised_amount=result["raised_amount"],
        target_amount=result["target_amount"],
        shortfall=result["shortfall"],
        expected_slippage_pct=result["expected_slippage_pct"],
        expected_realized_loss=result["expected_realized_loss"],
        residual_portfolio_weights=result["residual_portfolio_weights"],
        days_required=result["days_required"],
        feasible=result["feasible"],
        model_version=_model_version(),
        fuzzy_rule_trace=result["fuzzy_rule_trace"],
        agent_reasoning_trace=agent_trace,
        generated_at=_now(),
        market_context=market_context,
        currency=currency,
        cost_breakdown=result.get("cost_breakdown"),
        baseline_comparison=_baseline_comparison(portfolio_state, result, inputs),
        effective_inputs={**{k: v for k, v in inputs.items() if k != "liquidity_target"},
                          "decision": decision_meta},
    )

    publish_decision("withdraw", request.model_dump(mode="json"),
                     response.model_dump(mode="json"), response.model_version)
    return response


def _plan_summary(method: str, plan: dict) -> dict:
    return {
        "method": method,
        "feasible": plan["feasible"],
        "raised_amount": round(plan["raised_amount"], 2),
        "expected_realized_loss": round(plan["expected_realized_loss"], 2),
        "expected_slippage_pct": round(plan["expected_slippage_pct"], 6),
        "crystallized_loss": round((plan.get("cost_breakdown") or {}).get("crystallized_loss", 0.0), 2),
        "days_required": plan["days_required"],
    }


def _baseline_comparison(portfolio_state: dict, plan: dict, inputs: dict) -> list[dict]:
    """The GA plan beside the three naive baselines, costed by the same simulator with the
    same risk aversion -- so any difference is the schedule, not the accounting."""
    from agent.tools import plan_to_dict
    from optimization.ga_withdrawal import risk_aversion_for
    from optimization.naive_liquidation import BASELINES

    rows = [_plan_summary("fuzzy_ga", plan)]
    for name, baseline in BASELINES.items():
        try:
            costed = baseline(portfolio_state, inputs["liquidity_target"], deadline_days=inputs["deadline_days"],
                              timing_risk_aversion=risk_aversion_for(inputs["risk_tolerance"]))
        except Exception as exc:  # noqa: BLE001 - a comparison row must not sink the plan
            logger.warning("baseline %s failed: %s", name, exc)
            continue
        rows.append(_plan_summary(name, plan_to_dict(costed)))
    return rows


def _run_agent(request: WithdrawalRequest, portfolio_state: dict[str, dict[str, float]]
               ) -> tuple[dict | None, list[dict], dict]:
    """Run the agent; return (its grounded plan, trace, the inputs it chose) or (None, [], {})."""
    from agent.tool_schema import GROUNDING_TOOL_NAME

    transcript = _agent_transcript(request, portfolio_state)
    if transcript is None:
        return None, [], {}
    for step in transcript.steps:
        if step.tool_name == GROUNDING_TOOL_NAME and step.observation and "error" not in step.observation:
            arguments = step.tool_arguments or {}
            chosen = {k: arguments[k] for k in ("urgency", "risk_tolerance", "deadline_days") if k in arguments}
            return step.observation, _trace_rows(transcript), chosen
    return None, _trace_rows(transcript), {}


def _agent_transcript(request: WithdrawalRequest, portfolio_state: dict[str, dict[str, float]]):  # noqa: ANN202
    """The agent's episode for a withdrawal, or None when it fails or is not grounded.

    Uses the deterministic scripted policy, NOT the Ollama reference agent: the reference
    agent is scaffolding for validating the tool interface and must never sit in a request
    path. When the Colab-tuned model is ready it replaces the driver here.
    """
    from agent.trajectory_generation import ScenarioSpec, scripted_policy

    try:
        transcript = scripted_policy(
            ScenarioSpec(
                scenario_id="live", portfolio=portfolio_state,
                target_amount=request.target_amount, urgency=request.urgency,
                deadline_days=request.deadline_days,
                stress_scenario="live", market_regime="live",
            )
        )
    except Exception as exc:  # noqa: BLE001 - the agent is auxiliary; the direct GA still runs
        logger.warning("agent failed: %s", exc)
        return None

    if not transcript.grounded:
        # A non-grounded episode must never be used -- it would be exactly the fabricated
        # reasoning the constraint exists to prevent.
        logger.error("agent trace was not grounded: %s", transcript.grounding_error)
        return None
    return transcript


def _trace_rows(transcript) -> list[dict]:  # noqa: ANN001
    return [
        {
            "step": step.step_index,
            "thought": step.thought,
            "tool": step.tool_name,
            "tool_arguments": {k: v for k, v in (step.tool_arguments or {}).items()
                               if k != "portfolio_state"},
        }
        for step in transcript.steps
    ]
