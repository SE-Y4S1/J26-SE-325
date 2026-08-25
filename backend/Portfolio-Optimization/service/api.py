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
    SymbolForecast,
    WithdrawalRequest,
    WithdrawalResponse,
)
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


def _to_portfolio_state(holdings: list[Holding]) -> dict[str, dict[str, float]]:
    """Contract models -> the plain dict shape the optimizers consume."""
    state: dict[str, dict[str, float]] = {}
    for holding in holdings:
        value = holding.quantity * holding.current_price
        state[holding.symbol] = {
            "value": value,
            "price": holding.current_price,
            "quantity": holding.quantity,
            "adv_usd": holding.avg_daily_volume,
            "daily_volatility": 0.0126,
            "volatility_pct": 0.5,
            "cost_basis": holding.cost_basis if holding.cost_basis is not None else holding.current_price,
        }
    return state


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
    """Per-symbol quantile forecast from the active hybrid model."""
    from service.deps import get_active_forecaster

    forecaster = get_active_forecaster()
    if forecaster is None:
        raise HTTPException(
            status_code=503,
            detail="No trained forecaster is registered. Run Phase 3/4 training first.",
        )

    from agent.tools import get_forecast as forecast_tool

    forecasts: list[SymbolForecast] = []
    for symbol in request.symbols:
        result = forecast_tool(symbol, request.horizon)
        if "error" in result:
            raise HTTPException(status_code=422, detail=f"{symbol}: {result['error']}")
        forecasts.append(
            SymbolForecast(
                symbol=symbol, horizon=request.horizon,
                p10=result["p10"], p50=result["p50"], p90=result["p90"],
            )
        )

    return ForecastResponse(
        forecasts=forecasts, model_version=_model_version(), generated_at=_now()
    )


@app.post("/portfolio/optimize", response_model=PortfolioResponse)
def optimize_portfolio(
    request: PortfolioRequest,
    caller: CallerIdentity | None = Depends(require_user),
) -> PortfolioResponse:
    """Long-term allocation via MOEA/D. Publishes a decision event."""
    from agent.tools import run_moead_rebalance

    if not request.holdings:
        raise HTTPException(status_code=422, detail="holdings must not be empty")

    result = run_moead_rebalance(
        _to_portfolio_state(request.holdings),
        risk_preference=request.risk_preference,
        selection_rule=request.selection_rule,
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

    portfolio_state = _to_portfolio_state(request.holdings)
    total_value = sum(h["value"] for h in portfolio_state.values())
    if request.target_amount > total_value:
        raise HTTPException(
            status_code=422,
            detail=(
                f"target_amount {request.target_amount:,.2f} exceeds total portfolio value "
                f"{total_value:,.2f}"
            ),
        )

    result = run_fuzzy_ga_withdrawal(
        urgency=request.urgency,
        risk_tolerance=request.risk_tolerance,
        liquidity_target=request.target_amount,
        portfolio_state=portfolio_state,
        deadline_days=request.deadline_days,
    )
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    agent_trace: list[dict] = []
    if request.use_agent:
        agent_trace = _run_agent_trace(request, portfolio_state, result)

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
    )

    publish_decision("withdraw", request.model_dump(mode="json"),
                     response.model_dump(mode="json"), response.model_version)
    return response


def _run_agent_trace(
    request: WithdrawalRequest,
    portfolio_state: dict[str, dict[str, float]],
    plan: dict,
) -> list[dict]:
    """Produce the agent's internal reasoning trace for a withdrawal.

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
    except Exception as exc:  # noqa: BLE001 - the trace is auxiliary, the plan is not
        logger.warning("agent trace failed: %s", exc)
        return []

    if not transcript.grounded:
        # A non-grounded trace must never be attached -- it would be exactly the fabricated
        # reasoning the constraint exists to prevent.
        logger.error("agent trace was not grounded: %s", transcript.grounding_error)
        return []

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
