"""The JSON contract. FROZEN once Phase 6 lands; later changes are ADDITIVE and optional only.

A separately fine-tuned Llama (Colab), Component 3 (blockchain provenance) and Component 4
(explanation layer) all bind to these field names. Renaming one after the fine-tuning run
means retraining, so the names are descriptive and stable by design -- `assets_to_sell`, not
`ats`; `expected_slippage_pct`, not `slip`.

Three fields exist purely to serve other components and must be present on every response:
  model_version        -> Component 3 anchors model provenance on-chain against this
  fuzzy_rule_trace     -> Component 4 turns fired rules into a user-facing explanation
  agent_reasoning_trace-> Component 4 shows the decision path in the Trust Panel

Balances are SIMULATED platform balances, per the TAF: "user funds throughout the platform
are modeled as simulated platform balances rather than real brokerage holdings".

Response models set `protected_namespaces=()` because `model_version` collides with
pydantic v2's reserved `model_` prefix. The field name is fixed by the cross-component
contract, so we disable the protection rather than rename it.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Holding(BaseModel):
    """One position in the simulated portfolio."""

    symbol: str
    quantity: float = Field(ge=0)
    current_price: float = Field(gt=0)
    avg_daily_volume: float = Field(ge=0)
    cost_basis: float | None = None
    # Currency of current_price, avg_daily_volume and cost_basis (LKR for CSE, USD for Yahoo,
    # USDT for Binance). None = already in the request's base currency (the old behaviour).
    currency: str | None = None


class ForecastRequest(BaseModel):
    symbols: list[str]
    horizon: int = Field(default=5, ge=1, le=21)


class SymbolForecast(BaseModel):
    symbol: str
    horizon: int
    p10: float
    p50: float
    p90: float


class ForecastResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    forecasts: list[SymbolForecast]
    model_version: str
    generated_at: datetime


class PortfolioRequest(BaseModel):
    holdings: list[Holding]
    risk_preference: float = Field(default=0.5, ge=0.0, le=1.0)
    selection_rule: str = Field(default="knee")
    max_weight: float = Field(default=0.25, gt=0.0, le=1.0)
    allow_shorting: bool = False
    base_currency: str | None = None    # None = the service's BASE_CURRENCY


class PortfolioResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    recommended_weights: dict[str, float]
    expected_return: float
    expected_cvar: float
    expected_liquidity_cost: float
    pareto_front_size: int
    selection_rule: str
    selection_rationale: str
    model_version: str
    generated_at: datetime


class WithdrawalRequest(BaseModel):
    holdings: list[Holding]
    target_amount: float = Field(gt=0)
    urgency: float = Field(default=0.5, ge=0.0, le=1.0)
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0)
    deadline_days: int = Field(default=1, ge=1, le=30)
    use_agent: bool = False     # route through the Phase 5c agent instead of calling the GA directly
    base_currency: str | None = None    # target_amount's currency; None = the service's BASE_CURRENCY


class AssetSale(BaseModel):
    """One line of the liquidation plan. Every number here originates in the fuzzy GA."""

    symbol: str
    sell_fraction: float = Field(ge=0.0, le=1.0)
    quantity: float = Field(ge=0)
    expected_price: float
    expected_slippage_pct: float
    execution_day: int


class WithdrawalResponse(BaseModel):
    """The instant-withdrawal plan -- this component's headline output."""

    model_config = ConfigDict(protected_namespaces=())

    assets_to_sell: list[AssetSale]
    raised_amount: float
    target_amount: float
    shortfall: float
    expected_slippage_pct: float
    expected_realized_loss: float
    residual_portfolio_weights: dict[str, float]
    days_required: int
    feasible: bool

    # Cross-component fields -- required on every response.
    model_version: str
    fuzzy_rule_trace: list[dict] = Field(default_factory=list)
    agent_reasoning_trace: list[dict] = Field(default_factory=list)

    generated_at: datetime

    # Additive (market-data phases). Absent fields keep older clients working unchanged.
    # Per holding: forecast band, sentiment tiers, technical/fundamental summary, data age,
    # and the numeric signals the planner actually used.
    market_context: dict[str, dict] = Field(default_factory=dict)
    # {"base": "LKR", "rates": {"USD": {"rate": 300.1, "as_of": ..., "spread": 0.0075}}}
    currency: dict | None = None
    # slippage / timing / drift / fx (sum = expected_realized_loss) and realized P&L.
    cost_breakdown: dict | None = None
    # The same request costed by the naive baselines through the same simulator (RQ3).
    baseline_comparison: list[dict] = Field(default_factory=list)
    # The inputs the plan was produced with (the agent may adjust urgency/risk tolerance).
    effective_inputs: dict | None = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_version: str | None
    kafka_connected: bool
    available_forecasters: list[str]
