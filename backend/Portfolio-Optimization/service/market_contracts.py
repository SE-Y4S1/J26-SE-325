"""Contracts for the market-data and active-set endpoints.

Kept OUT of service/contracts.py on purpose: that file is frozen because other components
and the fine-tuned agent bind to it. Everything here is new and additive.

Money fields come in pairs: `*_native` in the instrument's own currency, `*_base` in the
platform base currency (LKR), with the rate and its timestamp alongside so a reader can see
exactly how one became the other.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class InstrumentOut(BaseModel):
    symbol: str
    name: str
    exchange: str              # cse | binance | yahoo
    asset_class: str           # cse_equity | crypto | equity | etf | forex
    currency: str
    last_price: float | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[InstrumentOut]


class QuoteRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=32)
    # Overrides symbol-shape routing when shapes collide.
    exchange: str | None = Field(default=None, pattern="^(cse|binance|yahoo)$")
    quantity: float | None = Field(default=None, ge=0)


class QuoteResponse(BaseModel):
    instrument: InstrumentOut
    price: float
    as_of: str                       # date of the last bar
    currency: str
    base_currency: str
    fx_rate_to_base: float
    fx_as_of: datetime
    # Average daily traded value. `avg_daily_volume` uses the same name and meaning as
    # Holding.avg_daily_volume, so the Portfolio screen can copy it straight across.
    avg_daily_volume: float
    avg_daily_volume_base: float
    daily_volatility: float
    volatility_pct: float
    zero_volume_share: float
    sessions_per_week: float | None
    history_days: int
    # False when volatility figures are fallbacks because history is too short.
    estimated: bool
    # True when the exchange was unreachable and stored bars were served.
    stale: bool
    value_native: float | None = None
    value_base: float | None = None


class TrackRequest(BaseModel):
    symbols: list[str] = Field(min_length=1, max_length=200)


class SymbolStatus(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbol: str
    exchange: str
    name: str
    currency: str
    status: str                      # preparing | ready | partial | failed
    steps: dict[str, str] = Field(default_factory=dict)
    prepared_at: str | None = None
    last_seen: str
    error: str | None = None


class TrackResponse(BaseModel):
    tracked: list[SymbolStatus]
    # Symbols no exchange recognised, with the reason. Not an error for the whole request:
    # one typo must not stop the rest of a portfolio being prepared.
    rejected: dict[str, str] = Field(default_factory=dict)


class StatusResponse(BaseModel):
    symbols: list[SymbolStatus]


# --------------------------------------------------------------------------------------
# Phase 2: keywords, channels, sentiment
# --------------------------------------------------------------------------------------

class ChannelStatusOut(BaseModel):
    name: str
    kind: str                        # official | news | social
    available: bool
    reason: str | None = None        # why not, e.g. "JINA_API_KEY not set"
    requires_login: bool


class ChannelsResponse(BaseModel):
    channels: list[ChannelStatusOut]


class KeywordProfileOut(BaseModel):
    symbol: str
    exchange: str
    name: str
    sector: str | None = None
    company: list[str]
    people_brands: list[str]
    sector_terms: list[str]
    macro_terms: list[str]
    exclude: list[str]
    local_names: list[str]
    linkedin_url: str | None = None
    source: str                      # auto | reviewed
    generated_at: datetime | None = None
    provenance: list[str]


class KeywordProfileUpdate(BaseModel):
    """Only the fields being changed. Lists replace the stored list."""

    model_config = ConfigDict(extra="forbid")

    sector: str | None = None
    company: list[str] | None = None
    people_brands: list[str] | None = None
    sector_terms: list[str] | None = None
    macro_terms: list[str] | None = None
    exclude: list[str] | None = None
    local_names: list[str] | None = None
    linkedin_url: str | None = None


class TierSentiment(BaseModel):
    mean_sentiment: float | None = None
    momentum: float | None = None
    items: int
    scored: int
    attention_spike: float
    recent_items: int


class SentimentSource(BaseModel):
    channel: str
    kind: str
    tier: str
    title: str | None = None         # None for social posts: text is never stored
    outlet: str | None = None
    url: str | None = None
    published_at: str
    polarity: float | None = None
    language: str


class SentimentOut(BaseModel):
    model_config = ConfigDict(extra="ignore")

    as_of: str
    window_days: int
    overall_sentiment: float | None = None
    tiers: dict[str, TierSentiment]
    items_collected: int
    items_relevant: int
    spam_dropped: int
    languages: dict[str, int]
    relevance_method: str            # keyword | keyword+laya
    scoring_backend: str | None = None
    top_sources: list[SentimentSource]
    channels: dict[str, dict[str, str | int | None]]


class SentimentResponse(BaseModel):
    symbol: str
    status: str
    sentiment: SentimentOut | None = None
    # Why there is no sentiment yet, when there is none.
    detail: str | None = None


class RefreshRequest(BaseModel):
    symbols: list[str] | None = Field(default=None, max_length=200)


class FundamentalsReport(BaseModel):
    title: str | None = None
    url: str | None = None
    kind: str | None = None
    period_end: str | None = None
    available_from: str
    available_from_estimated: bool = False
    age_days: int


class OutlookTone(BaseModel):
    tone: str                        # positive | neutral | negative
    confidence: float
    method: str                      # laya | ollama | lexicon


class FundamentalsOut(BaseModel):
    """Statements (CSE PDF / yfinance) or tokenomics (crypto) for one held symbol."""

    model_config = ConfigDict(extra="ignore")

    source: str | None = None        # cse_pdf | yfinance | coingecko
    kind: str                        # statements | tokenomics
    usable: bool
    report: FundamentalsReport | None = None
    template: str | None = None      # general | bank | finance | insurance
    confidence: float | None = None
    checks: dict[str, bool] | None = None
    summary: dict[str, float] | None = None
    ratios: dict[str, float | int | str | bool | dict[str, bool] | None] | None = None
    sector_relative: dict[str, float | None] | None = None
    flags: dict[str, bool | float | None] | None = None
    outlook: OutlookTone | None = None
    reports_stored: int | None = None
    tokenomics: dict[str, float | int | str | None] | None = None


class FundamentalsResponse(BaseModel):
    symbol: str
    status: str
    fundamentals: FundamentalsOut | None = None
    detail: str | None = None


class TechnicalOut(BaseModel):
    """Technical snapshot for one held symbol (features.technical.technical_snapshot)."""

    model_config = ConfigDict(extra="ignore")

    sessions: int
    as_of: str | None = None
    close: float | None = None
    trend: str | None = None                 # bullish | neutral | bearish
    sma_20: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    price_vs_sma_50: float | None = None
    price_vs_sma_200: float | None = None
    ma_cross_recent: str | None = None       # golden | death
    rsi: float | None = None
    macd_hist: float | None = None
    mfi: float | None = None
    atr_pct: float | None = None
    bollinger_pct_b: float | None = None
    adx: float | None = None
    obv_trend: float | None = None
    return_20d: float | None = None
    return_63d: float | None = None
    drawdown_52w: float | None = None
    support: float | None = None
    resistance: float | None = None
    dist_to_support: float | None = None
    dist_to_resistance: float | None = None
    benchmark_return_63d: float | None = None
    relative_strength_63d: float | None = None


class TechnicalResponse(BaseModel):
    symbol: str
    status: str
    technical: TechnicalOut | None = None
    detail: str | None = None


class ForecastBand(BaseModel):
    """One horizon's band. method: hybrid (TimesFM + LSTM head), empirical (the symbol's own
    historical h-day returns; used when no head/base can run), or unavailable."""

    model_config = ConfigDict(protected_namespaces=())

    horizon: int
    method: str
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None
    model_version: str | None = None
    base: list[float] | None = None
    residual: list[float] | None = None
    as_of: str | None = None
    reason: str | None = None


class AnalysisForecastResponse(BaseModel):
    symbol: str
    status: str
    bands: list[ForecastBand] = Field(default_factory=list)
    detail: str | None = None
