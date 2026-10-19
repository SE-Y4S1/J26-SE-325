/**
 * Convenience aliases over the generated OpenAPI types.
 *
 * `generated/*.ts` is produced by `npm run gen:api` and must never be hand-edited — it is
 * the single source of truth for the wire format, derived from each service's live
 * /openapi.json. Aliasing here keeps component code readable without anyone being tempted
 * to redeclare an interface that would then drift from the backend silently.
 *
 * `service/contracts.py` in Component 1 is a frozen contract that Components 3 and 4 also
 * bind to, so a change there should show up as a diff in the generated file.
 */

import type { components as PlatformComponents } from "./generated/platform";
import type { components as Component1Components } from "./generated/component1";

type PlatformSchemas = PlatformComponents["schemas"];
type Component1Schemas = Component1Components["schemas"];

// --- Platform: identity ---------------------------------------------------------------
export type User = PlatformSchemas["UserResponse"];
export type TokenResponse = PlatformSchemas["TokenResponse"];
export type RegisterRequest = PlatformSchemas["RegisterRequest"];
export type LoginRequest = PlatformSchemas["LoginRequest"];

// --- Platform: portfolios -------------------------------------------------------------
export type Holding = PlatformSchemas["HoldingModel"];
export type Portfolio = PlatformSchemas["PortfolioResponse"];
export type PortfolioSummary = PlatformSchemas["PortfolioSummary"];
export type PortfolioCreate = PlatformSchemas["PortfolioCreate"];
export type PortfolioUpdate = PlatformSchemas["PortfolioUpdate"];

// --- Component 1 ----------------------------------------------------------------------
export type WithdrawalRequest = Component1Schemas["WithdrawalRequest"];
export type WithdrawalResponse = Component1Schemas["WithdrawalResponse"];
export type AssetSale = Component1Schemas["AssetSale"];
export type OptimizeRequest = Component1Schemas["PortfolioRequest"];
export type OptimizeResponse = Component1Schemas["PortfolioResponse"];
export type ForecastRequest = Component1Schemas["ForecastRequest"];
export type ForecastResponse = Component1Schemas["ForecastResponse"];
export type SymbolForecast = Component1Schemas["SymbolForecast"];
export type Component1Health = Component1Schemas["HealthResponse"];

// --- Component 1: market data and the active set -----------------------------------------
export type Instrument = Component1Schemas["InstrumentOut"];
export type MarketSearchResponse = Component1Schemas["SearchResponse"];
export type QuoteRequest = Component1Schemas["QuoteRequest"];
export type QuoteResponse = Component1Schemas["QuoteResponse"];
export type SymbolStatus = Component1Schemas["SymbolStatus"];
export type TrackResponse = Component1Schemas["TrackResponse"];
export type AnalysisStatusResponse = Component1Schemas["StatusResponse"];

// --- Component 1: per-stock news and social sentiment ----------------------------------
export type ChannelStatus = Component1Schemas["ChannelStatusOut"];
export type KeywordProfile = Component1Schemas["KeywordProfileOut"];
export type KeywordProfileUpdate = Component1Schemas["KeywordProfileUpdate"];
export type TierSentiment = Component1Schemas["TierSentiment"];
export type SentimentSource = Component1Schemas["SentimentSource"];
export type SentimentResponse = Component1Schemas["SentimentResponse"];

// --- Component 1: fundamentals, technicals, forecast bands ---------------------------------
export type FundamentalsOut = Component1Schemas["FundamentalsOut"];
export type FundamentalsResponse = Component1Schemas["FundamentalsResponse"];
export type TechnicalOut = Component1Schemas["TechnicalOut"];
export type TechnicalResponse = Component1Schemas["TechnicalResponse"];
export type ForecastBand = Component1Schemas["ForecastBand"];
export type AnalysisForecastResponse = Component1Schemas["AnalysisForecastResponse"];

/**
 * One entry of a withdrawal's `fuzzy_rule_trace`.
 *
 * Typed by hand because the backend declares this field as `list[dict]` — Pydantic emits an
 * untyped object, so OpenAPI cannot describe the shape. The structure is produced by
 * `optimization/fuzzy_withdrawal.py::rule_trace_to_dict`; if that changes, change this too.
 * The trace is the auditability evidence the TAF's Legal Impact section calls for, so it is
 * rendered rather than hidden.
 */
export interface FuzzyRule {
  rule_id: string;
  if: string;
  then: string;
  strength: number;
}

export interface FuzzyRuleTraceEntry {
  symbol: string;
  sell_priority: number;
  rules: FuzzyRule[];
  /** Market-signal rule base (news x forecast outlook x attention): priority points. */
  signal_adjustment?: number;
  signal_rules?: FuzzyRule[];
  /** Laya Stage 2 action/downside answer, when confident: priority points. */
  laya_adjustment?: number;
}

/**
 * One holding's entry in `WithdrawalResponse.market_context` (declared `dict` on the
 * backend, so typed by hand). Built by `optimization/signals.py::market_context` plus the
 * `currency` and `decision` keys `service/api.py` adds.
 */
export interface MarketContext {
  status: string;
  prepared_at?: string | null;
  data_age_hours?: number | null;
  currency?: string;
  signals_used: Record<string, number>;
  forecast?: {
    horizon: number;
    method: string;
    p10: number;
    p50: number;
    p90: number;
    model_version?: string | null;
    reason?: string | null;
  };
  sentiment?: {
    overall: number | null;
    company: number | null;
    sector: number | null;
    macro: number | null;
    attention_spike: number | null;
    items: number | null;
    top_sources: Array<{ channel: string; title: string | null; url: string | null; polarity: number | null }>;
  };
  technical?: Record<string, number | string | null>;
  fundamentals?: Record<string, unknown> & { kind: string; red_flags?: string[] };
  decision?: {
    source: string;
    reason?: string;
    model_version?: string;
    action?: { choice: string; probabilities?: Record<string, number>; confidence: number } | null;
    downside_risk?: { noul: number; confidence: number } | null;
    applied?: string[];
  };
}

/** `WithdrawalResponse.baseline_comparison` rows: the GA beside the naive baselines. */
export interface BaselineRow {
  method: string;
  feasible: boolean;
  raised_amount: number;
  expected_realized_loss: number;
  expected_slippage_pct: number;
  crystallized_loss: number;
  days_required: number;
}

export interface CostBreakdown {
  slippage: number;
  timing_risk: number;
  forecast_drift: number;
  fx_conversion: number;
  realized_pnl: number;
  crystallized_loss: number;
}

export interface CurrencyBlock {
  base: string;
  rates: Record<string, { rate: number; as_of: string; source: string; spread: number }>;
}

/** One step of the agent's internal reasoning, when `use_agent` was set. */
export interface AgentTraceStep {
  step: number;
  thought: string | null;
  tool: string | null;
  tool_arguments: Record<string, unknown> | null;
}
