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

/**
 * One entry of a withdrawal's `fuzzy_rule_trace`.
 *
 * Typed by hand because the backend declares this field as `list[dict]` — Pydantic emits an
 * untyped object, so OpenAPI cannot describe the shape. The structure is produced by
 * `optimization/fuzzy_withdrawal.py::rule_trace_to_dict`; if that changes, change this too.
 * The trace is the auditability evidence the TAF's Legal Impact section calls for, so it is
 * rendered rather than hidden.
 */
export interface FuzzyRuleTraceEntry {
  symbol: string;
  sell_priority: number;
  rules: Array<{
    rule_id: string;
    if: string;
    then: string;
    strength: number;
  }>;
}

/** One step of the agent's internal reasoning, when `use_agent` was set. */
export interface AgentTraceStep {
  step: number;
  thought: string | null;
  tool: string | null;
  tool_arguments: Record<string, unknown> | null;
}
