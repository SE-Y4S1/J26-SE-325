/**
 * Typed client for Component 1: forecasting and portfolio optimization.
 *
 * `Holding` is stored by the platform service in exactly the shape this service expects, so
 * a saved portfolio's holdings pass straight through with no mapping. That is deliberate —
 * a translation layer between two contracts that are meant to be identical is where field
 * drift hides.
 */

import { request } from "./client";
import type {
  Component1Health,
  ForecastResponse,
  Holding,
  OptimizeResponse,
  WithdrawalResponse,
} from "./types";

export type SelectionRule = "knee" | "max_sharpe" | "scalarized";

export interface WithdrawParams {
  holdings: Holding[];
  targetAmount: number;
  /** 0 = no rush, optimize for cost. 1 = immediate. Drives the fuzzy layer. */
  urgency?: number;
  riskTolerance?: number;
  deadlineDays?: number;
  /** Route through the Phase 5c agent, which adds an `agent_reasoning_trace`. */
  useAgent?: boolean;
}

/**
 * Plan an instant, loss-minimized withdrawal — the component's headline capability.
 *
 * An infeasible request returns HTTP 200 with `feasible: false` and a `shortfall`, NOT an
 * error. "You can raise 80k of the 300k you asked for, and here is the cheapest way" is a
 * successful answer, and RQ4 depends on infeasible plans being observable.
 */
export function planWithdrawal(params: WithdrawParams): Promise<WithdrawalResponse> {
  return request<WithdrawalResponse>("portfolio", "/portfolio/withdraw", {
    method: "POST",
    body: {
      holdings: params.holdings,
      target_amount: params.targetAmount,
      urgency: params.urgency ?? 0.5,
      risk_tolerance: params.riskTolerance ?? 0.5,
      deadline_days: params.deadlineDays ?? 1,
      use_agent: params.useAgent ?? false,
    },
  });
}

export interface OptimizeParams {
  holdings: Holding[];
  riskPreference?: number;
  selectionRule?: SelectionRule;
  maxWeight?: number;
  allowShorting?: boolean;
}

/**
 * Long-term allocation via MOEA/D.
 *
 * `selectionRule` picks which point on the Pareto front is returned. Exposing all three in
 * the UI surfaces RQ2's sensitivity analysis directly rather than burying it in a notebook.
 */
export function optimizeAllocation(params: OptimizeParams): Promise<OptimizeResponse> {
  return request<OptimizeResponse>("portfolio", "/portfolio/optimize", {
    method: "POST",
    body: {
      holdings: params.holdings,
      risk_preference: params.riskPreference ?? 0.5,
      selection_rule: params.selectionRule ?? "knee",
      max_weight: params.maxWeight ?? 0.25,
      allow_shorting: params.allowShorting ?? false,
    },
  });
}

/**
 * Quantile forecasts from the hybrid model.
 *
 * Throws ApiError with status 503 until a model is registered. That is the expected state
 * until Colab fine-tuning has run, so callers should render it as an explanatory empty
 * state rather than a failure.
 */
export function getForecast(symbols: string[], horizon = 5): Promise<ForecastResponse> {
  return request<ForecastResponse>("portfolio", "/forecast", {
    method: "POST",
    body: { symbols, horizon },
  });
}

export function health(): Promise<Component1Health> {
  return request<Component1Health>("portfolio", "/health", { auth: false });
}
