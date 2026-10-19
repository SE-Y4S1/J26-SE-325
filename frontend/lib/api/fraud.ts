/**
 * Component 2 — Real-time fraud and behavioural anomaly engine (Dhushanthini R).
 *
 * Types are hand-written here rather than generated, because `npm run gen:api` needs the
 * service running and this component is not always up. They mirror
 * `backend/Fraud-Detection/src/fraud_engine/api/schemas.py` and `service.py`; if those
 * change, this file must too.
 */

import { request } from "./client";

/** Mirrors schemas.py::TransactionRequest. The service rejects names, emails, card
 *  numbers and raw network addresses, so none of those exist here. */
export interface TransactionInput {
  transaction_id: string;
  amount: number;
  currency?: string;
  product_code?: string;
  account_key?: string;
  device_key?: string;
  /** 0 (worst) to 1 (trusted). Omitted means unknown, which never blocks on its own. */
  ip_reputation?: number;
}

export interface TopFactor {
  name: string;
  contribution: number;
}

/** Mirrors FraudService.score()'s return value. */
export interface ScoreResponse {
  transaction_id: string;
  fraud_score: number;
  decision: "ALLOW" | "STEP_UP" | "BLOCK";
  model_version: string;
  /** False for the untrained heuristic: that score is not a research result. */
  benchmark_model: boolean;
  explanation: {
    rules: string[];
    streams: Record<string, number>;
    top_factors: TopFactor[];
    notes: string[];
  };
}

export interface FraudHealth {
  status: string;
  model_version: string;
  benchmark_model: boolean;
  kafka_enabled: boolean;
}

export function scoreTransaction(tx: TransactionInput): Promise<ScoreResponse> {
  // auth:false -- this service does not verify the platform JWT, so sending one would be
  // noise.
  return request<ScoreResponse>("fraud", "/v1/transactions/score", {
    method: "POST",
    body: tx,
    auth: false,
  });
}

/** The decision already returned for a transaction (read-only). */
export function getDecision(transactionId: string): Promise<ScoreResponse> {
  return request<ScoreResponse>("fraud", `/v1/transactions/${encodeURIComponent(transactionId)}`, {
    auth: false,
  });
}

export function decisionSummary(): Promise<Record<"ALLOW" | "STEP_UP" | "BLOCK", number>> {
  return request("fraud", "/v1/decisions/summary", { auth: false });
}

export function fraudHealth(): Promise<FraudHealth> {
  return request<FraudHealth>("fraud", "/health", { auth: false });
}
