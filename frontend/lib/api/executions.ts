/**
 * Executions: a Component 1 plan carried through Component 2 (fraud gateway) and
 * Component 3 (audit anchor) by the platform service, which alone changes holdings.
 *
 * Hand-written to mirror backend/Platform/service/contracts.py (Execution*). The flow and
 * its failure policy are documented in backend/Platform/service/pipeline.py.
 */

import { request } from "./client";

export type ExecutionKind = "withdrawal" | "rebalance";
export type ExecutionStatus = "EXECUTED" | "BLOCKED" | "PENDING_REVIEW" | "HELD" | "REJECTED";

export interface ExecutionOrder {
  symbol: string;
  side: "BUY" | "SELL";
  quantity: number;
  price: number;
}

/** Component 2's /v1/transactions/score response, kept whole in the trace. */
export interface FraudStage {
  transaction_id?: string;
  fraud_score?: number;
  decision?: "ALLOW" | "STEP_UP" | "BLOCK";
  model_version?: string;
  benchmark_model?: boolean;
  explanation?: {
    rules: string[];
    streams: Record<string, number>;
    top_factors: { name: string; contribution: number }[];
    notes: string[];
  };
  // Present instead when Component 2 gave no usable answer.
  reachable?: boolean;
  status_code?: number | null;
  error?: string | null;
}

/** What the platform keeps from Component 3's /api/audit/evaluate (and review) responses. */
export interface AuditStage {
  reachable: boolean;
  status_code: number | null;
  status: string;
  message?: string | null;
  policy?: { action?: string; reason?: string; policyId?: string; policyVersion?: string } | null;
  record?: {
    finalDecision?: string;
    recordHash?: string;
    auditId?: string;
    verificationStatus?: string;
    blockchainTxId?: string | null;
    blockNumber?: number | null;
  } | null;
  blockchain?: { isMock?: boolean; blockchainTxId?: string; blockNumber?: number } | null;
  review?: AuditStage;
}

export interface Execution {
  id: string;
  portfolio_id: number | null;
  kind: ExecutionKind;
  status: ExecutionStatus;
  status_reason: string;
  amount: number;
  currency: string;
  source_model_version: string;
  plan_hash: string;
  plan: Record<string, unknown>;
  orders: ExecutionOrder[];
  fraud: FraudStage | null;
  audit: AuditStage | null;
  created_at: string;
  updated_at: string;
}

export function createExecution(
  portfolioId: number,
  kind: ExecutionKind,
  plan: unknown,
): Promise<Execution> {
  return request<Execution>("platform", `/portfolios/${portfolioId}/executions`, {
    method: "POST",
    body: { kind, plan },
  });
}

export function getExecution(id: string): Promise<Execution> {
  return request<Execution>("platform", `/executions/${encodeURIComponent(id)}`);
}

/** Step-up: the account holder re-enters their password to approve or reject. */
export function confirmExecution(id: string, password: string, approve: boolean): Promise<Execution> {
  return request<Execution>("platform", `/executions/${encodeURIComponent(id)}/confirm`, {
    method: "POST",
    body: { password, approve },
  });
}

export function retryExecution(id: string): Promise<Execution> {
  return request<Execution>("platform", `/executions/${encodeURIComponent(id)}/retry`, {
    method: "POST",
  });
}
