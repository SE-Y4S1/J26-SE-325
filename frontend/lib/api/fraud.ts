/**
 * Component 2 — Real-Time Fraud & Behavioral Anomaly Engine (Dhushanthini R).
 *
 * Types are hand-written here rather than generated, because `npm run gen:api` needs the
 * service running and this component is not always up. They mirror the shapes declared in
 * `backend/Fraud-Detection/schemas.py`; if that file changes, this one must too.
 */

import { request } from "./client";

/** Mirrors schemas.py::Transaction. Only the fields the UI actually sends. */
export interface TransactionInput {
  user_id: string;
  amount: number;
  location: string;
  device_id: string;
  device_change: boolean;
  typing_speed: number;
  navigation_pattern: string;
  transaction_frequency: number;
  beneficiary_change: boolean;
  previous_transaction_amount: number;
  account_age: number;
}

export interface Signal {
  name: string;
  value: number;
  weight: number;
  message: string;
}

/** Mirrors schemas.py::ScoreResponse. */
export interface ScoreResponse {
  transaction_id: string;
  user_id: string;
  timestamp: number;
  behavioral_score: number;
  graph_score: number;
  risk_score: number;
  decision: "ALLOW" | "STEP-UP" | "BLOCK";
  reason: string;
  signals?: Signal[];
}

export function scoreTransaction(tx: TransactionInput): Promise<ScoreResponse> {
  // auth:false -- this service does not verify the platform JWT, so sending one would be
  // noise. Component 1 is the only backend that enforces it today.
  return request<ScoreResponse>("fraud", "/score", { method: "POST", body: tx, auth: false });
}

export function auditTrail(): Promise<unknown> {
  return request("fraud", "/audit", { auth: false });
}

export function fraudHealth(): Promise<{ status: string }> {
  return request<{ status: string }>("fraud", "/health", { auth: false });
}


// --- adversarial robustness -------------------------------------------------------------
// /simulate-attack is the evidence behind Component 2's second research gap: that a
// dual-stream model survives evasion a single-stream one does not. Contract read from
// schemas.py::AttackResponse and confirmed against the running service.

export type AttackType = "camouflage" | "slow_drift" | "structuring";

export interface AttackStep {
  step: number;
  label: string;
  amount: number;
  behavioral_score: number;
  graph_score: number;
  risk_score: number;
  decision: string;
  reason: string;
}

export interface AttackResponse {
  attack_type: string;
  title: string;
  description: string;
  baseline: AttackStep;
  steps: AttackStep[];
  summary: string;
  detected: boolean;
  /** True when a behaviour-only model would have let the attack through -- the ablation. */
  single_stream_would_have_missed: boolean;
  disclaimer: string;
}

export function simulateAttack(attack_type: AttackType, steps = 8): Promise<AttackResponse> {
  return request<AttackResponse>("fraud", "/simulate-attack", {
    method: "POST",
    body: { attack_type, steps },
    auth: false,
  });
}
