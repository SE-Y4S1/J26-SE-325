/**
 * Component 3 — Privacy-Preserving AI-to-Smart-Contract Audit Bridge (Abeysekara W C S M).
 *
 * An Express service, not FastAPI, so there is no OpenAPI document to generate from. Routes
 * mirror `backend/Blockchain-Auditability/backend/src/routes/auditRoutes.js`.
 */

import { request } from "./client";

export interface AuditRecord {
  transactionId?: string;
  decision?: string;
  hash?: string;
  timestamp?: string;
  [key: string]: unknown;
}

/** The service wraps its payload: {success, count, records}. Verified against the running
 *  Express service, not assumed -- a client that expected a bare array rendered an empty
 *  table however many records existed. */
interface RecordsEnvelope {
  success: boolean;
  count: number;
  records: AuditRecord[];
}

export async function auditRecords(): Promise<AuditRecord[]> {
  const payload = await request<RecordsEnvelope | AuditRecord[]>(
    "audit",
    "/api/audit/records",
    { auth: false },
  );
  // Tolerates both shapes, so a future change back to a bare array does not break the page.
  if (Array.isArray(payload)) return payload;
  return Array.isArray(payload?.records) ? payload.records : [];
}

export function auditStats(): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>("audit", "/api/audit/stats", { auth: false });
}

export function verifyAudit(transactionId: string): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>(
    "audit",
    `/api/audit/${encodeURIComponent(transactionId)}/verify`,
    { auth: false },
  );
}

export function auditHealth(): Promise<{ status: string }> {
  return request<{ status: string }>("audit", "/api/health", { auth: false });
}


// --- anchoring, verification and tamper-evidence -----------------------------------------
// The point of an audit bridge: anchor a decision, verify it matches the chain, then show
// that altering the record breaks that match. All three contracts confirmed against the
// running Express service, including the exact field names its validator demands.

export interface EvaluateInput {
  transactionId: string;
  riskScore: number;
  /** 0-100, not 0-1. The service rejects anything outside that range. */
  confidence: number;
  amount: number;
  transactionType: string;
  modelVersion: string;
}

export interface EvaluateResult {
  success: boolean;
  status: string;
  policyResult?: { action: string; reason: string; policyId: string; policyVersion: string };
  blockchain?: {
    success: boolean;
    blockchainTxId: string;
    blockNumber: number;
    isMock: boolean;
    network: string;
  };
  record?: Record<string, unknown>;
}

export interface VerifyResult {
  verified: boolean;
  status: string;
  message: string;
  details?: {
    storedOffChainHash?: string;
    calculatedCurrentHash?: string;
    blockchainOnChainHash?: string;
  };
}

export function evaluateDecision(input: EvaluateInput): Promise<EvaluateResult> {
  return request<EvaluateResult>("audit", "/api/audit/evaluate", {
    method: "POST",
    body: input,
    auth: false,
  });
}

export function verifyRecord(transactionId: string): Promise<VerifyResult> {
  return request<VerifyResult>(
    "audit",
    `/api/audit/${encodeURIComponent(transactionId)}/verify`,
    { auth: false },
  );
}

export function simulateTampering(
  transactionId: string,
  field: string,
  tamperedValue: unknown,
): Promise<{ success: boolean; message: string }> {
  return request<{ success: boolean; message: string }>(
    "audit",
    `/api/audit/${encodeURIComponent(transactionId)}/tamper`,
    { method: "POST", body: { field, tamperedValue }, auth: false },
  );
}
