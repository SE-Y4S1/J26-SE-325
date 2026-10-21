/**
 * Component 3 — AI Decision Assurance Engine and evidence ledger (Abeysekara W C S M).
 *
 * An Express service, not FastAPI, so there is no OpenAPI document to generate from. Routes
 * mirror `backend/Blockchain-Auditability/src/routes/auditRoutes.js`.
 *
 * Requests carry the session token: with AUTH_REQUIRED on, the service lets reviewers (its
 * AUDIT_REVIEWER_IDS) read records and decide held decisions; with it off, it ignores tokens.
 */

import { request } from "./client";

// --- shapes --------------------------------------------------------------------------------

export type Route = "AUTO" | "VERIFY" | "HUMAN_REVIEW";
export type SignalQuality = "FULL" | "DEGRADED" | "LEGACY";

/** The assurance engine's result, as committed in the record. */
export interface Assurance {
  engine: string;
  policyId: string;
  policyVersion: string;
  policyHash: string;
  policyStatus?: string;
  aiDecision: "APPROVE" | "REJECT";
  route: Route;
  assessedRoute: Route;
  assuranceScore: number | null;
  components: { confidence: number; disagreement: number; context: number } | null;
  context: { amount: number; upstreamRules: number; modelValidation: number; transactionType: number } | null;
  vetoes: { code: string; detail: string }[];
  signalQuality: SignalQuality;
  degradedSignals: string[];
  unreliableSignals: string[];
  reasons: string[];
  escalation: { source: string; decision: string; from?: string; to?: string; note?: string } | null;
}

export interface AuditRecord {
  transactionId: string;
  auditId: string;
  saltMasked?: string;
  policyHash: string;
  policyId: string;
  policyVersion: string;
  recordHash: string;
  modelVersion: string;
  aiDecision: string;
  finalDecision: string;
  reviewRequired: boolean;
  reviewerDecision: string | null;
  riskScore: number;
  confidence: number;
  amount: number;
  transactionType: string;
  reason?: string;
  timestamp: string;
  reviewedAt?: string | null;
  blockchainTxId?: string | null;
  blockNumber?: number | null;
  verificationStatus: string;
  isTampered: boolean;
  upstreamSource?: string | null;
  upstreamDecision?: string | null;
  // Record v2 only (decisions anchored by the assurance engine):
  recordVersion?: number;
  assurance?: Assurance | null;
  decisionFinalDecision?: string;
  reviewAuditId?: string | null;
  reviewRecordHash?: string | null;
  parentRecordHash?: string | null;
  reviewerKind?: string | null;
  reviewerRef?: string | null;
  reviewReason?: string | null;
  reviewBlockchainTxId?: string | null;
  reviewBlockNumber?: number | null;
  [key: string]: unknown;
}

export interface AuditStats {
  totalDecisions: number;
  approved: number;
  rejected: number;
  humanReview: number;
  verified: number;
  integrityFailures: number;
  onChainMissing: number;
  pendingReview?: number;
  routes?: Record<Route | "LEGACY", number>;
  signalQuality?: Record<SignalQuality, number>;
  activePolicyId?: string;
  blockchainStatus: {
    isMock: boolean;
    mode: string;
    rpcUrl: string | null;
    contractAddress: string | null;
    chainId?: string | null;
    isConfigured: boolean;
  };
}

export interface PolicySummary {
  id: string;
  version: string;
  engine: string;
  status: string;
  baseline: boolean;
  routesSignalRequests: boolean;
  hashScheme: "v1" | "v2";
  policyHash: string;
  description: string;
  document: Record<string, unknown>;
  onChain: { active: boolean | null; error: string | null };
}

export interface CommitmentCheck {
  auditId: string;
  calculatedHash: string;
  onChainHash: string | null;
  matches: boolean;
  blockTimestamp: number | null;
  parentRecordHash?: string;
  linkIntact?: boolean;
}

export interface VerifyResult {
  verified: boolean;
  status: string;
  message: string;
  redacted?: boolean;
  details?: {
    transactionId?: string;
    auditId?: string;
    recordVersion?: number;
    lifecycle?: "DECIDED" | "AWAITING_REVIEW" | "REVIEWED";
    storedOffChainHash?: string;
    calculatedCurrentHash?: string;
    blockchainOnChainHash?: string | null;
    policyHash?: string;
    blockTimestamp?: number | null;
    commitments?: { decision: CommitmentCheck; review: CommitmentCheck | null };
  };
}

export interface RecordDetail {
  record: AuditRecord;
  policy: { id: string; version: string; engine: string; status: string; hashScheme: string; policyHash: string } | null;
  evidenceExports: { actorRef: string; exportedAt: string }[];
}

// --- reads ---------------------------------------------------------------------------------

/** The service wraps its payload: {success, count, records}. Verified against the running
 *  Express service, not assumed -- a client that expected a bare array rendered an empty
 *  table however many records existed. */
interface RecordsEnvelope {
  success: boolean;
  count: number;
  records: AuditRecord[];
}

export async function auditRecords(): Promise<AuditRecord[]> {
  const payload = await request<RecordsEnvelope | AuditRecord[]>("audit", "/api/audit/records");
  // Tolerates both shapes, so a future change back to a bare array does not break the page.
  if (Array.isArray(payload)) return payload;
  return Array.isArray(payload?.records) ? payload.records : [];
}

/** The service returns {success, stats}; the stats object is what every screen reads. */
export async function auditStats(): Promise<AuditStats> {
  const payload = await request<{ stats: AuditStats }>("audit", "/api/audit/stats");
  return payload.stats;
}

export function auditRecord(transactionId: string): Promise<RecordDetail> {
  return request<RecordDetail>("audit", `/api/audit/records/${encodeURIComponent(transactionId)}`);
}

export async function auditQueue(): Promise<(AuditRecord & { route: Route })[]> {
  const payload = await request<{ queue: (AuditRecord & { route: Route })[] }>("audit", "/api/audit/queue");
  return payload.queue ?? [];
}

export async function auditPolicies(): Promise<{ activePolicyId: string; policies: PolicySummary[] }> {
  return request("audit", "/api/audit/policies");
}

/** Verify by transaction id or by on-chain auditId (the service tries both). */
export function verifyRecord(id: string): Promise<VerifyResult> {
  return request<VerifyResult>("audit", `/api/audit/${encodeURIComponent(id)}/verify`);
}

/** Kept for existing callers; same endpoint as verifyRecord. */
export function verifyAudit(transactionId: string): Promise<VerifyResult> {
  return verifyRecord(transactionId);
}

export function auditHealth(): Promise<{ status: string }> {
  return request<{ status: string }>("audit", "/api/health", { auth: false });
}

/** The independently verifiable evidence bundle (restricted; every export is logged). */
export function auditEvidence(transactionId: string): Promise<Record<string, unknown>> {
  return request("audit", `/api/audit/${encodeURIComponent(transactionId)}/evidence`);
}

export interface MethodResult {
  incorrect_automatic_rate: number;
  human_review_rate: number;
  verify_rate: number;
  fraud_auto_approved: number;
  legit_auto_rejected: number;
  aurc: number | null;
  ci95: { incorrect_automatic_rate: [number, number]; human_review_rate?: [number, number] };
  verify_effectiveness_sweep?: Record<string, number>;
}

export interface ResearchResults {
  status: "NOT_RUN" | "FINAL" | "NOT_FINAL_UNCOMMITTED_POLICY";
  generatedAt?: string;
  policy?: { id: string; version: string; hash: string };
  policyBudget?: number;
  data?: { rows: Record<string, number>; fraudRate: Record<string, number>; testIndexDigest: string };
  results?: Record<string, { methods: Record<string, MethodResult>; paired: Record<string, { difference: number; ci95: [number, number]; p_reference_not_better: number }> }>;
  curves?: Record<string, { human_review_rate: number; incorrect_automatic_rate: number }[]>;
}

/** The IEEE-CIS evaluation (aggregates only), or status NOT_RUN. */
export function researchResults(): Promise<ResearchResults> {
  return request<ResearchResults>("audit", "/api/audit/research/results", { auth: false });
}

// --- writes --------------------------------------------------------------------------------

export interface EvaluateInput {
  transactionId: string;
  riskScore: number;
  /** 0-100, not 0-1. The service rejects anything outside that range. */
  confidence: number;
  amount: number;
  transactionType: string;
  modelVersion: string;
  /** Assurance signals. Without them the service routes with the legacy P001 policy. */
  signals?: {
    modelScores?: Record<string, number>;
    rules?: string[];
    benchmarkModel?: boolean;
    sourceModelVersion?: string;
  };
}

export interface EvaluateResult {
  success: boolean;
  status: string;
  policyResult?: { action: string; reason: string; policyId: string; policyVersion: string };
  assurance?: Assurance;
  blockchain?: {
    success: boolean;
    blockchainTxId: string;
    blockNumber: number;
    isMock: boolean;
    network: string;
  };
  record?: AuditRecord;
}

export function evaluateDecision(input: EvaluateInput): Promise<EvaluateResult> {
  return request<EvaluateResult>("audit", "/api/audit/evaluate", { method: "POST", body: input });
}

export interface ReviewResult {
  success: boolean;
  message: string;
  separationOfDuties?: string;
  record: AuditRecord;
  blockchain?: { blockNumber: number; isMock: boolean; blockchainTxId: string };
}

export function submitReview(transactionId: string, reviewerDecision: "APPROVE" | "REJECT", reason?: string): Promise<ReviewResult> {
  return request<ReviewResult>("audit", "/api/audit/review", {
    method: "POST",
    body: { transactionId, reviewerDecision, ...(reason ? { reason } : {}) },
  });
}

export function simulateTampering(
  transactionId: string,
  field: string,
  tamperedValue: unknown,
): Promise<{ success: boolean; message: string }> {
  return request<{ success: boolean; message: string }>(
    "audit",
    `/api/audit/${encodeURIComponent(transactionId)}/tamper`,
    { method: "POST", body: { field, tamperedValue } },
  );
}
