/**
 * Component 3 — Privacy-Preserving AI-to-Smart-Contract Audit Bridge (Abeysekara W C S M).
 *
 * An Express service, not FastAPI, so there is no OpenAPI document to generate from. Routes
 * mirror `Blockchain-Auditability/backend/src/routes/auditRoutes.js`.
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
