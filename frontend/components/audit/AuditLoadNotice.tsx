"use client";

import { Notice, Spinner } from "@/components/ui";

import type { AuditLoadState } from "./useAuditLoad";

/** The non-data states of an audit screen, worded the same everywhere. */
export function AuditLoadNotice({ state, what }: { state: AuditLoadState<unknown>; what: string }) {
  if (state.loading && !state.data) return <Spinner label={`Loading ${what}`} />;
  if (state.unavailable) {
    return (
      <Notice tone="info" title="Audit bridge is not running">
        Start it with <code>docker compose up component3</code>, or <code>npm start</code> from{" "}
        <code>backend/Blockchain-Auditability</code>. Every other screen works without it.
      </Notice>
    );
  }
  if (state.forbidden) {
    return (
      <Notice tone="warn" title="Reviewer access required">
        The audit service enforces access control, and your account is not one of its reviewers
        (<code>AUDIT_REVIEWER_IDS</code>). Verification stays public on the Verify tab.
      </Notice>
    );
  }
  if (state.error) {
    return <Notice tone="error" title="Could not load">{state.error}</Notice>;
  }
  return null;
}
