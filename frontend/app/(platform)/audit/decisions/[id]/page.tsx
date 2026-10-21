"use client";

/**
 * One decision: why it was routed the way it was, the policy that routed it, its life on the
 * ledger, and the evidence an outsider can check without trusting this service.
 */

import Link from "next/link";
import { useParams } from "next/navigation";
import { useState } from "react";

import { AuditLoadNotice } from "@/components/audit/AuditLoadNotice";
import { SignalBreakdown } from "@/components/audit/SignalBreakdown";
import { IntegrityBadge, RouteBadge, outcomeLabel, routeOf, shortHash } from "@/components/audit/badges";
import { useAuditLoad } from "@/components/audit/useAuditLoad";
import { Button, Card, Notice, Stat, money } from "@/components/ui";
import { auditEvidence, auditPolicies, auditRecord, verifyRecord, type VerifyResult } from "@/lib/api/audit";

const REVIEWER_KIND: Record<string, string> = {
  ACCOUNT_HOLDER_STEP_UP: "the account holder, at step-up",
  ANALYST: "an analyst",
  UNAUTHENTICATED: "an unauthenticated reviewer (access control off)",
};

/** Route params may arrive encoded or already decoded; decoding twice must not throw. */
function safeDecode(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

export default function DecisionDetailPage() {
  const { id } = useParams<{ id: string }>();
  const transactionId = safeDecode(id);
  const state = useAuditLoad(async () => {
    const [detail, policies] = await Promise.all([auditRecord(transactionId), auditPolicies()]);
    return { detail, policies };
  }, [transactionId]);

  const [verify, setVerify] = useState<VerifyResult | null>(null);
  const [busy, setBusy] = useState<"verify" | "evidence" | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  async function act(kind: "verify" | "evidence") {
    setBusy(kind);
    setActionError(null);
    try {
      if (kind === "verify") {
        setVerify(await verifyRecord(transactionId));
        state.refresh();
      } else {
        const bundle = await auditEvidence(transactionId);
        const url = URL.createObjectURL(new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json" }));
        const link = document.createElement("a");
        link.href = url;
        link.download = `evidence-${transactionId}.json`;
        link.click();
        URL.revokeObjectURL(url);
        state.refresh();
      }
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "Rejected");
    } finally {
      setBusy(null);
    }
  }

  const back = <Link href="/audit/decisions" className="text-sm underline underline-offset-2">← All decisions</Link>;
  if (!state.data) return <div className="space-y-4">{back}<AuditLoadNotice state={state} what="the decision" /></div>;

  const { record, policy, evidenceExports } = state.data.detail;
  const policyDoc = state.data.policies.policies.find((p) => p.policyHash === record.policyHash);
  const routing = (policyDoc?.document as { routing?: { autoMin: number; verifyMin: number } } | undefined)?.routing;
  const a = record.assurance;

  return (
    <div className="space-y-6">
      {back}
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="font-mono text-base font-semibold">{record.transactionId}</h2>
        <RouteBadge route={routeOf(a)} />
        <IntegrityBadge status={record.verificationStatus} />
      </div>

      <Card>
        <div className="grid grid-cols-2 gap-6 sm:grid-cols-4">
          <Stat label="Amount" value={money(record.amount)} hint={record.transactionType} />
          <Stat label="AI decision" value={a?.aiDecision ?? record.aiDecision} hint={`risk score ${record.riskScore}`} />
          <Stat label="Outcome" value={outcomeLabel(record.finalDecision)} />
          <Stat label="Fraud gateway" value={record.upstreamDecision ?? "—"} hint={record.upstreamSource ?? undefined} />
        </div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-5">
        <Card title="Why this route" className="lg:col-span-3">
          {a ? <SignalBreakdown assurance={a} thresholds={routing} /> : (
            <p className="text-sm text-neutral-500">Anchored before the assurance engine existed (record v1); it carries no assurance result.</p>
          )}
          {a?.escalation && (
            <p className="mt-4 text-sm text-neutral-600 dark:text-neutral-400">
              {a.escalation.source} enforced <strong>{a.escalation.decision}</strong>
              {a.escalation.decision === "BLOCK" ? ", so the rejection was applied automatically." : ", so an automatic approval was held for verification."}
            </p>
          )}
          {a && a.reasons.length > 0 && (
            <details className="mt-4 text-sm">
              <summary className="cursor-pointer text-neutral-500">Engine reasoning</summary>
              <ul className="mt-2 list-disc space-y-1 pl-5 text-neutral-600 dark:text-neutral-400">
                {a.reasons.map((r) => <li key={r}>{r}</li>)}
              </ul>
            </details>
          )}
        </Card>

        <Card title="Policy" className="lg:col-span-2">
          <dl className="space-y-2 text-sm">
            <div><dt className="text-xs uppercase tracking-wide text-neutral-500">Policy</dt>
              <dd>{policy ? `${policy.id} v${policy.version}` : `${record.policyId} v${record.policyVersion}`}
                {policy?.status === "untuned" && <span className="ml-2 text-xs text-amber-700 dark:text-amber-400">untuned</span>}</dd></div>
            <div><dt className="text-xs uppercase tracking-wide text-neutral-500">Policy hash</dt>
              <dd className="break-all font-mono text-xs">{record.policyHash}</dd></div>
            <div><dt className="text-xs uppercase tracking-wide text-neutral-500">Signal quality</dt>
              <dd>{a?.signalQuality ?? "—"}{a?.degradedSignals.length ? ` · missing: ${a.degradedSignals.join(", ")}` : ""}</dd></div>
          </dl>
          <Link href="/audit/policies" className="mt-3 inline-block text-sm underline underline-offset-2">See the policy document</Link>
        </Card>
      </div>

      <Card title="On the ledger" subtitle="Only hashes are on-chain; amounts and scores stay here.">
        <ol className="space-y-4 border-l border-black/10 pl-5 text-sm dark:border-white/15">
          <li>
            <p className="font-medium">Decision anchored</p>
            <p className="text-xs text-neutral-500">
              {new Date(record.timestamp).toLocaleString()} · block {record.blockNumber ?? "—"} · audit id <span className="font-mono">{shortHash(record.auditId)}</span>
            </p>
            <p className="text-xs text-neutral-500">commits the decision, the assurance result and the policy hash · <span className="font-mono">{shortHash(record.recordHash)}</span></p>
          </li>
          {record.reviewAuditId ? (
            <li>
              <p className="font-medium">{outcomeLabel(record.finalDecision)} by {REVIEWER_KIND[record.reviewerKind ?? ""] ?? record.reviewerKind}</p>
              <p className="text-xs text-neutral-500">
                {record.reviewedAt ? new Date(record.reviewedAt).toLocaleString() : ""} · block {record.reviewBlockNumber ?? "—"} · linked to the decision by its hash
              </p>
              {record.reviewReason && <p className="text-xs text-neutral-600 dark:text-neutral-400">“{record.reviewReason}”</p>}
            </li>
          ) : record.finalDecision === "PENDING_HUMAN_REVIEW" ? (
            <li><p className="font-medium text-amber-700 dark:text-amber-400">Awaiting {routeOf(a) === "VERIFY" ? "step-up verification" : "an analyst"}</p>
              <Link href="/audit/review" className="text-xs underline underline-offset-2">Review queue</Link></li>
          ) : null}
        </ol>

        <div className="mt-6 flex flex-wrap gap-2">
          <Button onClick={() => act("verify")} disabled={busy !== null}>{busy === "verify" ? "Verifying…" : "Verify against the ledger"}</Button>
          <Button variant="secondary" onClick={() => act("evidence")} disabled={busy !== null}>{busy === "evidence" ? "Preparing…" : "Download evidence bundle"}</Button>
        </div>
        {actionError && <div className="mt-4"><Notice tone="error" title="Rejected">{actionError}</Notice></div>}
        {verify && (
          <div className="mt-4">
            <Notice tone={verify.verified ? "success" : "error"} title={verify.verified ? "Integrity verified" : "Integrity check failed"}>
              {verify.message}
              {verify.details?.commitments && (
                <span className="mt-1 block text-xs">
                  Decision commitment {verify.details.commitments.decision.matches ? "matches" : "does not match"}
                  {verify.details.commitments.review && <>; review commitment {verify.details.commitments.review.matches ? "matches" : "does not match"}, link {verify.details.commitments.review.linkIntact ? "intact" : "broken"}</>}.
                </span>
              )}
            </Notice>
          </div>
        )}
        <p className="mt-4 text-xs text-neutral-500">
          The bundle holds the committed payloads, salt and policy document, and no personal data. Check it independently with{" "}
          <code>node scripts/verify-evidence.js evidence.json --rpc &lt;url&gt;</code>.
          {evidenceExports.length > 0 && ` Exported ${evidenceExports.length} time${evidenceExports.length === 1 ? "" : "s"}; every export is logged.`}
        </p>
      </Card>
    </div>
  );
}
