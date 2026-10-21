"use client";

/**
 * Decisions the assurance engine held. HUMAN_REVIEW items wait for an analyst here; VERIFY
 * items are normally cleared by the account holder's step-up on the Withdraw screen.
 *
 * Every decision is anchored as a second commitment linked to the first. The service refuses
 * a review of a decision about the reviewer themselves, and says so.
 */

import Link from "next/link";
import { useState } from "react";

import { AuditLoadNotice } from "@/components/audit/AuditLoadNotice";
import { RouteBadge, routeOf } from "@/components/audit/badges";
import { useAuditLoad } from "@/components/audit/useAuditLoad";
import { Button, Card, Notice, money } from "@/components/ui";
import { auditQueue, submitReview, type AuditRecord } from "@/lib/api/audit";

export default function ReviewQueuePage() {
  const state = useAuditLoad(auditQueue);
  const [done, setDone] = useState<string | null>(null);
  const queue = state.data ?? [];
  const analyst = queue.filter((r) => routeOf(r.assurance) !== "VERIFY");
  const verify = queue.filter((r) => routeOf(r.assurance) === "VERIFY");

  const decided = (message: string) => {
    setDone(message);
    state.refresh();
  };

  return (
    <div className="space-y-6">
      <AuditLoadNotice state={state} what="the review queue" />
      {done && <Notice tone="success" title="Review anchored">{done}</Notice>}

      {state.data && (
        <>
          <Card title="Waiting for an analyst" subtitle="Routed to human review, or held by the legacy policy.">
            {analyst.length === 0 ? <p className="text-sm text-neutral-500">Nothing waiting.</p> : (
              <ul className="divide-y divide-black/5 dark:divide-white/10">
                {analyst.map((r) => <QueueItem key={r.transactionId} record={r} onDecided={decided} />)}
              </ul>
            )}
          </Card>

          <Card title="Waiting for step-up verification" subtitle="The account holder clears these by re-authenticating; an analyst can also decide.">
            {verify.length === 0 ? <p className="text-sm text-neutral-500">Nothing waiting.</p> : (
              <ul className="divide-y divide-black/5 dark:divide-white/10">
                {verify.map((r) => <QueueItem key={r.transactionId} record={r} onDecided={decided} />)}
              </ul>
            )}
          </Card>
        </>
      )}
    </div>
  );
}

function QueueItem({ record, onDecided }: { record: AuditRecord; onDecided: (message: string) => void }) {
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const a = record.assurance;

  async function decide(decision: "APPROVE" | "REJECT") {
    setBusy(true);
    setError(null);
    try {
      const result = await submitReview(record.transactionId, decision, reason.trim() || undefined);
      onDecided(`${record.transactionId} ${decision === "APPROVE" ? "approved" : "rejected"}${result.blockchain ? ` · block ${result.blockchain.blockNumber}` : ""}.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The review was not recorded");
    } finally {
      setBusy(false);
    }
  }

  return (
    <li className="py-4 first:pt-0 last:pb-0">
      <div className="flex flex-wrap items-center gap-3">
        <Link href={`/audit/decisions/${encodeURIComponent(record.transactionId)}`} className="font-mono text-xs underline-offset-2 hover:underline">
          {record.transactionId}
        </Link>
        <RouteBadge route={routeOf(a)} />
        <span className="text-sm tabular-nums">{money(record.amount)}</span>
        <span className="text-sm text-neutral-500">AI: {a?.aiDecision ?? record.aiDecision}</span>
        {a?.assuranceScore != null && <span className="text-sm tabular-nums text-neutral-500">assurance {a.assuranceScore.toFixed(2)}</span>}
        <span className="ml-auto text-xs text-neutral-500">held since {new Date(record.timestamp).toLocaleString()}</span>
      </div>
      {a && (a.vetoes.length > 0 || a.degradedSignals.length > 0) && (
        <p className="mt-1 text-xs text-neutral-600 dark:text-neutral-400">
          {[...a.vetoes.map((v) => v.detail), ...(a.degradedSignals.length ? [`Missing signals: ${a.degradedSignals.join(", ")}.`] : [])].join(" ")}
        </p>
      )}
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <input
          aria-label={`Reason for ${record.transactionId}`}
          value={reason}
          maxLength={500}
          onChange={(e) => setReason(e.target.value)}
          placeholder="Reason (anchored with the review)"
          className="min-w-0 flex-1 rounded-lg border border-black/15 bg-transparent px-3 py-1.5 text-sm outline-none focus:border-neutral-500 dark:border-white/20"
        />
        <Button onClick={() => decide("APPROVE")} disabled={busy}>Approve</Button>
        <Button variant="secondary" onClick={() => decide("REJECT")} disabled={busy}>Reject</Button>
      </div>
      {error && <div className="mt-2"><Notice tone="error" title="Not recorded">{error}</Notice></div>}
    </li>
  );
}
