"use client";

/**
 * Verification, open to anyone: look a decision up by its transaction id or its on-chain
 * audit id and check it against the ledger, then the tamper-evidence walkthrough.
 */

import { useState } from "react";

import { IntegrityBadge } from "@/components/audit/badges";
import { Button, Card, Field, Input, Notice } from "@/components/ui";
import { verifyRecord, type CommitmentCheck, type VerifyResult } from "@/lib/api/audit";
import { ApiError } from "@/lib/api/client";

import { TamperDemo } from "../TamperDemo";

function commitmentRows(c: { decision: CommitmentCheck; review: CommitmentCheck | null }) {
  const rows = [{ label: "decision commitment", check: c.decision }];
  if (c.review) rows.push({ label: "review commitment", check: c.review });
  return rows;
}

export default function VerifyPage() {
  const [id, setId] = useState("");
  const [result, setResult] = useState<VerifyResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function check(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      setResult(await verifyRecord(id.trim()));
    } catch (cause) {
      setResult(null);
      setError(cause instanceof ApiError && cause.status === 0 ? "The audit bridge is not running." : cause instanceof Error ? cause.message : "Could not verify");
    } finally {
      setBusy(false);
    }
  }

  const d = result?.details;
  return (
    <div className="space-y-6">
      <Card title="Verify a decision" subtitle="Recomputes the record's hashes and compares them with the on-chain commitments.">
        <form onSubmit={check} className="flex flex-wrap items-end gap-3">
          <div className="min-w-[16rem] flex-1">
            <Field label="Transaction id or audit id">
              <Input value={id} onChange={(e) => setId(e.target.value)} placeholder="EXE-… or 0x…" required />
            </Field>
          </div>
          <Button type="submit" disabled={busy || !id.trim()}>{busy ? "Checking…" : "Check"}</Button>
        </form>

        {error && <div className="mt-4"><Notice tone="error" title="Could not verify">{error}</Notice></div>}

        {result && (
          <div className="mt-4 space-y-3">
            <div className="flex flex-wrap items-center gap-3">
              <IntegrityBadge status={result.status} />
              {d?.lifecycle && <span className="text-sm text-neutral-500">{d.lifecycle.replace("_", " ").toLowerCase()}</span>}
            </div>
            <p className="text-sm">{result.message}</p>
            {d?.commitments ? (
              <dl className="space-y-2 text-xs">
                {commitmentRows(d.commitments).map(({ label, check: c }) => (
                  <div key={label} className="rounded-lg border border-black/10 p-3 dark:border-white/15">
                    <dt className="font-medium">
                      {label} · {c.matches ? "matches the chain" : "does not match"}
                      {c.linkIntact === false && " · link to the decision broken"}
                    </dt>
                    <dd className="mt-1 break-all font-mono text-neutral-500">recalculated {c.calculatedHash}</dd>
                    <dd className="break-all font-mono text-neutral-500">on-chain&nbsp;&nbsp;&nbsp;&nbsp; {c.onChainHash ?? "—"}</dd>
                  </div>
                ))}
              </dl>
            ) : d?.calculatedCurrentHash ? (
              <dl className="space-y-1 break-all font-mono text-xs text-neutral-500">
                <dd>recalculated {d.calculatedCurrentHash}</dd>
                <dd>on-chain&nbsp;&nbsp;&nbsp;&nbsp; {d.blockchainOnChainHash ?? "—"}</dd>
              </dl>
            ) : null}
            {result.redacted && <p className="text-xs text-neutral-500">Record contents are hidden; only hashes and the result are public.</p>}
          </div>
        )}
      </Card>

      <TamperDemo />
    </div>
  );
}
