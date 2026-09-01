"use client";

/**
 * Component 3's tamper-evidence demonstration.
 *
 * The reason an audit bridge exists at all: a decision is anchored, its hash is committed
 * on-chain, and altering the stored record afterwards makes verification fail. Without this
 * on screen the audit page is just a table, and the research claim is invisible.
 *
 * Every step calls his real service — the anchoring, the SHA-256 comparison and the tamper
 * simulation are all his, and verified against the running backend.
 */

import { useState } from "react";

import { Button, Card, Notice, Stat } from "@/components/ui";
import {
  evaluateDecision,
  simulateTampering,
  verifyRecord,
  type EvaluateResult,
  type VerifyResult,
} from "@/lib/api/audit";
import { ApiError } from "@/lib/api/client";

type Stage = "idle" | "anchored" | "verified" | "tampered" | "detected";

export function TamperDemo({ onChanged }: { onChanged?: () => void }) {
  const [stage, setStage] = useState<Stage>("idle");
  const [txId, setTxId] = useState<string | null>(null);
  const [anchor, setAnchor] = useState<EvaluateResult | null>(null);
  const [verify, setVerify] = useState<VerifyResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [unavailable, setUnavailable] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function guard(work: () => Promise<void>) {
    setBusy(true);
    setError(null);
    setUnavailable(false);
    try {
      await work();
    } catch (cause) {
      if (cause instanceof ApiError && (cause.isUnavailable || cause.status === 0)) {
        setUnavailable(true);
      } else {
        setError(cause instanceof Error ? cause.message : "The audit bridge rejected that");
      }
    } finally {
      setBusy(false);
    }
  }

  const anchorDecision = () =>
    guard(async () => {
      const id = `tx_demo_${Date.now()}`;
      // confidence is 0-100 here, not 0-1: his validator rejects anything else.
      const result = await evaluateDecision({
        transactionId: id,
        riskScore: 0.91,
        confidence: 93,
        amount: 250_000,
        transactionType: "withdrawal",
        modelVersion: "fraud-v1",
      });
      setTxId(id);
      setAnchor(result);
      setVerify(null);
      setStage("anchored");
      onChanged?.();
    });

  const verifyNow = (expectTampered: boolean) =>
    guard(async () => {
      if (!txId) return;
      const result = await verifyRecord(txId);
      setVerify(result);
      setStage(expectTampered ? "detected" : "verified");
    });

  const tamper = () =>
    guard(async () => {
      if (!txId) return;
      // Change the amount to something absurd. The record still exists and still looks
      // plausible -- only the hash disagrees, which is the entire point.
      await simulateTampering(txId, "amount", 1);
      setVerify(null);
      setStage("tampered");
      onChanged?.();
    });

  return (
    <div className="space-y-4">
      <Card
        title="Tamper-evidence"
        subtitle="Anchor a decision, verify it against the chain, alter the record, then watch verification fail."
      >
        <div className="flex flex-wrap gap-2">
          <Button onClick={anchorDecision} disabled={busy}>
            1 · Anchor a decision
          </Button>
          <Button
            variant="secondary"
            onClick={() => verifyNow(false)}
            disabled={busy || stage === "idle"}
          >
            2 · Verify
          </Button>
          <Button
            variant="secondary"
            onClick={tamper}
            disabled={busy || stage === "idle"}
          >
            3 · Tamper with the record
          </Button>
          <Button
            variant="danger"
            onClick={() => verifyNow(true)}
            disabled={busy || stage === "idle"}
          >
            4 · Verify again
          </Button>
        </div>
        {txId && (
          <p className="mt-3 font-mono text-xs text-neutral-500">transaction: {txId}</p>
        )}
      </Card>

      {unavailable && (
        <Notice tone="info" title="Audit bridge is not running">
          Start it with <code>docker compose up component3</code>.
        </Notice>
      )}

      {error && (
        <Notice tone="error" title="Rejected">
          {error}
        </Notice>
      )}

      {anchor?.blockchain && (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Stat label="Policy action" value={anchor.policyResult?.action ?? "—"} />
          <Stat label="Block" value={anchor.blockchain.blockNumber} />
          <Stat
            label="Network"
            value={anchor.blockchain.isMock ? "simulated" : "live"}
            hint={anchor.blockchain.network}
          />
          <Stat label="Status" value={anchor.status} />
        </div>
      )}

      {stage === "tampered" && !verify && (
        <Notice tone="warn" title="Record altered">
          The stored amount has been changed. The record still looks perfectly plausible —
          verify it again and see whether that is enough to get away with.
        </Notice>
      )}

      {verify && (
        <Notice
          tone={verify.verified ? "success" : "error"}
          title={verify.verified ? "Integrity verified" : "Tampering detected"}
        >
          {verify.message}
        </Notice>
      )}

      {verify?.details && (
        <Card title="Hashes" subtitle="Verification is this comparison, and nothing more.">
          <dl className="space-y-2 text-xs">
            {[
              ["stored off-chain", verify.details.storedOffChainHash],
              ["recalculated now", verify.details.calculatedCurrentHash],
              ["committed on-chain", verify.details.blockchainOnChainHash],
            ].map(([label, value]) => (
              <div key={label} className="flex flex-col gap-1 sm:flex-row sm:gap-3">
                <dt className="w-40 shrink-0 uppercase tracking-wide text-neutral-500">
                  {label}
                </dt>
                <dd className="break-all font-mono">{value ?? "—"}</dd>
              </div>
            ))}
          </dl>
        </Card>
      )}
    </div>
  );
}
