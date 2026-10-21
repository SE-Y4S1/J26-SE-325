"use client";

/**
 * Sends one AI decision through the assurance engine, so the routing can be seen happening.
 *
 * The presets are the situations the engine is meant to tell apart. The values go to the real
 * service and are anchored like any other decision.
 */

import Link from "next/link";
import { useState } from "react";

import { Button, Card, Field, Input, Notice } from "@/components/ui";
import { evaluateDecision, type EvaluateResult } from "@/lib/api/audit";
import { ApiError } from "@/lib/api/client";

import { RouteBadge, routeOf } from "./badges";

interface Draft {
  riskScore: number;
  amount: number;
  lstm: string;
  gnn: string;
  validated: boolean;
}

const PRESETS: { label: string; hint: string; draft: Draft }[] = [
  { label: "Confident, models agree", hint: "expected: Automatic", draft: { riskScore: 12, amount: 2500, lstm: "0.10", gnn: "0.14", validated: true } },
  { label: "Near the boundary", hint: "expected: Verification", draft: { riskScore: 42, amount: 2500, lstm: "0.38", gnn: "0.44", validated: true } },
  { label: "Models disagree", hint: "expected: Human review", draft: { riskScore: 35, amount: 2500, lstm: "0.10", gnn: "0.75", validated: true } },
  { label: "Untrained model only", hint: "never Automatic", draft: { riskScore: 1, amount: 2500, lstm: "", gnn: "", validated: false } },
  { label: "Large amount", hint: "expected: Human review", draft: { riskScore: 8, amount: 150000, lstm: "0.07", gnn: "0.09", validated: true } },
];

export function TryDecision({ onAnchored }: { onAnchored?: () => void }) {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState<Draft>(PRESETS[0].draft);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<EvaluateResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const modelScores: Record<string, number> = {};
      if (draft.lstm !== "") modelScores.lstm = Number(draft.lstm);
      if (draft.gnn !== "") modelScores.gnn = Number(draft.gnn);
      const response = await evaluateDecision({
        transactionId: `TRY-${Date.now()}`,
        riskScore: draft.riskScore,
        confidence: 90,
        amount: draft.amount,
        transactionType: "PAYMENT",
        modelVersion: draft.validated ? "fraud-bundle-demo" : "heuristic-untrained",
        signals: {
          modelScores,
          rules: [],
          benchmarkModel: draft.validated,
          sourceModelVersion: draft.validated ? "fraud-bundle-demo" : "heuristic-untrained",
        },
      });
      setResult(response);
      onAnchored?.();
    } catch (cause) {
      setResult(null);
      setError(cause instanceof ApiError && cause.status === 0 ? "The audit bridge is not running." : cause instanceof Error ? cause.message : "Rejected");
    } finally {
      setBusy(false);
    }
  }

  if (!open) {
    return (
      <div className="flex justify-end">
        <Button variant="secondary" onClick={() => setOpen(true)}>Try a decision</Button>
      </div>
    );
  }

  return (
    <Card title="Try a decision" subtitle="Route one AI decision through the assurance engine and anchor it.">
      <div className="mb-4 flex flex-wrap gap-1.5">
        {PRESETS.map((p) => (
          <button
            key={p.label}
            type="button"
            title={p.hint}
            onClick={() => { setDraft(p.draft); setResult(null); }}
            className="rounded-full border border-black/15 px-3 py-1 text-xs hover:bg-black/5 dark:border-white/20 dark:hover:bg-white/10"
          >
            {p.label}
          </button>
        ))}
      </div>
      <form onSubmit={submit} className="grid gap-4 sm:grid-cols-5">
        <Field label="Risk score" hint="0–100, the model's output">
          <Input type="number" min={0} max={100} step="0.1" value={draft.riskScore}
            onChange={(e) => setDraft({ ...draft, riskScore: Number(e.target.value) })} required />
        </Field>
        <Field label="Amount">
          <Input type="number" min={1} step="1" value={draft.amount}
            onChange={(e) => setDraft({ ...draft, amount: Number(e.target.value) })} required />
        </Field>
        <Field label="LSTM score" hint="0–1, blank = absent">
          <Input type="number" min={0} max={1} step="0.01" value={draft.lstm}
            onChange={(e) => setDraft({ ...draft, lstm: e.target.value })} />
        </Field>
        <Field label="GNN score" hint="0–1, blank = absent">
          <Input type="number" min={0} max={1} step="0.01" value={draft.gnn}
            onChange={(e) => setDraft({ ...draft, gnn: e.target.value })} />
        </Field>
        <label className="flex items-center gap-2 self-center text-sm">
          <input type="checkbox" checked={draft.validated} onChange={(e) => setDraft({ ...draft, validated: e.target.checked })} />
          Benchmark-validated model
        </label>
        <div className="flex gap-2 sm:col-span-5">
          <Button type="submit" disabled={busy}>{busy ? "Anchoring…" : "Route and anchor"}</Button>
          <Button type="button" variant="secondary" onClick={() => setOpen(false)}>Close</Button>
        </div>
      </form>

      {error && <div className="mt-4"><Notice tone="error" title="Rejected">{error}</Notice></div>}

      {result?.assurance && result.record && (
        <div className="mt-4 flex flex-wrap items-center gap-3 rounded-lg border border-black/10 p-3 text-sm dark:border-white/15">
          <RouteBadge route={routeOf(result.assurance)} />
          {result.assurance.assuranceScore != null && (
            <span className="tabular-nums">assurance {result.assurance.assuranceScore.toFixed(2)}</span>
          )}
          {result.assurance.signalQuality === "DEGRADED" && <span className="text-amber-700 dark:text-amber-400">degraded signals</span>}
          <span className="text-neutral-500">
            {result.blockchain ? `anchored in block ${result.blockchain.blockNumber}${result.blockchain.isMock ? " (simulated)" : ""}` : ""}
          </span>
          <Link href={`/audit/decisions/${encodeURIComponent(result.record.transactionId)}`} className="ml-auto underline underline-offset-2">
            Why?
          </Link>
        </div>
      )}
    </Card>
  );
}
