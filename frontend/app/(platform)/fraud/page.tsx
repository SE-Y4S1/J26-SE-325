"use client";

/**
 * Component 2 — Real-time fraud and behavioural anomaly engine (Dhushanthini R).
 *
 * Deliberately a thin screen over her real service: a transaction goes to
 * /v1/transactions/score and the gateway's verdict comes back. The models, the thresholds
 * and the wording of the rules are all hers — this page renders them and adds nothing.
 *
 * Executed portfolio plans reach this engine without this screen: the platform service
 * screens every Component 1 execution here before Component 3 anchors it.
 */

import { useEffect, useState } from "react";

import { Button, Card, Field, Input, Notice, Stat, money } from "@/components/ui";
import { ApiError } from "@/lib/api/client";
import {
  fraudHealth,
  scoreTransaction,
  type FraudHealth,
  type ScoreResponse,
  type TransactionInput,
} from "@/lib/api/fraud";

const TONE: Record<ScoreResponse["decision"], "success" | "warn" | "error"> = {
  ALLOW: "success",
  STEP_UP: "warn",
  BLOCK: "error",
};

const LABEL: Record<ScoreResponse["decision"], string> = {
  ALLOW: "ALLOW",
  STEP_UP: "STEP-UP verification required",
  BLOCK: "BLOCK",
};

function newTransactionId(): string {
  return `ui-${Date.now().toString(36)}`;
}

export default function FraudPage() {
  const [tx, setTx] = useState<TransactionInput>({
    transaction_id: newTransactionId(),
    amount: 1200,
    currency: "USD",
    account_key: "acct-demo-1",
    device_key: "dev-trusted-01",
    ip_reputation: 0.9,
  });
  const [health, setHealth] = useState<FraudHealth | null>(null);
  const [result, setResult] = useState<ScoreResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [unavailable, setUnavailable] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fraudHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  async function handleScore() {
    setBusy(true);
    setError(null);
    setUnavailable(false);
    try {
      setResult(await scoreTransaction(tx));
      // Each score is one transaction; reusing the id would merge two in the engine's history.
      setTx((current) => ({ ...current, transaction_id: newTransactionId() }));
    } catch (cause) {
      setResult(null);
      // A service that is not running is a state to explain, not an error to alarm anyone with.
      if (cause instanceof ApiError && (cause.isUnavailable || cause.status === 0)) {
        setUnavailable(true);
      } else {
        setError(cause instanceof Error ? cause.message : "Could not score the transaction");
      }
    } finally {
      setBusy(false);
    }
  }

  const set = <K extends keyof TransactionInput>(key: K, value: TransactionInput[K]) =>
    setTx((current) => ({ ...current, [key]: value }));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Fraud detection</h1>
        <p className="mt-1 text-sm text-neutral-500">
          Scores a transaction with the LSTM-autoencoder and graph streams, then applies the
          deterministic gateway. No language model chooses the action. Component 2 · Dhushanthini R.
        </p>
      </div>

      {health && !health.benchmark_model && (
        <Notice tone="info" title="Untrained heuristic">
          The service is running <code>{health.model_version}</code>: no trained bundle is loaded,
          so scores come from a fixed heuristic and are not research results. Train one with{" "}
          <code>scripts/train_benchmark.py</code>.
        </Notice>
      )}

      <Card
        title="Transaction"
        subtitle="Identifiers are pseudonymous keys. The service rejects names, emails, card numbers and IP addresses."
      >
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Amount">
            <Input
              type="number"
              value={tx.amount}
              onChange={(e) => set("amount", Number(e.target.value))}
            />
          </Field>
          <Field label="Currency">
            <Input
              value={tx.currency ?? ""}
              maxLength={3}
              onChange={(e) => set("currency", e.target.value.toUpperCase())}
            />
          </Field>
          <Field label="Account key" hint="Per-account history drives the behavioural stream.">
            <Input value={tx.account_key ?? ""} onChange={(e) => set("account_key", e.target.value)} />
          </Field>
          <Field label="Device key">
            <Input value={tx.device_key ?? ""} onChange={(e) => set("device_key", e.target.value)} />
          </Field>
          <Field
            label={`IP reputation — ${(tx.ip_reputation ?? 0).toFixed(2)}`}
            hint="0 is worst, 1 is trusted. Low reputation can only make the action stricter."
          >
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={tx.ip_reputation ?? 1}
              onChange={(e) => set("ip_reputation", Number(e.target.value))}
              className="w-full"
            />
          </Field>
        </div>

        <div className="mt-4 flex justify-end">
          <Button onClick={handleScore} disabled={busy}>
            {busy ? "Scoring…" : "Score transaction"}
          </Button>
        </div>
      </Card>

      {unavailable && (
        <Notice tone="info" title="Fraud service is not running">
          Start it with <code>docker compose up component2</code>, or run{" "}
          <code>py scripts\run_api.py</code> from <code>backend/Fraud-Detection</code>. Every
          other screen works without it; executions are held until it is back.
        </Notice>
      )}

      {error && (
        <Notice tone="error" title="Could not score">
          {error}
        </Notice>
      )}

      {result && <ScoreResult result={result} amount={tx.amount} currency={tx.currency} />}
    </div>
  );
}

function ScoreResult({
  result,
  amount,
  currency = "USD",
}: {
  result: ScoreResponse;
  amount?: number;
  currency?: string;
}) {
  const { rules, streams, top_factors: factors, notes } = result.explanation;
  return (
    <div className="space-y-4">
      <Notice tone={TONE[result.decision]} title={LABEL[result.decision]}>
        {rules.length > 0 ? rules.join(" · ") : "No gateway rule fired; the score alone decided."}
      </Notice>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label="Fraud score" value={result.fraud_score.toFixed(3)} />
        {Object.entries(streams).map(([name, value]) => (
          <Stat key={name} label={name.toUpperCase()} value={value.toFixed(3)} />
        ))}
        {amount !== undefined && <Stat label="Amount" value={money(amount, currency)} />}
      </div>

      {factors.length > 0 && (
        <Card title="Top factors" subtitle="The engine's own feature attributions, largest first.">
          <table className="w-full text-sm">
            <tbody>
              {factors.map((factor) => (
                <tr key={factor.name} className="border-t border-neutral-200/60">
                  <td className="py-2 font-medium">{factor.name}</td>
                  <td className="py-2 text-right tabular-nums">
                    {factor.contribution > 0 ? "+" : ""}
                    {factor.contribution.toFixed(4)}
                  </td>
                  <td className="py-2 pl-4 text-neutral-500">
                    {factor.contribution > 0 ? "increases" : "decreases"} risk
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {notes.length > 0 && (
        <ul className="list-disc space-y-1 pl-5 text-xs text-neutral-500">
          {notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}

      <p className="text-xs text-neutral-500">
        Model <code>{result.model_version}</code>
        {result.benchmark_model ? " (benchmark-validated)" : " (not a benchmark model)"} · transaction{" "}
        <code>{result.transaction_id}</code>
      </p>
    </div>
  );
}
