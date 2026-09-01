"use client";

/**
 * Component 2 — Real-Time Fraud & Behavioral Anomaly Engine (Dhushanthini R).
 *
 * Deliberately a thin screen over her real service: a transaction goes to /score and the
 * gateway's verdict comes back. The detection logic, the thresholds and the wording of the
 * reason are all hers — this page renders them and adds nothing of its own.
 */

import { useState } from "react";

import { Button, Card, Field, Input, Notice, Stat, money } from "@/components/ui";
import { ApiError } from "@/lib/api/client";
import { scoreTransaction, type ScoreResponse, type TransactionInput } from "@/lib/api/fraud";

import { AttackSimulator } from "./AttackSimulator";

const DEFAULT_TX: TransactionInput = {
  user_id: "u_1001",
  amount: 1200,
  location: "Colombo, LK",
  device_id: "dev_trusted_01",
  device_change: false,
  typing_speed: 4.2,
  navigation_pattern: "normal",
  transaction_frequency: 3,
  beneficiary_change: false,
  previous_transaction_amount: 900,
  account_age: 365,
};

const TONE: Record<ScoreResponse["decision"], "success" | "warn" | "error"> = {
  ALLOW: "success",
  "STEP-UP": "warn",
  BLOCK: "error",
};

export default function FraudPage() {
  const [tab, setTab] = useState<"score" | "attack">("score");
  const [tx, setTx] = useState<TransactionInput>(DEFAULT_TX);
  const [result, setResult] = useState<ScoreResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [unavailable, setUnavailable] = useState(false);
  const [busy, setBusy] = useState(false);

  async function handleScore() {
    setBusy(true);
    setError(null);
    setUnavailable(false);
    try {
      setResult(await scoreTransaction(tx));
    } catch (cause) {
      setResult(null);
      // Same treatment the Forecast screen gives a 503: a service that is not running is a
      // state to explain, not an error to alarm anyone with.
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
          Scores a transaction through the dual-stream engine, then applies the deterministic
          gateway. Component 2 · Dhushanthini R.
        </p>
      </div>

      <div className="flex gap-2 border-b border-neutral-200">
        {([["score", "Score a transaction"], ["attack", "Adversarial robustness"]] as const).map(
          ([key, label]) => (
            <button
              key={key}
              onClick={() => setTab(key)}
              className={
                tab === key
                  ? "border-b-2 border-neutral-900 px-3 py-2 text-sm font-medium"
                  : "px-3 py-2 text-sm text-neutral-500 hover:text-neutral-900"
              }
            >
              {label}
            </button>
          ),
        )}
      </div>

      {tab === "attack" && <AttackSimulator />}

      {tab === "score" && (
      <>
      <Card title="Transaction" subtitle="Behavioural signals matter as much as the amount.">
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="User">
            <Input value={tx.user_id} onChange={(e) => set("user_id", e.target.value)} />
          </Field>
          <Field label="Amount">
            <Input
              type="number"
              value={tx.amount}
              onChange={(e) => set("amount", Number(e.target.value))}
            />
          </Field>
          <Field label="Location">
            <Input value={tx.location} onChange={(e) => set("location", e.target.value)} />
          </Field>
          <Field label="Device">
            <Input value={tx.device_id} onChange={(e) => set("device_id", e.target.value)} />
          </Field>
          <Field label="Typing speed" hint="keystrokes/second — a behavioural biometric">
            <Input
              type="number"
              step="0.1"
              value={tx.typing_speed}
              onChange={(e) => set("typing_speed", Number(e.target.value))}
            />
          </Field>
          <Field label="Account age (days)">
            <Input
              type="number"
              value={tx.account_age}
              onChange={(e) => set("account_age", Number(e.target.value))}
            />
          </Field>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={tx.device_change}
              onChange={(e) => set("device_change", e.target.checked)}
            />
            New device
          </label>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={tx.beneficiary_change}
              onChange={(e) => set("beneficiary_change", e.target.checked)}
            />
            New beneficiary
          </label>
          <div className="flex-1" />
          <Button onClick={handleScore} disabled={busy}>
            {busy ? "Scoring…" : "Score transaction"}
          </Button>
        </div>
      </Card>

      {unavailable && (
        <Notice tone="info" title="Fraud service is not running">
          Start it with <code>docker compose up component2</code>, or run{" "}
          <code>uvicorn app:app --port 8001</code> from{" "}
          <code>backend/Fraud-Detection</code>. Every other screen works without it.
        </Notice>
      )}

      {error && (
        <Notice tone="error" title="Could not score">
          {error}
        </Notice>
      )}

      {result && (
        <div className="space-y-4">
          <Notice tone={TONE[result.decision]} title={result.decision}>
            {result.reason}
          </Notice>

          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat label="Risk score" value={result.risk_score.toFixed(3)} />
            <Stat label="Behavioural" value={result.behavioral_score.toFixed(3)} />
            <Stat label="Graph" value={result.graph_score.toFixed(3)} />
            <Stat label="Amount" value={money(tx.amount)} />
          </div>

          {result.signals && result.signals.length > 0 && (
            <Card title="Signals" subtitle="What drove the score, straight from the engine.">
              <table className="w-full text-sm">
                <thead className="text-left text-xs uppercase tracking-wide text-neutral-500">
                  <tr>
                    <th className="pb-2">Signal</th>
                    <th className="pb-2 text-right">Value</th>
                    <th className="pb-2 text-right">Weight</th>
                    <th className="pb-2">Explanation</th>
                  </tr>
                </thead>
                <tbody>
                  {result.signals.map((signal) => (
                    <tr key={signal.name} className="border-t border-neutral-200/60">
                      <td className="py-2 font-medium">{signal.name}</td>
                      <td className="py-2 text-right">{signal.value.toFixed(3)}</td>
                      <td className="py-2 text-right">{signal.weight.toFixed(2)}</td>
                      <td className="py-2 text-neutral-500">{signal.message}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}
        </div>
      )}
      </>
      )}
    </div>
  );
}
