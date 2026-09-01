"use client";

/**
 * Component 2's adversarial robustness demonstration.
 *
 * This is the evidence behind her second research gap — that benchmarks assume static,
 * non-evasive fraudsters, and a dual-stream model survives evasion a single-stream one does
 * not. The engine reports `single_stream_would_have_missed` directly, so the ablation is a
 * fact from the service rather than a claim made by this page.
 */

import { useState } from "react";

import { Button, Card, Field, Notice } from "@/components/ui";
import { ApiError } from "@/lib/api/client";
import { simulateAttack, type AttackResponse, type AttackType } from "@/lib/api/fraud";

const ATTACKS: { value: AttackType; label: string; blurb: string }[] = [
  {
    value: "camouflage",
    label: "Camouflage",
    blurb: "The attacker imitates normal behaviour to suppress the behavioural signal.",
  },
  {
    value: "slow_drift",
    label: "Slow drift",
    blurb: "Risk is introduced gradually, so no single step looks anomalous.",
  },
  {
    value: "structuring",
    label: "Structuring",
    blurb: "One large transfer is split into many small ones to stay under thresholds.",
  },
];

export function AttackSimulator() {
  const [attack, setAttack] = useState<AttackType>("camouflage");
  const [result, setResult] = useState<AttackResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [unavailable, setUnavailable] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    setUnavailable(false);
    try {
      setResult(await simulateAttack(attack, 8));
    } catch (cause) {
      setResult(null);
      if (cause instanceof ApiError && (cause.isUnavailable || cause.status === 0)) {
        setUnavailable(true);
      } else {
        setError(cause instanceof Error ? cause.message : "The simulation failed");
      }
    } finally {
      setBusy(false);
    }
  }

  const selected = ATTACKS.find((a) => a.value === attack);

  return (
    <div className="space-y-4">
      <Card
        title="Adversarial robustness"
        subtitle="Static benchmarks assume fraudsters do not adapt. These three attacks assume they do."
      >
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Attack" hint={selected?.blurb}>
            <select
              className="w-full rounded border border-neutral-300 bg-transparent px-3 py-2 text-sm"
              value={attack}
              onChange={(e) => setAttack(e.target.value as AttackType)}
            >
              {ATTACKS.map((a) => (
                <option key={a.value} value={a.value}>
                  {a.label}
                </option>
              ))}
            </select>
          </Field>
          <div className="flex items-end">
            <Button onClick={run} disabled={busy}>
              {busy ? "Simulating…" : "Run attack"}
            </Button>
          </div>
        </div>
      </Card>

      {unavailable && (
        <Notice tone="info" title="Fraud service is not running">
          Start it with <code>docker compose up component2</code>.
        </Notice>
      )}

      {error && (
        <Notice tone="error" title="Simulation failed">
          {error}
        </Notice>
      )}

      {result && (
        <div className="space-y-4">
          <Notice tone={result.detected ? "success" : "error"} title={result.title}>
            {result.summary}
          </Notice>

          {result.single_stream_would_have_missed && (
            // The whole argument for two streams, stated by the engine itself.
            <Notice tone="warn" title="A single-stream model would have missed this">
              The behavioural score was successfully suppressed by the attack, but the
              relational (graph) score was not — so the fused decision still caught it. This
              is the per-stream ablation her design predicts.
            </Notice>
          )}

          <Card
            title="Step by step"
            subtitle="Watch the behavioural score fall while the graph score holds."
          >
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="text-left text-xs uppercase tracking-wide text-neutral-500">
                  <tr>
                    <th className="pb-2">Step</th>
                    <th className="pb-2">Stage</th>
                    <th className="pb-2 text-right">Behavioural</th>
                    <th className="pb-2 text-right">Graph</th>
                    <th className="pb-2 text-right">Fused</th>
                    <th className="pb-2">Decision</th>
                  </tr>
                </thead>
                <tbody>
                  {[result.baseline, ...result.steps].map((step, index) => (
                    <tr
                      key={`${step.step}-${index}`}
                      className={index === 0 ? "border-t-2 border-neutral-300" : "border-t border-neutral-200/60"}
                    >
                      <td className="py-2">{index === 0 ? "base" : step.step}</td>
                      <td className="py-2 text-neutral-500">{step.label}</td>
                      <td className="py-2 text-right font-mono">
                        {step.behavioral_score.toFixed(3)}
                      </td>
                      <td className="py-2 text-right font-mono">{step.graph_score.toFixed(3)}</td>
                      <td className="py-2 text-right font-mono">{step.risk_score.toFixed(3)}</td>
                      <td className="py-2">{step.decision}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-xs text-neutral-500">{result.disclaimer}</p>
          </Card>
        </div>
      )}
    </div>
  );
}
