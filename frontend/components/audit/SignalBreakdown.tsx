"use client";

/**
 * Why a decision got its route: the three uncertainties the engine combined, the score
 * against the policy's thresholds, and any veto or degraded signal.
 *
 * Each signal is its own labelled bar on the same 0-1 scale (longer = less certain), with
 * its value printed, so nothing is read from colour. Bars use the neutral ink; only the score
 * scale uses the route colours, because there they mean the route.
 */

import type { Assurance } from "@/lib/api/audit";
import { ROUTE_FILL } from "./badges";

const SIGNALS: { key: "confidence" | "disagreement" | "context"; label: string; help: string }[] = [
  { key: "confidence", label: "Confidence", help: "How close the score is to the decision boundary" },
  { key: "disagreement", label: "Model disagreement", help: "Spread between independent models" },
  { key: "context", label: "Contextual risk", help: "Amount, fraud-gateway rules, model validation, type" },
];

const fmt = (x: number) => x.toFixed(2);

export function SignalBreakdown({
  assurance,
  thresholds,
}: {
  assurance: Assurance;
  thresholds?: { autoMin: number; verifyMin: number };
}) {
  if (!assurance.components || assurance.assuranceScore === null) {
    return (
      <p className="text-sm text-neutral-600 dark:text-neutral-400">
        Routed by the legacy P001 policy (risk and confidence thresholds). The caller sent no
        assurance signals, so no confidence, disagreement or context breakdown exists.
      </p>
    );
  }
  const degraded = new Set(assurance.degradedSignals.map((s) => s.split(".")[0]));
  const unreliable = new Set(assurance.unreliableSignals);

  return (
    <div className="space-y-5">
      <div>
        <div className="flex items-baseline justify-between">
          <p className="text-sm font-medium">Assurance score</p>
          <p className="text-xl font-semibold tabular-nums">{fmt(assurance.assuranceScore)}</p>
        </div>
        {thresholds && (
          <ScoreScale score={assurance.assuranceScore} {...thresholds} />
        )}
      </div>

      <ul className="space-y-3" aria-label="Uncertainty by signal (0 = certain, 1 = uncertain)">
        {SIGNALS.map(({ key, label, help }) => {
          const value = assurance.components![key];
          const flag = degraded.has(key) ? "missing — scored conservatively" : unreliable.has(key) ? "unvalidated model — floored" : null;
          return (
            <li key={key}>
              <div className="flex items-baseline justify-between gap-3 text-sm">
                <span>
                  {label}
                  <span className="ml-2 text-xs text-neutral-500">{help}</span>
                </span>
                <span className="tabular-nums">{fmt(value)}</span>
              </div>
              <div className="mt-1 h-1.5 w-full rounded bg-black/5 dark:bg-white/10">
                <div className="h-full rounded bg-neutral-700 dark:bg-neutral-300" style={{ width: `${Math.max(value * 100, 1)}%` }} />
              </div>
              {flag && <p className="mt-1 text-xs text-amber-700 dark:text-amber-400">{flag}</p>}
            </li>
          );
        })}
      </ul>

      {assurance.vetoes.length > 0 && (
        <div>
          <p className="text-sm font-medium">Vetoes</p>
          <ul className="mt-1 space-y-1 text-sm text-neutral-600 dark:text-neutral-400">
            {assurance.vetoes.map((v) => (
              <li key={v.code}><span className="font-mono text-xs">{v.code}</span> · {v.detail}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

/** The 0-1 score axis split at the policy's two thresholds, with the score marked on it. */
function ScoreScale({ score, autoMin, verifyMin }: { score: number; autoMin: number; verifyMin: number }) {
  const zones = [
    { from: 0, to: verifyMin, fill: ROUTE_FILL.HUMAN_REVIEW, label: "Human review" },
    { from: verifyMin, to: autoMin, fill: ROUTE_FILL.VERIFY, label: "Verify" },
    { from: autoMin, to: 1, fill: ROUTE_FILL.AUTO, label: "Automatic" },
  ];
  return (
    <div className="mt-2">
      <div className="relative">
        <div className="flex h-2 gap-0.5">
          {zones.map((z) => (
            <div key={z.label} className={`h-full opacity-30 first:rounded-l last:rounded-r ${z.fill}`} style={{ width: `${(z.to - z.from) * 100}%` }} />
          ))}
        </div>
        <div
          className="absolute -top-1 h-4 w-1 -translate-x-1/2 rounded-full bg-neutral-900 ring-2 ring-white dark:bg-white dark:ring-neutral-900"
          style={{ left: `${score * 100}%` }}
          title={`Score ${fmt(score)}`}
        />
      </div>
      <div className="mt-1 flex text-xs text-neutral-500">
        {zones.map((z) => (
          <span key={z.label} style={{ width: `${(z.to - z.from) * 100}%` }} className="truncate">
            {z.label} {z.from > 0 ? `≥ ${fmt(z.from)}` : ""}
          </span>
        ))}
      </div>
    </div>
  );
}
