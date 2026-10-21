"use client";

/**
 * The research result: does the assurance engine automate fewer incorrect decisions than
 * simpler routing, at the same human-review workload? Read from the IEEE-CIS evaluation's
 * aggregate output; nothing here is computed in the browser.
 */

import { useState } from "react";

import { AuditLoadNotice } from "@/components/audit/AuditLoadNotice";
import { DeferralChart } from "@/components/audit/DeferralChart";
import { useAuditLoad } from "@/components/audit/useAuditLoad";
import { Card, Notice } from "@/components/ui";
import { researchResults } from "@/lib/api/audit";

const LABELS: Record<string, string> = {
  assurance: "Assurance engine (P002)",
  assurance_no_confidence: "ablation: no confidence",
  assurance_no_disagreement: "ablation: no disagreement",
  assurance_no_context: "ablation: no context",
  learned_deferral: "Learned deferral (logistic)",
  risk_band: "Risk only (fraud-gateway style)",
  confidence_only: "Confidence only",
  p001_retuned: "P001 thresholds, re-tuned",
  p001_deployed: "P001 as deployed",
  random_review: "Random review",
  auto_all: "Automate everything",
};
const ORDER = Object.keys(LABELS);
const pct = (v: number, d = 2) => `${(v * 100).toFixed(d)}%`;

export default function EvaluationPage() {
  const state = useAuditLoad(researchResults);
  const [budget, setBudget] = useState<string | null>(null);

  if (!state.data) return <AuditLoadNotice state={state} what="the evaluation" />;
  const r = state.data;

  if (r.status === "NOT_RUN" || !r.results) {
    return (
      <Card title="IEEE-CIS evaluation" subtitle="Assurance engine vs simpler decision routing, at matched human-review budgets">
        <Notice tone="info" title="Not run yet">
          No result is shown until the evaluation has run on the real dataset. Place the Kaggle IEEE-CIS files in{" "}
          <code>backend/Fraud-Detection/data/raw/ieee-cis/</code>, then from <code>backend/Blockchain-Auditability/research</code>{" "}
          run <code>uv run python -m assurance.run tune</code>, commit the exported policy, and run{" "}
          <code>uv run python -m assurance.run final</code>.
        </Notice>
        <dl className="mt-4 grid gap-3 text-sm sm:grid-cols-2">
          <div><dt className="font-medium">Compared</dt><dd className="text-neutral-600 dark:text-neutral-400">assurance engine and its ablations, learned deferral, risk-only, confidence-only, P001 (as deployed and re-tuned), random review, automate-everything</dd></div>
          <div><dt className="font-medium">Primary metric</dt><dd className="text-neutral-600 dark:text-neutral-400">incorrect automatic decisions, with VERIFY counted as automatic (no credit for step-up)</dd></div>
          <div><dt className="font-medium">Budgets</dt><dd className="text-neutral-600 dark:text-neutral-400">5, 10, 15 and 20% of decisions to human review</dd></div>
          <div><dt className="font-medium">Rigour</dt><dd className="text-neutral-600 dark:text-neutral-400">time-ordered splits, test split used once, 95% bootstrap intervals, paired tests</dd></div>
        </dl>
      </Card>
    );
  }

  const budgets = Object.keys(r.results).sort((a, b) => Number(a) - Number(b));
  const active = budget ?? String(r.policyBudget ?? budgets[0]);
  const block = r.results[active];
  const methods = ORDER.filter((m) => block.methods[m]);

  return (
    <div className="space-y-6">
      {r.status !== "FINAL" && (
        <Notice tone="warn" title="Not a final result">This run evaluated an uncommitted policy (a smoke test); do not report it.</Notice>
      )}

      <Card title="Incorrect automatic decisions vs review workload" subtitle="Held-out test split; each method defers in its own order">
        <DeferralChart curves={r.curves ?? {}} />
      </Card>

      <Card title="At a fixed human-review budget" subtitle={`Policy ${r.policy?.id} v${r.policy?.version} · ${r.data?.rows.test.toLocaleString()} test transactions`}>
        <div className="mb-4 flex flex-wrap gap-1.5" role="group" aria-label="Human-review budget">
          {budgets.map((b) => (
            <button key={b} type="button" onClick={() => setBudget(b)} aria-pressed={active === b}
              className={`rounded-full px-3 py-1 text-xs transition ${active === b ? "bg-neutral-900 text-white dark:bg-white dark:text-neutral-900" : "border border-black/15 hover:bg-black/5 dark:border-white/20 dark:hover:bg-white/10"}`}>
              {pct(Number(b), 0)} review
            </button>
          ))}
        </div>
        <div className="-mx-5 overflow-x-auto">
          <table className="w-full min-w-[720px] text-sm">
            <thead>
              <tr className="border-b border-black/10 text-left text-xs uppercase tracking-wide text-neutral-500 dark:border-white/15">
                <th className="px-5 py-2 font-medium">Method</th>
                <th className="px-3 py-2 text-right font-medium">Incorrect automatic</th>
                <th className="px-3 py-2 font-medium">95% CI</th>
                <th className="px-3 py-2 text-right font-medium">Human review</th>
                <th className="px-3 py-2 text-right font-medium">Verify</th>
                <th className="px-5 py-2 text-right font-medium">Fraud auto-approved</th>
              </tr>
            </thead>
            <tbody>
              {methods.map((m) => {
                const v = block.methods[m];
                return (
                  <tr key={m} className={`border-b border-black/5 last:border-0 dark:border-white/10 ${m === "assurance" ? "font-medium" : ""}`}>
                    <td className={`px-5 py-2 ${m.startsWith("assurance_") ? "pl-9 text-neutral-500" : ""}`}>{LABELS[m]}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{pct(v.incorrect_automatic_rate, 3)}</td>
                    <td className="px-3 py-2 tabular-nums text-neutral-500">{pct(v.ci95.incorrect_automatic_rate[0], 3)} – {pct(v.ci95.incorrect_automatic_rate[1], 3)}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{pct(v.human_review_rate)}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{pct(v.verify_rate)}</td>
                    <td className="px-5 py-2 text-right tabular-nums">{v.fraud_auto_approved.toLocaleString()}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Is the difference real?" subtitle="Paired bootstrap: assurance engine minus each comparator (negative = fewer incorrect automatic decisions)">
        <ul className="space-y-1.5 text-sm">
          {Object.entries(block.paired).map(([m, p]) => (
            <li key={m} className="flex flex-wrap justify-between gap-2">
              <span>vs {LABELS[m] ?? m}</span>
              <span className="tabular-nums text-neutral-600 dark:text-neutral-400">
                {pct(p.difference, 3)} [{pct(p.ci95[0], 3)}, {pct(p.ci95[1], 3)}] · P(not better) {p.p_reference_not_better.toFixed(3)}
              </span>
            </li>
          ))}
        </ul>
        <p className="mt-4 text-xs text-neutral-500">
          Assumptions, not measurements: VERIFY is counted as automatic in the primary metric; how often step-up
          catches an error and how accurate analysts are are reported as sensitivity sweeps in the full report.
        </p>
      </Card>
    </div>
  );
}
