"use client";

/**
 * The policies that route decisions, as the documents whose hashes are on-chain. Anyone can
 * recompute a hash from the document shown here; a record commits to the hash of the policy
 * that routed it.
 */

import { AuditLoadNotice } from "@/components/audit/AuditLoadNotice";
import { useAuditLoad } from "@/components/audit/useAuditLoad";
import { Card, Notice } from "@/components/ui";
import { auditPolicies } from "@/lib/api/audit";

export default function PoliciesPage() {
  const state = useAuditLoad(auditPolicies);

  return (
    <div className="space-y-6">
      <AuditLoadNotice state={state} what="policies" />
      {state.data && (
        <>
          <Notice tone="info" title="Evaluation not run yet">
            P002&apos;s weights and thresholds are hand-set and conservative. They will be tuned and
            evaluated on the IEEE-CIS dataset against simpler routing baselines; until then no
            claim is made about its error rate or review workload.
          </Notice>
          {state.data.policies.map((p) => (
            <Card key={`${p.id}@${p.version}`} title={`${p.id} v${p.version}${p.routesSignalRequests ? " · in force" : ""}`} subtitle={p.description}>
              <dl className="grid gap-3 text-sm sm:grid-cols-4">
                <div><dt className="text-xs uppercase tracking-wide text-neutral-500">Engine</dt><dd>{p.engine}</dd></div>
                <div><dt className="text-xs uppercase tracking-wide text-neutral-500">Status</dt><dd>{p.baseline ? "baseline" : p.status}</dd></div>
                <div><dt className="text-xs uppercase tracking-wide text-neutral-500">Hash scheme</dt><dd>{p.hashScheme}</dd></div>
                <div><dt className="text-xs uppercase tracking-wide text-neutral-500">On-chain</dt>
                  <dd>{p.onChain.active === true ? "registered ✓" : p.onChain.active === false ? "not registered" : "unknown"}</dd></div>
                <div className="sm:col-span-4"><dt className="text-xs uppercase tracking-wide text-neutral-500">Policy hash</dt>
                  <dd className="break-all font-mono text-xs">{p.policyHash}</dd></div>
              </dl>
              <details className="mt-4">
                <summary className="cursor-pointer text-sm text-neutral-500">Policy document</summary>
                <pre className="mt-2 max-h-96 overflow-auto rounded-lg bg-neutral-50 p-3 text-xs dark:bg-neutral-950">{JSON.stringify(p.document, null, 2)}</pre>
              </details>
            </Card>
          ))}
        </>
      )}
    </div>
  );
}
