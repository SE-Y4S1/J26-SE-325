"use client";

import Link from "next/link";

import { money } from "@/components/ui";
import type { AuditRecord } from "@/lib/api/audit";

import { IntegrityBadge, RouteBadge, outcomeLabel, routeOf } from "./badges";

/** Anchored decisions, newest first; each row links to its detail page. */
export function DecisionTable({ records }: { records: AuditRecord[] }) {
  return (
    <div className="-mx-5 overflow-x-auto">
      <table className="w-full min-w-[720px] text-sm">
        <thead>
          <tr className="border-b border-black/10 text-left text-xs uppercase tracking-wide text-neutral-500 dark:border-white/15">
            <th className="px-5 py-2 font-medium">Transaction</th>
            <th className="px-3 py-2 text-right font-medium">Amount</th>
            <th className="px-3 py-2 font-medium">Route</th>
            <th className="px-3 py-2 text-right font-medium">Assurance</th>
            <th className="px-3 py-2 font-medium">Outcome</th>
            <th className="px-5 py-2 font-medium">Integrity</th>
          </tr>
        </thead>
        <tbody>
          {records.map((r) => (
            <tr key={r.transactionId} className="border-b border-black/5 last:border-0 dark:border-white/10">
              <td className="px-5 py-2.5">
                <Link href={`/audit/decisions/${encodeURIComponent(r.transactionId)}`} className="font-mono text-xs underline-offset-2 hover:underline">
                  {r.transactionId.length > 24 ? `${r.transactionId.slice(0, 24)}…` : r.transactionId}
                </Link>
                <p className="text-xs text-neutral-500">{new Date(r.timestamp).toLocaleString()}</p>
              </td>
              <td className="px-3 py-2.5 text-right tabular-nums">{money(r.amount)}</td>
              <td className="px-3 py-2.5"><RouteBadge route={routeOf(r.assurance)} /></td>
              <td className="px-3 py-2.5 text-right tabular-nums">
                {r.assurance?.assuranceScore != null ? r.assurance.assuranceScore.toFixed(2) : "—"}
              </td>
              <td className="px-3 py-2.5">{outcomeLabel(r.finalDecision)}</td>
              <td className="px-5 py-2.5"><IntegrityBadge status={r.verificationStatus} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
