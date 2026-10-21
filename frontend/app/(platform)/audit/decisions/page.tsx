"use client";

import { useState } from "react";

import { AuditLoadNotice } from "@/components/audit/AuditLoadNotice";
import { DecisionTable } from "@/components/audit/DecisionTable";
import { TryDecision } from "@/components/audit/TryDecision";
import { ROUTE_LABEL, routeOf } from "@/components/audit/badges";
import { useAuditLoad } from "@/components/audit/useAuditLoad";
import { Card } from "@/components/ui";
import { auditRecords, type Route } from "@/lib/api/audit";

type Filter = "ALL" | Route | "LEGACY";
const FILTERS: Filter[] = ["ALL", "AUTO", "VERIFY", "HUMAN_REVIEW", "LEGACY"];

export default function DecisionsPage() {
  const state = useAuditLoad(auditRecords);
  const [filter, setFilter] = useState<Filter>("ALL");
  const records = (state.data ?? []).filter((r) => filter === "ALL" || routeOf(r.assurance) === filter);

  return (
    <div className="space-y-6">
      <TryDecision onAnchored={state.refresh} />

      <Card title="Anchored decisions" subtitle="Open one to see why it was routed that way and to export its evidence.">
        <AuditLoadNotice state={state} what="decisions" />
        {state.data && (
          <>
            <div className="mb-4 flex flex-wrap gap-1.5" role="group" aria-label="Filter by route">
              {FILTERS.map((f) => (
                <button
                  key={f}
                  type="button"
                  onClick={() => setFilter(f)}
                  aria-pressed={filter === f}
                  className={`rounded-full px-3 py-1 text-xs transition ${
                    filter === f
                      ? "bg-neutral-900 text-white dark:bg-white dark:text-neutral-900"
                      : "border border-black/15 hover:bg-black/5 dark:border-white/20 dark:hover:bg-white/10"
                  }`}
                >
                  {f === "ALL" ? "All" : ROUTE_LABEL[f]}
                </button>
              ))}
            </div>
            {records.length === 0 ? (
              <p className="text-sm text-neutral-500">
                {state.data.length === 0 ? "No decisions anchored yet." : "No decisions with this route."}
              </p>
            ) : (
              <DecisionTable records={records} />
            )}
          </>
        )}
      </Card>
    </div>
  );
}
