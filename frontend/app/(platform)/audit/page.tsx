"use client";

/**
 * Audit overview: how decisions are being routed, whether the ledger and the policy in force
 * are what they should be, and the latest anchored decisions.
 */

import Link from "next/link";

import { AuditLoadNotice } from "@/components/audit/AuditLoadNotice";
import { DecisionTable } from "@/components/audit/DecisionTable";
import { RouteDistribution } from "@/components/audit/RouteDistribution";
import { shortHash } from "@/components/audit/badges";
import { useAuditLoad } from "@/components/audit/useAuditLoad";
import { Button, Card, Stat } from "@/components/ui";
import { auditPolicies, auditRecords, auditStats } from "@/lib/api/audit";

const pct = (part: number, whole: number) => (whole ? `${Math.round((part / whole) * 100)}%` : "—");

export default function AuditOverviewPage() {
  const state = useAuditLoad(async () => {
    const [stats, records, policies] = await Promise.all([auditStats(), auditRecords(), auditPolicies()]);
    return { stats, records, policies };
  });

  const notice = <AuditLoadNotice state={state} what="the audit trail" />;
  if (!state.data) return notice;
  const { stats, records, policies } = state.data;
  const active = policies.policies.find((p) => p.routesSignalRequests);
  const routes = stats.routes ?? { AUTO: 0, VERIFY: 0, HUMAN_REVIEW: 0, LEGACY: stats.totalDecisions };
  const routed = routes.AUTO + routes.VERIFY + routes.HUMAN_REVIEW;

  return (
    <div className="space-y-6">
      {notice}
      <div className="flex justify-end">
        <Button variant="secondary" onClick={state.refresh} disabled={state.loading}>
          {state.loading ? "Refreshing…" : "Refresh"}
        </Button>
      </div>

      <Card>
        <div className="grid grid-cols-2 gap-6 sm:grid-cols-4">
          <Stat label="Decisions" value={stats.totalDecisions} hint={`${routed} routed by the assurance engine`} />
          <Stat label="Automated" value={pct(routes.AUTO, routed)} hint={`${routes.AUTO} of ${routed} routed`} />
          <Stat label="Held for review" value={stats.pendingReview ?? 0} hint="verification or analyst" />
          <Stat
            label="Integrity"
            value={stats.integrityFailures ? `${stats.integrityFailures} failed` : "intact"}
            hint={`${stats.verified} verified on last check`}
          />
        </div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card title="How decisions were routed" subtitle="Share of all anchored decisions" className="lg:col-span-2">
          <RouteDistribution routes={routes} />
          {stats.signalQuality && stats.signalQuality.DEGRADED > 0 && (
            <p className="mt-4 text-xs text-neutral-500">
              {stats.signalQuality.DEGRADED} decision{stats.signalQuality.DEGRADED === 1 ? "" : "s"} had missing or
              unvalidated signals and were kept away from automatic processing.
            </p>
          )}
        </Card>

        <Card title="Ledger and policy">
          <dl className="space-y-3 text-sm">
            <div>
              <dt className="text-xs uppercase tracking-wide text-neutral-500">Ledger</dt>
              <dd>
                {stats.blockchainStatus.isMock
                  ? "Simulated ledger (no chain connected)"
                  : `EVM chain ${stats.blockchainStatus.chainId ?? ""}`}
              </dd>
              {stats.blockchainStatus.contractAddress && (
                <dd className="font-mono text-xs text-neutral-500">{shortHash(stats.blockchainStatus.contractAddress, 12)}</dd>
              )}
            </div>
            {active && (
              <div>
                <dt className="text-xs uppercase tracking-wide text-neutral-500">Policy in force</dt>
                <dd>
                  {active.id} v{active.version}
                  {active.status === "untuned" && <span className="ml-2 text-xs text-amber-700 dark:text-amber-400">untuned</span>}
                </dd>
                <dd className="font-mono text-xs text-neutral-500">{shortHash(active.policyHash, 12)}</dd>
                <dd className="text-xs text-neutral-500">
                  {active.onChain.active === true ? "Registered on-chain ✓" : active.onChain.active === false ? "Not registered on-chain" : "On-chain status unknown"}
                </dd>
              </div>
            )}
            <Link href="/audit/policies" className="inline-block text-sm underline underline-offset-2">
              Inspect policies
            </Link>
          </dl>
        </Card>
      </div>

      <Card title="Anchored decisions" subtitle="The five most recent">
        {records.length === 0 ? (
          <p className="text-sm text-neutral-500">
            No decisions anchored yet. Execute a withdrawal, try one on the Decisions tab, or run
            the tamper-evidence walkthrough on the Verify tab.
          </p>
        ) : (
          <>
            <DecisionTable records={records.slice(0, 5)} />
            <Link href="/audit/decisions" className="mt-4 inline-block text-sm underline underline-offset-2">
              All decisions
            </Link>
          </>
        )}
      </Card>
    </div>
  );
}
