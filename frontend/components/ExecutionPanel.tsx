"use client";

/**
 * Executes a Component 1 plan and shows it moving through the platform:
 *
 *   1 · Plan (Component 1)  ->  2 · Fraud gateway (Component 2)  ->  3 · Audit anchor (Component 3)
 *
 * Holdings change only when the platform service reports EXECUTED. Every stage shown here
 * is read from the execution trace the service stores; nothing is computed in the browser.
 */

import { useState } from "react";

import { RouteBadge, routeOf } from "@/components/audit/badges";
import { Button, Card, Field, Input, Notice, money } from "@/components/ui";
import {
  confirmExecution,
  createExecution,
  retryExecution,
  type Execution,
  type ExecutionKind,
  type ExecutionStatus,
} from "@/lib/api/executions";

const STATUS_TONE: Record<ExecutionStatus, "success" | "warn" | "error" | "info"> = {
  EXECUTED: "success",
  PENDING_REVIEW: "warn",
  HELD: "info",
  BLOCKED: "error",
  REJECTED: "error",
};

const STATUS_TITLE: Record<ExecutionStatus, string> = {
  EXECUTED: "Executed",
  PENDING_REVIEW: "Verification required",
  HELD: "Held — not executed",
  BLOCKED: "Blocked",
  REJECTED: "Rejected — not executed",
};

export function ExecutionPanel({
  portfolioId,
  kind,
  plan,
  disabledReason,
  onExecuted,
}: {
  portfolioId: number;
  kind: ExecutionKind;
  plan: unknown;
  /** Set when the plan should not be executed (e.g. nothing to sell). */
  disabledReason?: string;
  /** Called once holdings have changed, so the screen can reload the portfolio. */
  onExecuted?: () => void;
}) {
  const [execution, setExecution] = useState<Execution | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [password, setPassword] = useState("");

  async function run(action: () => Promise<Execution>) {
    setBusy(true);
    setError(null);
    try {
      const next = await action();
      setExecution(next);
      setPassword("");
      if (next.status === "EXECUTED") onExecuted?.();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The request failed");
    } finally {
      setBusy(false);
    }
  }

  const label = kind === "withdrawal" ? "Execute withdrawal" : "Execute rebalance";

  return (
    <Card
      title="Execute"
      subtitle="Screened by the fraud gateway (Component 2) and anchored by the audit bridge (Component 3) before any holding changes."
    >
      {!execution && (
        <div className="flex flex-wrap items-center gap-4">
          <Button
            onClick={() => run(() => createExecution(portfolioId, kind, plan))}
            disabled={busy || Boolean(disabledReason)}
          >
            {busy ? "Executing…" : label}
          </Button>
          {disabledReason && <span className="text-sm text-neutral-500">{disabledReason}</span>}
        </div>
      )}

      {error && (
        <div className="mt-4">
          <Notice tone="error" title="Could not complete">
            {error}
          </Notice>
        </div>
      )}

      {execution && (
        <div className="space-y-4" data-testid="execution-trace">
          <Notice tone={STATUS_TONE[execution.status]} title={STATUS_TITLE[execution.status]}>
            {execution.status_reason}
          </Notice>

          <ol className="grid gap-3 sm:grid-cols-3">
            <PlanStage execution={execution} />
            <FraudStageCard execution={execution} />
            <AuditStageCard execution={execution} />
          </ol>

          {execution.status === "PENDING_REVIEW" && (
            <form
              className="flex flex-wrap items-end gap-3"
              onSubmit={(event) => {
                event.preventDefault();
                run(() => confirmExecution(execution.id, password, true));
              }}
            >
              <Field label="Re-enter your password to verify it is you">
                <Input
                  type="password"
                  autoComplete="current-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </Field>
              <Button type="submit" disabled={busy || !password}>
                Verify and execute
              </Button>
              <Button
                type="button"
                variant="secondary"
                disabled={busy || !password}
                onClick={() => run(() => confirmExecution(execution.id, password, false))}
              >
                Reject
              </Button>
            </form>
          )}

          {execution.status === "HELD" && (
            <Button variant="secondary" disabled={busy} onClick={() => run(() => retryExecution(execution.id))}>
              {busy ? "Retrying…" : "Retry"}
            </Button>
          )}

          <p className="text-xs text-neutral-500">
            Execution <code>{execution.id}</code>. Ask the Assistant{" "}
            <q>Explain {execution.id}</q> for the plan, the fraud decision and the audit proof in
            plain language.
          </p>
        </div>
      )}
    </Card>
  );
}

function Stage({ step, title, children }: { step: number; title: string; children: React.ReactNode }) {
  return (
    <li className="rounded-lg border border-black/10 p-3 text-sm dark:border-white/15">
      <p className="text-xs uppercase tracking-wide text-neutral-500">
        {step} · {title}
      </p>
      <div className="mt-2 space-y-1">{children}</div>
    </li>
  );
}

function PlanStage({ execution }: { execution: Execution }) {
  return (
    <Stage step={1} title="Plan · Component 1">
      <p className="font-medium">{money(execution.amount, execution.currency)}</p>
      <ul className="text-xs text-neutral-500">
        {execution.orders.map((order) => (
          <li key={`${order.side}-${order.symbol}`}>
            {order.side} {order.quantity.toLocaleString(undefined, { maximumFractionDigits: 4 })}{" "}
            {order.symbol}
          </li>
        ))}
      </ul>
      <p className="text-xs text-neutral-500">
        model <code>{execution.source_model_version}</code>
      </p>
    </Stage>
  );
}

function FraudStageCard({ execution }: { execution: Execution }) {
  const fraud = execution.fraud;
  if (!fraud || !fraud.decision) {
    return (
      <Stage step={2} title="Fraud gateway · Component 2">
        <p className="font-medium">No decision</p>
        <p className="text-xs text-neutral-500">{fraud?.error ?? "The fraud service did not answer."}</p>
      </Stage>
    );
  }
  const rules = fraud.explanation?.rules ?? [];
  return (
    <Stage step={2} title="Fraud gateway · Component 2">
      <p className="font-medium">
        {fraud.decision} · score {(fraud.fraud_score ?? 0).toFixed(3)}
      </p>
      {rules.length > 0 && <p className="text-xs">{rules.join(" · ")}</p>}
      <p className="text-xs text-neutral-500">
        model <code>{fraud.model_version}</code>
        {fraud.benchmark_model ? "" : " (not a benchmark model)"}
      </p>
    </Stage>
  );
}

function AuditStageCard({ execution }: { execution: Execution }) {
  const audit = execution.audit;
  const review = audit?.review;
  const record = review?.record ?? audit?.record;
  if (!audit) {
    return (
      <Stage step={3} title="Audit anchor · Component 3">
        <p className="font-medium">Not anchored</p>
        <p className="text-xs text-neutral-500">Nothing reaches the ledger without a fraud decision.</p>
      </Stage>
    );
  }
  const assurance = audit.assurance;
  return (
    <Stage step={3} title="Audit anchor · Component 3">
      <p className="font-medium">{record?.finalDecision ?? audit.status}</p>
      {assurance?.route && (
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <RouteBadge route={routeOf({ route: assurance.route, signalQuality: assurance.signalQuality ?? "FULL" })} />
          {assurance.assuranceScore != null && <span className="tabular-nums text-neutral-500">assurance {assurance.assuranceScore.toFixed(2)}</span>}
          {assurance.signalQuality === "DEGRADED" && <span className="text-neutral-500">degraded signals</span>}
        </div>
      )}
      {record?.recordHash && record.recordHash.startsWith("0x") && (
        <p className="break-all text-xs">
          hash <code>{record.recordHash.slice(0, 18)}…</code>
        </p>
      )}
      <p className="text-xs text-neutral-500">
        {record?.verificationStatus ?? audit.message ?? audit.status}
      </p>
      <p className="break-all text-xs text-neutral-500">
        plan hash <code>{execution.plan_hash.slice(0, 18)}…</code>
      </p>
    </Stage>
  );
}
