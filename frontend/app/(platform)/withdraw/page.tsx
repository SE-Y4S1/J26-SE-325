"use client";

/**
 * The withdrawal planner — this component's novelty claim made visible.
 *
 * The TAF frames the contribution as operationalizing liquidity-aware withdrawal planning as
 * "a real-time, user-facing service". This screen is that service.
 */

import { Suspense, useState } from "react";

import { Button, Card, Field, Input, Notice, Spinner, Stat, money, percent } from "@/components/ui";
import { planWithdrawal } from "@/lib/api/portfolio";
import type { FuzzyRuleTraceEntry, WithdrawalResponse } from "@/lib/api/types";
import { usePortfolio } from "@/lib/usePortfolio";

function WithdrawScreen() {
  const { portfolio, loading, error } = usePortfolio();

  const [amount, setAmount] = useState(150_000);
  const [urgency, setUrgency] = useState(0.5);
  const [deadlineDays, setDeadlineDays] = useState(3);
  const [useAgent, setUseAgent] = useState(false);

  const [plan, setPlan] = useState<WithdrawalResponse | null>(null);
  const [planError, setPlanError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const totalValue = portfolio?.total_value ?? 0;

  async function handlePlan(event: React.FormEvent) {
    event.preventDefault();
    if (!portfolio) return;

    setBusy(true);
    setPlanError(null);
    setPlan(null);
    try {
      setPlan(
        await planWithdrawal({
          holdings: portfolio.holdings,
          targetAmount: amount,
          urgency,
          deadlineDays,
          useAgent,
        }),
      );
    } catch (cause) {
      setPlanError(cause instanceof Error ? cause.message : "Planning failed");
    } finally {
      setBusy(false);
    }
  }

  if (loading) return <Spinner label="Loading portfolio" />;
  if (error) return <Notice tone="error">{error}</Notice>;
  if (!portfolio) {
    return <Notice tone="info">Create a portfolio first, on the Portfolio screen.</Notice>;
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Instant withdrawal</h1>
        <p className="mt-1 text-sm text-neutral-500">
          Plans a loss-minimized liquidation across {portfolio.holdings.length} holdings worth{" "}
          {money(totalValue)}.
        </p>
      </div>

      <Card title="Request">
        <form onSubmit={handlePlan} className="grid gap-4 sm:grid-cols-2">
          <Field label="Amount to raise" hint={`Portfolio value ${money(totalValue)}`}>
            <Input
              type="number"
              value={amount}
              min={1}
              max={Math.floor(totalValue)}
              onChange={(e) => setAmount(Number(e.target.value))}
              required
            />
          </Field>

          <Field label="Deadline (trading days)" hint="Fewer days means higher market impact.">
            <Input
              type="number"
              value={deadlineDays}
              min={1}
              max={30}
              onChange={(e) => setDeadlineDays(Number(e.target.value))}
            />
          </Field>

          <Field
            label={`Urgency — ${urgency.toFixed(2)}`}
            hint="Fed to the fuzzy inference system as a linguistic variable, not a hard threshold."
          >
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={urgency}
              onChange={(e) => setUrgency(Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-neutral-500">
              <span>relaxed</span>
              <span>immediate</span>
            </div>
          </Field>

          <Field label="Options">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={useAgent}
                onChange={(e) => setUseAgent(e.target.checked)}
              />
              Route through the agent (adds a reasoning trace)
            </label>
          </Field>

          <div className="sm:col-span-2">
            <Button type="submit" disabled={busy}>
              {busy ? "Planning…" : "Plan withdrawal"}
            </Button>
          </div>
        </form>
      </Card>

      {planError && <Notice tone="error" title="Could not plan">{planError}</Notice>}
      {plan && <PlanResult plan={plan} />}
    </div>
  );
}

function PlanResult({ plan }: { plan: WithdrawalResponse }) {
  const trace = (plan.fuzzy_rule_trace ?? []) as unknown as FuzzyRuleTraceEntry[];

  return (
    <div className="space-y-6">
      {/* Infeasible is a legitimate answer, not a failure: the service returns HTTP 200 with
          feasible=false so the shortfall is observable. RQ4 depends on that being visible. */}
      {plan.feasible ? (
        <Notice tone="success" title="Feasible">
          The full {money(plan.target_amount)} can be raised in {plan.days_required}{" "}
          {plan.days_required === 1 ? "day" : "days"}.
        </Notice>
      ) : (
        <Notice tone="warn" title="Cannot raise the full amount">
          Only {money(plan.raised_amount)} of {money(plan.target_amount)} can be liquidated
          within the deadline — a shortfall of {money(plan.shortfall)}. This is limited by the
          daily participation cap on the least liquid holdings, not by an error.
        </Notice>
      )}

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label="Raised" value={money(plan.raised_amount)} />
        <Stat label="Expected slippage" value={percent(plan.expected_slippage_pct)} />
        <Stat label="Expected loss" value={money(plan.expected_realized_loss)} />
        <Stat label="Days required" value={plan.days_required} />
      </div>

      <Card title="Liquidation plan" subtitle="Execution order and day matter: spreading a sale reduces market impact.">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-left text-xs uppercase tracking-wide text-neutral-500">
              <tr>
                <th className="pb-2">Day</th>
                <th className="pb-2">Symbol</th>
                <th className="pb-2 text-right">Fraction</th>
                <th className="pb-2 text-right">Quantity</th>
                <th className="pb-2 text-right">Est. price</th>
                <th className="pb-2 text-right">Slippage</th>
              </tr>
            </thead>
            <tbody>
              {plan.assets_to_sell.map((sale, index) => (
                <tr key={index} className="border-t border-black/5 dark:border-white/10">
                  <td className="py-2">{sale.execution_day}</td>
                  <td className="py-2 font-medium">{sale.symbol}</td>
                  <td className="py-2 text-right tabular-nums">
                    {(sale.sell_fraction * 100).toFixed(2)}%
                  </td>
                  <td className="py-2 text-right tabular-nums">
                    {sale.quantity.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                  </td>
                  <td className="py-2 text-right tabular-nums">{money(sale.expected_price)}</td>
                  <td className="py-2 text-right tabular-nums">
                    {percent(sale.expected_slippage_pct, 4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {plan.assets_to_sell.length === 0 && (
          <p className="text-sm text-neutral-500">No sales were required.</p>
        )}
      </Card>

      {trace.length > 0 && (
        <Card
          title="Fuzzy rule trace"
          subtitle="Which rules fired, and how strongly. This is the auditable derivation behind every number above."
        >
          <div className="space-y-4">
            {trace.map((entry) => (
              <div key={entry.symbol}>
                <div className="flex items-baseline justify-between">
                  <span className="font-medium">{entry.symbol}</span>
                  <span className="text-sm text-neutral-500">
                    sell priority {entry.sell_priority.toFixed(1)}
                  </span>
                </div>
                <ul className="mt-1 space-y-1">
                  {entry.rules.map((rule) => (
                    <li key={rule.rule_id} className="text-xs">
                      <code className="rounded bg-black/5 px-1 py-0.5 dark:bg-white/10">
                        {rule.rule_id}
                      </code>{" "}
                      <span className="text-neutral-500">IF</span> {rule.if}{" "}
                      <span className="text-neutral-500">THEN</span> {rule.then}
                      <span className="ml-2 text-neutral-500">
                        (strength {rule.strength.toFixed(3)})
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </Card>
      )}

      {plan.agent_reasoning_trace && plan.agent_reasoning_trace.length > 0 && (
        <Card
          title="Agent reasoning"
          subtitle="Internal trace only. User-facing explanation is Component 4's responsibility."
        >
          <ol className="space-y-2 text-sm">
            {(plan.agent_reasoning_trace as Array<Record<string, unknown>>).map((step, i) => (
              <li key={i}>
                {typeof step.thought === "string" && <p>{step.thought}</p>}
                {typeof step.tool === "string" && (
                  <code className="text-xs text-neutral-500">called {step.tool}</code>
                )}
              </li>
            ))}
          </ol>
        </Card>
      )}

      <p className="text-xs text-neutral-500">
        Model version <code>{plan.model_version}</code> — anchored on-chain by Component 3.
      </p>
    </div>
  );
}

export default function Page() {
  return (
    <Suspense fallback={<Spinner label="Loading" />}>
      <WithdrawScreen />
    </Suspense>
  );
}
