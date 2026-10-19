"use client";

/**
 * The withdrawal planner — this component's novelty claim made visible.
 *
 * The TAF frames the contribution as operationalizing liquidity-aware withdrawal planning as
 * "a real-time, user-facing service". This screen is that service.
 */

import { Suspense, useState } from "react";

import { BaselineComparison, CostBreakdownCard, MarketContextPanel } from "@/components/analysis/MarketContext";
import { Button, Card, Field, Input, Notice, Spinner, Stat, money, percent } from "@/components/ui";
import { planWithdrawal } from "@/lib/api/portfolio";
import type {
  BaselineRow,
  CostBreakdown,
  CurrencyBlock,
  FuzzyRuleTraceEntry,
  MarketContext,
  WithdrawalResponse,
} from "@/lib/api/types";

/** Currencies the plan can be expressed in. Holdings keep their own trading currency (CSE
 *  LKR, US USD, Binance USDT); the service converts each into this one. */
const BASE_CURRENCIES = ["LKR", "USD"] as const;
import { usePortfolio } from "@/lib/usePortfolio";

function WithdrawScreen() {
  const { portfolio, loading, error } = usePortfolio();

  const [amount, setAmount] = useState(150_000);
  const [urgency, setUrgency] = useState(0.5);
  const [deadlineDays, setDeadlineDays] = useState(3);
  const [riskTolerance, setRiskTolerance] = useState(0.5);
  const [baseCurrency, setBaseCurrency] = useState<string>("LKR");
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
          riskTolerance,
          deadlineDays,
          useAgent,
          baseCurrency,
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
          Plans a loss-minimized liquidation across {portfolio.holdings.length} holdings, each in its own
          trading currency, converted to {baseCurrency} with the current rate and conversion spread.
        </p>
      </div>

      <Card title="Request">
        <form onSubmit={handlePlan} className="grid gap-4 sm:grid-cols-2">
          <Field
            label={`Amount to raise (${baseCurrency})`}
            hint={`Sum of native prices ${totalValue.toLocaleString()} — the service checks the converted total.`}
          >
            <div className="flex gap-2">
              <Input
                type="number"
                value={amount}
                min={1}
                onChange={(e) => setAmount(Number(e.target.value))}
                required
              />
              <select
                aria-label="Base currency"
                value={baseCurrency}
                onChange={(e) => setBaseCurrency(e.target.value)}
                className="rounded-lg border border-black/15 bg-transparent px-2 text-sm dark:border-white/20"
              >
                {BASE_CURRENCIES.map((code) => (
                  <option key={code} value={code}>
                    {code}
                  </option>
                ))}
              </select>
            </div>
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

          <Field
            label={`Risk tolerance — ${riskTolerance.toFixed(2)}`}
            hint="Higher waits for cheaper fills and accepts more price risk; lower sells sooner."
          >
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={riskTolerance}
              onChange={(e) => setRiskTolerance(Number(e.target.value))}
              className="w-full"
            />
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

// Exported for tests: the feasible/infeasible distinction is a research-relevant
// behaviour, and asserting it through the whole page would mean mocking auth, the
// portfolio hook and the API to reach it.
export function PlanResult({ plan }: { plan: WithdrawalResponse }) {
  const trace = (plan.fuzzy_rule_trace ?? []) as unknown as FuzzyRuleTraceEntry[];
  const currencyBlock = (plan.currency ?? null) as CurrencyBlock | null;
  // Older responses carry no currency block; they were single-currency, shown as before.
  const currency = currencyBlock?.base ?? "USD";
  const fmt = (value: number) => money(value, currency);
  const contexts = (plan.market_context ?? {}) as unknown as Record<string, MarketContext>;
  const baselines = (plan.baseline_comparison ?? []) as unknown as BaselineRow[];
  const breakdown = (plan.cost_breakdown ?? null) as CostBreakdown | null;
  const inputs = (plan.effective_inputs ?? null) as Record<string, unknown> | null;
  const decision = (inputs?.decision ?? null) as Record<string, unknown> | null;

  return (
    <div className="space-y-6">
      {/* Infeasible is a legitimate answer, not a failure: the service returns HTTP 200 with
          feasible=false so the shortfall is observable. RQ4 depends on that being visible. */}
      {plan.feasible ? (
        <Notice tone="success" title="Feasible">
          The full {fmt(plan.target_amount)} can be raised in {plan.days_required}{" "}
          {plan.days_required === 1 ? "day" : "days"}.
        </Notice>
      ) : (
        <Notice tone="warn" title="Cannot raise the full amount">
          Only {fmt(plan.raised_amount)} of {fmt(plan.target_amount)} can be liquidated
          within the deadline — a shortfall of {fmt(plan.shortfall)}. This is limited by the
          daily participation cap on the least liquid holdings, not by an error.
        </Notice>
      )}

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label="Raised" value={fmt(plan.raised_amount)} />
        <Stat label="Expected slippage" value={percent(plan.expected_slippage_pct)} />
        <Stat label="Expected cost" value={fmt(plan.expected_realized_loss)} />
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
                  <td className="py-2 text-right tabular-nums">{fmt(sale.expected_price)}</td>
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

      {currencyBlock && Object.keys(currencyBlock.rates).length > 0 && (
        <p className="text-xs text-neutral-500">
          Converted to {currencyBlock.base}:{" "}
          {Object.entries(currencyBlock.rates)
            .map(([code, r]) => `1 ${code} = ${r.rate.toFixed(2)} (${r.source}, spread ${(r.spread * 100).toFixed(2)}%)`)
            .join(" · ")}
        </p>
      )}

      {inputs && (
        <p className="text-xs text-neutral-500" data-testid="effective-inputs">
          Planned with urgency {Number(inputs.urgency).toFixed(2)}, risk tolerance{" "}
          {Number(inputs.risk_tolerance).toFixed(2)}, deadline {String(inputs.deadline_days)}d · decisions:{" "}
          {decision?.source === "laya"
            ? `Laya (${String(decision.model_version)})`
            : `fuzzy rules${decision?.reason ? ` — ${String(decision.reason)}` : ""}`}
        </p>
      )}

      {breakdown && <CostBreakdownCard breakdown={breakdown} currency={currency} />}
      <BaselineComparison rows={baselines} currency={currency} />
      <MarketContextPanel contexts={contexts} />

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
                    {entry.signal_adjustment ? ` · market signals ${entry.signal_adjustment > 0 ? "+" : ""}${entry.signal_adjustment.toFixed(1)}` : ""}
                    {entry.laya_adjustment ? ` · Laya ${entry.laya_adjustment > 0 ? "+" : ""}${entry.laya_adjustment.toFixed(0)}` : ""}
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
                  {(entry.signal_rules ?? []).map((rule) => (
                    <li key={rule.rule_id} className="text-xs">
                      <code className="rounded bg-amber-500/10 px-1 py-0.5">{rule.rule_id}</code>{" "}
                      <span className="text-neutral-500">IF</span> {rule.if}{" "}
                      <span className="text-neutral-500">THEN</span> {rule.then}
                      <span className="ml-2 text-neutral-500">(strength {rule.strength.toFixed(3)})</span>
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
