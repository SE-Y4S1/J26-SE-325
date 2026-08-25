"use client";

/**
 * Long-term allocation via MOEA/D.
 *
 * The selection-rule switch is the interesting part: choosing a single point from a Pareto
 * front is a value judgement, so all three rules are exposed and RQ2's sensitivity analysis
 * becomes something you can see rather than something buried in a notebook.
 */

import { Suspense, useState } from "react";

import { Button, Card, Field, Notice, Spinner, Stat, percent } from "@/components/ui";
import { optimizeAllocation, type SelectionRule } from "@/lib/api/portfolio";
import type { OptimizeResponse } from "@/lib/api/types";
import { usePortfolio } from "@/lib/usePortfolio";

const RULES: Array<{ value: SelectionRule; label: string; blurb: string }> = [
  {
    value: "knee",
    label: "Knee point",
    blurb:
      "Maximum trade-off curvature — where giving up more return stops buying a proportionate reduction in risk. Parameter-free, so no preference weights have to be justified.",
  },
  {
    value: "max_sharpe",
    label: "Max Sharpe",
    blurb:
      "Best return per unit of CVaR. Directly comparable to the mean-variance tangency portfolio, though it optimizes the same metric RQ2 scores on.",
  },
  {
    value: "scalarized",
    label: "Scalarized",
    blurb:
      "Weighted sum under explicit preferences. Maps to a real risk-profile slider, but the weights are an input someone must defend.",
  },
];

function OptimizeScreen() {
  const { portfolio, loading, error } = usePortfolio();

  const [rule, setRule] = useState<SelectionRule>("knee");
  const [riskPreference, setRiskPreference] = useState(0.5);
  const [maxWeight, setMaxWeight] = useState(0.25);

  const [result, setResult] = useState<OptimizeResponse | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function handleRun() {
    if (!portfolio) return;
    setBusy(true);
    setRunError(null);
    setResult(null);
    try {
      setResult(
        await optimizeAllocation({
          holdings: portfolio.holdings,
          riskPreference,
          selectionRule: rule,
          maxWeight,
        }),
      );
    } catch (cause) {
      setRunError(cause instanceof Error ? cause.message : "Optimization failed");
    } finally {
      setBusy(false);
    }
  }

  if (loading) return <Spinner label="Loading portfolio" />;
  if (error) return <Notice tone="error">{error}</Notice>;
  if (!portfolio) return <Notice tone="info">Create a portfolio first.</Notice>;

  const current = new Map(
    portfolio.holdings.map((h) => [h.symbol, (h.quantity * h.current_price) / portfolio.total_value]),
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Long-term allocation</h1>
        <p className="mt-1 text-sm text-neutral-500">
          Three objectives at once: maximize return, minimize CVaR, minimize liquidity cost.
          A mean-variance optimizer optimizes the first two and is blind to the third.
        </p>
      </div>

      <Card title="Parameters">
        <div className="grid gap-4 sm:grid-cols-3">
          <Field label="Pareto selection rule">
            <select
              value={rule}
              onChange={(e) => setRule(e.target.value as SelectionRule)}
              className="w-full rounded-lg border border-black/15 bg-transparent px-3 py-2 text-sm dark:border-white/20"
            >
              {RULES.map((r) => (
                <option key={r.value} value={r.value}>
                  {r.label}
                </option>
              ))}
            </select>
          </Field>

          <Field label={`Risk preference — ${riskPreference.toFixed(2)}`}>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={riskPreference}
              onChange={(e) => setRiskPreference(Number(e.target.value))}
              className="w-full"
            />
          </Field>

          <Field label={`Per-asset cap — ${(maxWeight * 100).toFixed(0)}%`} hint="Prevents degenerate single-name solutions.">
            <input
              type="range"
              min={0.1}
              max={1}
              step={0.05}
              value={maxWeight}
              onChange={(e) => setMaxWeight(Number(e.target.value))}
              className="w-full"
            />
          </Field>
        </div>

        <p className="mt-3 text-xs text-neutral-500">
          {RULES.find((r) => r.value === rule)?.blurb}
        </p>

        <Button onClick={handleRun} disabled={busy} className="mt-4">
          {busy ? "Optimizing…" : "Run MOEA/D"}
        </Button>
      </Card>

      {runError && <Notice tone="error" title="Could not optimize">{runError}</Notice>}

      {result && (
        <>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat label="Expected return" value={percent(result.expected_return, 4)} />
            <Stat label="CVaR (95%)" value={percent(result.expected_cvar, 4)} hint="expected shortfall" />
            <Stat
              label="Liquidity cost"
              value={percent(result.expected_liquidity_cost, 4)}
              hint="charged on the trade, not the holding"
            />
            <Stat label="Pareto front" value={`${result.pareto_front_size} pts`} />
          </div>

          <Card title="Recommended weights" subtitle={result.selection_rationale}>
            <div className="space-y-2">
              {Object.entries(result.recommended_weights)
                .sort(([, a], [, b]) => b - a)
                .map(([symbol, weight]) => {
                  const now = current.get(symbol) ?? 0;
                  const delta = weight - now;
                  return (
                    <div key={symbol} className="flex items-center gap-3">
                      <span className="w-16 font-medium">{symbol}</span>
                      <div className="h-2 flex-1 overflow-hidden rounded bg-black/5 dark:bg-white/10">
                        <div
                          className="h-full bg-neutral-900 dark:bg-white"
                          style={{ width: `${Math.min(weight * 100, 100)}%` }}
                        />
                      </div>
                      <span className="w-16 text-right text-sm tabular-nums">
                        {(weight * 100).toFixed(1)}%
                      </span>
                      <span
                        className={`w-20 text-right text-xs tabular-nums ${
                          Math.abs(delta) < 0.005
                            ? "text-neutral-400"
                            : delta > 0
                              ? "text-emerald-600"
                              : "text-red-600"
                        }`}
                      >
                        {delta >= 0 ? "+" : ""}
                        {(delta * 100).toFixed(1)}%
                      </span>
                    </div>
                  );
                })}
            </div>
            <p className="mt-3 text-xs text-neutral-500">
              The right-hand column is the change from your current allocation. Liquidity cost
              is charged on that <em>delta</em>, not on the holding — so a large position left
              untouched costs nothing to keep.
            </p>
          </Card>

          <p className="text-xs text-neutral-500">
            Rule <code>{result.selection_rule}</code> · model{" "}
            <code>{result.model_version}</code>
          </p>
        </>
      )}
    </div>
  );
}

export default function Page() {
  return (
    <Suspense fallback={<Spinner label="Loading" />}>
      <OptimizeScreen />
    </Suspense>
  );
}
