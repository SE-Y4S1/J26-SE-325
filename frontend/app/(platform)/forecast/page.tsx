"use client";

/**
 * Quantile forecasts from the hybrid model.
 *
 * This endpoint returns 503 until a model is registered, which is the EXPECTED state until
 * Colab fine-tuning has run. The screen therefore treats 503 as an explanatory empty state
 * rather than an error — showing a red failure for a known, documented condition trains
 * people to ignore real errors.
 */

import { Suspense, useState } from "react";

import { Button, Card, Field, Notice, Spinner, percent } from "@/components/ui";
import { ApiError } from "@/lib/api/client";
import { getForecast } from "@/lib/api/portfolio";
import type { ForecastResponse } from "@/lib/api/types";
import { usePortfolio } from "@/lib/usePortfolio";

function ForecastScreen() {
  const { portfolio, loading, error } = usePortfolio();

  const [horizon, setHorizon] = useState(5);
  const [result, setResult] = useState<ForecastResponse | null>(null);
  const [unavailable, setUnavailable] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function handleRun() {
    if (!portfolio) return;
    setBusy(true);
    setRunError(null);
    setUnavailable(false);
    setResult(null);
    try {
      setResult(await getForecast(portfolio.holdings.map((h) => h.symbol), horizon));
    } catch (cause) {
      if (cause instanceof ApiError && cause.isUnavailable) {
        setUnavailable(true);
      } else {
        setRunError(cause instanceof Error ? cause.message : "Forecast failed");
      }
    } finally {
      setBusy(false);
    }
  }

  if (loading) return <Spinner label="Loading portfolio" />;
  if (error) return <Notice tone="error">{error}</Notice>;
  if (!portfolio) return <Notice tone="info">Create a portfolio first.</Notice>;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Forecasts</h1>
        <p className="mt-1 text-sm text-neutral-500">
          p10 / p50 / p90 return quantiles. The interval matters more than the midpoint — the
          CVaR objective on the Optimize screen is computed from it.
        </p>
      </div>

      <Card title="Request">
        <div className="flex flex-wrap items-end gap-4">
          <Field label="Horizon (trading days)">
            <select
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))}
              className="rounded-lg border border-black/15 bg-transparent px-3 py-2 text-sm dark:border-white/20"
            >
              {[1, 5, 21].map((h) => (
                <option key={h} value={h}>
                  {h}
                </option>
              ))}
            </select>
          </Field>
          <Button onClick={handleRun} disabled={busy}>
            {busy ? "Forecasting…" : `Forecast ${portfolio.holdings.length} symbols`}
          </Button>
        </div>
      </Card>

      {unavailable && (
        <Notice tone="info" title="No trained forecaster is registered yet">
          <p>
            This is expected. The baseline LSTM trains locally, but the hybrid model needs
            LoRA-fine-tuned foundation weights, and those require a GPU and a fast connection
            — neither of which the development machine has.
          </p>
          <p className="mt-2">
            Run <code>experiments/colab_finetune.ipynb</code> on Colab, download the adapters,
            and register them. Every other screen works without it.
          </p>
        </Notice>
      )}

      {runError && <Notice tone="error" title="Could not forecast">{runError}</Notice>}

      {result && (
        <Card title="Quantile forecasts" subtitle={`Horizon ${horizon} trading days`}>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-xs uppercase tracking-wide text-neutral-500">
                <tr>
                  <th className="pb-2">Symbol</th>
                  <th className="pb-2 text-right">p10</th>
                  <th className="pb-2 text-right">p50</th>
                  <th className="pb-2 text-right">p90</th>
                  <th className="pb-2">Interval</th>
                </tr>
              </thead>
              <tbody>
                {result.forecasts.map((forecast) => (
                  <tr key={forecast.symbol} className="border-t border-black/5 dark:border-white/10">
                    <td className="py-2 font-medium">{forecast.symbol}</td>
                    <td className="py-2 text-right tabular-nums text-red-600">
                      {percent(forecast.p10)}
                    </td>
                    <td className="py-2 text-right tabular-nums">{percent(forecast.p50)}</td>
                    <td className="py-2 text-right tabular-nums text-emerald-600">
                      {percent(forecast.p90)}
                    </td>
                    <td className="py-2">
                      <QuantileBar p10={forecast.p10} p50={forecast.p50} p90={forecast.p90} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-neutral-500">
            Model <code>{result.model_version}</code>
          </p>
        </Card>
      )}
    </div>
  );
}

/** A minimal p10–p90 range with the median marked. Scaled to a fixed ±10% window so bars
 *  are comparable between symbols rather than each being normalised to its own range. */
function QuantileBar({ p10, p50, p90 }: { p10: number; p50: number; p90: number }) {
  const SCALE = 0.1;
  const toPct = (value: number) => ((value + SCALE) / (2 * SCALE)) * 100;

  const left = Math.max(0, Math.min(100, toPct(p10)));
  const right = Math.max(0, Math.min(100, toPct(p90)));
  const mid = Math.max(0, Math.min(100, toPct(p50)));

  return (
    <div className="relative h-2 w-40 rounded bg-black/5 dark:bg-white/10">
      <div
        className="absolute h-full rounded bg-neutral-400"
        style={{ left: `${left}%`, width: `${Math.max(right - left, 1)}%` }}
      />
      <div
        className="absolute top-[-2px] h-3 w-0.5 bg-neutral-900 dark:bg-white"
        style={{ left: `${mid}%` }}
      />
    </div>
  );
}

export default function Page() {
  return (
    <Suspense fallback={<Spinner label="Loading" />}>
      <ForecastScreen />
    </Suspense>
  );
}
