"use client";

import { Suspense, useState } from "react";

import { Button, Card, Field, Input, Notice, Spinner, Stat, compact, money } from "@/components/ui";
import * as platform from "@/lib/api/platform";
import type { Holding, Portfolio, PortfolioSummary } from "@/lib/api/types";
import { usePortfolio } from "@/lib/usePortfolio";

/**
 * A realistic starting book. It deliberately spans the liquidity spectrum — SPY has ~4e10
 * daily volume while THIN has 8e5 — because on an all-liquid portfolio every liquidation
 * strategy performs identically and the Withdraw screen has nothing to show.
 */
const SEED_HOLDINGS: Holding[] = [
  { symbol: "SPY", quantity: 500, current_price: 580, avg_daily_volume: 4.0e10, cost_basis: 420 },
  { symbol: "AAPL", quantity: 1000, current_price: 230, avg_daily_volume: 2.5e9, cost_basis: 150 },
  { symbol: "QQQ", quantity: 400, current_price: 490, avg_daily_volume: 2.0e10, cost_basis: 350 },
  { symbol: "XLE", quantity: 1500, current_price: 92, avg_daily_volume: 8.0e8, cost_basis: 75 },
  { symbol: "THIN", quantity: 6000, current_price: 15, avg_daily_volume: 8.0e5, cost_basis: 22 },
];

const BLANK: Holding = {
  symbol: "",
  quantity: 0,
  current_price: 0,
  avg_daily_volume: 0,
  cost_basis: null,
};

function PortfolioScreen() {
  const { summaries, portfolio, loading, error, select, reload } = usePortfolio();
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [screenError, setScreenError] = useState<string | null>(null);

  async function handleCreate() {
    setBusy(true);
    setScreenError(null);
    try {
      const created = await platform.createPortfolio({
        name: `Portfolio ${summaries.length + 1}`,
        base_currency: "USD",
        holdings: SEED_HOLDINGS,
      });
      await reload();
      select(created.id);
      setMessage("Created a portfolio seeded with a mixed-liquidity book.");
    } catch (cause) {
      setScreenError(cause instanceof Error ? cause.message : "Could not create the portfolio");
    } finally {
      setBusy(false);
    }
  }

  if (loading) return <Spinner label="Loading portfolios" />;
  if (error) return <Notice tone="error" title="Cannot reach the platform service">{error}</Notice>;

  if (!portfolio) {
    return (
      <Card title="No portfolios yet" subtitle="Create one to start planning withdrawals.">
        {screenError && <Notice tone="error">{screenError}</Notice>}
        <Button onClick={handleCreate} disabled={busy} className="mt-3">
          {busy ? "Creating…" : "Create a demo portfolio"}
        </Button>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Header
        summaries={summaries}
        portfolio={portfolio}
        onSelect={select}
        onCreate={handleCreate}
        busy={busy}
      />

      {/* Keyed on the portfolio id so switching portfolios REMOUNTS the editor, which
          re-initialises its draft from the new props. The alternative -- syncing props into
          state inside an effect -- is what React 19 flags as cascading renders, and it also
          silently discards unsaved edits one render later. */}
      <PortfolioEditor
        key={portfolio.id}
        portfolio={portfolio}
        onSaved={reload}
        onMessage={setMessage}
      />

      {message && <Notice tone="success">{message}</Notice>}
      {screenError && <Notice tone="error">{screenError}</Notice>}
    </div>
  );
}

function Header({
  summaries,
  portfolio,
  onSelect,
  onCreate,
  busy,
}: {
  summaries: PortfolioSummary[];
  portfolio: Portfolio;
  onSelect: (id: number) => void;
  onCreate: () => void;
  busy: boolean;
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3">
      <h1 className="text-xl font-semibold">Portfolio</h1>
      <div className="flex items-center gap-2">
        {summaries.length > 1 && (
          <select
            value={portfolio.id}
            onChange={(e) => onSelect(Number(e.target.value))}
            className="rounded-lg border border-black/15 bg-transparent px-3 py-2 text-sm dark:border-white/20"
          >
            {summaries.map((summary) => (
              <option key={summary.id} value={summary.id}>
                {summary.name}
              </option>
            ))}
          </select>
        )}
        <Button variant="secondary" onClick={onCreate} disabled={busy}>
          New
        </Button>
      </div>
    </div>
  );
}

function PortfolioEditor({
  portfolio,
  onSaved,
  onMessage,
}: {
  portfolio: Portfolio;
  onSaved: () => Promise<void>;
  onMessage: (message: string) => void;
}) {
  // Initialised from props exactly once, because the parent remounts this on id change.
  const [draft, setDraft] = useState<Holding[]>(portfolio.holdings);
  const [name, setName] = useState(portfolio.name);
  const [busy, setBusy] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const totalValue = draft.reduce((sum, h) => sum + h.quantity * h.current_price, 0);
  const leastLiquid = draft.length
    ? draft.reduce((worst, h) => (h.avg_daily_volume < worst.avg_daily_volume ? h : worst)).symbol
    : "—";

  function updateRow(index: number, patch: Partial<Holding>) {
    setDraft((rows) => rows.map((row, i) => (i === index ? { ...row, ...patch } : row)));
  }

  async function handleSave() {
    setBusy(true);
    setSaveError(null);
    try {
      await platform.updatePortfolio(portfolio.id, { name, holdings: draft });
      await onSaved();
      onMessage("Saved.");
    } catch (cause) {
      setSaveError(cause instanceof Error ? cause.message : "Could not save");
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete() {
    setBusy(true);
    setSaveError(null);
    try {
      await platform.deletePortfolio(portfolio.id);
      await onSaved();
      onMessage("Deleted.");
    } catch (cause) {
      setSaveError(cause instanceof Error ? cause.message : "Could not delete");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label="Total value" value={money(totalValue)} />
        <Stat label="Holdings" value={draft.length} />
        <Stat label="Least liquid" value={leastLiquid} hint="drives withdrawal cost" />
        <Stat label="Currency" value={portfolio.base_currency} />
      </div>

      <Card
        title="Holdings"
        subtitle="Average daily volume is what makes a position cheap or expensive to exit."
        className="mt-6"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-left text-xs uppercase tracking-wide text-neutral-500">
              <tr>
                <th className="pb-2">Symbol</th>
                <th className="pb-2">Quantity</th>
                <th className="pb-2">Price</th>
                <th className="pb-2">ADV (USD)</th>
                <th className="pb-2">Cost basis</th>
                <th className="pb-2 text-right">Value</th>
                <th className="pb-2" />
              </tr>
            </thead>
            <tbody>
              {draft.map((holding, index) => (
                <tr key={index} className="border-t border-black/5 dark:border-white/10">
                  <td className="py-2 pr-2">
                    <Input
                      value={holding.symbol}
                      onChange={(e) => updateRow(index, { symbol: e.target.value.toUpperCase() })}
                      className="w-24"
                    />
                  </td>
                  <td className="py-2 pr-2">
                    <Input
                      type="number"
                      value={holding.quantity}
                      onChange={(e) => updateRow(index, { quantity: Number(e.target.value) })}
                      className="w-28"
                    />
                  </td>
                  <td className="py-2 pr-2">
                    <Input
                      type="number"
                      step="0.01"
                      value={holding.current_price}
                      onChange={(e) => updateRow(index, { current_price: Number(e.target.value) })}
                      className="w-28"
                    />
                  </td>
                  <td className="py-2 pr-2">
                    <Input
                      type="number"
                      value={holding.avg_daily_volume}
                      onChange={(e) =>
                        updateRow(index, { avg_daily_volume: Number(e.target.value) })
                      }
                      className="w-32"
                    />
                    <span className="mt-0.5 block text-xs text-neutral-500">
                      {compact(holding.avg_daily_volume)}
                    </span>
                  </td>
                  <td className="py-2 pr-2">
                    <Input
                      type="number"
                      step="0.01"
                      value={holding.cost_basis ?? ""}
                      onChange={(e) =>
                        updateRow(index, {
                          cost_basis: e.target.value === "" ? null : Number(e.target.value),
                        })
                      }
                      className="w-28"
                    />
                  </td>
                  <td className="py-2 text-right tabular-nums">
                    {money(holding.quantity * holding.current_price)}
                  </td>
                  <td className="py-2 pl-2 text-right">
                    <button
                      onClick={() => setDraft((rows) => rows.filter((_, i) => i !== index))}
                      className="text-sm text-red-600 hover:underline"
                    >
                      Remove
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-2">
          <Button variant="secondary" onClick={() => setDraft((rows) => [...rows, { ...BLANK }])}>
            Add holding
          </Button>
          <Button variant="secondary" onClick={() => setDraft(SEED_HOLDINGS)}>
            Reset to demo book
          </Button>
          <div className="flex-1" />
          <Button variant="danger" onClick={handleDelete} disabled={busy}>
            Delete portfolio
          </Button>
          <Button onClick={handleSave} disabled={busy}>
            {busy ? "Saving…" : "Save"}
          </Button>
        </div>
      </Card>

      <Card title="Name" className="mt-6">
        <Field label="">
          <Input value={name} onChange={(e) => setName(e.target.value)} className="w-64" />
        </Field>
      </Card>

      {saveError && <Notice tone="error">{saveError}</Notice>}
    </>
  );
}

export default function Page() {
  // usePortfolio reads useSearchParams, which Next requires inside a Suspense boundary or
  // the whole route is forced to client-side rendering at build time.
  return (
    <Suspense fallback={<Spinner label="Loading" />}>
      <PortfolioScreen />
    </Suspense>
  );
}
