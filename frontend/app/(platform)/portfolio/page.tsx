"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";

import { EXCHANGE_LABEL, InstrumentSearch } from "@/components/analysis/InstrumentSearch";
import { Button, Card, Field, Input, Notice, Spinner, Stat, compact, money } from "@/components/ui";
import { analysisStatus, getQuote, trackSymbols, type ExchangeName } from "@/lib/api/market";
import * as platform from "@/lib/api/platform";
import type { Holding, Portfolio, PortfolioSummary, QuoteResponse, SymbolStatus } from "@/lib/api/types";
import { usePortfolio } from "@/lib/usePortfolio";

/**
 * A realistic starting book across all three markets. It deliberately spans the liquidity
 * spectrum — SPY trades ~USD 30bn a day while COMB and JKH trade LKR 100-200m — because on an
 * all-liquid portfolio every liquidation strategy performs identically and the Withdraw
 * screen has nothing to show. Prices are a starting point; "Refresh quotes" replaces them
 * with live ones. Each holding is in its own trading currency (LKR / USD / USDT).
 */
const SEED_HOLDINGS: Holding[] = [
  { symbol: "JKH.N0000", quantity: 200_000, current_price: 18.4, avg_daily_volume: 9.5e7, cost_basis: 21.0 },
  { symbol: "COMB.N0000", quantity: 20_000, current_price: 150, avg_daily_volume: 2.0e8, cost_basis: 120 },
  { symbol: "AAPL", quantity: 100, current_price: 336.6, avg_daily_volume: 1.5e10, cost_basis: 250 },
  { symbol: "SPY", quantity: 50, current_price: 660, avg_daily_volume: 3.0e10, cost_basis: 560 },
  { symbol: "BTCUSDT", quantity: 0.1, current_price: 82_570, avg_daily_volume: 2.3e10, cost_basis: 60_000 },
];

/** How often to re-check preparation, and for how long (news + reports take minutes). */
const STATUS_POLL_MS = 5_000;
const STATUS_POLL_LIMIT = 120;

const STATUS_TONE: Record<string, string> = {
  ready: "bg-emerald-500/15 text-emerald-800 dark:text-emerald-200",
  partial: "bg-amber-500/15 text-amber-900 dark:text-amber-200",
  preparing: "bg-blue-500/15 text-blue-900 dark:text-blue-200",
  failed: "bg-red-500/15 text-red-800 dark:text-red-200",
};

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
        base_currency: "LKR",
        holdings: SEED_HOLDINGS,
      });
      await reload();
      select(created.id);
      setMessage("Created a portfolio seeded with a mixed CSE / US / crypto book.");
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
  // Live quote per symbol (currency, exchange, FX to LKR). Not persisted: it is market data.
  const [quotes, setQuotes] = useState<Record<string, QuoteResponse>>({});
  const [statuses, setStatuses] = useState<Record<string, SymbolStatus>>({});
  const [quoteError, setQuoteError] = useState<string | null>(null);

  // Value in LKR only when every row has a quote (each carries its own FX rate); otherwise a
  // mixed-currency sum would be meaningless, so it says so instead.
  const allQuoted = draft.length > 0 && draft.every((h) => quotes[h.symbol]);
  const totalBase = draft.reduce(
    (sum, h) => sum + h.quantity * h.current_price * (quotes[h.symbol]?.fx_rate_to_base ?? 0),
    0,
  );
  const leastLiquid = draft.length
    ? draft.reduce((worst, h) => {
        const adv = (row: Holding) => row.avg_daily_volume * (quotes[row.symbol]?.fx_rate_to_base ?? 1);
        return adv(h) < adv(worst) ? h : worst;
      }).symbol
    : "—";

  // Polling outlives a render; stop it when the editor unmounts (portfolio switch, navigation).
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const pollStatus = useCallback(async (symbols: string[]) => {
    for (let attempt = 0; attempt < STATUS_POLL_LIMIT && symbols.length && alive.current; attempt++) {
      try {
        const response = await analysisStatus(symbols);
        setStatuses((current) => ({
          ...current,
          ...Object.fromEntries(response.symbols.map((s) => [s.symbol, s])),
        }));
        if (!response.symbols.some((s) => s.status === "preparing")) return;
      } catch {
        return; // status is informational; never block the editor on it
      }
      await new Promise((resolve) => setTimeout(resolve, STATUS_POLL_MS));
    }
  }, []);

  useEffect(() => {
    // Started from a timer so no state update can run synchronously inside the effect.
    const timer = setTimeout(() => void pollStatus(portfolio.holdings.map((h) => h.symbol).filter(Boolean)), 0);
    return () => clearTimeout(timer);
  }, [portfolio.holdings, pollStatus]);

  function updateRow(index: number, patch: Partial<Holding>) {
    setDraft((rows) => rows.map((row, i) => (i === index ? { ...row, ...patch } : row)));
  }

  async function quoteRow(index: number, symbol: string, exchange?: ExchangeName) {
    try {
      const quote = await getQuote(symbol, { exchange });
      setQuotes((current) => ({ ...current, [quote.instrument.symbol]: quote }));
      updateRow(index, {
        symbol: quote.instrument.symbol,
        current_price: quote.price,
        avg_daily_volume: Math.round(quote.avg_daily_volume),
      });
      return true;
    } catch (cause) {
      setQuoteError(`${symbol}: ${cause instanceof Error ? cause.message : "quote unavailable"} — enter the price manually.`);
      return false;
    }
  }

  async function refreshQuotes() {
    setBusy(true);
    setQuoteError(null);
    await Promise.all(draft.map((h, i) => (h.symbol ? quoteRow(i, h.symbol) : Promise.resolve(false))));
    setBusy(false);
  }

  async function handleSave() {
    setBusy(true);
    setSaveError(null);
    try {
      await platform.updatePortfolio(portfolio.id, { name, holdings: draft });
      await onSaved();
      // Saving is what makes symbols "held": only they get history, news, reports,
      // forecasts and Laya analysis. Preparation runs in the background.
      const symbols = draft.map((h) => h.symbol).filter(Boolean);
      const tracked = await trackSymbols(symbols);
      const rejected = Object.entries(tracked.rejected ?? {});
      onMessage(
        rejected.length
          ? `Saved. Not recognised on any exchange: ${rejected.map(([s]) => s).join(", ")}.`
          : "Saved. Analysis is being prepared for each holding.",
      );
      void pollStatus(symbols);
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
        <Stat
          label="Total value (LKR)"
          value={allQuoted ? money(totalBase, "LKR") : "—"}
          hint={allQuoted ? "at live FX rates" : "refresh quotes to convert"}
        />
        <Stat label="Holdings" value={draft.length} />
        <Stat label="Least liquid" value={leastLiquid} hint="drives withdrawal cost" />
        <Stat label="Plan currency" value={portfolio.base_currency} />
      </div>

      <Card
        title="Holdings"
        subtitle="Search any CSE, Binance or Yahoo instrument. Prices and daily traded value are in the instrument's own currency."
        className="mt-6"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-left text-xs uppercase tracking-wide text-neutral-500">
              <tr>
                <th className="pb-2">Symbol</th>
                <th className="pb-2">Market</th>
                <th className="pb-2">Quantity</th>
                <th className="pb-2">Price</th>
                <th className="pb-2">Daily traded value</th>
                <th className="pb-2">Cost basis</th>
                <th className="pb-2 text-right">Value</th>
                <th className="pb-2">Analysis</th>
                <th className="pb-2" />
              </tr>
            </thead>
            <tbody>
              {draft.map((holding, index) => (
                <tr key={index} className="border-t border-black/5 dark:border-white/10">
                  <td className="py-2 pr-2">
                    <InstrumentSearch
                      value={holding.symbol}
                      onChange={(symbol) => updateRow(index, { symbol })}
                      onPick={(instrument) => {
                        updateRow(index, { symbol: instrument.symbol });
                        void quoteRow(index, instrument.symbol, instrument.exchange as ExchangeName);
                      }}
                    />
                  </td>
                  <td className="py-2 pr-2 text-xs text-neutral-500">
                    {quotes[holding.symbol]
                      ? `${EXCHANGE_LABEL[quotes[holding.symbol].instrument.exchange] ?? ""} · ${quotes[holding.symbol].currency}`
                      : "—"}
                    {quotes[holding.symbol]?.stale && <span className="block text-amber-600">stale</span>}
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
                    {(holding.quantity * holding.current_price).toLocaleString(undefined, { maximumFractionDigits: 2 })}
                  </td>
                  <td className="py-2 pl-2">
                    {statuses[holding.symbol] ? (
                      <span
                        title={statuses[holding.symbol].error ?? Object.entries(statuses[holding.symbol].steps ?? {}).map(([k, v]) => `${k}: ${v}`).join(", ")}
                        className={`rounded px-1.5 py-0.5 text-xs font-medium ${STATUS_TONE[statuses[holding.symbol].status] ?? ""}`}
                      >
                        {statuses[holding.symbol].status}
                      </span>
                    ) : (
                      <span className="text-xs text-neutral-500">not saved</span>
                    )}
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
          <Button variant="secondary" onClick={refreshQuotes} disabled={busy}>
            Refresh quotes
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

      {quoteError && <Notice tone="warn">{quoteError}</Notice>}
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
