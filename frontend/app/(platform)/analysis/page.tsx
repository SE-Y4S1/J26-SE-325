"use client";

/**
 * Per-holding analysis: WHY a stock's sentiment, technical and fundamental read-out is what
 * it is. Shows the keyword profile the news search used (editable -- a corrected profile is
 * marked reviewed and never regenerated over), the channels that ran, the latest sources,
 * and the technical and fundamental panels the planner and Laya read.
 */

import { Suspense, useEffect, useState } from "react";

import { Button, Card, Field, Input, Notice, Spinner } from "@/components/ui";
import {
  getFundamentals,
  getKeywords,
  getSentiment,
  getTechnical,
  listChannels,
  updateKeywords,
} from "@/lib/api/market";
import type {
  ChannelStatus,
  FundamentalsResponse,
  KeywordProfile,
  SentimentResponse,
  TechnicalResponse,
} from "@/lib/api/types";
import { usePortfolio } from "@/lib/usePortfolio";

const LIST_FIELDS = ["company", "people_brands", "sector_terms", "macro_terms", "exclude", "local_names"] as const;
type ListField = (typeof LIST_FIELDS)[number];

const fmt = (value: unknown, digits = 3) =>
  typeof value === "number" ? value.toFixed(digits) : value == null ? "—" : String(value);

function AnalysisScreen() {
  const { portfolio, loading, error } = usePortfolio();
  const [symbol, setSymbol] = useState<string | null>(null);
  const [channels, setChannels] = useState<ChannelStatus[] | null>(null);

  useEffect(() => {
    listChannels().then(setChannels).catch(() => setChannels([]));
  }, []);

  if (loading) return <Spinner label="Loading portfolio" />;
  if (error) return <Notice tone="error">{error}</Notice>;
  if (!portfolio || portfolio.holdings.length === 0) {
    return <Notice tone="info">Add holdings on the Portfolio screen and save to start analysis.</Notice>;
  }
  const selected = symbol ?? portfolio.holdings[0].symbol;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-xl font-semibold">Holding analysis</h1>
        <select
          aria-label="Holding"
          value={selected}
          onChange={(e) => setSymbol(e.target.value)}
          className="rounded-lg border border-black/15 bg-transparent px-3 py-2 text-sm dark:border-white/20"
        >
          {portfolio.holdings.map((h) => (
            <option key={h.symbol} value={h.symbol}>
              {h.symbol}
            </option>
          ))}
        </select>
      </div>

      <HoldingPanels key={selected} symbol={selected} />

      {channels && (
        <Card title="News and social channels" subtitle="Logged-in channels need a dedicated account; see the README.">
          <ul className="grid gap-1 text-sm sm:grid-cols-2">
            {channels.map((c) => (
              <li key={c.name}>
                <span className={c.available ? "text-emerald-600" : "text-neutral-500"}>{c.available ? "●" : "○"}</span>{" "}
                <span className="font-medium">{c.name}</span> <span className="text-neutral-500">({c.kind})</span>
                {!c.available && c.reason && <span className="block text-xs text-neutral-500">{c.reason}</span>}
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  );
}

function HoldingPanels({ symbol }: { symbol: string }) {
  const [technical, setTechnical] = useState<TechnicalResponse | null>(null);
  const [fundamentals, setFundamentals] = useState<FundamentalsResponse | null>(null);
  const [sentiment, setSentiment] = useState<SentimentResponse | null>(null);
  const [failure, setFailure] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [t, f, s] = await Promise.all([getTechnical(symbol), getFundamentals(symbol), getSentiment(symbol)]);
        if (cancelled) return;
        setTechnical(t);
        setFundamentals(f);
        setSentiment(s);
      } catch (cause) {
        if (!cancelled) setFailure(cause instanceof Error ? cause.message : "Analysis unavailable");
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, [symbol]);

  if (failure) return <Notice tone="warn" title={`${symbol} is not analysed yet`}>{failure}</Notice>;
  if (!technical || !fundamentals || !sentiment) return <Spinner label={`Loading ${symbol}`} />;

  const t = technical.technical;
  const f = fundamentals.fundamentals;
  const ratios = (f?.ratios ?? {}) as Record<string, unknown>;

  return (
    <>
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Technical" subtitle={t ? `${t.sessions} sessions to ${t.as_of}` : undefined}>
          {t ? (
            <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
              {([
                ["Trend", t.trend],
                ["MA cross (20d)", t.ma_cross_recent ?? "none"],
                ["vs 50d / 200d MA", `${fmt(t.price_vs_sma_50)} / ${fmt(t.price_vs_sma_200)}`],
                ["RSI / MFI", `${fmt(t.rsi, 1)} / ${fmt(t.mfi, 1)}`],
                ["ADX", fmt(t.adx, 1)],
                ["Bollinger %B", fmt(t.bollinger_pct_b, 2)],
                ["OBV trend", fmt(t.obv_trend, 2)],
                ["From 52w high", fmt(t.drawdown_52w)],
                ["To support / resistance", `${fmt(t.dist_to_support)} / ${fmt(t.dist_to_resistance)}`],
                ["vs market, 63d", fmt(t.relative_strength_63d)],
              ] as const).map(([label, value]) => (
                <div key={label} className="contents">
                  <dt className="text-neutral-500">{label}</dt>
                  <dd className="text-right tabular-nums">{String(value ?? "—")}</dd>
                </div>
              ))}
            </dl>
          ) : (
            <p className="text-sm text-neutral-500">{technical.detail}</p>
          )}
        </Card>

        <Card
          title="Fundamentals"
          subtitle={f?.report ? `${f.report.title ?? "report"} · ${f.report.age_days} days old` : undefined}
        >
          {!f ? (
            <p className="text-sm text-neutral-500">{fundamentals.detail}</p>
          ) : f.kind === "tokenomics" ? (
            <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
              {Object.entries(f.tokenomics ?? {}).map(([k, v]) => (
                <div key={k} className="contents">
                  <dt className="text-neutral-500">{k.replaceAll("_", " ")}</dt>
                  <dd className="text-right tabular-nums">{fmt(v)}</dd>
                </div>
              ))}
            </dl>
          ) : (
            <>
              <p className="mb-2 text-xs text-neutral-500">
                {f.template} template · extraction confidence {fmt(f.confidence, 2)} ·{" "}
                {Object.entries(f.checks ?? {})
                  .map(([k, ok]) => `${k.replaceAll("_", " ")} ${ok ? "✓" : "✗"}`)
                  .join(" · ")}
              </p>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
                {["roe", "net_margin", "revenue_growth_yoy", "earnings_growth_yoy", "debt_to_equity", "current_ratio",
                  "interest_cover", "pe", "pb", "dividend_yield", "nim_on_assets", "cost_to_income", "z_score", "z_zone"]
                  .filter((k) => ratios[k] != null)
                  .map((k) => (
                    <div key={k} className="contents">
                      <dt className="text-neutral-500">
                        {k.replaceAll("_", " ")}
                        {f.sector_relative?.[k] != null && (
                          <span className="ml-1 text-xs">(p{Math.round((f.sector_relative[k] ?? 0) * 100)} in sector)</span>
                        )}
                      </dt>
                      <dd className="text-right tabular-nums">{fmt(ratios[k])}</dd>
                    </div>
                  ))}
                <dt className="text-neutral-500">Piotroski F</dt>
                <dd className="text-right">{ratios.f_score != null ? `${ratios.f_score}/${ratios.f_score_max}` : "—"}</dd>
                <dt className="text-neutral-500">Management outlook</dt>
                <dd className="text-right">{f.outlook ? `${f.outlook.tone} (${f.outlook.method})` : "—"}</dd>
              </dl>
              {Object.entries(f.flags ?? {}).some(([k, v]) => v === true && k !== "unaudited") && (
                <Notice tone="warn" title="Red flags">
                  {Object.entries(f.flags ?? {})
                    .filter(([k, v]) => v === true && k !== "unaudited")
                    .map(([k]) => k.replaceAll("_", " "))
                    .join(", ")}
                </Notice>
              )}
              {f.report?.url && (
                <a href={f.report.url} target="_blank" rel="noreferrer" className="mt-2 block text-xs hover:underline">
                  Source report (PDF)
                </a>
              )}
            </>
          )}
        </Card>
      </div>

      <Card title="Latest sources" subtitle={sentiment.sentiment ? `${sentiment.sentiment.items_relevant} relevant items` : undefined}>
        {sentiment.sentiment ? (
          <ul className="space-y-1 text-sm">
            {sentiment.sentiment.top_sources.map((s, i) => (
              <li key={i} className="truncate">
                <span className="text-xs text-neutral-500">{s.channel} · {s.tier} · {fmt(s.polarity, 2)}</span>{" "}
                {s.url ? (
                  <a href={s.url} target="_blank" rel="noreferrer" className="hover:underline">{s.title ?? s.url}</a>
                ) : (
                  s.title ?? "social post"
                )}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-neutral-500">{sentiment.detail}</p>
        )}
      </Card>

      <KeywordEditor symbol={symbol} />
    </>
  );
}

function KeywordEditor({ symbol }: { symbol: string }) {
  const [profile, setProfile] = useState<KeywordProfile | null>(null);
  const [draft, setDraft] = useState<Record<ListField, string>>({} as Record<ListField, string>);
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getKeywords(symbol)
      .then((p) => {
        setProfile(p);
        setDraft(Object.fromEntries(LIST_FIELDS.map((k) => [k, (p[k] ?? []).join(", ")])) as Record<ListField, string>);
      })
      .catch((cause) => setMessage(cause instanceof Error ? cause.message : "No keyword profile yet"));
  }, [symbol]);

  async function save() {
    setBusy(true);
    try {
      const changes = Object.fromEntries(
        LIST_FIELDS.map((k) => [k, draft[k].split(",").map((s) => s.trim()).filter(Boolean)]),
      );
      setProfile(await updateKeywords(symbol, changes));
      setMessage("Saved and marked reviewed. Refresh analysis to search with it.");
    } catch (cause) {
      setMessage(cause instanceof Error ? cause.message : "Could not save");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card
      title="Keyword profile"
      subtitle={profile ? `${profile.source} · from ${profile.provenance.join(", ")}` : "What the news and social search looks for"}
    >
      {profile && (
        <div className="grid gap-3 sm:grid-cols-2">
          {LIST_FIELDS.map((field) => (
            <Field key={field} label={field.replaceAll("_", " ")}>
              <Input
                value={draft[field] ?? ""}
                onChange={(e) => setDraft((d) => ({ ...d, [field]: e.target.value }))}
              />
            </Field>
          ))}
          <div className="sm:col-span-2">
            <Button onClick={save} disabled={busy}>
              {busy ? "Saving…" : "Save profile"}
            </Button>
          </div>
        </div>
      )}
      {message && <p className="mt-2 text-sm text-neutral-500">{message}</p>}
    </Card>
  );
}

export default function Page() {
  return (
    <Suspense fallback={<Spinner label="Loading" />}>
      <AnalysisScreen />
    </Suspense>
  );
}
