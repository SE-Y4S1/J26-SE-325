"use client";

/**
 * What the planner knew about each holding when it planned: the forecast band (and what
 * produced it), news sentiment per tier, the technical and fundamental read-out, the Laya
 * decision, and how old the data is. Every number the GA used is listed under "signals used",
 * so a reader can tie a sell order back to its inputs.
 *
 * The forecast's `method` is always shown: an "empirical" band is the symbol's own past
 * returns (used when no trained head can run), and must never read as a model forecast.
 */

import type { BaselineRow, CostBreakdown, MarketContext } from "@/lib/api/types";
import { Card, money, percent } from "@/components/ui";

const pct = (value: number | null | undefined, digits = 1) =>
  value == null ? "—" : `${(value * 100).toFixed(digits)}%`;

const signed = (value: number | null | undefined, digits = 2) =>
  value == null ? "—" : `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;

function Badge({ children, tone = "neutral" }: { children: React.ReactNode; tone?: "neutral" | "good" | "bad" | "warn" }) {
  const styles = {
    neutral: "bg-black/5 text-neutral-700 dark:bg-white/10 dark:text-neutral-200",
    good: "bg-emerald-500/15 text-emerald-800 dark:text-emerald-200",
    bad: "bg-red-500/15 text-red-800 dark:text-red-200",
    warn: "bg-amber-500/15 text-amber-900 dark:text-amber-200",
  }[tone];
  return <span className={`rounded px-1.5 py-0.5 text-xs font-medium ${styles}`}>{children}</span>;
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3 text-sm">
      <span className="text-neutral-500">{label}</span>
      <span className="text-right tabular-nums">{children}</span>
    </div>
  );
}

const TREND_TONE: Record<string, "good" | "bad" | "neutral"> = { bullish: "good", bearish: "bad", neutral: "neutral" };

export function HoldingContext({ symbol, context }: { symbol: string; context: MarketContext }) {
  const forecast = context.forecast;
  const sentiment = context.sentiment;
  const technical = context.technical;
  const fundamentals = context.fundamentals;
  const decision = context.decision;

  return (
    <div className="rounded-lg border border-black/10 p-4 dark:border-white/15" data-testid={`context-${symbol}`}>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <span className="font-semibold">{symbol}</span>
        <div className="flex flex-wrap gap-1">
          {context.currency && <Badge>{context.currency}</Badge>}
          <Badge tone={context.status === "ready" ? "good" : context.status === "untracked" ? "warn" : "neutral"}>
            {context.status}
          </Badge>
          {context.data_age_hours != null && <Badge>data {context.data_age_hours.toFixed(1)}h old</Badge>}
        </div>
      </div>

      {context.status === "untracked" && (
        <p className="text-sm text-neutral-500">
          Not analysed yet — save the portfolio on the Portfolio screen to start preparation. Planned on
          liquidity and volatility rules only.
        </p>
      )}

      <div className="grid gap-4 sm:grid-cols-2">
        {forecast && (
          <div className="space-y-1">
            <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
              Forecast · {forecast.horizon}d{" "}
              <Badge tone={forecast.method === "hybrid" ? "good" : "warn"}>
                {forecast.method === "hybrid" ? "TimesFM + LSTM" : "empirical (no model)"}
              </Badge>
            </p>
            <Row label="p10 / p50 / p90">
              {pct(forecast.p10)} / {pct(forecast.p50)} / {pct(forecast.p90)}
            </Row>
            {forecast.reason && forecast.method !== "hybrid" && (
              <p className="text-xs text-neutral-500">{forecast.reason}</p>
            )}
          </div>
        )}

        {sentiment && (
          <div className="space-y-1">
            <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
              News · {sentiment.items ?? 0} items
            </p>
            <Row label="company / sector / macro">
              {signed(sentiment.company)} / {signed(sentiment.sector)} / {signed(sentiment.macro)}
            </Row>
            <Row label="attention spike">
              {sentiment.attention_spike != null && sentiment.attention_spike >= 2 ? (
                <Badge tone="warn">{signed(sentiment.attention_spike, 1)}</Badge>
              ) : (
                signed(sentiment.attention_spike, 1)
              )}
            </Row>
            <ul className="mt-1 space-y-0.5">
              {sentiment.top_sources.map((source, i) => (
                <li key={i} className="truncate text-xs">
                  <Badge>{source.channel}</Badge>{" "}
                  {source.url ? (
                    <a href={source.url} target="_blank" rel="noreferrer" className="hover:underline">
                      {source.title ?? source.url}
                    </a>
                  ) : (
                    source.title ?? "social post"
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}

        {technical && (
          <div className="space-y-1">
            <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">Technical</p>
            <Row label="trend">
              {technical.trend ? <Badge tone={TREND_TONE[String(technical.trend)] ?? "neutral"}>{String(technical.trend)}</Badge> : "—"}
            </Row>
            <Row label="RSI / ADX">
              {technical.rsi ?? "—"} / {technical.adx ?? "—"}
            </Row>
            <Row label="vs market (63d)">{pct(technical.relative_strength_63d as number | null)}</Row>
            <Row label="from 52w high">{pct(technical.drawdown_52w as number | null)}</Row>
          </div>
        )}

        {fundamentals && (
          <div className="space-y-1">
            <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
              Fundamentals{fundamentals.kind === "tokenomics" ? " · tokenomics" : ""}
            </p>
            {fundamentals.kind === "tokenomics" ? (
              <>
                <Row label="market-cap rank">{String(fundamentals.market_cap_rank ?? "—")}</Row>
                <Row label="supply not yet circulating">{pct(fundamentals.supply_overhang as number | null)}</Row>
              </>
            ) : (
              <>
                <Row label="ROE / P/B">
                  {pct(fundamentals.roe as number | null)} / {fundamentals.pb != null ? Number(fundamentals.pb).toFixed(2) : "—"}
                </Row>
                <Row label="F-score / Z'' zone">
                  {fundamentals.f_score != null ? `${fundamentals.f_score}/${fundamentals.f_score_max}` : "—"} /{" "}
                  {fundamentals.z_zone ? (
                    <Badge tone={fundamentals.z_zone === "distress" ? "bad" : fundamentals.z_zone === "safe" ? "good" : "warn"}>
                      {String(fundamentals.z_zone)}
                    </Badge>
                  ) : (
                    "n/a"
                  )}
                </Row>
                {(fundamentals.red_flags ?? []).length > 0 && (
                  <Row label="red flags">
                    {(fundamentals.red_flags ?? []).map((flag) => (
                      <Badge key={flag} tone="bad">
                        {flag.replaceAll("_", " ")}
                      </Badge>
                    ))}
                  </Row>
                )}
                {typeof fundamentals.report_url === "string" && (
                  <a href={fundamentals.report_url} target="_blank" rel="noreferrer" className="text-xs hover:underline">
                    latest report ({String(fundamentals.report_age_days ?? "?")} days old)
                  </a>
                )}
              </>
            )}
          </div>
        )}
      </div>

      {decision && (
        <div className="mt-3 border-t border-black/5 pt-3 text-sm dark:border-white/10">
          {decision.source === "laya" && decision.action ? (
            <p>
              <span className="font-medium">Laya:</span> {decision.action.choice.replaceAll("_", " ")}{" "}
              <span className="text-neutral-500">(confidence {decision.action.confidence.toFixed(2)})</span>
              {decision.downside_risk && (
                <span className="text-neutral-500">
                  {" "}
                  · downside risk {pct(decision.downside_risk.noul, 0)}
                </span>
              )}
              {(decision.applied ?? []).length > 0 && (
                <span className="block text-xs text-neutral-500">{(decision.applied ?? []).join("; ")}</span>
              )}
            </p>
          ) : (
            <p className="text-neutral-500">Fuzzy rules only{decision.reason ? ` — ${decision.reason}` : ""}.</p>
          )}
        </div>
      )}

      {Object.keys(context.signals_used).length > 0 && (
        <p className="mt-2 text-xs text-neutral-500">
          Signals used:{" "}
          {Object.entries(context.signals_used)
            .map(([key, value]) => `${key.replaceAll("_", " ")} ${value.toFixed(4)}`)
            .join(" · ")}
        </p>
      )}
    </div>
  );
}

export function MarketContextPanel({ contexts }: { contexts: Record<string, MarketContext> }) {
  const entries = Object.entries(contexts);
  if (entries.length === 0) return null;
  return (
    <Card
      title="What the planner knew"
      subtitle="Per holding: forecast, news, technical and fundamental analysis, and the decision model's view."
    >
      <div className="grid gap-4 lg:grid-cols-2">
        {entries.map(([symbol, context]) => (
          <HoldingContext key={symbol} symbol={symbol} context={context} />
        ))}
      </div>
    </Card>
  );
}

const METHOD_LABEL: Record<string, string> = {
  fuzzy_ga: "Fuzzy GA (this plan)",
  pro_rata: "Pro-rata",
  largest_first: "Largest first",
  most_liquid_first: "Most liquid first",
};

export function BaselineComparison({ rows, currency }: { rows: BaselineRow[]; currency: string }) {
  if (rows.length === 0) return null;
  return (
    <Card
      title="Against naive liquidation"
      subtitle="The same request costed by three heuristics through the same simulator, so differences are the schedule's."
    >
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-xs uppercase tracking-wide text-neutral-500">
            <tr>
              <th className="pb-2">Method</th>
              <th className="pb-2 text-right">Expected cost</th>
              <th className="pb-2 text-right">Slippage</th>
              <th className="pb-2 text-right">Loss crystallised</th>
              <th className="pb-2 text-right">Days</th>
              <th className="pb-2 text-right">Feasible</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.method}
                className={`border-t border-black/5 dark:border-white/10 ${row.method === "fuzzy_ga" ? "font-medium" : ""}`}
              >
                <td className="py-2">{METHOD_LABEL[row.method] ?? row.method}</td>
                <td className="py-2 text-right tabular-nums">{money(row.expected_realized_loss, currency)}</td>
                <td className="py-2 text-right tabular-nums">{percent(row.expected_slippage_pct, 4)}</td>
                <td className="py-2 text-right tabular-nums">{money(row.crystallized_loss, currency)}</td>
                <td className="py-2 text-right tabular-nums">{row.days_required}</td>
                <td className="py-2 text-right">{row.feasible ? "yes" : "no"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

export function CostBreakdownCard({ breakdown, currency }: { breakdown: CostBreakdown; currency: string }) {
  const items: Array<[string, number, string]> = [
    ["Market impact (slippage)", breakdown.slippage, "square-root law on daily participation"],
    ["Timing risk", breakdown.timing_risk, "price risk while a slice waits; forecast band when available"],
    ["Forecast drift", breakdown.forecast_drift, "forecast return given up (+) or loss avoided (−) by selling"],
    ["FX conversion", breakdown.fx_conversion, "spread to bring proceeds into the base currency"],
  ];
  return (
    <Card title="Where the expected cost comes from" subtitle={`All figures in ${currency}.`}>
      <div className="space-y-1">
        {items.map(([label, value, hint]) => (
          <div key={label} className="flex items-baseline justify-between gap-3 text-sm">
            <span>
              {label} <span className="text-xs text-neutral-500">— {hint}</span>
            </span>
            <span className="tabular-nums">{money(value, currency)}</span>
          </div>
        ))}
        <div className="mt-2 flex items-baseline justify-between border-t border-black/5 pt-2 text-sm dark:border-white/10">
          <span>Realised gain/loss vs cost basis</span>
          <span className={`tabular-nums ${breakdown.realized_pnl < 0 ? "text-red-600" : "text-emerald-600"}`}>
            {money(breakdown.realized_pnl, currency)}
          </span>
        </div>
      </div>
    </Card>
  );
}
