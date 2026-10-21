"use client";

/**
 * Incorrect automatic decisions against human-review workload, one line per routing method.
 *
 * One y-axis; the method under test is the only coloured line (emerald, the app's accent),
 * baselines are neutral and told apart by dash pattern and a direct label, so identity never
 * rests on colour. Hovering a line's point shows its exact values. The methods table below the
 * chart on the Evaluation page is its table view.
 */

import { useState } from "react";

type Point = { human_review_rate: number; incorrect_automatic_rate: number };

const SERIES: { key: string; label: string; stroke: string; dash?: string; width: number }[] = [
  { key: "assurance", label: "Assurance engine", stroke: "stroke-emerald-700 dark:stroke-emerald-500", width: 2.5 },
  { key: "learned_deferral", label: "Learned deferral", stroke: "stroke-neutral-600 dark:stroke-neutral-300", dash: "6 3", width: 1.5 },
  { key: "confidence_only", label: "Confidence only", stroke: "stroke-neutral-500 dark:stroke-neutral-400", dash: "2 3", width: 1.5 },
  { key: "risk_band", label: "Risk only", stroke: "stroke-neutral-400 dark:stroke-neutral-500", dash: "8 3 2 3", width: 1.5 },
];

const W = 640, H = 300, L = 56, R = 120, T = 16, B = 40;

export function DeferralChart({ curves }: { curves: Record<string, Point[]> }) {
  const [hover, setHover] = useState<{ label: string; p: Point; x: number; y: number } | null>(null);
  const series = SERIES.filter((s) => curves[s.key]?.length);
  if (!series.length) return <p className="text-sm text-neutral-500">No curves in this result.</p>;

  const all = series.flatMap((s) => curves[s.key]);
  const xMax = Math.max(...all.map((p) => p.human_review_rate)) || 1;
  const yMax = Math.max(...all.map((p) => p.incorrect_automatic_rate)) * 1.05 || 1;
  const x = (v: number) => L + (v / xMax) * (W - L - R);
  const y = (v: number) => T + (1 - v / yMax) * (H - T - B);
  const ticks = (max: number) => [0, 0.25, 0.5, 0.75, 1].map((f) => f * max);
  const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img"
        aria-label="Incorrect automatic decisions by human-review rate for each routing method">
        {ticks(yMax).map((v) => (
          <g key={`y${v}`}>
            <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} className="stroke-black/5 dark:stroke-white/10" />
            <text x={L - 6} y={y(v)} textAnchor="end" dominantBaseline="middle" className="fill-neutral-500 text-[10px]">{pct(v, 2)}</text>
          </g>
        ))}
        {ticks(xMax).map((v) => (
          <text key={`x${v}`} x={x(v)} y={H - B + 16} textAnchor="middle" className="fill-neutral-500 text-[10px]">{pct(v, 0)}</text>
        ))}
        <text x={(L + W - R) / 2} y={H - 6} textAnchor="middle" className="fill-neutral-500 text-[11px]">Decisions sent to human review</text>
        {series.map((s) => {
          const pts = curves[s.key];
          const last = pts[pts.length - 1];
          return (
            <g key={s.key}>
              <polyline fill="none" className={s.stroke} strokeWidth={s.width} strokeDasharray={s.dash}
                strokeLinejoin="round" strokeLinecap="round"
                points={pts.map((p) => `${x(p.human_review_rate)},${y(p.incorrect_automatic_rate)}`).join(" ")} />
              <text x={x(last.human_review_rate) + 6} y={y(last.incorrect_automatic_rate)} dominantBaseline="middle"
                className="fill-neutral-700 text-[10px] dark:fill-neutral-300">{s.label}</text>
              {pts.map((p, i) => (
                <circle key={i} cx={x(p.human_review_rate)} cy={y(p.incorrect_automatic_rate)} r={8} fill="transparent"
                  onMouseEnter={() => setHover({ label: s.label, p, x: x(p.human_review_rate), y: y(p.incorrect_automatic_rate) })}
                  onMouseLeave={() => setHover(null)} />
              ))}
            </g>
          );
        })}
        {hover && <circle cx={hover.x} cy={hover.y} r={4} className="fill-neutral-900 stroke-white dark:fill-white dark:stroke-neutral-900" strokeWidth={2} />}
      </svg>
      {hover && (
        <div className="pointer-events-none absolute rounded-md border border-black/10 bg-white px-2 py-1 text-xs shadow-sm dark:border-white/15 dark:bg-neutral-900"
          style={{ left: `${(hover.x / W) * 100}%`, top: `${(hover.y / H) * 100}%`, transform: "translate(8px, -110%)" }}>
          <p className="font-medium">{hover.label}</p>
          <p className="tabular-nums text-neutral-600 dark:text-neutral-400">
            review {pct(hover.p.human_review_rate)} · incorrect automatic {pct(hover.p.incorrect_automatic_rate, 3)}
          </p>
        </div>
      )}
    </div>
  );
}
