/**
 * Labels for Component 3's routes and integrity states.
 *
 * Route colours are a status palette validated for both modes (dataviz validator, light:
 * emerald-700 / amber-600 / red-700; dark: emerald-600 / amber-600 / rose-600). Colour is
 * never the only carrier: every badge shows its words, and the same words the landing page
 * uses for these outcomes.
 */

import type { Route } from "@/lib/api/audit";

export const ROUTE_LABEL: Record<Route | "LEGACY", string> = {
  AUTO: "Automatic",
  VERIFY: "Verification required",
  HUMAN_REVIEW: "Human review",
  LEGACY: "Legacy policy",
};

/** Solid fills for bars and dots. */
export const ROUTE_FILL: Record<Route | "LEGACY", string> = {
  AUTO: "bg-emerald-700 dark:bg-emerald-600",
  VERIFY: "bg-amber-600",
  HUMAN_REVIEW: "bg-red-700 dark:bg-rose-600",
  LEGACY: "bg-neutral-400 dark:bg-neutral-500",
};

const ROUTE_TINT: Record<Route | "LEGACY", string> = {
  AUTO: "bg-emerald-50 text-emerald-800 ring-emerald-700/20 dark:bg-emerald-950 dark:text-emerald-300 dark:ring-emerald-500/30",
  VERIFY: "bg-amber-50 text-amber-800 ring-amber-600/25 dark:bg-amber-950 dark:text-amber-300 dark:ring-amber-500/30",
  HUMAN_REVIEW: "bg-red-50 text-red-800 ring-red-700/20 dark:bg-rose-950 dark:text-rose-300 dark:ring-rose-500/30",
  LEGACY: "bg-neutral-100 text-neutral-700 ring-black/10 dark:bg-neutral-800 dark:text-neutral-300 dark:ring-white/15",
};

const pill = "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset whitespace-nowrap";

/** The route a record was assigned; records routed by the legacy P001 policy read "Legacy". */
export function routeOf(assurance: { route: Route; signalQuality: string } | null | undefined): Route | "LEGACY" {
  if (!assurance || assurance.signalQuality === "LEGACY") return "LEGACY";
  return assurance.route;
}

export function RouteBadge({ route }: { route: Route | "LEGACY" }) {
  return (
    <span className={`${pill} ${ROUTE_TINT[route]}`}>
      <span aria-hidden className={`size-1.5 rounded-full ${ROUTE_FILL[route]}`} />
      {ROUTE_LABEL[route]}
    </span>
  );
}

type IntegrityTone = "good" | "bad" | "pending" | "neutral";

const INTEGRITY: Record<string, { label: string; tone: IntegrityTone }> = {
  VERIFIED_ON_CHAIN: { label: "Verified on-chain", tone: "good" },
  VERIFIED_SIMULATED: { label: "Verified (simulated ledger)", tone: "good" },
  RECORDED_ON_CHAIN: { label: "Anchored", tone: "good" },
  PENDING_HUMAN_REVIEW: { label: "Anchored · awaiting review", tone: "pending" },
  ANCHOR_PENDING: { label: "Not yet anchored", tone: "pending" },
  BLOCKCHAIN_FAILED: { label: "Anchor failed", tone: "bad" },
  ON_CHAIN_MISSING: { label: "Missing on-chain", tone: "bad" },
  MODIFIED: { label: "Tampering detected", tone: "bad" },
  CHAIN_BROKEN: { label: "Review link broken", tone: "bad" },
  TAMPER_SUSPECTED: { label: "Altered · verify", tone: "bad" },
  NOT_FOUND: { label: "Not found", tone: "neutral" },
  BLOCKCHAIN_UNAVAILABLE: { label: "Ledger unreachable", tone: "neutral" },
};

const TONE: Record<IntegrityTone, string> = {
  good: "bg-emerald-50 text-emerald-800 ring-emerald-700/20 dark:bg-emerald-950 dark:text-emerald-300 dark:ring-emerald-500/30",
  bad: "bg-red-50 text-red-800 ring-red-700/20 dark:bg-rose-950 dark:text-rose-300 dark:ring-rose-500/30",
  pending: "bg-amber-50 text-amber-800 ring-amber-600/25 dark:bg-amber-950 dark:text-amber-300 dark:ring-amber-500/30",
  neutral: "bg-neutral-100 text-neutral-700 ring-black/10 dark:bg-neutral-800 dark:text-neutral-300 dark:ring-white/15",
};

export function integrityLabel(status: string): string {
  return INTEGRITY[status]?.label ?? status.replace(/_/g, " ").toLowerCase();
}

export function IntegrityBadge({ status }: { status: string }) {
  const entry = INTEGRITY[status] ?? { label: integrityLabel(status), tone: "neutral" as const };
  return <span className={`${pill} ${TONE[entry.tone]}`}>{entry.label}</span>;
}

const OUTCOME: Record<string, string> = {
  APPROVE: "Approved",
  REJECT: "Rejected",
  PENDING_HUMAN_REVIEW: "Held",
  HUMAN_APPROVED: "Approved after review",
  HUMAN_REJECTED: "Rejected after review",
};

export function outcomeLabel(finalDecision: string): string {
  return OUTCOME[finalDecision] ?? finalDecision;
}

export function shortHash(hash: string | null | undefined, size = 10): string {
  if (!hash) return "—";
  return hash.length > size + 6 ? `${hash.slice(0, size)}…${hash.slice(-4)}` : hash;
}
