"use client";

/**
 * How decisions were routed: one stacked bar plus a labelled legend with counts and shares.
 *
 * A single part-to-whole bar rather than a pie: the shares are compared along one baseline,
 * segments are separated by a 2px surface gap, and every segment is named in the legend, so
 * the reading never depends on telling colours apart. Hovering a segment shows its count.
 */

import { ROUTE_FILL, ROUTE_LABEL } from "./badges";
import type { Route } from "@/lib/api/audit";

const ORDER: (Route | "LEGACY")[] = ["AUTO", "VERIFY", "HUMAN_REVIEW", "LEGACY"];

export function RouteDistribution({ routes }: { routes: Partial<Record<Route | "LEGACY", number>> }) {
  const total = ORDER.reduce((sum, r) => sum + (routes[r] ?? 0), 0);
  if (!total) {
    return <p className="text-sm text-neutral-500">No decisions routed yet.</p>;
  }
  const present = ORDER.filter((r) => (routes[r] ?? 0) > 0);
  const share = (r: Route | "LEGACY") => (routes[r] ?? 0) / total;

  return (
    <figure>
      <div className="flex h-3 w-full gap-0.5 overflow-hidden rounded" role="img"
        aria-label={present.map((r) => `${ROUTE_LABEL[r]} ${routes[r]}`).join(", ")}>
        {present.map((r) => (
          <div
            key={r}
            title={`${ROUTE_LABEL[r]}: ${routes[r]} of ${total} (${Math.round(share(r) * 100)}%)`}
            className={`h-full first:rounded-l last:rounded-r ${ROUTE_FILL[r]}`}
            style={{ width: `${share(r) * 100}%`, minWidth: 4 }}
          />
        ))}
      </div>
      <figcaption>
        <ul className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1.5 text-sm sm:grid-cols-4">
          {ORDER.map((r) => (
            <li key={r} className="flex items-center gap-2">
              <span aria-hidden className={`size-2 shrink-0 rounded-full ${ROUTE_FILL[r]}`} />
              <span className="text-neutral-600 dark:text-neutral-400">{ROUTE_LABEL[r]}</span>
              <span className="ml-auto font-medium tabular-nums">
                {routes[r] ?? 0}
                <span className="ml-1 text-xs font-normal text-neutral-500">{Math.round(share(r) * 100)}%</span>
              </span>
            </li>
          ))}
        </ul>
      </figcaption>
    </figure>
  );
}
