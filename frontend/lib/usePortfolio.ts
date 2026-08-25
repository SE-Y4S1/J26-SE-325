"use client";

/**
 * Portfolio selection shared across the Withdraw, Optimize and Forecast screens.
 *
 * All three operate on "the portfolio the user is currently looking at", and the selection
 * lives in the URL (`?portfolio=<id>`) rather than in React state so a result page can be
 * linked to or refreshed without losing context.
 */

import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import * as platform from "@/lib/api/platform";
import type { Portfolio, PortfolioSummary } from "@/lib/api/types";

interface UsePortfolioResult {
  summaries: PortfolioSummary[];
  portfolio: Portfolio | null;
  loading: boolean;
  error: string | null;
  select: (id: number) => void;
  reload: () => Promise<void>;
}

export function usePortfolio(): UsePortfolioResult {
  const router = useRouter();
  const searchParams = useSearchParams();
  const selectedId = searchParams.get("portfolio");

  const [summaries, setSummaries] = useState<PortfolioSummary[]>([]);
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // Bumping this re-runs the effect; it is how `reload()` triggers a refetch without the
  // effect having to depend on a function identity that changes every render.
  const [refreshToken, setRefreshToken] = useState(0);

  useEffect(() => {
    // Guards against a slow response for a portfolio the user has already navigated away
    // from overwriting the newer one.
    let cancelled = false;

    // Everything that touches state happens after an await, so nothing runs synchronously in
    // the effect body -- that is what causes the cascading renders React 19 warns about.
    async function load() {
      try {
        const list = await platform.listPortfolios();
        if (cancelled) return;

        // Fall back to the first portfolio so the screens are never blank just because the
        // URL has no id yet, and so a deleted id does not strand the user on an empty page.
        //
        // Written as an explicit undefined check rather than `selectedId && ...`: an empty
        // `?portfolio=` makes that expression evaluate to "", which `??` does not treat as
        // missing, so the empty string would reach getPortfolio.
        const requested = selectedId
          ? list.find((p) => String(p.id) === selectedId)?.id
          : undefined;
        const target = requested ?? list[0]?.id;

        const selected =
          typeof target === "number" ? await platform.getPortfolio(target) : null;
        if (cancelled) return;

        setSummaries(list);
        setPortfolio(selected);
        setError(null);
      } catch (cause) {
        if (cancelled) return;
        setError(cause instanceof Error ? cause.message : "Could not load portfolios");
        setPortfolio(null);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, [selectedId, refreshToken]);

  const reload = useCallback(async () => {
    setRefreshToken((token) => token + 1);
  }, []);

  const select = useCallback(
    (id: number) => {
      const params = new URLSearchParams(searchParams.toString());
      params.set("portfolio", String(id));
      router.replace(`?${params.toString()}`);
    },
    [router, searchParams],
  );

  return { summaries, portfolio, loading, error, select, reload };
}
