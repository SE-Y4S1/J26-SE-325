"use client";

/**
 * Loads one thing from the audit service and classifies failures the way every audit screen
 * shows them: the service not running, access refused (a reviewer-only view while the service
 * enforces AUTH_REQUIRED), or an error with the service's own message.
 *
 * Same refresh-token pattern as usePortfolio: bumping the token re-runs the effect, and the
 * loader lives inside it, so no setState runs synchronously in the effect body.
 */

import { useEffect, useState } from "react";

import { ApiError } from "@/lib/api/client";

export type AuditLoadState<T> = {
  data: T | null;
  loading: boolean;
  unavailable: boolean;
  forbidden: boolean;
  error: string | null;
  refresh: () => void;
};

export function useAuditLoad<T>(load: () => Promise<T>, deps: unknown[] = []): AuditLoadState<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [unavailable, setUnavailable] = useState(false);
  const [forbidden, setForbidden] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [token, setToken] = useState(0);

  useEffect(() => {
    let cancelled = false;
    async function run() {
      try {
        const result = await load();
        if (cancelled) return;
        setData(result);
        setError(null);
        setUnavailable(false);
        setForbidden(false);
      } catch (cause) {
        if (cancelled) return;
        if (cause instanceof ApiError && (cause.isUnavailable || cause.status === 0)) setUnavailable(true);
        else if (cause instanceof ApiError && cause.isAuthError) setForbidden(true);
        else setError(cause instanceof Error ? cause.message : "Could not load");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    run();
    return () => {
      cancelled = true;
    };
    // `load` is recreated each render; callers pass what it depends on in `deps`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, ...deps]);

  return { data, loading, unavailable, forbidden, error, refresh: () => setToken((t) => t + 1) };
}
