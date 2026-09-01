"use client";

/**
 * Component 3 — Privacy-Preserving AI-to-Smart-Contract Audit Bridge (Abeysekara W C S M).
 *
 * Reads the bridge's own records and statistics. Anchoring, hashing and verification all
 * happen in his Express service against the Solidity registry; this screen only displays
 * what it reports.
 *
 * The table is rendered from whatever keys the service returns rather than a fixed column
 * list, because the record shape is his to change and this page should not break when it does.
 */

import { useCallback, useEffect, useState } from "react";

import { Button, Card, Notice, Spinner, Stat } from "@/components/ui";
import { auditRecords, auditStats, type AuditRecord } from "@/lib/api/audit";
import { ApiError } from "@/lib/api/client";

export default function AuditPage() {
  const [records, setRecords] = useState<AuditRecord[]>([]);
  const [stats, setStats] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [unavailable, setUnavailable] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    setUnavailable(false);
    try {
      const [rows, summary] = await Promise.all([auditRecords(), auditStats()]);
      setRecords(Array.isArray(rows) ? rows : []);
      setStats(summary);
    } catch (cause) {
      // Same treatment the Forecast screen gives a 503: a service that is not running is a
      // state to explain, not an error to alarm anyone with.
      if (cause instanceof ApiError && (cause.isUnavailable || cause.status === 0)) {
        setUnavailable(true);
      } else {
        setError(cause instanceof Error ? cause.message : "Could not load the audit trail");
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const columns = records.length > 0 ? Object.keys(records[0]).slice(0, 5) : [];

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold">Audit trail</h1>
          <p className="mt-1 text-sm text-neutral-500">
            Tamper-evident record of AI decisions anchored on-chain. Component 3 · Abeysekara
            W C S M.
          </p>
        </div>
        <Button variant="secondary" onClick={() => void load()} disabled={loading}>
          Refresh
        </Button>
      </div>

      {loading && <Spinner label="Loading audit records" />}

      {unavailable && (
        <Notice tone="info" title="Audit bridge is not running">
          Start it with <code>docker compose up component3</code>, or{" "}
          <code>node src/index.js</code> from <code>Blockchain-Auditability/backend</code>.
          Every other screen works without it.
        </Notice>
      )}

      {error && (
        <Notice tone="error" title="Could not load">
          {error}
        </Notice>
      )}

      {!loading && !unavailable && !error && (
        <>
          {stats && Object.keys(stats).length > 0 && (
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              {Object.entries(stats)
                .filter(([, value]) => typeof value === "number" || typeof value === "string")
                .slice(0, 4)
                .map(([key, value]) => (
                  <Stat key={key} label={key.replace(/_/g, " ")} value={String(value)} />
                ))}
            </div>
          )}

          <Card
            title="Anchored decisions"
            subtitle="Each row is a decision committed to the registry with its model version."
          >
            {records.length === 0 ? (
              <p className="text-sm text-neutral-500">
                No decisions anchored yet. The bridge records them as the platform&apos;s
                engines emit decisions.
              </p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="text-left text-xs uppercase tracking-wide text-neutral-500">
                    <tr>
                      {columns.map((column) => (
                        <th key={column} className="pb-2">
                          {column.replace(/_/g, " ")}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {records.slice(0, 25).map((row, index) => (
                      <tr key={index} className="border-t border-neutral-200/60">
                        {columns.map((column) => (
                          <td key={column} className="py-2 font-mono text-xs">
                            {String(row[column] ?? "—").slice(0, 48)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </>
      )}
    </div>
  );
}
