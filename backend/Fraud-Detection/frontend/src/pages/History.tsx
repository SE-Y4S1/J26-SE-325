import { useEffect, useMemo, useState } from "react";
import { ApiError, getDecisions, type DecisionName, type StoredDecision } from "../api";
import { DecisionTable } from "./Overview";

export function History() {
  const [rows, setRows] = useState<StoredDecision[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [decision, setDecision] = useState<DecisionName | "ALL">("ALL");
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    getDecisions()
      .then((body) => {
        setRows(body.decisions);
        setError(null);
      })
      .catch((reason: unknown) => {
        setRows(null);
        setError(reason instanceof ApiError ? reason.message : "The fraud API is not reachable.");
      });
  }, []);

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return (rows ?? []).filter((row) => {
      const matchesDecision = decision === "ALL" || row.decision === decision;
      const haystack = `${row.transaction_id} ${row.model_version} ${(row.reasons ?? []).join(" ")}`.toLowerCase();
      return matchesDecision && (!needle || haystack.includes(needle));
    });
  }, [rows, query, decision]);

  const detail = visible.find((row) => row.transaction_id === selected) ?? visible[0];

  return (
    <section>
      <h1>Transaction history</h1>
      <p className="muted">These rows are the decisions kept in memory by the running API. Account keys and device keys are not stored in this list.</p>
      {error ? <p className="error">{error}</p> : null}
      <div className="filters">
        <input aria-label="Search decisions" value={query} placeholder="Search id, model, or reason" onChange={(event) => setQuery(event.target.value)} />
        <select aria-label="Filter by decision" value={decision} onChange={(event) => setDecision(event.target.value as DecisionName | "ALL")}>
          <option value="ALL">All decisions</option>
          <option value="ALLOW">ALLOW</option>
          <option value="STEP_UP">STEP_UP</option>
          <option value="BLOCK">BLOCK</option>
        </select>
      </div>
      {rows === null && !error ? <p>Loading decisions…</p> : (
        <DecisionTable
          rows={visible}
          empty={error ? "Decisions are unavailable." : "No decisions match this filter."}
          onSelect={setSelected}
          selectedId={detail?.transaction_id}
        />
      )}
      {detail ? (
        <article className="card" style={{ marginTop: "0.85rem" }}>
          <h2>{detail.transaction_id}</h2>
          <p className={detail.decision}>{detail.decision}</p>
          <p>Score {detail.fraud_score.toFixed(4)}</p>
          <p>Scored at {detail.scored_at ?? "not recorded"}</p>
          <p>Reasons: {detail.reasons?.length ? detail.reasons.join(", ") : "not recorded"}</p>
        </article>
      ) : null}
    </section>
  );
}
