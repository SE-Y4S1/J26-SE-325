import { useEffect, useState } from "react";
import { ApiError, getDecisions, getHealth, type DecisionList, type Health } from "../api";
import { formatMetric, RESEARCH_STUDIES } from "../studies";

const primary = RESEARCH_STUDIES[0];

export function Overview() {
  const [health, setHealth] = useState<Health | null>(null);
  const [history, setHistory] = useState<DecisionList | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    Promise.all([getHealth(), getDecisions()])
      .then(([nextHealth, nextHistory]) => {
        if (!active) return;
        setHealth(nextHealth);
        setHistory(nextHistory);
        setError(null);
      })
      .catch((reason: unknown) => {
        if (!active) return;
        setError(reason instanceof ApiError ? reason.message : "The fraud API is not reachable.");
      });
    return () => {
      active = false;
    };
  }, []);

  const counts = history?.counts ?? { ALLOW: 0, STEP_UP: 0, BLOCK: 0 };
  const total = counts.ALLOW + counts.STEP_UP + counts.BLOCK;
  const max = Math.max(total, 1);

  return (
    <section>
      <h1>Overview</h1>
      <p className="muted">
        Counts below are decisions made by this running API process. They are not the offline IEEE or Elliptic benchmarks.
      </p>
      {error ? <p className="error">{error}</p> : null}
      <div className="grid cards">
        <article className="card">
          <h2>Transactions scored</h2>
          <p className="metric">{history ? total : "…"}</p>
          <p className="scope">{history?.scope ?? "Waiting for the API"}</p>
        </article>
        <article className="card">
          <h2>Allowed</h2>
          <p className="metric ALLOW">{history ? counts.ALLOW : "…"}</p>
        </article>
        <article className="card">
          <h2>Step-up</h2>
          <p className="metric STEP_UP">{history ? counts.STEP_UP : "…"}</p>
        </article>
        <article className="card">
          <h2>Blocked</h2>
          <p className="metric BLOCK">{history ? counts.BLOCK : "…"}</p>
        </article>
      </div>
      <div className="grid two" style={{ marginTop: "0.85rem" }}>
        <article className="card">
          <h2>Decision distribution</h2>
          {total === 0 ? (
            <p>No transactions have been scored in this API process.</p>
          ) : (
            <div className="bars">
              {(["ALLOW", "STEP_UP", "BLOCK"] as const).map((name) => (
                <div className="bar-row" key={name}>
                  <span className={name}>{name}</span>
                  <div className="track" aria-hidden="true">
                    <div className="fill" style={{ width: `${(counts[name] / max) * 100}%` }} />
                  </div>
                  <span>{counts[name]}</span>
                </div>
              ))}
            </div>
          )}
        </article>
        <article className="card">
          <h2>What this session cannot show</h2>
          <p>Fraud detection rate and false-positive rate need labelled outcomes. This live API does not receive fraud labels.</p>
          <p>
            Offline single-row timing on the IEEE 20,000-row logistic regression scorer:{" "}
            {formatMetric(primary.singleRowMs, 3)} ms median. That figure is an in-memory timing from the saved experiment, not a load test of this API.
          </p>
          <p className="scope">
            Connected model: {health ? `${health.model_version} (benchmark model: ${health.benchmark_model ? "yes" : "no"})` : "unavailable"}
            . Kafka: {health ? (health.kafka_enabled ? "enabled" : "disabled") : "unavailable"}.
          </p>
        </article>
      </div>
      <article className="card" style={{ marginTop: "0.85rem" }}>
        <h2>Recent decisions</h2>
        <DecisionTable rows={history?.decisions.slice(0, 8) ?? []} empty="Score a transaction on the Live page to fill this table." />
      </article>
    </section>
  );
}

export function DecisionTable({
  rows,
  empty,
  onSelect,
  selectedId,
}: {
  rows: DecisionList["decisions"];
  empty: string;
  onSelect?: (id: string) => void;
  selectedId?: string;
}) {
  if (!rows.length) return <p>{empty}</p>;
  return (
    <table>
      <thead>
        <tr>
          <th>Transaction</th>
          <th>Scored at</th>
          <th>Score</th>
          <th>Decision</th>
          <th>Model</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr
            key={`${row.transaction_id}-${row.scored_at ?? ""}`}
            className={onSelect ? "clickable" : undefined}
            aria-selected={selectedId === row.transaction_id}
            onClick={() => onSelect?.(row.transaction_id)}
          >
            <td>{row.transaction_id}</td>
            <td>{row.scored_at ?? "not recorded"}</td>
            <td>{row.fraud_score.toFixed(4)}</td>
            <td className={row.decision}>{row.decision}</td>
            <td>{row.model_version}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
