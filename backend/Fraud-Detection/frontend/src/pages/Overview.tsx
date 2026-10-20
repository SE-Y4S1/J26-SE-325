import { useEffect, useState } from "react";
import { ApiError, getDecisions, getHealth, type DecisionList, type Health } from "../api";
import { formatMetric, RESEARCH_STUDIES } from "../studies";

const primary = RESEARCH_STUDIES[0];
const fused = primary.models.find((model) => model.model === "fused_lstm_gnn");

function countOrMissing(value: number | null | undefined): string {
  return value === null || value === undefined ? "not measured" : String(value);
}

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

  const session = history?.counts ?? { ALLOW: 0, STEP_UP: 0, BLOCK: 0 };
  const sessionTotal = session.ALLOW + session.STEP_UP + session.BLOCK;
  const dataset =
    fused?.allow != null && fused.stepUp != null && fused.block != null
      ? { ALLOW: fused.allow, STEP_UP: fused.stepUp, BLOCK: fused.block }
      : null;
  const datasetTotal = dataset ? dataset.ALLOW + dataset.STEP_UP + dataset.BLOCK : 0;
  const datasetMax = Math.max(datasetTotal, 1);

  return (
    <section>
      <h1>Overview</h1>
      <p className="muted">
        The decision cards use the saved IEEE-CIS test split for the fused LSTM + GNN model. They come from the executed experiment file, not from the transactions scored in this browser session.
      </p>
      {error ? <p className="error">{error}</p> : null}
      <div className="grid cards">
        <article className="card">
          <h2>Transactions analysed</h2>
          <p className="metric">{dataset ? datasetTotal : "not measured"}</p>
          <p className="scope">{primary.subset ?? primary.title}</p>
        </article>
        <article className="card">
          <h2>Allowed</h2>
          <p className="metric ALLOW">{countOrMissing(dataset?.ALLOW)}</p>
        </article>
        <article className="card">
          <h2>Step-up</h2>
          <p className="metric STEP_UP">{countOrMissing(dataset?.STEP_UP)}</p>
        </article>
        <article className="card">
          <h2>Blocked</h2>
          <p className="metric BLOCK">{countOrMissing(dataset?.BLOCK)}</p>
        </article>
      </div>
      <div className="grid two" style={{ marginTop: "0.85rem" }}>
        <article className="card">
          <h2>Dataset decision distribution</h2>
          {dataset ? (
            <div className="bars">
              {(["ALLOW", "STEP_UP", "BLOCK"] as const).map((name) => (
                <div className="bar-row" key={name}>
                  <span className={name}>{name}</span>
                  <div className="track" aria-hidden="true">
                    <div className={`fill ${name}`} style={{ width: `${(dataset[name] / datasetMax) * 100}%` }} />
                  </div>
                  <span>{dataset[name]}</span>
                </div>
              ))}
            </div>
          ) : (
            <p>The saved IEEE file does not contain gateway counts for the fused model.</p>
          )}
          <p className="scope">Model: {fused?.label ?? "Fused LSTM + GNN"}. Study: {primary.title}.</p>
        </article>
        <article className="card">
          <h2>Labelled outcomes on the same split</h2>
          <p>
            Fraud blocked: {countOrMissing(fused?.fraudBlocked)} of {countOrMissing(fused?.fraudRows)}. Fraud sent to step-up: {countOrMissing(fused?.fraudStepUp)} of {countOrMissing(fused?.fraudRows)}.
          </p>
          <p>
            Legitimate transactions blocked: {countOrMissing(fused?.legitimateBlocked)} of {countOrMissing(fused?.legitimateRows)}. Legitimate transactions challenged, meaning step-up or block: {countOrMissing(fused?.legitimateChallenged)} of {countOrMissing(fused?.legitimateRows)}.
          </p>
          <p>
            Offline single-row timing on the IEEE 20,000-row logistic regression scorer: {formatMetric(primary.singleRowMs, 3)} ms median. That figure is an in-memory timing from the saved experiment, not a load test of this API.
          </p>
          <p className="scope">
            Connected model: {health ? `${health.model_version} (benchmark model: ${health.benchmark_model ? "yes" : "no"})` : "unavailable"}
            . Kafka: {health ? (health.kafka_enabled ? "enabled" : "disabled") : "unavailable"}.
          </p>
        </article>
      </div>
      <article className="card" style={{ marginTop: "0.85rem" }}>
        <h2>This API session</h2>
        <p className="scope">{history?.scope ?? "Waiting for the API"}. Allowed {history ? session.ALLOW : "…"}, step-up {history ? session.STEP_UP : "…"}, blocked {history ? session.BLOCK : "…"}. Session total: {history ? sessionTotal : "…"}.</p>
      </article>
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
