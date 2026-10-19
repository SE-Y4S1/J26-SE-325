import { useState } from "react";
import { formatMetric, PIPELINE_CHECK, RESEARCH_STUDIES, type Study } from "../studies";

export function Research() {
  const [studyId, setStudyId] = useState(RESEARCH_STUDIES[0].id);
  const study = RESEARCH_STUDIES.find((item) => item.id === studyId) ?? RESEARCH_STUDIES[0];
  const maxAuroc = Math.max(...study.models.map((model) => model.auroc ?? 0), 0.01);

  return (
    <section>
      <h1>Research and benchmark results</h1>
      <p className="muted">
        Numbers come from the saved experiment files. A model is marked “not measured” when that file skipped it. The synthetic pipeline check is kept separate because it is not a public-dataset benchmark.
      </p>
      <label>
        Executed study
        <select value={study.id} onChange={(event) => setStudyId(event.target.value)}>
          {RESEARCH_STUDIES.map((item) => (
            <option key={item.id} value={item.id}>
              {item.title}
            </option>
          ))}
        </select>
      </label>
      <div className="grid cards" style={{ marginTop: "0.85rem" }}>
        <article className="card">
          <h2>Rows used</h2>
          <p className="metric">{study.rows ?? "not recorded"}</p>
          <p className="scope">{study.subset ?? study.split}</p>
        </article>
        <article className="card">
          <h2>Highest measured AUROC</h2>
          <p className="metric">{study.bestModel ?? "not measured"}</p>
          <p className="scope">Research benchmark: {study.research ? "yes" : "no"}</p>
        </article>
        <article className="card">
          <h2>LSTM fusion weight</h2>
          <p className="metric">{formatMetric(study.fusionAlpha, 1)}</p>
          <p className="scope">{study.fusionNote ?? "Fusion was not fit for this study."}</p>
        </article>
        <article className="card">
          <h2>Device-field lift</h2>
          <p className="metric">{formatMetric(study.deviceLift)}</p>
          <p className="scope">
            {study.deviceWith === null
              ? "This file does not contain a device-feature ablation."
              : `Logistic AUROC ${formatMetric(study.deviceWithout)} without identity fields, ${formatMetric(study.deviceWith)} with them.`}
          </p>
        </article>
      </div>
      <article className="card" style={{ marginTop: "0.85rem" }}>
        <h2>Test AUROC</h2>
        <svg className="chart" viewBox="0 0 680 220" role="img" aria-label="Bar chart of test AUROC by model">
          {study.models.map((model, index) => {
            const height = ((model.auroc ?? 0) / maxAuroc) * 160;
            const x = 36 + index * 78;
            return (
              <g key={model.model}>
                <rect x={x} y={190 - height} width="46" height={height} fill={model.skipped ? "#3a465c" : "#7eb0ff"}>
                  <title>{model.skipped ? `${model.label}: not measured` : `${model.label}: ${formatMetric(model.auroc)}`}</title>
                </rect>
                <text x={x + 23} y="208" textAnchor="middle" fill="#9aafcc" fontSize="10">
                  {model.label.split(" ")[0]}
                </text>
              </g>
            );
          })}
        </svg>
        <p className="legend">Blue bars are measured AUROC. Grey bars are skipped models and are not scores.</p>
      </article>
      <article className="card" style={{ marginTop: "0.85rem" }}>
        <h2>Comparison</h2>
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>AUROC</th>
              <th>Average precision</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1</th>
              <th>Fraud blocked</th>
              <th>Legitimate blocked</th>
            </tr>
          </thead>
          <tbody>
            {study.models.map((model) => (
              <tr key={model.model}>
                <td>{model.label}</td>
                <td>{model.skipped ? "skipped" : formatMetric(model.auroc)}</td>
                <td>{formatMetric(model.averagePrecision)}</td>
                <td>{formatMetric(model.precision)}</td>
                <td>{formatMetric(model.recall)}</td>
                <td>{formatMetric(model.f1)}</td>
                <td>{pair(model.fraudBlocked, model.fraudRows)}</td>
                <td>{pair(model.legitimateBlocked, model.legitimateRows)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {study.models.filter((model) => model.skipped).map((model) => (
          <p key={model.model} className="scope">{model.label}: {model.notes[0] ?? "Skipped in this experiment file."}</p>
        ))}
        <p className="scope">
          Logistic single-row median {formatMetric(study.singleRowMs, 3)} ms. Batch throughput {formatMetric(study.batchRowsPerSecond, 1)} rows/s.
          Graph forward {formatMetric(study.graphRowsPerSecond, 1)} rows/s. {study.latencyNote ?? "No separate latency note was stored."}
        </p>
      </article>
      <PipelineCheck />
    </section>
  );
}

function pair(part: number | null, total: number | null): string {
  if (part === null || total === null) return "not measured";
  return `${part} / ${total}`;
}

function PipelineCheck() {
  const study: Study = PIPELINE_CHECK;
  return (
    <article className="card" style={{ marginTop: "0.85rem" }}>
      <h2>Synthetic pipeline check</h2>
      <p className="notice">
        This folder is a pipeline check. results_are_research_benchmark is {study.research ? "true" : "false"}. Do not cite these scores as IEEE-CIS, Elliptic, or DGraphFin results.
      </p>
      <p className="scope">{study.rows ?? "unknown"} rows. Highest measured AUROC: {study.bestModel ?? "not measured"}.</p>
    </article>
  );
}
