import elliptic from "../../experiments/results/elliptic/metrics.json";
import ieeeWide from "../../experiments/results/ieee-cis-100000/metrics.json";
import ieeeDev from "../../experiments/results/ieee-cis/metrics.json";
import synthetic from "../../experiments/results/synthetic-fixture/metrics.json";

export const MODEL_LABELS: Record<string, string> = {
  logistic_regression: "Logistic regression",
  svm: "Linear SVM",
  random_forest: "Random forest",
  static_gcn: "Static GCN",
  static_gat: "Static GAT",
  lstm_only: "LSTM autoencoder",
  gnn_only: "GNN-only",
  fused_lstm_gnn: "Fused LSTM + GNN",
};

export type ModelResult = {
  model: string;
  label: string;
  auroc: number | null;
  averagePrecision: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  skipped: boolean;
  notes: string[];
  fraudBlocked: number | null;
  fraudRows: number | null;
  fraudStepUp: number | null;
  legitimateBlocked: number | null;
  legitimateRows: number | null;
  legitimateChallenged: number | null;
  allow: number | null;
  stepUp: number | null;
  block: number | null;
};

export type AttackResult = {
  name: string;
  auroc: number | null;
  cleanAuroc: number | null;
  change: number | null;
  fraudStillBlocked: number | null;
  fraudRows: number | null;
  cases: number | null;
  parts: number | null;
  fullyAllowed: number | null;
  anyPartBlocked: number | null;
  notes: string[];
};

export type Study = {
  id: string;
  title: string;
  research: boolean;
  rows: number | null;
  split: string;
  subset: string | null;
  models: ModelResult[];
  bestModel: string | null;
  latencyNote: string | null;
  singleRowMs: number | null;
  batchRowsPerSecond: number | null;
  graphRowsPerSecond: number | null;
  fusionAlpha: number | null;
  fusionNote: string | null;
  deviceLift: number | null;
  deviceWith: number | null;
  deviceWithout: number | null;
  attacks: AttackResult[];
  attackModel: string | null;
};

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

function num(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function textList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function parseModel(raw: unknown): ModelResult {
  const row = asRecord(raw);
  const gateway = asRecord(row.gateway);
  const name = String(row.model ?? "unknown");
  const fusion = asRecord(asRecord(row.parameters).fusion);
  const notes = textList(row.notes);
  if (typeof fusion.alpha_on_lstm === "number") {
    notes.push(`Validation mixture weight on the LSTM stream: ${fusion.alpha_on_lstm}.`);
  }
  return {
    model: name,
    label: MODEL_LABELS[name] ?? name,
    auroc: num(row.auroc),
    averagePrecision: num(row.average_precision),
    precision: num(row.precision),
    recall: num(row.recall),
    f1: num(row.f1),
    skipped: row.skipped === true || row.auroc === null,
    notes,
    fraudBlocked: num(gateway.fraud_blocked),
    fraudRows: num(gateway.fraud_rows),
    fraudStepUp: num(gateway.fraud_step_up),
    legitimateBlocked: num(gateway.legitimate_blocked),
    legitimateRows: num(gateway.legitimate_rows),
    legitimateChallenged: num(gateway.legitimate_challenged),
    allow: num(gateway.allow),
    stepUp: num(gateway.step_up),
    block: num(gateway.block),
  };
}

function parseAttack(name: string, raw: unknown): AttackResult | null {
  const row = asRecord(raw);
  if (!Object.keys(row).length) return null;
  return {
    name,
    auroc: num(row.auroc),
    cleanAuroc: num(row.clean_auroc),
    change: num(row.auroc_change),
    fraudStillBlocked: num(row.fraud_still_blocked),
    fraudRows: num(row.fraud_rows),
    cases: num(row.cases),
    parts: num(row.parts),
    fullyAllowed: num(row.fully_allowed),
    anyPartBlocked: num(row.any_part_blocked),
    notes: textList(row.notes),
  };
}

function parseStudy(id: string, title: string, raw: unknown): Study {
  const data = asRecord(raw);
  const models = Array.isArray(data.models) ? data.models.map(parseModel) : [];
  const ranked = models.filter((model) => model.auroc !== null);
  const best = ranked.reduce<ModelResult | null>(
    (winner, model) => (winner === null || (model.auroc ?? -1) > (winner.auroc ?? -1) ? model : winner),
    null,
  );
  const latency = asRecord(data.scoring_latency);
  const logistic = asRecord(latency.logistic_regression);
  const graph = asRecord(latency.gnn_only_full_graph);
  const fused = models.find((model) => model.model === "fused_lstm_gnn");
  const alphaNote = fused?.notes.find((note) => note.includes("weight on the LSTM")) ?? null;
  const lift = asRecord(data.device_feature_lift);
  const attacksRaw = asRecord(data.adversarial);
  const attacks = ["camouflage", "slow_drift", "structuring"]
    .map((name) => parseAttack(name, attacksRaw[name]))
    .filter((attack): attack is AttackResult => attack !== null);
  const source = asRecord(data.source_metadata);
  return {
    id,
    title,
    research: data.results_are_research_benchmark === true,
    rows: num(data.rows),
    split: typeof data.split_strategy === "string" ? data.split_strategy : "not recorded",
    subset: typeof source.subset === "string" ? source.subset : null,
    models,
    bestModel: best?.label ?? null,
    latencyNote: textList(logistic.notes)[0] ?? textList(graph.notes)[0] ?? null,
    singleRowMs: num(logistic.single_row_latency_ms_p50),
    batchRowsPerSecond: num(logistic.batch_throughput_rows_per_s),
    graphRowsPerSecond: num(graph.batch_throughput_rows_per_s),
    fusionAlpha: fused ? readAlpha(data) : null,
    fusionNote: alphaNote,
    deviceLift: num(lift.lift),
    deviceWith: num(lift.auroc_with_device_features),
    deviceWithout: num(lift.auroc_without_device_features),
    attacks,
    attackModel: typeof attacksRaw.scored_model === "string" ? attacksRaw.scored_model : null,
  };
}

function readAlpha(data: Record<string, unknown>): number | null {
  const models = Array.isArray(data.models) ? data.models : [];
  for (const raw of models) {
    const row = asRecord(raw);
    if (row.model !== "fused_lstm_gnn") continue;
    return num(asRecord(asRecord(row.parameters).fusion).alpha_on_lstm);
  }
  return null;
}

export const RESEARCH_STUDIES: Study[] = [
  parseStudy("ieee-cis", "IEEE-CIS, earliest 20,000 labelled rows", ieeeDev),
  parseStudy("ieee-cis-100000", "IEEE-CIS, earliest 100,000 labelled rows", ieeeWide),
  parseStudy("elliptic", "Elliptic development subgraph", elliptic),
];

export const PIPELINE_CHECK: Study = parseStudy("synthetic-fixture", "Synthetic pipeline check", synthetic);

export function formatMetric(value: number | null, digits = 4): string {
  if (value === null) return "not measured";
  return value.toFixed(digits);
}
