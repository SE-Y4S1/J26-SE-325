export const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8082").replace(/\/$/, "");

export type DecisionName = "ALLOW" | "STEP_UP" | "BLOCK";

export type Health = {
  status: string;
  component: string;
  version: string;
  model_version: string;
  benchmark_model: boolean;
  kafka_enabled: boolean;
};

export type ScoreResponse = {
  transaction_id: string;
  fraud_score: number;
  decision: DecisionName;
  model_version: string;
  benchmark_model: boolean;
  explanation: {
    rules: string[];
    streams: Record<string, number>;
    top_factors: { name: string; contribution: number }[];
    notes: string[];
  };
};

export type StoredDecision = {
  transaction_id: string;
  scored_at?: string;
  decision: DecisionName;
  fraud_score: number;
  model_version: string;
  reasons: string[];
};

export type DecisionList = {
  decisions: StoredDecision[];
  counts: Record<DecisionName, number>;
  scope: string;
};

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function readError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return "The request was rejected. Check the form fields.";
  } catch {
    return `The API returned HTTP ${response.status}.`;
  }
  return `The API returned HTTP ${response.status}.`;
}

export async function getHealth(): Promise<Health> {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) throw new ApiError(response.status, await readError(response));
  return response.json() as Promise<Health>;
}

export async function getDecisions(): Promise<DecisionList> {
  const response = await fetch(`${API_BASE}/v1/decisions`);
  if (!response.ok) throw new ApiError(response.status, await readError(response));
  return response.json() as Promise<DecisionList>;
}

export type ScoreRequest = {
  transaction_id: string;
  amount: number;
  currency: string;
  product_code?: string;
  account_key?: string;
  device_key?: string;
  ip_reputation?: number;
  simulated_signals?: {
    origin: "simulated";
    typing_interval_ms?: number;
    pointer_jitter?: number;
    session_duration_s?: number;
  };
};

export async function scoreTransaction(body: ScoreRequest): Promise<ScoreResponse> {
  const response = await fetch(`${API_BASE}/v1/transactions/score`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new ApiError(response.status, await readError(response));
  return response.json() as Promise<ScoreResponse>;
}
