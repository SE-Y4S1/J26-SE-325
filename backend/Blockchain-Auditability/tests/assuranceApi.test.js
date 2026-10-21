const request = require("supertest");
const path = require("path");
const fs = require("fs");

const testDbPath = path.join(__dirname, "test_assurance.sqlite");
process.env.DB_PATH = testDbPath;
process.env.BLOCKCHAIN_MODE = "mock";
process.env.ALLOW_DEMO_TAMPER = "true";

function removeTestDb() {
  [testDbPath, `${testDbPath}-wal`, `${testDbPath}-shm`].forEach((f) => {
    if (fs.existsSync(f)) { try { fs.unlinkSync(f); } catch {} }
  });
}
removeTestDb();

const app = require("../src/index");
const { getDatabase } = require("../src/db/database");
const { policyRegistry } = require("../src/services/policyRegistry");

const VALIDATED = { benchmarkModel: true, sourceModelVersion: "fraud-bundle-2026-10" };

function payload(transactionId, overrides = {}) {
  return {
    transactionId,
    riskScore: 10,
    confidence: 90,
    amount: 500,
    transactionType: "PAYMENT",
    modelVersion: "c2:fraud-bundle-2026-10",
    ...overrides
  };
}

/**
 * The shape captured from the API before the assurance engine existed
 * (tests/fixtures/api-contract-v1.json). The current API must still produce every key, with
 * the same type and the same enum value; adding keys is allowed.
 */
const ENUM_KEYS = new Set(["status", "action", "finalDecision", "aiDecision", "verificationStatus",
  "reviewerDecision", "policyId", "policyVersion", "upstreamSource", "upstreamDecision", "success",
  "verified", "reviewRequired", "isTampered", "isMock", "reconciled"]);

function shape(value, key) {
  if (value === null) return "null";
  if (Array.isArray(value)) return value.length ? [shape(value[0], key)] : [];
  if (typeof value === "object") {
    const out = {};
    for (const k of Object.keys(value).sort()) out[k] = shape(value[k], k);
    return out;
  }
  if (ENUM_KEYS.has(key)) return `=${JSON.stringify(value)}`;
  return typeof value;
}

function incompatibilities(expected, actual, at = "$", inArray = false) {
  if (Array.isArray(expected)) {
    if (!Array.isArray(actual)) return [`${at}: expected array`];
    if (!expected.length || !actual.length) return [];
    return incompatibilities(expected[0], actual[0], `${at}[0]`, true);
  }
  if (expected !== null && typeof expected === "object") {
    if (actual === null || typeof actual !== "object") return [`${at}: expected object, got ${JSON.stringify(actual)}`];
    return Object.keys(expected).flatMap((k) =>
      k in actual ? incompatibilities(expected[k], actual[k], `${at}.${k}`, inArray) : [`${at}.${k}: missing`]);
  }
  if (expected === "null" || inArray) return []; // nullable field, or list items that vary by row
  // The capture tool froze any value under an enum-named key; a number there (stats.verified
  // is a count) is a quantity, so only its type is part of the contract.
  if (expected.startsWith("=") && typeof JSON.parse(expected.slice(1)) === "number") {
    return actual.startsWith("=") && typeof JSON.parse(actual.slice(1)) === "number" ? [] : [`${at}: expected a number, got ${actual}`];
  }
  return expected === actual ? [] : [`${at}: expected ${expected}, got ${JSON.stringify(actual)}`];
}

describe("assurance engine over the API", () => {
  let db;

  beforeAll(() => {
    db = getDatabase(testDbPath);
  });

  afterAll(() => {
    if (db) db.close();
    removeTestDb();
  });

  test("full signals route a clean decision to AUTO and return the assurance object", async () => {
    const res = await request(app).post("/api/audit/evaluate").send(payload("TX-AS-AUTO", {
      signals: { modelScores: { lstm: 0.1, gnn: 0.12 }, rules: [], ...VALIDATED }
    }));
    expect(res.status).toBe(200);
    expect(res.body.status).toBe("COMPLETED");
    expect(res.body.policyResult).toMatchObject({ action: "APPROVE", policyId: "P002" });
    expect(res.body.policyResult.policyHash).toBe(policyRegistry.get("P002").policyHash);
    expect(res.body.assurance).toMatchObject({ route: "AUTO", signalQuality: "FULL", policyId: "P002", aiDecision: "APPROVE" });
    expect(res.body.record.policyHash).toBe(policyRegistry.get("P002").policyHash);
  });

  test("VERIFY keeps the legacy HUMAN_REVIEW_REQUIRED status, so existing callers hold it", async () => {
    const res = await request(app).post("/api/audit/evaluate").send(payload("TX-AS-VERIFY", {
      riskScore: 42,
      signals: { modelScores: { lstm: 0.38, gnn: 0.44 }, rules: [], ...VALIDATED }
    }));
    expect(res.status).toBe(200);
    expect(res.body.assurance.route).toBe("VERIFY");
    expect(res.body.status).toBe("HUMAN_REVIEW_REQUIRED");
    expect(res.body.policyResult.action).toBe("HUMAN_REVIEW");
    expect(res.body.record.finalDecision).toBe("PENDING_HUMAN_REVIEW");
  });

  test("a heuristic-only score (Component 2 untrained) is DEGRADED and never AUTO", async () => {
    const res = await request(app).post("/api/audit/evaluate").send(payload("TX-AS-HEUR", {
      riskScore: 0.5,
      signals: { modelScores: {}, rules: ["score_below_step_up_threshold"], benchmarkModel: false, sourceModelVersion: "heuristic-untrained" }
    }));
    expect(res.status).toBe(200);
    expect(res.body.assurance.signalQuality).toBe("DEGRADED");
    expect(res.body.assurance.route).not.toBe("AUTO");
    expect(res.body.status).toBe("HUMAN_REVIEW_REQUIRED");
  });

  test("an upstream BLOCK is anchored as REJECT under P002 too", async () => {
    const res = await request(app).post("/api/audit/evaluate").send(payload("TX-AS-BLOCK", {
      signals: { modelScores: { lstm: 0.1, gnn: 0.12 }, rules: ["rate_limit_block"], ...VALIDATED },
      upstreamDecision: { source: "component2", decision: "BLOCK" }
    }));
    expect(res.body.status).toBe("COMPLETED");
    expect(res.body.record.finalDecision).toBe("REJECT");
    expect(res.body.assurance.escalation.decision).toBe("BLOCK");
  });

  test("a request without signals keeps P001 behaviour exactly", async () => {
    const res = await request(app).post("/api/audit/evaluate").send(payload("TX-AS-LEGACY", { riskScore: 90, confidence: 40 }));
    expect(res.body.status).toBe("HUMAN_REVIEW_REQUIRED");
    expect(res.body.policyResult.policyId).toBe("P001");
    expect(res.body.assurance).toMatchObject({ signalQuality: "LEGACY", route: "HUMAN_REVIEW", policyId: "P001" });
  });

  test("ASSURANCE_REQUIRE_SIGNALS routes signal-less requests through P002 on a degraded basis", async () => {
    process.env.ASSURANCE_REQUIRE_SIGNALS = "true";
    try {
      const res = await request(app).post("/api/audit/evaluate").send(payload("TX-AS-STRICT"));
      expect(res.body.policyResult.policyId).toBe("P002");
      expect(res.body.assurance.signalQuality).toBe("DEGRADED");
      expect(res.body.assurance.route).not.toBe("AUTO");
    } finally {
      delete process.env.ASSURANCE_REQUIRE_SIGNALS;
    }
  });

  test.each([
    ["not an object", "yes"],
    ["an unknown field", { modelScores: {}, confidenceOverride: 1 }],
    ["a score above 1", { modelScores: { a: 1.2 } }],
    ["a non-numeric score", { modelScores: { a: "0.3" } }],
    ["NaN smuggled as null", { modelScores: { a: null } }],
    ["a bad model name", { modelScores: { "a b": 0.1 } }],
    ["rules that are not strings", { rules: [1] }],
    ["a string benchmark flag", { benchmarkModel: "true" }],
    ["a subjectRef that is not a keyed hex digest", { subjectRef: "user-17" }]
  ])("rejects signals with %s (400, never coerced)", async (_, signals) => {
    const res = await request(app).post("/api/audit/evaluate").send(payload(`TX-AS-BAD-${Math.random()}`, { signals }));
    expect(res.status).toBe(400);
    expect(res.body.errors.some((e) => e.startsWith("signals"))).toBe(true);
  });

  test("GET /api/audit/policies returns each document with a hash a reader can recompute", async () => {
    const { computePolicyHashForScheme } = require("../src/services/policyDefinition");
    const res = await request(app).get("/api/audit/policies");
    expect(res.status).toBe(200);
    expect(res.body.activePolicyId).toBe("P002");
    const byId = Object.fromEntries(res.body.policies.map((p) => [p.id, p]));
    expect(Object.keys(byId).sort()).toEqual(["P001", "P002"]);
    for (const p of res.body.policies) {
      expect(computePolicyHashForScheme(p.hashScheme, p.document)).toBe(p.policyHash);
      expect(p.onChain.active).toBe(true); // mock ledger accepts every policy
    }
    expect(byId.P002).toMatchObject({ routesSignalRequests: true, status: "untuned", hashScheme: "v2" });
    expect(byId.P001).toMatchObject({ routesSignalRequests: false, baseline: true, hashScheme: "v1" });
  });
});

describe("research results endpoint", () => {
  afterEach(() => { delete process.env.RESEARCH_RESULTS_PATH; });

  test("reports NOT_RUN until the IEEE-CIS evaluation has produced metrics", async () => {
    process.env.RESEARCH_RESULTS_PATH = path.join(__dirname, "does-not-exist.json");
    const res = await request(app).get("/api/audit/research/results");
    expect(res.status).toBe(200);
    expect(res.body.status).toBe("NOT_RUN");
  });

  test("serves aggregates only, never row-level data", async () => {
    const file = path.join(__dirname, "test_research_metrics.json");
    const method = { incorrect_automatic_rate: 0.01, human_review_rate: 0.1, verify_rate: 0.05, fraud_auto_approved: 3,
      legit_auto_rejected: 4, aurc: 0.02, ci95: { incorrect_automatic_rate: [0.009, 0.011] }, params: { secret: "not served" } };
    fs.writeFileSync(file, JSON.stringify({
      status: "FINAL", generated_at: "2026-10-21T00:00:00Z", policy: { id: "P002", version: "1.1", hash: "0x1" }, policy_budget: 0.1,
      meta: { rows: { test: 10 }, fraud_rate: { test: 0.03 }, test_index_digest: "sha256:x", calibration_split_model_quality: {} },
      assumptions: {}, results: { "0.1": { methods: { assurance: method }, paired_vs_assurance: {} } }, curves_at_policy_budget: {}
    }));
    process.env.RESEARCH_RESULTS_PATH = file;
    try {
      const res = await request(app).get("/api/audit/research/results");
      expect(res.body.status).toBe("FINAL");
      expect(res.body.results["0.1"].methods.assurance.incorrect_automatic_rate).toBe(0.01);
      expect(JSON.stringify(res.body)).not.toContain("not served");
    } finally {
      fs.unlinkSync(file);
    }
  });
});

describe("API contract captured before the engine existed", () => {
  const captured = require("./fixtures/api-contract-v1.json");
  const base = { amount: 1500, transactionType: "TRANSFER", modelVersion: "fraud-v1" };

  // Replays the exact capture sequence, with fresh transaction ids.
  test("every captured response keeps its keys, types and enum values", async () => {
    const r = {};
    const call = async (name, req) => { const res = await req; r[name] = { httpStatus: res.status, body: shape(res.body) }; };
    const id = (n) => `TX-C-${n}-${process.pid}`;
    await call("evaluate_approve", request(app).post("/api/audit/evaluate").send({ ...base, transactionId: id(1), riskScore: 20, confidence: 95 }));
    await call("evaluate_reject", request(app).post("/api/audit/evaluate").send({ ...base, transactionId: id(2), riskScore: 90, confidence: 95 }));
    await call("evaluate_review", request(app).post("/api/audit/evaluate").send({ ...base, transactionId: id(3), riskScore: 90, confidence: 40 }));
    await call("evaluate_upstream", request(app).post("/api/audit/evaluate").send({ ...base, transactionId: id(4), riskScore: 20, confidence: 95,
      upstreamDecision: { source: "component2", decision: "STEP_UP", modelVersion: "m" }, planHash: "0x" + "a".repeat(64) }));
    await call("evaluate_invalid", request(app).post("/api/audit/evaluate").send({}));
    await call("evaluate_duplicate", request(app).post("/api/audit/evaluate").send({ ...base, transactionId: id(1), riskScore: 20, confidence: 95 }));
    await call("review", request(app).post("/api/audit/review").send({ transactionId: id(3), reviewerDecision: "APPROVE" }));
    await call("verify_ok", request(app).get(`/api/audit/${id(1)}/verify`));
    await call("verify_missing", request(app).get("/api/audit/NOPE/verify"));
    await call("tamper", request(app).post(`/api/audit/${id(2)}/tamper`).send({ field: "riskScore", tamperedValue: 10 }));
    await call("verify_tampered", request(app).get(`/api/audit/${id(2)}/verify`));
    await call("retry", request(app).post(`/api/audit/${id(1)}/retry`));
    await call("records", request(app).get("/api/audit/records"));
    await call("stats", request(app).get("/api/audit/stats"));
    await call("health", request(app).get("/api/health"));

    for (const [name, expected] of Object.entries(captured)) {
      expect([name, r[name].httpStatus]).toEqual([name, expected.httpStatus]);
      expect([name, incompatibilities(expected.body, r[name].body)]).toEqual([name, []]);
    }
  });
});
