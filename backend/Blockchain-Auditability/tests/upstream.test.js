const request = require("supertest");
const path = require("path");
const fs = require("fs");

// Isolated database and mock chain, as in e2e.test.js.
const testDbPath = path.join(__dirname, "test_upstream.sqlite");
process.env.DB_PATH = testDbPath;
process.env.BLOCKCHAIN_MODE = "mock";

// Before the app is required: it opens this file on load, and Windows will not delete an
// open file, so cleaning up in beforeAll would leave the last run's rows behind.
function removeTestDb() {
  [testDbPath, `${testDbPath}-wal`, `${testDbPath}-shm`].forEach((f) => {
    if (fs.existsSync(f)) { try { fs.unlinkSync(f); } catch {} }
  });
}
removeTestDb();

const app = require("../src/index");
const { getDatabase } = require("../src/db/database");
const hashService = require("../src/services/hashService");

const PLAN_HASH = "0x" + "ab".repeat(32);

function payload(transactionId, overrides = {}) {
  return {
    transactionId,
    riskScore: 30,
    confidence: 90,
    amount: 120000,
    transactionType: "PORTFOLIO_WITHDRAWAL",
    modelVersion: "c1:abc123|c2:heuristic-untrained",
    planHash: PLAN_HASH,
    ...overrides
  };
}

describe("Upstream decisions from Component 2", () => {
  let db;

  beforeAll(() => {
    db = getDatabase(testDbPath);
  });

  afterAll(() => {
    if (db) db.close();
    removeTestDb();
  });

  test("a BLOCK at a low score is anchored as REJECT, not APPROVE", async () => {
    const res = await request(app).post("/api/audit/evaluate").send(payload("TX-UP-BLOCK", {
      upstreamDecision: { source: "component2", decision: "BLOCK", modelVersion: "heuristic-untrained" }
    }));

    expect(res.status).toBe(200);
    expect(res.body.status).toBe("COMPLETED");
    expect(res.body.record.finalDecision).toBe("REJECT");
    expect(res.body.record.upstreamDecision).toBe("BLOCK");
    expect(res.body.record.planHash).toBe(PLAN_HASH);
    expect(res.body.policyResult.reason).toContain("component2 enforced BLOCK");
  });

  test("a STEP_UP holds the record for review", async () => {
    const res = await request(app).post("/api/audit/evaluate").send(payload("TX-UP-STEP", {
      upstreamDecision: { source: "component2", decision: "STEP_UP" }
    }));

    expect(res.status).toBe(200);
    expect(res.body.status).toBe("HUMAN_REVIEW_REQUIRED");
    expect(res.body.record.finalDecision).toBe("PENDING_HUMAN_REVIEW");
  });

  test("an ALLOW never relaxes P001", async () => {
    const res = await request(app).post("/api/audit/evaluate").send(payload("TX-UP-ALLOW", {
      riskScore: 95,
      confidence: 95,
      upstreamDecision: { source: "component2", decision: "ALLOW" }
    }));

    expect(res.body.record.finalDecision).toBe("REJECT");
  });

  test("provenance is part of the commitment and verifies", async () => {
    await request(app).post("/api/audit/evaluate").send(payload("TX-UP-VERIFY", {
      upstreamDecision: { source: "component2", decision: "ALLOW" }
    }));

    const verify = await request(app).get("/api/audit/TX-UP-VERIFY/verify");
    expect(verify.status).toBe(200);
    expect(verify.body.verified).toBe(true);
    expect(verify.body.details.canonicalPayload.upstreamDecision).toBe("ALLOW");

    const stored = db.getRecord("TX-UP-VERIFY");
    const withProvenance = hashService.generateCanonicalHash(stored).hash;
    const withoutPlan = hashService.generateCanonicalHash({ ...stored, planHash: null }).hash;
    expect(withProvenance).toBe(stored.recordHash);
    expect(withoutPlan).not.toBe(stored.recordHash);
  });

  test("records without provenance hash exactly as before", () => {
    const base = {
      transactionId: "TX-LEGACY", auditId: "0x" + "11".repeat(32), salt: "22".repeat(32),
      riskScore: 10, confidence: 90, amount: 5, transactionType: "TRANSFER",
      aiDecision: "APPROVE", finalDecision: "APPROVE", modelVersion: "v1",
      timestamp: "2026-10-01T00:00:00.000Z", reviewRequired: false, reviewerDecision: null
    };
    const { canonicalPayload } = hashService.generateCanonicalHash(base);
    expect(Object.keys(canonicalPayload)).toHaveLength(16);
  });

  test("an unknown upstream decision is rejected", async () => {
    const res = await request(app).post("/api/audit/evaluate").send(payload("TX-UP-BAD", {
      upstreamDecision: { source: "component2", decision: "MAYBE" }
    }));
    expect(res.status).toBe(400);
  });
});
