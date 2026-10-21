const request = require("supertest");
const { newDb } = require("pg-mem");

// The Postgres store over an in-process Postgres, behind the real app and mock chain.
process.env.BLOCKCHAIN_MODE = "mock";

const { useDatabase } = require("../src/db/database");
const { PostgresAuditDatabase } = require("../src/db/postgresDatabase");
const { DuplicateTransactionError } = require("../src/db/errors");
const { blockchainService } = require("../src/services/blockchainService");

const { Pool } = newDb().adapters.createPg();
const db = new PostgresAuditDatabase(null, new Pool());
useDatabase(db);
const app = require("../src/index");

function payload(transactionId, overrides = {}) {
  return {
    transactionId,
    riskScore: 20,
    confidence: 90,
    amount: 500,
    transactionType: "PORTFOLIO_WITHDRAWAL",
    modelVersion: "c1:v|c2:heuristic-untrained",
    upstreamDecision: { source: "component2", decision: "ALLOW" },
    planHash: "0x" + "ef".repeat(32),
    ...overrides
  };
}

describe("Postgres audit store", () => {
  test("evaluate, then verify, round-trips through Postgres", async () => {
    const res = await request(app).post("/api/audit/evaluate").send(payload("PG-1"));
    expect(res.status).toBe(200);
    expect(res.body.record.finalDecision).toBe("APPROVE");

    const verify = await request(app).get("/api/audit/PG-1/verify");
    expect(verify.body.verified).toBe(true);
    expect(verify.body.details.canonicalPayload.planHash).toBe("0x" + "ef".repeat(32));
  });

  test("a restart that empties the in-memory mock ledger still verifies", async () => {
    await request(app).post("/api/audit/evaluate").send(payload("PG-2"));
    blockchainService.mockLedger.clear();   // what a Render spin-down does to process memory

    const verify = await request(app).get("/api/audit/PG-2/verify");
    expect(verify.body.verified).toBe(true);
    expect(verify.body.status).toBe("VERIFIED_SIMULATED");
  });

  test("a duplicate transaction is a 409, via the shared error class", async () => {
    const res = await request(app).post("/api/audit/evaluate").send(payload("PG-1"));
    expect(res.status).toBe(409);
    await expect(db.createRecord({ ...(await db.getRecord("PG-1")) }))
      .rejects.toBeInstanceOf(DuplicateTransactionError);
  });

  test("human review updates the stored record and re-anchors it", async () => {
    await request(app).post("/api/audit/evaluate").send(payload("PG-3", {
      upstreamDecision: { source: "component2", decision: "STEP_UP" }
    }));
    const review = await request(app).post("/api/audit/review")
      .send({ transactionId: "PG-3", reviewerDecision: "APPROVE" });
    expect(review.status).toBe(200);
    expect((await db.getRecord("PG-3")).finalDecision).toBe("HUMAN_APPROVED");

    const verify = await request(app).get("/api/audit/PG-3/verify");
    expect(verify.body.verified).toBe(true);
  });

  test("tampering is detected", async () => {
    await request(app).post("/api/audit/PG-1/tamper").send({ field: "amount", tamperedValue: 1 });
    const verify = await request(app).get("/api/audit/PG-1/verify");
    expect(verify.body.verified).toBe(false);
    expect(verify.body.status).toBe("MODIFIED");
  });

  test("records list and stats read from Postgres", async () => {
    const list = await request(app).get("/api/audit/records");
    expect(list.body.count).toBe(3);
    const stats = await request(app).get("/api/audit/stats");
    expect(stats.status).toBe(200);
  });

  test("record v2 in Postgres: assurance kept, review linked, decision fields immutable on update", async () => {
    await request(app).post("/api/audit/evaluate").send(payload("PG-V2", {
      riskScore: 42, upstreamDecision: undefined,
      signals: { modelScores: { lstm: 0.38, gnn: 0.44 }, rules: [], benchmarkModel: true, sourceModelVersion: "b1" }
    }));
    const stored = await db.getRecord("PG-V2");
    expect(stored).toMatchObject({ recordVersion: 2, decisionFinalDecision: "PENDING_HUMAN_REVIEW" });
    expect(stored.assurance.route).toBe("VERIFY");

    // An update cannot rewrite what the decision commitment covers.
    await db.updateRecord({ ...stored, assurance: { ...stored.assurance, route: "AUTO" }, decisionFinalDecision: "APPROVE" });
    expect((await db.getRecord("PG-V2")).assurance.route).toBe("VERIFY");

    const review = await request(app).post("/api/audit/review").send({ transactionId: "PG-V2", reviewerDecision: "APPROVE" });
    expect(review.status).toBe(200);
    const verify = await request(app).get("/api/audit/PG-V2/verify");
    expect(verify.body).toMatchObject({ verified: true });
    expect(verify.body.details.commitments.review.linkIntact).toBe(true);
  });

  test("evidence exports are logged in Postgres", async () => {
    const res = await request(app).get("/api/audit/PG-V2/evidence");
    expect(res.status).toBe(200);
    expect(await db.getEvidenceExports("PG-V2")).toHaveLength(1);
  });
});
