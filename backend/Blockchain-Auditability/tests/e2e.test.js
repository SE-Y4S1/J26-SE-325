const request = require("supertest");
const path = require("path");
const fs = require("fs");

// Set environment for isolated test database and mock blockchain mode
const testDbPath = path.join(__dirname, "test_audits.sqlite");
process.env.DB_PATH = testDbPath;
process.env.BLOCKCHAIN_MODE = "mock";
process.env.ALLOW_DEMO_TAMPER = "true";

const app = require("../src/index");
const { getDatabase } = require("../src/db/database");

describe("End-to-End API Integration Tests", () => {
  let db;

  beforeAll(() => {
    if (fs.existsSync(testDbPath)) {
      try { fs.unlinkSync(testDbPath); } catch {}
    }
    db = getDatabase(testDbPath);
  });

  afterAll(() => {
    if (db) {
      db.close();
    }
    // Clean up test sqlite files
    [testDbPath, `${testDbPath}-wal`, `${testDbPath}-shm`].forEach((f) => {
      if (fs.existsSync(f)) {
        try { fs.unlinkSync(f); } catch {}
      }
    });
  });

  test("GET /api/health returns online status", async () => {
    const res = await request(app).get("/api/health");
    expect(res.status).toBe(200);
    expect(res.body.status).toBe("online");
    expect(res.body.component).toContain("Component 03");
  });

  test("POST /api/audit/evaluate validates input correctly", async () => {
    const res = await request(app)
      .post("/api/audit/evaluate")
      .send({});
    expect(res.status).toBe(400);
    expect(res.body.success).toBe(false);
    expect(Array.isArray(res.body.errors)).toBe(true);
    expect(res.body.errors.length).toBeGreaterThan(0);
  });

  let tx1AuditId = "";

  test("Scenario 1: Automated Low-Risk Approval (APPROVE)", async () => {
    const payload = {
      transactionId: "TX-TEST-001",
      riskScore: 20,
      confidence: 95,
      amount: 1500,
      transactionType: "TRANSFER",
      modelVersion: "FraudModel-v2"
    };

    const res = await request(app)
      .post("/api/audit/evaluate")
      .send(payload);

    expect(res.status).toBe(200);
    expect(res.body.success).toBe(true);
    expect(res.body.status).toBe("COMPLETED");
    expect(res.body.record.finalDecision).toBe("APPROVE");
    expect(res.body.record.reviewRequired).toBe(false);
    expect(res.body.record.auditId.startsWith("0x")).toBe(true);
    expect(res.body.record.auditId).toHaveLength(66);
    expect(res.body.record.saltMasked).toBeDefined();
    expect(res.body.record.salt).toBeUndefined(); // Salt is masked
    expect(res.body.blockchain.success).toBe(true);

    tx1AuditId = res.body.record.auditId;
  });

  test("Duplicate transactionId is rejected with HTTP 409 Conflict", async () => {
    const payload = {
      transactionId: "TX-TEST-001",
      riskScore: 20,
      confidence: 95,
      amount: 1500,
      transactionType: "TRANSFER",
      modelVersion: "FraudModel-v2"
    };

    const res = await request(app)
      .post("/api/audit/evaluate")
      .send(payload);

    expect(res.status).toBe(409);
    expect(res.body.success).toBe(false);
    expect(res.body.message).toContain("already exists");
  });

  test("Scenario 2: Automated High-Risk Rejection (REJECT)", async () => {
    const payload = {
      transactionId: "TX-TEST-002",
      riskScore: 92,
      confidence: 94,
      amount: 50000,
      transactionType: "WITHDRAWAL",
      modelVersion: "FraudModel-v2"
    };

    const res = await request(app)
      .post("/api/audit/evaluate")
      .send(payload);

    expect(res.status).toBe(200);
    expect(res.body.success).toBe(true);
    expect(res.body.status).toBe("COMPLETED");
    expect(res.body.record.finalDecision).toBe("REJECT");
    expect(res.body.record.reviewRequired).toBe(false);
  });

  test("Scenario 3: Human Review Workflow (evaluate -> pending -> review -> approve)", async () => {
    const payload = {
      transactionId: "TX-TEST-003",
      riskScore: 90,
      confidence: 50,
      amount: 75000,
      transactionType: "LOAN_DISBURSEMENT",
      modelVersion: "FraudModel-v2"
    };

    const evalRes = await request(app)
      .post("/api/audit/evaluate")
      .send(payload);

    expect(evalRes.status).toBe(200);
    expect(evalRes.body.status).toBe("HUMAN_REVIEW_REQUIRED");
    expect(evalRes.body.record.finalDecision).toBe("PENDING_HUMAN_REVIEW");
    expect(evalRes.body.record.reviewRequired).toBe(true);

    // Submit Human Review Decision
    const reviewRes = await request(app)
      .post("/api/audit/review")
      .send({
        transactionId: "TX-TEST-003",
        reviewerDecision: "APPROVE"
      });

    expect(reviewRes.status).toBe(200);
    expect(reviewRes.body.success).toBe(true);
    expect(reviewRes.body.record.finalDecision).toBe("HUMAN_APPROVED");
    expect(reviewRes.body.record.reviewerDecision).toBe("APPROVE");
    expect(reviewRes.body.blockchain.success).toBe(true);
  });

  test("Cryptographic Verification by transactionId validates successfully", async () => {
    const res = await request(app)
      .get("/api/audit/TX-TEST-001/verify");

    expect(res.status).toBe(200);
    expect(res.body.verified).toBe(true);
    expect(res.body.status).toBe("VERIFIED_SIMULATED");
    expect(res.body.details.transactionId).toBe("TX-TEST-001");
    expect(res.body.details.auditId).toBe(tx1AuditId);
    expect(res.body.details.isTampered).toBe(false);
  });

  test("Cryptographic Verification by auditId validates successfully", async () => {
    const res = await request(app)
      .get(`/api/audit/by-audit-id/${tx1AuditId}/verify`);

    expect(res.status).toBe(200);
    expect(res.body.verified).toBe(true);
    expect(res.body.status).toBe("VERIFIED_SIMULATED");
  });

  test("Tamper Simulation detects cryptographic modification (non-repudiation)", async () => {
    // 1. Simulate tampering on riskScore field in off-chain database
    const tamperRes = await request(app)
      .post("/api/audit/TX-TEST-001/tamper")
      .send({ field: "riskScore", tamperedValue: 99 });

    expect(tamperRes.status).toBe(200);
    expect(tamperRes.body.success).toBe(true);
    expect(tamperRes.body.tamperedRecord.isTampered).toBe(true);

    // 2. Re-verify record - must fail cryptographic integrity check
    const verifyRes = await request(app)
      .get("/api/audit/TX-TEST-001/verify");

    expect(verifyRes.status).toBe(200);
    expect(verifyRes.body.verified).toBe(false);
    expect(verifyRes.body.status).toBe("MODIFIED");
    expect(verifyRes.body.message).toContain("integrity failure");
    expect(verifyRes.body.details.calculatedCurrentHash).not.toBe(
      verifyRes.body.details.blockchainOnChainHash
    );
  });

  test("Verification of non-existent record returns NOT_FOUND", async () => {
    const res = await request(app)
      .get("/api/audit/TX-DOES-NOT-EXIST/verify");

    expect(res.status).toBe(200);
    expect(res.body.verified).toBe(false);
    expect(res.body.status).toBe("NOT_FOUND");
  });

  test("GET /api/audit/records returns all off-chain records with masked salts", async () => {
    const res = await request(app).get("/api/audit/records");
    expect(res.status).toBe(200);
    expect(res.body.count).toBeGreaterThanOrEqual(3);
    expect(Array.isArray(res.body.records)).toBe(true);

    res.body.records.forEach((record) => {
      expect(record.salt).toBeUndefined();
      expect(record.saltMasked).toBeDefined();
    });
  });

  test("GET /api/audit/stats accurately reports aggregated metrics", async () => {
    const res = await request(app).get("/api/audit/stats");
    expect(res.status).toBe(200);
    expect(res.body.success).toBe(true);
    expect(res.body.stats.totalDecisions).toBeGreaterThanOrEqual(3);
    expect(res.body.stats.approved).toBeGreaterThanOrEqual(1);
    expect(res.body.stats.rejected).toBeGreaterThanOrEqual(1);
    expect(res.body.stats.humanReview).toBeGreaterThanOrEqual(1);
    expect(res.body.stats.integrityFailures).toBeGreaterThanOrEqual(1);
  });

  test("DELETE /api/audit/records clears all off-chain records", async () => {
    const delRes = await request(app).delete("/api/audit/records");
    expect(delRes.status).toBe(200);
    expect(delRes.body.success).toBe(true);

    const listRes = await request(app).get("/api/audit/records");
    expect(listRes.body.count).toBe(0);
    expect(listRes.body.records).toHaveLength(0);
  });
});
