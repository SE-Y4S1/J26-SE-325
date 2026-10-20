const request = require("supertest");
const path = require("path");
const fs = require("fs");

const testDbPath = path.join(__dirname, "test_ledger_v2.sqlite");
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
const hashService = require("../src/services/hashService");
const { blockchainService, BlockchainUnavailableError } = require("../src/services/blockchainService");

const VALIDATED = { benchmarkModel: true, sourceModelVersion: "fraud-bundle-2026-10" };

function payload(transactionId, overrides = {}) {
  return { transactionId, riskScore: 10, confidence: 90, amount: 500, transactionType: "PAYMENT", modelVersion: "m", ...overrides };
}
const held = (id) => payload(id, { riskScore: 90, confidence: 40 }); // P001 -> HUMAN_REVIEW
const verifyRoute = (id) => payload(id, { riskScore: 42, signals: { modelScores: { lstm: 0.38, gnn: 0.44 }, rules: [], ...VALIDATED } });

describe("record v2: decision-time anchoring and linked review", () => {
  let db;
  beforeAll(() => { db = getDatabase(testDbPath); });
  afterAll(() => { jest.restoreAllMocks(); if (db) db.close(); removeTestDb(); });

  test("a held decision is anchored immediately, before any review", async () => {
    const res = await request(app).post("/api/audit/evaluate").send(held("V2-HELD"));
    expect(res.status).toBe(200);
    expect(res.body.status).toBe("HUMAN_REVIEW_REQUIRED");
    expect(res.body.blockchain.success).toBe(true);
    expect(res.body.record.recordVersion).toBe(2);
    expect(res.body.record.recordHash).toMatch(/^0x[0-9a-f]{64}$/);
    expect(blockchainService.mockLedger.get(res.body.record.auditId).recordHash).toBe(res.body.record.recordHash);

    const verify = await request(app).get("/api/audit/V2-HELD/verify");
    expect(verify.body).toMatchObject({ verified: true, status: "VERIFIED_SIMULATED" });
    expect(verify.body.details.lifecycle).toBe("AWAITING_REVIEW");
  });

  test("the assurance result is part of the decision commitment", async () => {
    await request(app).post("/api/audit/evaluate").send(verifyRoute("V2-ASSURED"));
    const stored = db.getRecord("V2-ASSURED");
    expect(stored.assurance.route).toBe("VERIFY");
    const { canonicalPayload } = hashService.decisionCommitment(stored);
    expect(canonicalPayload.assurance).toMatchObject({ route: "VERIFY", policyId: "P002", signalQuality: "FULL" });

    // Rewriting the stored route (VERIFY -> AUTO) must break verification.
    const forged = { ...stored.assurance, route: "AUTO" };
    db.db.prepare("UPDATE audits SET assurance = ? WHERE transactionId = ?").run(JSON.stringify(forged), "V2-ASSURED");
    const verify = await request(app).get("/api/audit/V2-ASSURED/verify");
    expect(verify.body.status).toBe("MODIFIED");
  });

  test("a review is a second commitment that links to the first, and the first never changes", async () => {
    const before = db.getRecord("V2-HELD");
    const review = await request(app).post("/api/audit/review").send({ transactionId: "V2-HELD", reviewerDecision: "APPROVE", reason: "Known payee" });
    expect(review.status).toBe(200);
    expect(review.body.record.finalDecision).toBe("HUMAN_APPROVED");
    expect(review.body.separationOfDuties).toBe("NOT_ENFORCED_ANONYMOUS_REVIEWER");

    const after = db.getRecord("V2-HELD");
    expect(after.recordHash).toBe(before.recordHash);
    expect(after.parentRecordHash).toBe(before.recordHash);
    expect(after.reviewAuditId).toBe(hashService.computeReviewAuditId("V2-HELD", after.salt));
    expect(after.reviewAuditId).not.toBe(after.auditId);
    expect(blockchainService.mockLedger.get(after.reviewAuditId).recordHash).toBe(after.reviewRecordHash);

    const verify = await request(app).get("/api/audit/V2-HELD/verify");
    expect(verify.body).toMatchObject({ verified: true, status: "VERIFIED_SIMULATED" });
    expect(verify.body.details.lifecycle).toBe("REVIEWED");
    expect(verify.body.details.commitments.review).toMatchObject({ matches: true, linkIntact: true });
    expect(verify.body.details.canonicalPayload).toMatchObject({ commitment: "REVIEW", finalDecision: "HUMAN_APPROVED", reviewerDecision: "APPROVE", reviewReason: "Known payee" });
  });

  test("a second review of the same decision is refused", async () => {
    const res = await request(app).post("/api/audit/review").send({ transactionId: "V2-HELD", reviewerDecision: "REJECT" });
    expect(res.status).toBe(409);
  });

  test("altering the reviewed outcome is detected", async () => {
    await request(app).post("/api/audit/V2-HELD/tamper").send({ field: "reviewerDecision", tamperedValue: "REJECT" });
    const verify = await request(app).get("/api/audit/V2-HELD/verify");
    expect(verify.body.status).toBe("MODIFIED");
    expect(verify.body.details.commitments.decision.matches).toBe(true);
    expect(verify.body.details.commitments.review.matches).toBe(false);
  });

  test("altering decision data after a review is detected at the decision commitment", async () => {
    await request(app).post("/api/audit/evaluate").send(held("V2-TAMPER-DECISION"));
    await request(app).post("/api/audit/review").send({ transactionId: "V2-TAMPER-DECISION", reviewerDecision: "REJECT" });
    await request(app).post("/api/audit/V2-TAMPER-DECISION/tamper").send({ field: "riskScore", tamperedValue: 5 });
    const verify = await request(app).get("/api/audit/V2-TAMPER-DECISION/verify");
    expect(verify.body.status).toBe("MODIFIED");
    expect(verify.body.details.commitments.decision.matches).toBe(false);
  });

  test("a review re-pointed at another decision is CHAIN_BROKEN", async () => {
    await request(app).post("/api/audit/evaluate").send(held("V2-LINK"));
    await request(app).post("/api/audit/review").send({ transactionId: "V2-LINK", reviewerDecision: "APPROVE" });
    const bogusParent = "0x" + "ab".repeat(32);
    const record = db.getRecord("V2-LINK");
    // Re-anchor a review commitment consistent with the bogus parent, so only the link is wrong.
    const forged = { ...record, parentRecordHash: bogusParent };
    const forgedHash = hashService.reviewCommitment(forged).hash;
    blockchainService.mockLedger.set(record.reviewAuditId, { ...blockchainService.mockLedger.get(record.reviewAuditId), recordHash: forgedHash });
    db.db.prepare("UPDATE audits SET parentRecordHash = ?, reviewRecordHash = ? WHERE transactionId = ?").run(bogusParent, forgedHash, "V2-LINK");

    const verify = await request(app).get("/api/audit/V2-LINK/verify");
    expect(verify.body.status).toBe("CHAIN_BROKEN");
    expect(verify.body.details.commitments.review.linkIntact).toBe(false);
  });

  test("changing the outcome of an unreviewed decision is detected", async () => {
    await request(app).post("/api/audit/evaluate").send(payload("V2-OUTCOME"));
    await request(app).post("/api/audit/V2-OUTCOME/tamper").send({ field: "finalDecision", tamperedValue: "REJECT" });
    const verify = await request(app).get("/api/audit/V2-OUTCOME/verify");
    expect(verify.body.status).toBe("MODIFIED");
  });

  test("ledger down at decision time: 503, nothing reviewable until /retry publishes it", async () => {
    const spy = jest.spyOn(blockchainService, "recordAudit").mockRejectedValueOnce(new BlockchainUnavailableError("node offline"));
    const res = await request(app).post("/api/audit/evaluate").send(held("V2-OFFLINE"));
    expect(res.status).toBe(503);
    expect(db.getRecord("V2-OFFLINE").verificationStatus).toBe("BLOCKCHAIN_FAILED");
    spy.mockRestore();

    const early = await request(app).post("/api/audit/review").send({ transactionId: "V2-OFFLINE", reviewerDecision: "APPROVE" });
    expect(early.status).toBe(409);

    const retry = await request(app).post("/api/audit/V2-OFFLINE/retry");
    expect(retry.status).toBe(200);
    expect(retry.body).toMatchObject({ reconciled: false, status: "HUMAN_REVIEW_REQUIRED" });
    expect(retry.body.assurance.route).toBe("HUMAN_REVIEW");

    const review = await request(app).post("/api/audit/review").send({ transactionId: "V2-OFFLINE", reviewerDecision: "APPROVE" });
    expect(review.status).toBe(200);
  });

  test("ledger down at review time: the review is not saved and can be submitted again", async () => {
    await request(app).post("/api/audit/evaluate").send(held("V2-REVIEW-OFFLINE"));
    const spy = jest.spyOn(blockchainService, "recordAudit").mockRejectedValueOnce(new BlockchainUnavailableError("node offline"));
    const failed = await request(app).post("/api/audit/review").send({ transactionId: "V2-REVIEW-OFFLINE", reviewerDecision: "APPROVE" });
    expect(failed.status).toBe(503);
    spy.mockRestore();
    expect(db.getRecord("V2-REVIEW-OFFLINE")).toMatchObject({ finalDecision: "PENDING_HUMAN_REVIEW", reviewerDecision: null });

    const ok = await request(app).post("/api/audit/review").send({ transactionId: "V2-REVIEW-OFFLINE", reviewerDecision: "APPROVE" });
    expect(ok.status).toBe(200);
  });

  test("a duplicate transaction never reaches the chain", async () => {
    const spy = jest.spyOn(blockchainService, "recordAudit");
    const res = await request(app).post("/api/audit/evaluate").send(payload("V2-OUTCOME"));
    expect(res.status).toBe(409);
    expect(spy).not.toHaveBeenCalled();
    spy.mockRestore();
  });

  test("a held v1 record (from before the migration) is still reviewed the v1 way and verifies", async () => {
    const salt = hashService.generateSalt();
    db.saveRecord({
      transactionId: "V1-PENDING", auditId: hashService.computeAuditId("V1-PENDING", salt), salt,
      policyHash: require("../src/services/policyDefinition").computePolicyHash(), recordHash: "PENDING_REVIEW",
      riskScore: 90, confidence: 40, amount: 10, transactionType: "TRANSFER", aiDecision: "HUMAN_REVIEW",
      finalDecision: "PENDING_HUMAN_REVIEW", modelVersion: "m", policyId: "P001", policyVersion: "1.0",
      timestamp: new Date().toISOString(), reviewRequired: true, reviewerDecision: null, verificationStatus: "PENDING_HUMAN_REVIEW"
    });
    expect(db.getRecord("V1-PENDING").recordVersion).toBeUndefined();
    const retry = await request(app).post("/api/audit/V1-PENDING/retry");
    expect(retry.status).toBe(400);

    const review = await request(app).post("/api/audit/review").send({ transactionId: "V1-PENDING", reviewerDecision: "REJECT" });
    expect(review.status).toBe(200);
    const verify = await request(app).get("/api/audit/V1-PENDING/verify");
    expect(verify.body).toMatchObject({ verified: true, status: "VERIFIED_SIMULATED" });
    expect(verify.body.details.recordVersion).toBeUndefined();
  });

  test("the queue lists held decisions with their route, oldest first", async () => {
    await request(app).post("/api/audit/evaluate").send(verifyRoute("V2-QUEUE-VERIFY"));
    const res = await request(app).get("/api/audit/queue");
    expect(res.status).toBe(200);
    const ids = res.body.queue.map((q) => q.transactionId);
    expect(ids).toEqual(expect.arrayContaining(["V2-ASSURED", "V2-QUEUE-VERIFY"]));
    expect(ids).not.toContain("V2-HELD"); // reviewed
    expect(res.body.queue.find((q) => q.transactionId === "V2-QUEUE-VERIFY").route).toBe("VERIFY");
    res.body.queue.forEach((q) => { expect(q.salt).toBeUndefined(); expect(q.subjectRef).toBeUndefined(); });
  });

  test("stats report routes and signal quality", async () => {
    const res = await request(app).get("/api/audit/stats");
    // V2-QUEUE-VERIFY; V2-ASSURED was also VERIFY until the forgery test rewrote it to AUTO.
    expect(res.body.stats.routes.VERIFY).toBeGreaterThanOrEqual(1);
    expect(res.body.stats.routes.LEGACY).toBeGreaterThanOrEqual(1);
    expect(res.body.stats.signalQuality.FULL).toBeGreaterThanOrEqual(2);
    expect(res.body.stats.pendingReview).toBeGreaterThanOrEqual(1);
    expect(res.body.stats.activePolicyId).toBe("P002");
  });

  test("record detail returns the record, its policy and its evidence exports", async () => {
    const res = await request(app).get("/api/audit/records/V2-QUEUE-VERIFY");
    expect(res.status).toBe(200);
    expect(res.body.record.assurance.route).toBe("VERIFY");
    expect(res.body.policy).toMatchObject({ id: "P002", hashScheme: "v2" });
    expect(res.body.evidenceExports).toEqual([]);
    expect((await request(app).get("/api/audit/records/NOPE")).status).toBe(404);
  });
});
