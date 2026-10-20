const request = require("supertest");
const path = require("path");
const fs = require("fs");
const os = require("os");
const { execFileSync } = require("child_process");

const testDbPath = path.join(__dirname, "test_evidence.sqlite");
process.env.DB_PATH = testDbPath;
process.env.BLOCKCHAIN_MODE = "mock";

function removeTestDb() {
  [testDbPath, `${testDbPath}-wal`, `${testDbPath}-shm`].forEach((f) => {
    if (fs.existsSync(f)) { try { fs.unlinkSync(f); } catch {} }
  });
}
removeTestDb();

const app = require("../src/index");
const { getDatabase } = require("../src/db/database");
const { verifyBundleOffline } = require("../scripts/verify-evidence");

const VALIDATED = { benchmarkModel: true, sourceModelVersion: "fraud-bundle-2026-10" };
const SUBJECT_REF = "ab".repeat(32);

const clone = (x) => JSON.parse(JSON.stringify(x));
const failed = (checks) => checks.filter((c) => !c.ok).map((c) => c.name);

/** Every key anywhere in a JSON value. */
function keysOf(value, out = new Set()) {
  if (value && typeof value === "object") {
    for (const [k, v] of Object.entries(value)) { out.add(k); keysOf(v, out); }
  }
  return out;
}

describe("evidence bundles", () => {
  let db;
  let reviewed;
  let auto;

  beforeAll(async () => {
    db = getDatabase(testDbPath);
    await request(app).post("/api/audit/evaluate").send({
      transactionId: "EV-AUTO", riskScore: 10, confidence: 90, amount: 750, transactionType: "PAYMENT", modelVersion: "m",
      signals: { modelScores: { lstm: 0.1, gnn: 0.12 }, rules: [], ...VALIDATED, subjectRef: SUBJECT_REF }
    });
    await request(app).post("/api/audit/evaluate").send({
      transactionId: "EV-REVIEWED", riskScore: 90, confidence: 40, amount: 75000, transactionType: "LOAN_DISBURSEMENT", modelVersion: "m"
    });
    await request(app).post("/api/audit/review").send({ transactionId: "EV-REVIEWED", reviewerDecision: "REJECT", reason: "Mismatched payee" });
    auto = (await request(app).get("/api/audit/EV-AUTO/evidence")).body;
    reviewed = (await request(app).get("/api/audit/EV-REVIEWED/evidence")).body;
  });

  afterAll(() => { if (db) db.close(); removeTestDb(); });

  test("contain no personal data: no user id, email, name or account key anywhere", () => {
    for (const bundle of [auto, reviewed]) {
      // The policy document is public configuration (it has a "name": the policy's), so the
      // scan covers everything else -- the record data.
      const recordData = clone(bundle);
      delete recordData.policy.document;
      const keys = [...keysOf(recordData)].map((k) => k.toLowerCase());
      for (const forbidden of ["email", "userid", "user_id", "name", "accountkey", "account_key", "password", "displayname"]) {
        expect([forbidden, keys.includes(forbidden)]).toEqual([forbidden, false]);
      }
      expect(bundle.privacy.sensitiveFields).toEqual(["amount", "riskScore", "confidence", "salt"]);
    }
    // The only identity in a bundle is the keyed reference, never a raw id.
    expect(auto.commitments.decision.payload.subjectRef).toBe(SUBJECT_REF);
  });

  test("carry what an outsider needs: payloads, salt, policy document, anchors, ledger", () => {
    expect(auto.salt).toMatch(/^[0-9a-f]{64}$/);
    expect(auto.policy).toMatchObject({ id: "P002", hashScheme: "v2" });
    expect(auto.policy.document.weights).toBeDefined();
    expect(auto.commitments.decision.anchor.blockchainTxId).toMatch(/^0x/);
    expect(auto.commitments.decision.payload.assurance.route).toBe("AUTO");
    expect(reviewed.commitments.review.payload).toMatchObject({ reviewerDecision: "REJECT", reviewReason: "Mismatched payee", commitment: "REVIEW" });
    expect(auto.verificationAtExport).toMatchObject({ verified: true, status: "VERIFIED_SIMULATED" });
  });

  test("verify independently, without the service's code", () => {
    expect(failed(verifyBundleOffline(auto))).toEqual([]);
    expect(failed(verifyBundleOffline(reviewed))).toEqual([]);
  });

  test.each([
    ["an altered amount", (b) => { b.commitments.decision.payload.amount = 1; }, "decision recordHash recomputes from its payload"],
    ["an altered assurance route", (b) => { b.commitments.decision.payload.assurance.route = "HUMAN_REVIEW"; }, "decision recordHash recomputes from its payload"],
    ["an altered policy weight", (b) => { b.policy.document.weights.context = 0.3; }, "policyHash recomputes from the policy document"],
    ["a swapped salt", (b) => { b.salt = "00".repeat(32); }, "auditId = sha256(transactionId:salt)"],
    ["an altered review outcome", (b) => { b.commitments.review.payload.reviewerDecision = "APPROVE"; }, "review recordHash recomputes from its payload"],
    ["a review re-linked to another decision", (b) => { b.commitments.review.payload.parentRecordHash = "0x" + "11".repeat(32); }, "review links to the decision (parentRecordHash)"]
  ])("detect %s", (_, mutate, expectedFailure) => {
    const base = expectedFailure.startsWith("review") ? reviewed : auto;
    const bundle = clone(base);
    mutate(bundle);
    expect(failed(verifyBundleOffline(bundle))).toContain(expectedFailure);
  });

  test("the CLI exits 0 for an intact bundle and 1 for a tampered one", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "c3-evidence-"));
    const script = path.join(__dirname, "../scripts/verify-evidence.js");
    try {
      const good = path.join(dir, "good.json");
      fs.writeFileSync(good, JSON.stringify(reviewed));
      const out = execFileSync(process.execPath, [script, good], { encoding: "utf8" });
      expect(out).toContain("Evidence verified.");
      expect(out).toContain("SKIPPED: this record was anchored on the simulated ledger");

      const bad = path.join(dir, "bad.json");
      const tampered = clone(reviewed);
      tampered.commitments.decision.payload.riskScore = 1;
      fs.writeFileSync(bad, JSON.stringify(tampered));
      let status = 0;
      try { execFileSync(process.execPath, [script, bad], { encoding: "utf8" }); } catch (err) { status = err.status; }
      expect(status).toBe(1);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  test("a v1 (legacy) record exports a bundle that verifies with the legacy hash schemes", async () => {
    const { records } = require("./fixtures/legacy-records-v1.json");
    const legacy = records.find((r) => !r.isTampered);
    db.saveRecord({ ...legacy, reviewRequired: Boolean(legacy.reviewRequired), isTampered: false });
    const bundle = (await request(app).get(`/api/audit/${legacy.transactionId}/evidence`)).body;
    expect(bundle.recordVersion).toBe(1);
    expect(bundle.policy).toMatchObject({ id: "P001", hashScheme: "v1" });
    expect(failed(verifyBundleOffline(bundle))).toEqual([]);
  });

  test("every export is logged", async () => {
    const exports = db.getEvidenceExports("EV-AUTO");
    expect(exports.length).toBe(1);
    expect(exports[0].actorRef).toBe("anonymous (AUTH_REQUIRED off)");
  });

  test("an unknown transaction is a 404 and is not logged", async () => {
    const res = await request(app).get("/api/audit/NOPE/evidence");
    expect(res.status).toBe(404);
    expect(db.getEvidenceExports("NOPE")).toEqual([]);
  });
});
