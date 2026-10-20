const request = require("supertest");
const path = require("path");
const fs = require("fs");
const crypto = require("crypto");
const jwt = require("jsonwebtoken");

const testDbPath = path.join(__dirname, "test_auth.sqlite");
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
const { subjectRef } = require("../src/auth/auth");

const SECRET = "test-secret-only-for-jest";
const ISSUER = "j26-se-325-platform";
const VALIDATED = { benchmarkModel: true, sourceModelVersion: "fraud-bundle-2026-10" };

/** A token exactly as the platform service issues it (backend/Platform/auth/security.py). */
function token(sub, overrides = {}, options = {}) {
  const now = Math.floor(Date.now() / 1000);
  const claims = { sub: String(sub), iat: now, exp: now + 600, iss: ISSUER, email: `${sub}@example.test`, ...overrides };
  for (const [k, v] of Object.entries(claims)) if (v === undefined) delete claims[k];
  // jsonwebtoken adds iat itself unless noTimestamp is set (which also strips a supplied one),
  // so noTimestamp is used only to build the deliberately iat-less token.
  const omitIat = "iat" in overrides && overrides.iat === undefined;
  return jwt.sign(claims, options.secret || SECRET, { algorithm: options.algorithm || "HS256", noTimestamp: omitIat });
}
const bearer = (t) => ({ Authorization: `Bearer ${t}` });

const ANALYST = 7;
const SUBJECT = 17;

function enforce() {
  process.env.AUTH_REQUIRED = "true";
  process.env.JWT_SECRET = SECRET;
  process.env.AUDIT_REVIEWER_IDS = `${ANALYST},${SUBJECT}`;
}
function relax() {
  delete process.env.AUTH_REQUIRED;
  delete process.env.AUDIT_REVIEWER_IDS;
  process.env.JWT_SECRET = SECRET;
}

function decision(id, extra = {}) {
  return { transactionId: id, riskScore: 90, confidence: 40, amount: 500, transactionType: "PAYMENT", modelVersion: "m", ...extra };
}

describe("authentication mirrors the platform's shared JWT contract", () => {
  let db;
  beforeAll(() => { db = getDatabase(testDbPath); enforce(); });
  afterAll(() => { relax(); delete process.env.JWT_SECRET; if (db) db.close(); removeTestDb(); });
  beforeEach(() => enforce());

  test("a platform-issued token is accepted", async () => {
    const res = await request(app).get("/api/audit/records").set(bearer(token(ANALYST)));
    expect(res.status).toBe(200);
  });

  test.each([
    ["no token", null, 401],
    ["a token signed with another secret", token(ANALYST, {}, { secret: "wrong" }), 401],
    ["HS512 instead of the pinned HS256", token(ANALYST, {}, { algorithm: "HS512" }), 401],
    ["an unsigned alg:none token", `${Buffer.from('{"alg":"none","typ":"JWT"}').toString("base64url")}.${Buffer.from(JSON.stringify({ sub: String(ANALYST), iss: ISSUER, iat: 1, exp: 9999999999 })).toString("base64url")}.`, 401],
    ["the wrong issuer", token(ANALYST, { iss: "someone-else" }), 401],
    ["no iat claim", token(ANALYST, { iat: undefined }), 401],
    ["an expired token", token(ANALYST, { exp: Math.floor(Date.now() / 1000) - 10 }), 401],
    ["a valid token of a non-reviewer", token(99), 403]
  ])("refuses %s", async (_, t, status) => {
    const req = request(app).get("/api/audit/records");
    const res = await (t ? req.set(bearer(t)) : req);
    expect(res.status).toBe(status);
  });

  test("enforcement on without a secret refuses everything (500), never falls open", async () => {
    delete process.env.JWT_SECRET;
    const res = await request(app).get("/api/audit/records").set(bearer(token(ANALYST)));
    expect(res.status).toBe(500);
  });

  test("the platform service may anchor decisions but not read records", async () => {
    const service = bearer(token("service:platform"));
    expect((await request(app).post("/api/audit/evaluate").set(service).send(decision("AUTH-SVC"))).status).toBe(200);
    expect((await request(app).get("/api/audit/records").set(service)).status).toBe(403);
    expect((await request(app).post("/api/audit/AUTH-SVC/tamper").set(service).send({})).status).toBe(403);
  });

  test("health, policies and verification stay public; verification hides record contents", async () => {
    expect((await request(app).get("/api/health")).status).toBe(200);
    expect((await request(app).get("/api/audit/policies")).status).toBe(200);

    const anonymous = await request(app).get("/api/audit/AUTH-SVC/verify");
    expect(anonymous.status).toBe(200);
    expect(anonymous.body.verified).toBe(true);
    expect(anonymous.body.redacted).toBe(true);
    expect(anonymous.body.details.canonicalPayload).toBeUndefined();
    expect(JSON.stringify(anonymous.body)).not.toContain("\"amount\"");

    const analyst = await request(app).get("/api/audit/AUTH-SVC/verify").set(bearer(token(ANALYST)));
    expect(analyst.body.details.canonicalPayload.amount).toBe(500);
  });

  test("a HUMAN_REVIEW decision cannot be resolved by the platform service", async () => {
    const res = await request(app).post("/api/audit/review").set(bearer(token("service:platform")))
      .send({ transactionId: "AUTH-SVC", reviewerDecision: "APPROVE" });
    expect(res.status).toBe(403);
    expect(res.body.message).toContain("authorised analyst");
  });

  test("a VERIFY decision is resolved by the platform service as an account-holder step-up", async () => {
    const service = bearer(token("service:platform"));
    await request(app).post("/api/audit/evaluate").set(service).send({
      transactionId: "AUTH-VERIFY", riskScore: 42, confidence: 90, amount: 500, transactionType: "PAYMENT", modelVersion: "m",
      signals: { modelScores: { lstm: 0.38, gnn: 0.44 }, rules: [], ...VALIDATED }
    });
    const res = await request(app).post("/api/audit/review").set(service).send({ transactionId: "AUTH-VERIFY", reviewerDecision: "APPROVE" });
    expect(res.status).toBe(200);
    expect(db.getRecord("AUTH-VERIFY")).toMatchObject({ reviewerKind: "ACCOUNT_HOLDER_STEP_UP", reviewerRef: "service:platform" });
  });

  test("separation of duties: an analyst cannot review a decision about themselves", async () => {
    const service = bearer(token("service:platform"));
    await request(app).post("/api/audit/evaluate").set(service).send(decision("AUTH-SOD", {
      signals: { modelScores: { a: 0.9, b: 0.1 }, rules: [], ...VALIDATED, subjectRef: subjectRef(SUBJECT) }
    }));
    expect(db.getRecord("AUTH-SOD").assurance.route).toBe("HUMAN_REVIEW");

    const self = await request(app).post("/api/audit/review").set(bearer(token(SUBJECT))).send({ transactionId: "AUTH-SOD", reviewerDecision: "APPROVE" });
    expect(self.status).toBe(403);
    expect(self.body.message).toContain("Separation of duties");

    const other = await request(app).post("/api/audit/review").set(bearer(token(ANALYST))).send({ transactionId: "AUTH-SOD", reviewerDecision: "REJECT", reason: "Velocity pattern" });
    expect(other.status).toBe(200);
    expect(other.body.separationOfDuties).toBe("ENFORCED");
    expect(db.getRecord("AUTH-SOD")).toMatchObject({ reviewerKind: "ANALYST", reviewerRef: subjectRef(ANALYST) });
  });

  test("evidence exports are restricted and logged with the caller's keyed reference", async () => {
    expect((await request(app).get("/api/audit/AUTH-SOD/evidence")).status).toBe(401);
    const res = await request(app).get("/api/audit/AUTH-SOD/evidence").set(bearer(token(ANALYST)));
    expect(res.status).toBe(200);
    expect(res.headers["cache-control"]).toBe("no-store");
    expect(db.getEvidenceExports("AUTH-SOD")).toEqual([{ actorRef: subjectRef(ANALYST), exportedAt: expect.any(String) }]);
  });

  test("with enforcement off every route is open as before, and a bad token is ignored, not refused", async () => {
    relax();
    expect((await request(app).get("/api/audit/records")).status).toBe(200);
    expect((await request(app).get("/api/audit/records").set(bearer("not-a-jwt"))).status).toBe(200);
    const verify = await request(app).get("/api/audit/AUTH-SVC/verify");
    expect(verify.body.details.canonicalPayload).toBeDefined();
  });
});

describe("keyed pseudonymous subject references", () => {
  afterEach(() => { delete process.env.JWT_SECRET; delete process.env.AUDIT_PSEUDONYM_KEY; });

  test("match the cross-language vector (also asserted by the platform's Python tests)", () => {
    process.env.JWT_SECRET = "vector-secret-not-for-production";
    expect(subjectRef(17)).toBe("54a2a6c26d59c6e0d39d5e5e39a43a517fd0b79e2fb44ae435a4a4997dab9e00");
    expect(subjectRef("18")).toBe("38d28f38aa46b80df1cb544b0371ad1d33fd53d25abdf0746070bc3680183475");
  });

  test("cannot be recomputed without the key, unlike the unkeyed hash they replace", () => {
    process.env.JWT_SECRET = "secret-a";
    const a = subjectRef(17);
    process.env.JWT_SECRET = "secret-b";
    expect(subjectRef(17)).not.toBe(a);
    const unkeyed = crypto.createHash("sha256").update("platform-user:17").digest("hex");
    expect(a).not.toBe(unkeyed);
  });

  test("AUDIT_PSEUDONYM_KEY overrides the derived key, and no key means no reference", () => {
    process.env.JWT_SECRET = "secret-a";
    const derived = subjectRef(17);
    process.env.AUDIT_PSEUDONYM_KEY = "separate-key";
    expect(subjectRef(17)).not.toBe(derived);
    delete process.env.AUDIT_PSEUDONYM_KEY;
    delete process.env.JWT_SECRET;
    expect(subjectRef(17)).toBeNull();
  });
});
