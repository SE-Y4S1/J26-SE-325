/**
 * What the service does with missing, zero, malformed and hostile input: every case is either
 * refused with a JSON error that names the problem, or routed conservatively. Nothing reaches
 * the ledger on a refused request, and nothing answers with a stack trace.
 */
const request = require("supertest");
const path = require("path");
const fs = require("fs");

const testDbPath = path.join(__dirname, "test_input_validation.sqlite");
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

let seq = 0;
const id = () => `VAL-${seq++}`;
const base = (overrides = {}) => ({
  transactionId: id(), riskScore: 30, confidence: 90, amount: 2500,
  transactionType: "PAYMENT", modelVersion: "m1", ...overrides
});
const FULL = { modelScores: { lstm: 0.01, gnn: 0.02 }, rules: [], benchmarkModel: true, sourceModelVersion: "fraud-bundle" };

async function refused(body, pattern) {
  const before = (await getDatabase().getAllRecords()).length;
  const res = await request(app).post("/api/audit/evaluate").send(body);
  expect(res.status).toBe(400);
  expect(res.body.success).toBe(false);
  expect(res.body.errors.join(" ")).toMatch(pattern);
  expect((await getDatabase().getAllRecords()).length).toBe(before);   // nothing stored
  return res;
}

afterAll(() => {
  getDatabase().close();
  removeTestDb();
});

describe("refused input", () => {
  test("empty body names every missing field", async () => {
    const res = await request(app).post("/api/audit/evaluate").send({});
    expect(res.status).toBe(400);
    expect(res.body.errors).toHaveLength(6);
  });

  test("all zeros: a zero amount is refused (a zero risk score alone is valid)", () =>
    refused({ transactionId: id(), riskScore: 0, confidence: 0, amount: 0, transactionType: "PAYMENT", modelVersion: "m" }, /Amount/));

  test("strings where numbers belong", () => refused(base({ riskScore: "abc", amount: "lots" }), /Risk score.*Amount/));

  test("out-of-range and negative values", () => refused(base({ riskScore: -5, confidence: 250, amount: -1 }), /Risk score.*Confidence.*Amount/));

  test("nulls everywhere", () => refused({ transactionId: null, riskScore: null, confidence: null, amount: null, transactionType: null, modelVersion: null }, /Transaction ID/));

  test("an over-long transaction id", () => refused(base({ transactionId: "X".repeat(129) }), /at most 128/));

  test("a transaction id with markup or spaces", () => refused(base({ transactionId: "<b>tx 1</b>" }), /Transaction ID/));

  test("a transaction type carrying markup", () => refused(base({ transactionType: "<script>alert(1)</script>" }), /Transaction type/));

  test("an over-long model version", () => refused(base({ modelVersion: "v".repeat(257) }), /Model version/));

  test("unknown signal fields", () => refused(base({ signals: { foo: 1 } }), /unknown field/));

  test("model scores outside [0,1]", () => refused(base({ signals: { modelScores: { a: 7, b: -1 } } }), /between 0 and 1/));

  test("an upstream decision that is not ALLOW/STEP_UP/BLOCK", () =>
    refused(base({ upstreamDecision: { source: "component2", decision: "MAYBE" } }), /upstreamDecision.decision/));

  test("a JSON array instead of an object", async () => {
    const res = await request(app).post("/api/audit/evaluate").send([1, 2, 3]);
    expect(res.status).toBe(400);
    expect(res.body.errors).toEqual(["Request body must be a JSON object."]);
  });
});

describe("transport errors answer as JSON, never a stack trace", () => {
  test("malformed JSON", async () => {
    const res = await request(app).post("/api/audit/evaluate").set("Content-Type", "application/json").send("{not json");
    expect(res.status).toBe(400);
    expect(res.body).toEqual({ success: false, message: "Request body is not valid JSON." });
    expect(res.text).not.toMatch(/at JSON\.parse|node_modules/);
  });

  test("a body over the size limit", async () => {
    const res = await request(app).post("/api/audit/evaluate").send({ pad: "x".repeat(120 * 1024) });
    expect(res.status).toBe(413);
    expect(res.body.success).toBe(false);
  });

  test("an unknown route", async () => {
    const res = await request(app).get("/api/nothing-here");
    expect(res.status).toBe(404);
    expect(res.body.success).toBe(false);
  });

  test("a review with no body is a 400, not a crash", async () => {
    const res = await request(app).post("/api/audit/review").set("Content-Type", "text/plain").send("hello");
    expect(res.status).toBe(400);
  });
});

describe("accepted input is routed conservatively", () => {
  test("signals sent but every one missing: degraded, never automatic", async () => {
    const res = await request(app).post("/api/audit/evaluate").send(base({ signals: {} }));
    expect(res.status).toBe(200);
    expect(res.body.assurance.signalQuality).toBe("DEGRADED");
    expect(res.body.assurance.route).not.toBe("AUTO");
  });

  test("an absurd amount is held for a human by the amount veto", async () => {
    const res = await request(app).post("/api/audit/evaluate").send(base({ amount: 1e300, riskScore: 5, signals: FULL }));
    expect(res.status).toBe(200);
    expect(res.body.assurance.route).toBe("HUMAN_REVIEW");
    expect(res.body.assurance.vetoes.map((v) => v.code)).toContain("AMOUNT_REVIEW");
  });

  test("a confident decision with agreeing, validated models is automatic", async () => {
    const res = await request(app).post("/api/audit/evaluate").send(base({ riskScore: 1, signals: FULL }));
    expect(res.status).toBe(200);
    expect(res.body.assurance.route).toBe("AUTO");
    expect(res.body.assurance.signalQuality).toBe("FULL");
  });

  test("a resubmitted transaction id is refused before anything is anchored", async () => {
    const body = base({ signals: FULL });
    expect((await request(app).post("/api/audit/evaluate").send(body)).status).toBe(200);
    const again = await request(app).post("/api/audit/evaluate").send(body);
    expect(again.status).toBe(409);
  });

  test("the platform's execution id format is accepted", async () => {
    const res = await request(app).post("/api/audit/evaluate")
      .send(base({ transactionId: "EXE-0123456789abcdef0123456789abcdef", transactionType: "PORTFOLIO_WITHDRAWAL", modelVersion: "c1:abc|c2:fraud-bundle-2026-10" }));
    expect(res.status).toBe(200);
  });

  test("verifying an id that does not exist reports NOT_FOUND", async () => {
    const res = await request(app).get("/api/audit/NO-SUCH-TX/verify");
    expect(res.body.status).toBe("NOT_FOUND");
  });
});
