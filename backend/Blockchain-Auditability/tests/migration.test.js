const fs = require("fs");
const os = require("os");
const path = require("path");
const { AuditDatabase } = require("../src/db/database");
const hashService = require("../src/services/hashService");
const { rollback } = require("../scripts/db-rollback");

const { records: LEGACY } = require("./fixtures/legacy-records-v1.json");

// The schema exactly as v1 shipped it (before schema_meta and the v2 columns).
const V1_DDL = `
  CREATE TABLE audits (
    transactionId TEXT PRIMARY KEY, auditId TEXT UNIQUE NOT NULL, salt TEXT NOT NULL,
    policyHash TEXT NOT NULL, recordHash TEXT NOT NULL, policyId TEXT NOT NULL,
    policyVersion TEXT NOT NULL, modelVersion TEXT NOT NULL, aiDecision TEXT NOT NULL,
    finalDecision TEXT NOT NULL, reviewRequired INTEGER NOT NULL DEFAULT 0, reviewerDecision TEXT,
    riskScore REAL NOT NULL, confidence REAL NOT NULL, amount REAL NOT NULL,
    transactionType TEXT NOT NULL, reason TEXT, timestamp TEXT NOT NULL, reviewedAt TEXT,
    blockchainTxId TEXT, blockNumber INTEGER,
    verificationStatus TEXT NOT NULL DEFAULT 'RECORDED_ON_CHAIN', isTampered INTEGER NOT NULL DEFAULT 0,
    tamperedAt TEXT, tamperedField TEXT, previousValue TEXT, lastVerifiedAt TEXT,
    createdAt TEXT NOT NULL, updatedAt TEXT NOT NULL,
    upstreamSource TEXT, upstreamDecision TEXT, upstreamModelVersion TEXT, planHash TEXT
  );`;

const V1_COLUMNS = ["transactionId", "auditId", "salt", "policyHash", "recordHash", "policyId", "policyVersion", "modelVersion",
  "aiDecision", "finalDecision", "reviewRequired", "reviewerDecision", "riskScore", "confidence", "amount", "transactionType",
  "reason", "timestamp", "reviewedAt", "blockchainTxId", "blockNumber", "verificationStatus", "isTampered", "tamperedAt",
  "tamperedField", "previousValue", "lastVerifiedAt", "createdAt", "updatedAt",
  "upstreamSource", "upstreamDecision", "upstreamModelVersion", "planHash"];

function sqlite(file) {
  let Engine;
  try { Engine = require("better-sqlite3"); } catch { Engine = require("node:sqlite").DatabaseSync; }
  return new Engine(file);
}

/** A database file as v1 left it: v1 schema, the three records anchored on the live chain. */
function makeV1Database(dir) {
  const file = path.join(dir, "audits.sqlite");
  const db = sqlite(file);
  db.exec(V1_DDL);
  const insert = db.prepare(`INSERT INTO audits (${V1_COLUMNS.join(",")}) VALUES (${V1_COLUMNS.map(() => "?").join(",")})`);
  for (const r of LEGACY) insert.run(...V1_COLUMNS.map((c) => (r[c] === undefined ? null : r[c])));
  db.close();
  return file;
}

const columnsOf = (file) => {
  const db = sqlite(file);
  try { return db.prepare("PRAGMA table_info(audits)").all().map((c) => c.name); } finally { db.close(); }
};

describe("SQLite schema migration v1 -> v2", () => {
  let dir;
  beforeEach(() => { dir = fs.mkdtempSync(path.join(os.tmpdir(), "c3-migration-")); });
  afterEach(() => { fs.rmSync(dir, { recursive: true, force: true }); });

  test("backs up first, adds only nullable columns, and leaves every v1 record readable and verifiable", () => {
    const file = makeV1Database(dir);
    const db = new AuditDatabase(file);
    try {
      expect(db.migration.from).toBe(1);
      expect(db.migration.to).toBe(2);
      expect(fs.existsSync(db.migration.backupPath)).toBe(true);

      for (const legacy of LEGACY) {
        const record = db.getRecord(legacy.transactionId);
        expect(record.recordVersion).toBeUndefined(); // read exactly as v1 did
        expect(record.assurance).toBeUndefined();
        expect(record.recordHash).toBe(legacy.recordHash);
        const recomputed = hashService.generateCanonicalHash(record).hash;
        expect(recomputed === record.recordHash).toBe(!legacy.isTampered);
      }
    } finally {
      db.close();
    }
    expect(columnsOf(file)).toEqual(expect.arrayContaining(["recordVersion", "assurance", "reviewAuditId", "parentRecordHash"]));
  });

  test("is idempotent: reopening a v2 database migrates nothing and takes no second backup", () => {
    const file = makeV1Database(dir);
    new AuditDatabase(file).close();
    const backups = () => fs.readdirSync(dir).filter((f) => f.includes(".bak-v1-"));
    expect(backups()).toHaveLength(1);
    const again = new AuditDatabase(file);
    try {
      expect(again.migration).toEqual({ from: 2, to: 2, backupPath: null });
      expect(again.getAllRecords()).toHaveLength(LEGACY.length);
    } finally {
      again.close();
    }
    expect(backups()).toHaveLength(1);
  });

  test("v1 code paths still work on the migrated file (its INSERT names only v1 columns)", () => {
    const file = makeV1Database(dir);
    new AuditDatabase(file).close();
    const raw = sqlite(file);
    try {
      const r = { ...LEGACY[1], transactionId: "TX-V1-WRITER", auditId: "0x" + "77".repeat(32) };
      raw.prepare(`INSERT INTO audits (${V1_COLUMNS.join(",")}) VALUES (${V1_COLUMNS.map(() => "?").join(",")})`)
        .run(...V1_COLUMNS.map((c) => (r[c] === undefined ? null : r[c])));
    } finally {
      raw.close();
    }
    const db = new AuditDatabase(file);
    try {
      expect(db.getRecord("TX-V1-WRITER").recordVersion).toBeUndefined();
    } finally {
      db.close();
    }
  });

  test("a fresh database starts at v2 and takes no backup", () => {
    const db = new AuditDatabase(path.join(dir, "fresh.sqlite"));
    try {
      expect(db.migration.backupPath).toBeNull();
      expect(columnsOf(path.join(dir, "fresh.sqlite"))).toContain("recordVersion");
    } finally {
      db.close();
    }
  });

  test("scripts/db-rollback.js restores the v1 file exactly and keeps the migrated one", () => {
    const file = makeV1Database(dir);
    const migrated = new AuditDatabase(file);
    const backupPath = migrated.migration.backupPath;
    migrated.close();
    expect(columnsOf(file)).toContain("recordVersion");

    const result = rollback(file);
    expect(result.restoredFrom).toBe(backupPath);
    expect(fs.existsSync(result.previousKeptAs)).toBe(true);
    expect(columnsOf(file)).not.toContain("recordVersion");
    const restored = sqlite(file);
    try {
      expect(restored.prepare("SELECT COUNT(*) AS n FROM audits").get().n).toBe(LEGACY.length);
      const hashes = restored.prepare("SELECT recordHash FROM audits ORDER BY timestamp").all().map((r) => r.recordHash);
      expect(hashes).toEqual(LEGACY.map((r) => r.recordHash));
    } finally {
      restored.close();
    }
  });
});
