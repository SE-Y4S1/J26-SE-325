const path = require("path");
const fs = require("fs");

let DatabaseEngine;
let engineType = "";

/**
 * Resolved on first SQLite use, not at load: a Postgres deployment must boot even where
 * better-sqlite3 failed to build and node:sqlite does not exist (Node < 22.5).
 */
function loadSqliteEngine() {
  if (DatabaseEngine) return;
  try {
    DatabaseEngine = require("better-sqlite3");
    engineType = "better-sqlite3";
  } catch {
    try {
      const nodeSqlite = require("node:sqlite");
      DatabaseEngine = nodeSqlite.DatabaseSync;
      engineType = "node:sqlite";
    } catch (err) {
      throw new Error("Neither better-sqlite3 nor node:sqlite is available in this Node runtime: " + err.message);
    }
  }
}

// Component 2's enforced decision and Component 1's plan hash; see hashService.
const PROVENANCE_COLUMNS = ["upstreamSource", "upstreamDecision", "upstreamModelVersion", "planHash"];

const { DuplicateTransactionError } = require("./errors");

class AuditDatabase {
  constructor(customDbPath = null) {
    const defaultDataDir = path.join(__dirname, "../../data");
    if (!fs.existsSync(defaultDataDir)) {
      fs.mkdirSync(defaultDataDir, { recursive: true });
    }

    const dbPath = customDbPath || process.env.DB_PATH || path.join(defaultDataDir, "audits.sqlite");
    this.dbPath = dbPath;
    loadSqliteEngine();
    this.engineType = engineType;
    this.db = new DatabaseEngine(dbPath);

    // Performance and integrity pragmas
    if (typeof this.db.pragma === "function") {
      this.db.pragma("journal_mode = WAL");
      this.db.pragma("synchronous = NORMAL");
      this.db.pragma("foreign_keys = ON");
    } else {
      this.db.exec("PRAGMA journal_mode = WAL;");
      this.db.exec("PRAGMA synchronous = NORMAL;");
      this.db.exec("PRAGMA foreign_keys = ON;");
    }

    this.initSchema();
    this.prepareStatements();
  }

  initSchema() {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS audits (
        transactionId TEXT PRIMARY KEY,
        auditId TEXT UNIQUE NOT NULL,
        salt TEXT NOT NULL,
        policyHash TEXT NOT NULL,
        recordHash TEXT NOT NULL,
        policyId TEXT NOT NULL,
        policyVersion TEXT NOT NULL,
        modelVersion TEXT NOT NULL,
        aiDecision TEXT NOT NULL,
        finalDecision TEXT NOT NULL,
        reviewRequired INTEGER NOT NULL DEFAULT 0,
        reviewerDecision TEXT,
        riskScore REAL NOT NULL,
        confidence REAL NOT NULL,
        amount REAL NOT NULL,
        transactionType TEXT NOT NULL,
        reason TEXT,
        timestamp TEXT NOT NULL,
        reviewedAt TEXT,
        blockchainTxId TEXT,
        blockNumber INTEGER,
        verificationStatus TEXT NOT NULL DEFAULT 'RECORDED_ON_CHAIN',
        isTampered INTEGER NOT NULL DEFAULT 0,
        tamperedAt TEXT,
        tamperedField TEXT,
        previousValue TEXT,
        lastVerifiedAt TEXT,
        createdAt TEXT NOT NULL,
        updatedAt TEXT NOT NULL
      );

      CREATE UNIQUE INDEX IF NOT EXISTS idx_audits_auditId ON audits(auditId);
      CREATE INDEX IF NOT EXISTS idx_audits_timestamp ON audits(timestamp DESC);
      CREATE INDEX IF NOT EXISTS idx_audits_status ON audits(verificationStatus);
    `);

    // Cross-component provenance columns, added after the first release. Nullable, so rows
    // written before them stay valid, and added in place so an existing audits.sqlite upgrades.
    const columns = new Set(this.db.prepare("PRAGMA table_info(audits)").all().map((c) => c.name));
    for (const column of PROVENANCE_COLUMNS) {
      if (!columns.has(column)) {
        this.db.exec(`ALTER TABLE audits ADD COLUMN ${column} TEXT`);
      }
    }
  }

  prepareStatements() {
    this.stmts = {
      insert: this.db.prepare(`
        INSERT INTO audits (
          transactionId, auditId, salt, policyHash, recordHash,
          policyId, policyVersion, modelVersion, aiDecision, finalDecision,
          reviewRequired, reviewerDecision, riskScore, confidence, amount,
          transactionType, reason, timestamp, reviewedAt, blockchainTxId,
          blockNumber, verificationStatus, isTampered, tamperedAt, tamperedField,
          previousValue, lastVerifiedAt, createdAt, updatedAt,
          upstreamSource, upstreamDecision, upstreamModelVersion, planHash
        ) VALUES (
          @transactionId, @auditId, @salt, @policyHash, @recordHash,
          @policyId, @policyVersion, @modelVersion, @aiDecision, @finalDecision,
          @reviewRequired, @reviewerDecision, @riskScore, @confidence, @amount,
          @transactionType, @reason, @timestamp, @reviewedAt, @blockchainTxId,
          @blockNumber, @verificationStatus, @isTampered, @tamperedAt, @tamperedField,
          @previousValue, @lastVerifiedAt, @createdAt, @updatedAt,
          @upstreamSource, @upstreamDecision, @upstreamModelVersion, @planHash
        )
      `),
      update: this.db.prepare(`
        UPDATE audits SET
          recordHash = @recordHash,
          finalDecision = @finalDecision,
          reviewRequired = @reviewRequired,
          reviewerDecision = @reviewerDecision,
          reviewedAt = @reviewedAt,
          blockchainTxId = @blockchainTxId,
          blockNumber = @blockNumber,
          verificationStatus = @verificationStatus,
          isTampered = @isTampered,
          tamperedAt = @tamperedAt,
          tamperedField = @tamperedField,
          previousValue = @previousValue,
          lastVerifiedAt = @lastVerifiedAt,
          updatedAt = @updatedAt
        WHERE transactionId = @transactionId
      `),
      getByTxId: this.db.prepare(`
        SELECT * FROM audits WHERE transactionId = ?
      `),
      getByAuditId: this.db.prepare(`
        SELECT * FROM audits WHERE auditId = ?
      `),
      getAll: this.db.prepare(`
        SELECT * FROM audits ORDER BY timestamp DESC
      `),
      deleteAll: this.db.prepare(`
        DELETE FROM audits
      `)
    };
  }

  formatRow(row) {
    if (!row) return null;
    return {
      ...row,
      reviewRequired: Boolean(row.reviewRequired),
      isTampered: Boolean(row.isTampered),
      amount: Number(row.amount),
      riskScore: Number(row.riskScore),
      confidence: Number(row.confidence)
    };
  }

  toInsertPayload(record) {
    const now = new Date().toISOString();
    return {
      transactionId: record.transactionId,
      auditId: record.auditId,
      salt: record.salt,
      policyHash: record.policyHash,
      recordHash: record.recordHash || record.hash || "",
      policyId: record.policyId || "P001",
      policyVersion: record.policyVersion || "1.0",
      modelVersion: record.modelVersion || "1.0.0",
      aiDecision: record.aiDecision || "",
      finalDecision: record.finalDecision || "",
      reviewRequired: record.reviewRequired ? 1 : 0,
      reviewerDecision: record.reviewerDecision || null,
      riskScore: Number(record.riskScore),
      confidence: Number(record.confidence),
      amount: Number(record.amount),
      transactionType: record.transactionType || "TRANSFER",
      reason: record.reason || "",
      timestamp: record.timestamp || now,
      reviewedAt: record.reviewedAt || null,
      blockchainTxId: record.blockchainTxId || null,
      blockNumber: record.blockNumber !== undefined ? record.blockNumber : null,
      verificationStatus: record.verificationStatus || "RECORDED_ON_CHAIN",
      isTampered: record.isTampered ? 1 : 0,
      tamperedAt: record.tamperedAt || null,
      tamperedField: record.tamperedField || null,
      previousValue: record.previousValue !== undefined ? String(record.previousValue) : null,
      lastVerifiedAt: record.lastVerifiedAt || null,
      createdAt: record.createdAt || now,
      updatedAt: now,
      upstreamSource: record.upstreamSource || null,
      upstreamDecision: record.upstreamDecision || null,
      upstreamModelVersion: record.upstreamModelVersion || null,
      planHash: record.planHash || null
    };
  }

  toUpdatePayload(record) {
    const now = new Date().toISOString();
    return {
      transactionId: record.transactionId,
      recordHash: record.recordHash || record.hash || "",
      finalDecision: record.finalDecision || "",
      reviewRequired: record.reviewRequired ? 1 : 0,
      reviewerDecision: record.reviewerDecision || null,
      reviewedAt: record.reviewedAt || null,
      blockchainTxId: record.blockchainTxId || null,
      blockNumber: record.blockNumber !== undefined ? record.blockNumber : null,
      verificationStatus: record.verificationStatus || "RECORDED_ON_CHAIN",
      isTampered: record.isTampered ? 1 : 0,
      tamperedAt: record.tamperedAt || null,
      tamperedField: record.tamperedField || null,
      previousValue: record.previousValue !== undefined ? String(record.previousValue) : null,
      lastVerifiedAt: record.lastVerifiedAt || null,
      updatedAt: now
    };
  }

  createRecord(record) {
    const existing = this.stmts.getByTxId.get(record.transactionId);
    if (existing) {
      throw new DuplicateTransactionError(`Transaction ID '${record.transactionId}' already exists`);
    }

    const payload = this.toInsertPayload(record);
    try {
      this.stmts.insert.run(payload);
    } catch (err) {
      if (err.message && err.message.includes("UNIQUE constraint failed")) {
        throw new DuplicateTransactionError(`Transaction ID '${record.transactionId}' or Audit ID already exists`);
      }
      throw err;
    }

    return this.getRecord(record.transactionId);
  }

  updateRecord(record) {
    const existing = this.stmts.getByTxId.get(record.transactionId);
    if (!existing) {
      throw new Error(`Record ${record.transactionId} not found for update`);
    }

    const payload = this.toUpdatePayload(record);
    this.stmts.update.run(payload);
    return this.getRecord(record.transactionId);
  }

  saveRecord(record) {
    const existing = this.stmts.getByTxId.get(record.transactionId);
    if (existing) {
      return this.updateRecord(record);
    }
    return this.createRecord(record);
  }

  getRecord(transactionId) {
    const row = this.stmts.getByTxId.get(transactionId);
    return this.formatRow(row);
  }

  getRecordByAuditId(auditId) {
    const row = this.stmts.getByAuditId.get(auditId);
    return this.formatRow(row);
  }

  getAllRecords() {
    const rows = this.stmts.getAll.all();
    return rows.map((r) => this.formatRow(r));
  }

  clearAllRecords() {
    this.stmts.deleteAll.run();
    return { success: true, message: "All off-chain SQLite audit records cleared successfully." };
  }

  tamperRecord(transactionId, fieldToTamper, tamperedValue) {
    const allowedFields = ["riskScore", "confidence", "amount", "finalDecision", "aiDecision"];
    if (!allowedFields.includes(fieldToTamper)) {
      throw new Error(`Field '${fieldToTamper}' is not allowed for tampering simulation.`);
    }

    const current = this.getRecord(transactionId);
    if (!current) {
      throw new Error(`Record ${transactionId} not found.`);
    }

    const now = new Date().toISOString();
    const previousVal = current[fieldToTamper];
    const stmt = this.db.prepare(`
      UPDATE audits SET
        ${fieldToTamper} = ?,
        isTampered = 1,
        tamperedAt = ?,
        tamperedField = ?,
        previousValue = ?,
        verificationStatus = 'TAMPER_SUSPECTED',
        updatedAt = ?
      WHERE transactionId = ?
    `);
    stmt.run(tamperedValue, now, fieldToTamper, String(previousVal), now, transactionId);

    return {
      success: true,
      message: `Tampering simulated successfully on transaction ${transactionId}. Field '${fieldToTamper}' modified from ${previousVal} to ${tamperedValue}.`,
      tamperedRecord: this.getRecord(transactionId)
    };
  }

  close() {
    this.db.close();
  }
}

let instance = null;

/** Postgres when DATABASE_URL names one (hosted deployments); SQLite otherwise. */
function postgresUrl() {
  const url = process.env.DATABASE_URL || "";
  return /^postgres(ql)?:\/\//.test(url) ? url : null;
}

function getDatabase(customPath = null) {
  if (!customPath && postgresUrl()) {
    if (!instance) {
      // Required lazily so SQLite-only installs never need the pg package.
      const { PostgresAuditDatabase } = require("./postgresDatabase");
      instance = new PostgresAuditDatabase(postgresUrl());
    }
    return instance;
  }
  if (!instance || customPath) {
    const db = new AuditDatabase(customPath);
    if (!customPath) {
      instance = db;
    }
    return db;
  }
  return instance;
}

/** Tests only: install a store (e.g. Postgres over pg-mem) as the shared instance. */
function useDatabase(db) {
  instance = db;
}

module.exports = {
  AuditDatabase,
  DuplicateTransactionError,
  getDatabase,
  useDatabase
};
