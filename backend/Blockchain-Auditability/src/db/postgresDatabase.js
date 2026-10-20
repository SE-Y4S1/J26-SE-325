const { Pool } = require("pg");
const { DuplicateTransactionError } = require("./errors");

/**
 * Postgres-backed audit store, used when DATABASE_URL names a Postgres database.
 *
 * Same method names as the SQLite AuditDatabase, but every method returns a promise; the
 * services await both, so either store can sit behind them. Hosts with an ephemeral disk
 * (Render's free tier wipes it on every spin-down) need this: a SQLite file there loses
 * every off-chain record, and with it every later verification.
 *
 * Each record is kept whole as JSONB, with the fields the queries use pulled into indexed
 * columns. The record shape is owned by the controller and hashService; storing it whole
 * means a new field cannot be silently dropped by a column list here.
 */


const TAMPERABLE_FIELDS = ["riskScore", "confidence", "amount", "finalDecision", "aiDecision"];

class PostgresAuditDatabase {
  /** `pool` is for tests: an in-process Postgres (pg-mem) stands in for a server. */
  constructor(connectionString, pool = null) {
    this.engineType = "postgres";
    this.pool = pool || new Pool({
      connectionString,
      // Hosted Postgres (Prisma Postgres, Render, Neon) requires TLS; local servers do not.
      ssl: /localhost|127\.0\.0\.1/.test(connectionString) ? false : { rejectUnauthorized: false },
      max: 5
    });
    this.ready = this.initSchema();
  }

  async initSchema() {
    await this.pool.query(`
      CREATE TABLE IF NOT EXISTS audit_records (
        transaction_id TEXT PRIMARY KEY,
        audit_id TEXT UNIQUE NOT NULL,
        ts TEXT NOT NULL,
        record JSONB NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_audit_records_ts ON audit_records (ts DESC);
      CREATE TABLE IF NOT EXISTS audit_mock_ledger (
        audit_id TEXT PRIMARY KEY,
        entry JSONB NOT NULL
      );
    `);
  }

  async query(text, params) {
    await this.ready;
    return this.pool.query(text, params);
  }

  formatRow(row) {
    if (!row) return null;
    const record = row.record;
    return {
      ...record,
      reviewRequired: Boolean(record.reviewRequired),
      isTampered: Boolean(record.isTampered),
      amount: Number(record.amount),
      riskScore: Number(record.riskScore),
      confidence: Number(record.confidence)
    };
  }

  normalise(record, existing = null) {
    const now = new Date().toISOString();
    return {
      ...record,
      policyId: record.policyId || "P001",
      policyVersion: record.policyVersion || "1.0",
      modelVersion: record.modelVersion || "1.0.0",
      recordHash: record.recordHash || record.hash || "",
      transactionType: record.transactionType || "TRANSFER",
      reason: record.reason || "",
      timestamp: record.timestamp || now,
      verificationStatus: record.verificationStatus || "RECORDED_ON_CHAIN",
      reviewRequired: Boolean(record.reviewRequired),
      isTampered: Boolean(record.isTampered),
      reviewerDecision: record.reviewerDecision || null,
      createdAt: (existing && existing.createdAt) || record.createdAt || now,
      updatedAt: now
    };
  }

  async createRecord(record) {
    const row = this.normalise(record);
    try {
      await this.query(
        "INSERT INTO audit_records (transaction_id, audit_id, ts, record) VALUES ($1, $2, $3, $4)",
        [row.transactionId, row.auditId, row.timestamp, row]
      );
    } catch (err) {
      if (err.code === "23505") {
        throw new DuplicateTransactionError(`Transaction ID '${record.transactionId}' or Audit ID already exists`);
      }
      throw err;
    }
    return this.getRecord(record.transactionId);
  }

  async updateRecord(record) {
    const existing = await this.getRecord(record.transactionId);
    if (!existing) {
      throw new Error(`Record ${record.transactionId} not found for update`);
    }
    const row = this.normalise({ ...existing, ...record }, existing);
    await this.query("UPDATE audit_records SET record = $2 WHERE transaction_id = $1", [row.transactionId, row]);
    return this.getRecord(record.transactionId);
  }

  async saveRecord(record) {
    const existing = await this.getRecord(record.transactionId);
    return existing ? this.updateRecord(record) : this.createRecord(record);
  }

  async getRecord(transactionId) {
    const { rows } = await this.query("SELECT record FROM audit_records WHERE transaction_id = $1", [transactionId]);
    return this.formatRow(rows[0]);
  }

  async getRecordByAuditId(auditId) {
    const { rows } = await this.query("SELECT record FROM audit_records WHERE audit_id = $1", [auditId]);
    return this.formatRow(rows[0]);
  }

  async getAllRecords() {
    const { rows } = await this.query("SELECT record FROM audit_records ORDER BY ts DESC");
    return rows.map((r) => this.formatRow(r));
  }

  async clearAllRecords() {
    await this.query("DELETE FROM audit_records");
    return { success: true, message: "All off-chain audit records cleared successfully." };
  }

  async tamperRecord(transactionId, fieldToTamper, tamperedValue) {
    if (!TAMPERABLE_FIELDS.includes(fieldToTamper)) {
      throw new Error(`Field '${fieldToTamper}' is not allowed for tampering simulation.`);
    }
    const current = await this.getRecord(transactionId);
    if (!current) {
      throw new Error(`Record ${transactionId} not found.`);
    }
    const now = new Date().toISOString();
    const previousVal = current[fieldToTamper];
    const tampered = {
      ...current,
      [fieldToTamper]: tamperedValue,
      isTampered: true,
      tamperedAt: now,
      tamperedField: fieldToTamper,
      previousValue: String(previousVal),
      verificationStatus: "TAMPER_SUSPECTED",
      updatedAt: now
    };
    // Written directly, not through normalise(): the point is to alter the stored record.
    await this.query("UPDATE audit_records SET record = $2 WHERE transaction_id = $1", [transactionId, tampered]);
    return {
      success: true,
      message: `Tampering simulated successfully on transaction ${transactionId}. Field '${fieldToTamper}' modified from ${previousVal} to ${tamperedValue}.`,
      tamperedRecord: await this.getRecord(transactionId)
    };
  }

  // --- mock ledger persistence (BLOCKCHAIN_MODE=mock) ---------------------------------
  // In memory, the simulated chain forgets every commitment on restart, so every record
  // anchored before it would then verify as ON_CHAIN_MISSING.

  async putLedgerEntry(auditId, entry) {
    await this.query(
      `INSERT INTO audit_mock_ledger (audit_id, entry) VALUES ($1, $2)
       ON CONFLICT (audit_id) DO UPDATE SET entry = EXCLUDED.entry`,
      [auditId, entry]
    );
  }

  async getLedgerEntry(auditId) {
    const { rows } = await this.query("SELECT entry FROM audit_mock_ledger WHERE audit_id = $1", [auditId]);
    return rows[0] ? rows[0].entry : null;
  }

  async close() {
    await this.pool.end();
  }
}

module.exports = { PostgresAuditDatabase };
