const crypto = require("crypto");
const { computePolicyHash } = require("./policyDefinition");

/**
 * Deterministic Cryptographic Hash Service
 * Implements salted pseudonymous auditId and canonical SHA-256 commitments in bytes32 format.
 */
class HashService {
  /**
   * Generates a cryptographically secure random 32-byte salt (64 hex characters).
   */
  generateSalt() {
    return crypto.randomBytes(32).toString("hex");
  }

  /**
   * Computes a deterministic pseudonymous auditId from transactionId and salt.
   * Returns a 0x-prefixed 32-byte hex string (bytes32).
   */
  computeAuditId(transactionId, salt) {
    if (!transactionId || typeof transactionId !== "string") {
      throw new Error("Invalid transactionId for auditId computation");
    }
    if (!salt || typeof salt !== "string") {
      throw new Error("Invalid salt for auditId computation");
    }
    const hash = crypto.createHash("sha256").update(`${transactionId}:${salt}`).digest("hex");
    return "0x" + hash;
  }

  /**
   * Generates a deterministic canonical representation of the audit record fields and computes SHA-256.
   * Format: 0x-prefixed 32-byte hex string (bytes32 compatible with EVM).
   * 
   * Canonical fields (16 fields, sorted alphabetically):
   * - aiDecision
   * - amount
   * - auditId
   * - confidence
   * - finalDecision
   * - modelVersion
   * - policyHash
   * - policyId
   * - policyVersion
   * - reviewRequired
   * - reviewerDecision
   * - riskScore
   * - salt
   * - timestamp
   * - transactionId
   * - transactionType
   */
  generateCanonicalHash(record) {
    if (!record.salt) {
      throw new Error("Cannot hash record without salt");
    }
    if (!record.auditId) {
      throw new Error("Cannot hash record without auditId");
    }

    const defaultPolicyHash = computePolicyHash();

    const canonicalPayload = {
      aiDecision: String(record.aiDecision || ""),
      amount: Number(record.amount),
      auditId: String(record.auditId),
      confidence: Number(record.confidence),
      finalDecision: String(record.finalDecision || ""),
      modelVersion: String(record.modelVersion || ""),
      policyHash: String(record.policyHash || defaultPolicyHash),
      policyId: String(record.policyId || "P001"),
      policyVersion: String(record.policyVersion || "1.0"),
      reviewRequired: Boolean(record.reviewRequired),
      reviewerDecision: record.reviewerDecision !== null && record.reviewerDecision !== undefined ? String(record.reviewerDecision) : null,
      riskScore: Number(record.riskScore),
      salt: String(record.salt),
      timestamp: String(record.timestamp || ""),
      transactionId: String(record.transactionId || ""),
      transactionType: String(record.transactionType || "")
    };

    // Sort JSON keys deterministically
    const canonicalString = JSON.stringify(canonicalPayload, Object.keys(canonicalPayload).sort());

    // Generate SHA-256 hex digest prefixed with 0x (bytes32)
    const hash = "0x" + crypto.createHash("sha256").update(canonicalString).digest("hex");

    return {
      hash,
      canonicalPayload,
      canonicalString
    };
  }
}

module.exports = new HashService();
