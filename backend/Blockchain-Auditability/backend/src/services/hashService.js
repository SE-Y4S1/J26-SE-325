const crypto = require("crypto");

/**
 * Deterministic SHA-256 Hash Service for Audit Records
 * Ensures exact same audit inputs produce identical hash commitments across time and environments.
 */
class HashService {
  /**
   * Generates a deterministic canonical representation of the audit record fields and computes SHA-256.
   * Canonical fields:
   * - transactionId
   * - riskScore
   * - confidence
   * - amount
   * - transactionType
   * - aiDecision
   * - finalDecision
   * - modelVersion
   * - policyId
   * - policyVersion
   * - timestamp
   * - reviewRequired
   * - reviewerDecision
   */
  generateCanonicalHash(record) {
    const canonicalPayload = {
      aiDecision: record.aiDecision || "",
      amount: Number(record.amount),
      confidence: Number(record.confidence),
      finalDecision: record.finalDecision || "",
      modelVersion: String(record.modelVersion || ""),
      policyId: String(record.policyId || "P001"),
      policyVersion: String(record.policyVersion || "1.0"),
      reviewRequired: Boolean(record.reviewRequired),
      reviewerDecision: record.reviewerDecision ? String(record.reviewerDecision) : null,
      riskScore: Number(record.riskScore),
      timestamp: record.timestamp || "",
      transactionId: String(record.transactionId || ""),
      transactionType: String(record.transactionType || "")
    };

    // Sort JSON keys deterministically
    const canonicalString = JSON.stringify(canonicalPayload, Object.keys(canonicalPayload).sort());
    
    // Generate SHA-256 hex digest
    const hash = crypto.createHash("sha256").update(canonicalString).digest("hex");

    return {
      hash,
      canonicalPayload,
      canonicalString
    };
  }
}

module.exports = new HashService();
