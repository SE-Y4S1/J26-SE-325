const crypto = require("crypto");
const { computePolicyHash } = require("./policyDefinition");
const { canonicalJson } = require("./canonicalJson");

const PROVENANCE_FIELDS = ["upstreamSource", "upstreamDecision", "upstreamModelVersion", "planHash"];

const sha256 = (text) => "0x" + crypto.createHash("sha256").update(text).digest("hex");
const present = (v) => v !== null && v !== undefined && v !== "";

/**
 * The assurance result as committed: everything that decided the route, and nothing that is
 * only presentation (the free-text reasons are derived from these values).
 */
function assuranceCommitment(assurance) {
  const a = typeof assurance === "string" ? JSON.parse(assurance) : assurance;
  return {
    engine: String(a.engine),
    policyId: String(a.policyId),
    policyVersion: String(a.policyVersion),
    aiDecision: String(a.aiDecision),
    route: String(a.route),
    assessedRoute: String(a.assessedRoute),
    assuranceScore: a.assuranceScore === null || a.assuranceScore === undefined ? null : Number(a.assuranceScore),
    components: a.components
      ? { confidence: Number(a.components.confidence), disagreement: Number(a.components.disagreement), context: Number(a.components.context) }
      : null,
    signalQuality: String(a.signalQuality),
    degradedSignals: [...(a.degradedSignals || [])].map(String).sort(),
    unreliableSignals: [...(a.unreliableSignals || [])].map(String).sort(),
    vetoes: (a.vetoes || []).map((v) => String(v.code)).sort(),
    escalation: a.escalation ? { source: String(a.escalation.source), decision: String(a.escalation.decision) } : null
  };
}

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
    if (Number(record.recordVersion) === 2) {
      return record.reviewAuditId ? this.reviewCommitment(record) : this.decisionCommitment(record);
    }
    return this.generateCanonicalHashV1(record);
  }

  /** Pseudonymous on-chain id of a record's review commitment: sha256(txId:salt:review). */
  computeReviewAuditId(transactionId, salt) {
    return sha256(`${this.computeAuditIdInput(transactionId, salt)}:review`);
  }

  computeAuditIdInput(transactionId, salt) {
    if (!transactionId || typeof transactionId !== "string") throw new Error("Invalid transactionId for auditId computation");
    if (!salt || typeof salt !== "string") throw new Error("Invalid salt for auditId computation");
    return `${transactionId}:${salt}`;
  }

  /**
   * Record v2, commitment #1: the decision as made, anchored immediately -- including decisions
   * held for verification or review. Uses the decision-time outcome (decisionFinalDecision), so
   * a later review never changes this hash; the review is committed separately and linked.
   */
  decisionPayload(record) {
    if (!record.salt) throw new Error("Cannot hash record without salt");
    if (!record.auditId) throw new Error("Cannot hash record without auditId");
    const payload = {
      recordVersion: 2,
      commitment: "DECISION",
      aiDecision: String(record.aiDecision || ""),
      amount: Number(record.amount),
      auditId: String(record.auditId),
      confidence: Number(record.confidence),
      finalDecision: String(record.decisionFinalDecision || ""),
      modelVersion: String(record.modelVersion || ""),
      policyHash: String(record.policyHash),
      policyId: String(record.policyId),
      policyVersion: String(record.policyVersion),
      reviewRequired: Boolean(record.reviewRequired),
      // Kept (as null: no review exists when the decision is made) so the payload has every
      // key a v1 payload had; consumers read it.
      reviewerDecision: null,
      riskScore: Number(record.riskScore),
      salt: String(record.salt),
      timestamp: String(record.timestamp || ""),
      transactionId: String(record.transactionId || ""),
      transactionType: String(record.transactionType || "")
    };
    for (const field of PROVENANCE_FIELDS) {
      if (present(record[field])) payload[field] = String(record[field]);
    }
    if (present(record.subjectRef)) payload.subjectRef = String(record.subjectRef);
    if (present(record.assurance)) payload.assurance = assuranceCommitment(record.assurance);
    return payload;
  }

  decisionCommitment(record) {
    const canonicalPayload = this.decisionPayload(record);
    const canonicalString = canonicalJson(canonicalPayload);
    return { hash: sha256(canonicalString), canonicalPayload, canonicalString };
  }

  /**
   * Record v2, commitment #2: the review, linked to the decision by parentRecordHash. It repeats
   * the decision fields with the reviewed outcome, so the latest commitment alone describes the
   * record's current state.
   */
  reviewPayload(record) {
    if (!record.reviewAuditId || !record.parentRecordHash) {
      throw new Error("Cannot hash a review without reviewAuditId and parentRecordHash");
    }
    return {
      ...this.decisionPayload(record),
      commitment: "REVIEW",
      finalDecision: String(record.finalDecision || ""),
      reviewerDecision: String(record.reviewerDecision || ""),
      reviewerKind: String(record.reviewerKind || ""),
      reviewerRef: present(record.reviewerRef) ? String(record.reviewerRef) : null,
      reviewReason: present(record.reviewReason) ? String(record.reviewReason) : null,
      reviewedAt: String(record.reviewedAt || ""),
      reviewAuditId: String(record.reviewAuditId),
      parentRecordHash: String(record.parentRecordHash)
    };
  }

  reviewCommitment(record) {
    const canonicalPayload = this.reviewPayload(record);
    const canonicalString = canonicalJson(canonicalPayload);
    return { hash: sha256(canonicalString), canonicalPayload, canonicalString };
  }

  /** Record v1 (every record anchored before the assurance engine): unchanged. */
  generateCanonicalHashV1(record) {
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

    // Cross-component provenance (Component 2's enforced decision and Component 1's plan
    // hash). Hashed only when present, so records anchored before these fields existed
    // still verify against the commitments already on-chain.
    for (const field of PROVENANCE_FIELDS) {
      if (record[field] !== null && record[field] !== undefined && record[field] !== "") {
        canonicalPayload[field] = String(record[field]);
      }
    }

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
module.exports.assuranceCommitment = assuranceCommitment;
