const { getDatabase } = require("../db/database");
const hashService = require("./hashService");
const { blockchainService, BlockchainUnavailableError } = require("./blockchainService");
const { policyRegistry } = require("./policyRegistry");

const PENDING = "PENDING_HUMAN_REVIEW";

/** The status a caller of /evaluate would have seen for this record. */
function legacyStatus(record) {
  return record.finalDecision === PENDING ? "HUMAN_REVIEW_REQUIRED" : "COMPLETED";
}

/**
 * Off-Chain Audit Service & Verification Engine
 * Backed by durable SQLite storage and connected to live EVM smart contract commitments.
 */
class AuditService {
  constructor() {
    this.db = getDatabase();
  }

  /**
   * Helper to mask cryptographic salt before returning to external callers.
   * Full salt is stored in SQLite for integrity verification.
   */
  maskRecord(record) {
    if (!record) return null;
    const masked = { ...record };
    if (masked.salt && masked.salt.length > 8) {
      masked.saltMasked = `${masked.salt.substring(0, 8)}…`;
      delete masked.salt;
    }
    // A keyed pseudonym is not personal data, but it links every decision about one person;
    // it appears only in the restricted evidence bundle.
    delete masked.subjectRef;
    return masked;
  }

  async createRecord(record) {
    return this.db.createRecord(record);
  }

  async updateRecord(record) {
    return this.db.updateRecord(record);
  }

  async saveRecord(record) {
    return this.db.saveRecord(record);
  }

  async getRecord(transactionId) {
    return this.db.getRecord(transactionId);
  }

  async getRecordByAuditId(auditId) {
    return this.db.getRecordByAuditId(auditId);
  }

  async getAllRecords(masked = false) {
    const records = await this.db.getAllRecords();
    return masked ? records.map((r) => this.maskRecord(r)) : records;
  }

  async clearAllRecords() {
    return this.db.clearAllRecords();
  }

  async tamperRecord(transactionId, fieldToTamper, tamperedValue) {
    return this.db.tamperRecord(transactionId, fieldToTamper, tamperedValue);
  }

  /**
   * Strict Cryptographic Verification Engine
   * Validates:
   * 1. Audit ID derivation (sha256(txId:salt))
   * 2. Canonical SHA-256 record hash over current off-chain state
   * 3. On-chain commitment retrieval from AuditRegistry smart contract
   * 4. Cryptographic equivalence: calculatedHash === onChainRecordHash
   * 5. Policy specification equivalence: record.policyHash === onChainPolicyHash
   * 
   * Returns one of:
   * - VERIFIED_ON_CHAIN (Live blockchain match)
   * - VERIFIED_SIMULATED (Mock environment match)
   * - MODIFIED (Integrity failure / hash mismatch)
   * - ON_CHAIN_MISSING (No commitment found on-chain)
   * - NOT_FOUND (Transaction not in off-chain database)
   */
  async verifyRecord(transactionId) {
    let record = await this.getRecord(transactionId);
    if (!record) {
      // Check if transactionId might actually be an auditId
      record = await this.getRecordByAuditId(transactionId);
      if (!record) {
        return {
          verified: false,
          status: "NOT_FOUND",
          message: `Audit record for '${transactionId}' was not found in off-chain database.`
        };
      }
    }

    if (Number(record.recordVersion) === 2) {
      return this.verifyRecordV2(record);
    }

    // 1. Verify internal pseudonymous auditId derivation
    const expectedAuditId = hashService.computeAuditId(record.transactionId, record.salt);
    if (expectedAuditId.toLowerCase() !== record.auditId.toLowerCase()) {
      record.verificationStatus = "MODIFIED";
      record.lastVerifiedAt = new Date().toISOString();
      await this.db.saveRecord(record);

      return {
        verified: false,
        status: "MODIFIED",
        message: "ALERT: Cryptographic auditId derivation mismatch. The transaction salt or ID has been tampered with.",
        details: {
          transactionId: record.transactionId,
          auditId: record.auditId,
          expectedAuditId,
          isTampered: true
        }
      };
    }

    // 2. Reconstruct canonical representation & compute fresh SHA-256
    const { hash: calculatedCurrentHash, canonicalPayload } = hashService.generateCanonicalHash(record);

    // 3. Query on-chain commitment from AuditRegistry contract
    let onChainAudit;
    try {
      onChainAudit = await blockchainService.getAudit(record.auditId);
    } catch (err) {
      if (err instanceof BlockchainUnavailableError) {
        return {
          verified: false,
          status: "BLOCKCHAIN_UNAVAILABLE",
          message: `Cannot verify on-chain integrity: ${err.message}`,
          details: {
            transactionId: record.transactionId,
            auditId: record.auditId,
            calculatedCurrentHash
          }
        };
      }
      throw err;
    }

    // 4. Verification decisions
    if (!onChainAudit.exists) {
      record.verificationStatus = "ON_CHAIN_MISSING";
      record.lastVerifiedAt = new Date().toISOString();
      await this.db.saveRecord(record);

      return {
        verified: false,
        status: "ON_CHAIN_MISSING",
        message: `No on-chain commitment exists for auditId ${record.auditId}. Record has not been anchored to the blockchain.`,
        details: {
          transactionId: record.transactionId,
          auditId: record.auditId,
          storedOffChainHash: record.recordHash || record.hash,
          calculatedCurrentHash,
          onChainData: null
        }
      };
    }

    const hashesMatch = calculatedCurrentHash.toLowerCase() === onChainAudit.recordHash.toLowerCase();
    const policyMatches = record.policyHash.toLowerCase() === onChainAudit.policyHash.toLowerCase();
    const isIntegrityValid = hashesMatch && policyMatches;

    let finalStatus = "MODIFIED";
    if (isIntegrityValid) {
      finalStatus = onChainAudit.isMock ? "VERIFIED_SIMULATED" : "VERIFIED_ON_CHAIN";
    }

    record.verificationStatus = finalStatus;
    record.lastVerifiedAt = new Date().toISOString();
    await this.db.saveRecord(record);

    return {
      verified: isIntegrityValid,
      status: finalStatus,
      message: isIntegrityValid
        ? (onChainAudit.isMock
            ? "Record integrity verified against simulated blockchain adapter."
            : "Cryptographic integrity successfully verified against immutable on-chain smart contract commitment.")
        : "ALERT: Cryptographic integrity failure detected! Current record data does not match on-chain commitment.",
      details: {
        transactionId: record.transactionId,
        auditId: record.auditId,
        storedOffChainHash: record.recordHash || record.hash,
        calculatedCurrentHash,
        blockchainOnChainHash: onChainAudit.recordHash,
        policyHash: record.policyHash,
        onChainPolicyHash: onChainAudit.policyHash,
        blockTimestamp: onChainAudit.timestamp,
        submitter: onChainAudit.submitter,
        isTampered: record.isTampered || !isIntegrityValid,
        canonicalPayload,
        onChainData: onChainAudit
      }
    };
  }

  /**
   * Record v2 verification: the decision commitment, the review commitment when there is one,
   * and the link between them. Reported statuses:
   *   VERIFIED_ON_CHAIN / VERIFIED_SIMULATED  every commitment matches the ledger;
   *   MODIFIED          current data no longer hashes to what was anchored;
   *   CHAIN_BROKEN      the review does not point at this record's decision commitment;
   *   ON_CHAIN_MISSING  a commitment was never anchored;
   *   BLOCKCHAIN_UNAVAILABLE  the ledger could not be read.
   */
  async verifyRecordV2(record) {
    const now = new Date().toISOString();
    const persist = async (status) => {
      record.verificationStatus = status;
      record.lastVerifiedAt = now;
      await this.db.saveRecord(record);
    };

    const expectedAuditId = hashService.computeAuditId(record.transactionId, record.salt);
    const reviewed = Boolean(record.reviewAuditId);
    const expectedReviewId = reviewed ? hashService.computeReviewAuditId(record.transactionId, record.salt) : null;
    if (expectedAuditId.toLowerCase() !== String(record.auditId).toLowerCase()
        || (reviewed && expectedReviewId.toLowerCase() !== String(record.reviewAuditId).toLowerCase())) {
      await persist("MODIFIED");
      return {
        verified: false,
        status: "MODIFIED",
        message: "ALERT: Cryptographic auditId derivation mismatch. The transaction salt or ID has been tampered with.",
        details: { transactionId: record.transactionId, auditId: record.auditId, expectedAuditId, isTampered: true, recordVersion: 2 }
      };
    }

    const decision = hashService.decisionCommitment(record);
    const review = reviewed ? hashService.reviewCommitment(record) : null;

    let onDecision;
    let onReview = null;
    try {
      onDecision = await blockchainService.getAudit(record.auditId);
      if (reviewed) onReview = await blockchainService.getAudit(record.reviewAuditId);
    } catch (err) {
      if (err instanceof BlockchainUnavailableError) {
        return {
          verified: false,
          status: "BLOCKCHAIN_UNAVAILABLE",
          message: `Cannot verify on-chain integrity: ${err.message}`,
          details: { transactionId: record.transactionId, auditId: record.auditId, calculatedCurrentHash: (review || decision).hash, recordVersion: 2 }
        };
      }
      throw err;
    }

    const matches = (calculated, onChain) => Boolean(onChain && onChain.exists)
      && calculated.toLowerCase() === String(onChain.recordHash).toLowerCase()
      && String(record.policyHash).toLowerCase() === String(onChain.policyHash).toLowerCase();

    const decisionMatches = matches(decision.hash, onDecision);
    const linkIntact = !reviewed || String(record.parentRecordHash).toLowerCase() === decision.hash.toLowerCase();
    const reviewMatches = !reviewed || matches(review.hash, onReview);
    // Without a review, the outcome must still be the one committed at decision time.
    const outcomeUnchanged = reviewed || record.finalDecision === record.decisionFinalDecision;

    const isMock = Boolean(onDecision.isMock);
    let status;
    let message;
    if (!onDecision.exists || (reviewed && !onReview.exists)) {
      status = "ON_CHAIN_MISSING";
      message = `No on-chain commitment exists for the ${!onDecision.exists ? "decision" : "review"} of this record. It has not been anchored to the blockchain.`;
    } else if (!decisionMatches || !outcomeUnchanged) {
      status = "MODIFIED";
      message = "ALERT: Cryptographic integrity failure detected! Current record data does not match the decision committed on-chain.";
    } else if (!linkIntact) {
      status = "CHAIN_BROKEN";
      message = "ALERT: The review commitment does not link to this record's decision commitment.";
    } else if (!reviewMatches) {
      status = "MODIFIED";
      message = "ALERT: Cryptographic integrity failure detected! The review outcome does not match the review committed on-chain.";
    } else {
      status = isMock ? "VERIFIED_SIMULATED" : "VERIFIED_ON_CHAIN";
      message = isMock
        ? "Record integrity verified against simulated blockchain adapter."
        : "Cryptographic integrity successfully verified against immutable on-chain smart contract commitment.";
    }
    const verified = status === "VERIFIED_SIMULATED" || status === "VERIFIED_ON_CHAIN";
    await persist(status);

    const latest = reviewed ? review : decision;
    const latestOnChain = reviewed ? onReview : onDecision;
    const lifecycle = reviewed ? "REVIEWED" : record.finalDecision === PENDING ? "AWAITING_REVIEW" : "DECIDED";
    return {
      verified,
      status,
      message,
      details: {
        transactionId: record.transactionId,
        auditId: record.auditId,
        recordVersion: 2,
        lifecycle,
        storedOffChainHash: reviewed ? record.reviewRecordHash : record.recordHash,
        calculatedCurrentHash: latest.hash,
        blockchainOnChainHash: latestOnChain.exists ? latestOnChain.recordHash : null,
        policyHash: record.policyHash,
        onChainPolicyHash: latestOnChain.exists ? latestOnChain.policyHash : null,
        blockTimestamp: latestOnChain.exists ? latestOnChain.timestamp : null,
        submitter: latestOnChain.exists ? latestOnChain.submitter : null,
        isTampered: Boolean(record.isTampered) || !verified,
        canonicalPayload: latest.canonicalPayload,
        onChainData: latestOnChain,
        commitments: {
          decision: {
            auditId: record.auditId,
            calculatedHash: decision.hash,
            onChainHash: onDecision.exists ? onDecision.recordHash : null,
            matches: decisionMatches && outcomeUnchanged,
            blockTimestamp: onDecision.exists ? onDecision.timestamp : null
          },
          review: reviewed
            ? {
                auditId: record.reviewAuditId,
                calculatedHash: review.hash,
                onChainHash: onReview.exists ? onReview.recordHash : null,
                matches: reviewMatches,
                parentRecordHash: record.parentRecordHash,
                linkIntact,
                blockTimestamp: onReview.exists ? onReview.timestamp : null
              }
            : null
        }
      }
    };
  }

  /**
   * Publishes whatever a record has not yet anchored -- its decision commitment, and for a
   * reviewed v2 record its review commitment -- skipping anything already on-chain.
   */
  async retryBlockchainCommitment(transactionId) {
    const record = await this.getRecord(transactionId);
    if (!record) {
      throw new Error(`Record ${transactionId} not found`);
    }
    const v2 = Number(record.recordVersion) === 2;
    if (!v2 && record.recordHash === "PENDING_REVIEW") {
      throw new Error(`Record ${transactionId} is awaiting human review; there is no commitment to publish yet.`);
    }

    const pending = [{ kind: "decision", auditId: record.auditId, recordHash: record.recordHash }];
    if (v2 && record.reviewAuditId) {
      pending.push({ kind: "review", auditId: record.reviewAuditId, recordHash: record.reviewRecordHash });
    }

    let published = null;
    let isMock = false;
    for (const commitment of pending) {
      const onChain = await blockchainService.getAudit(commitment.auditId);
      isMock = Boolean(onChain.isMock);
      if (onChain.exists) continue;
      published = await blockchainService.recordAudit(commitment.auditId, commitment.recordHash, record.policyHash);
      isMock = Boolean(published.isMock);
      if (commitment.kind === "decision") {
        record.blockchainTxId = published.blockchainTxId;
        record.blockNumber = published.blockNumber;
      } else {
        record.reviewBlockchainTxId = published.blockchainTxId;
        record.reviewBlockNumber = published.blockNumber;
      }
    }

    record.verificationStatus = isMock ? "VERIFIED_SIMULATED" : "RECORDED_ON_CHAIN";
    await this.db.saveRecord(record);

    return {
      success: true,
      reconciled: !published,
      status: legacyStatus(record),
      message: published ? "Successfully published commitment to smart contract." : "Record was already committed on-chain.",
      ...(published ? { blockchain: published } : {}),
      ...(record.assurance ? { assurance: record.assurance } : {}),
      record: this.maskRecord(record)
    };
  }

  /** Decisions held for verification or review, oldest first. */
  async getQueue() {
    const records = await this.getAllRecords(true);
    return records
      .filter((r) => r.finalDecision === PENDING)
      .map((r) => ({ ...r, route: r.assurance ? r.assurance.route : "HUMAN_REVIEW" }))
      .sort((a, b) => String(a.timestamp).localeCompare(String(b.timestamp)));
  }

  /**
   * Everything an auditor needs to check one decision without trusting this service: the
   * committed payloads with the salt, the policy document, the anchors, and how each hash is
   * computed. Sensitive (it contains the amount, scores and salt) but holds no personal data;
   * every export is logged.
   */
  async buildEvidenceBundle(transactionId, actorRef) {
    const record = (await this.getRecord(transactionId)) || (await this.getRecordByAuditId(transactionId));
    if (!record) return null;
    const v2 = Number(record.recordVersion) === 2;
    const decision = v2 ? hashService.decisionCommitment(record) : hashService.generateCanonicalHashV1(record);
    const review = v2 && record.reviewAuditId ? hashService.reviewCommitment(record) : null;
    const policy = policyRegistry.getByHash(record.policyHash);
    const verification = await this.verifyRecord(record.transactionId);
    const ledger = blockchainService.getStatus();

    await this.db.logEvidenceExport(record.transactionId, actorRef);

    return {
      bundleVersion: 1,
      generatedAt: new Date().toISOString(),
      recordVersion: v2 ? 2 : 1,
      transactionId: record.transactionId,
      salt: record.salt,
      hashing: {
        auditId: "sha256(transactionId + ':' + salt)",
        reviewAuditId: "sha256(transactionId + ':' + salt + ':review')",
        recordHash: v2
          ? "sha256(canonical JSON of the payload: keys sorted at every depth, no whitespace)"
          : "sha256(JSON.stringify(payload, sorted top-level keys))",
        policyHash: policy && policy.hashScheme === "v1"
          ? "sha256(JSON.stringify(document, sorted top-level keys)) -- legacy scheme v1"
          : "sha256(canonical JSON of the policy document)"
      },
      commitments: {
        decision: {
          auditId: record.auditId,
          payload: decision.canonicalPayload,
          recordHash: decision.hash,
          anchor: { blockchainTxId: record.blockchainTxId || null, blockNumber: record.blockNumber ?? null }
        },
        review: review
          ? {
              auditId: record.reviewAuditId,
              payload: review.canonicalPayload,
              recordHash: review.hash,
              anchor: { blockchainTxId: record.reviewBlockchainTxId || null, blockNumber: record.reviewBlockNumber ?? null }
            }
          : null
      },
      policy: policy
        ? { id: policy.id, version: policy.version, hashScheme: policy.hashScheme, policyHash: policy.policyHash, document: policy.document }
        : { policyHash: record.policyHash, document: null, note: "This policy hash is not in the service's registry." },
      ledger: {
        isMock: ledger.isMock,
        network: ledger.isMock ? "simulated" : "evm",
        chainId: ledger.chainId,
        contractAddress: ledger.contractAddress
      },
      verificationAtExport: { status: verification.status, verified: verification.verified, message: verification.message },
      privacy: {
        personalData: "None. No name, email, account key or platform user id is included; subjectRef and reviewerRef are keyed pseudonyms that cannot be reversed without the service key.",
        sensitiveFields: ["amount", "riskScore", "confidence", "salt"],
        handling: "Restricted. Share only with an authorised auditor; the salt links this bundle to its on-chain auditId."
      }
    };
  }
}

module.exports = new AuditService();
