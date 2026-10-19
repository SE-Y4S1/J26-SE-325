const { getDatabase, DuplicateTransactionError } = require("../db/database");
const hashService = require("./hashService");
const { blockchainService, BlockchainUnavailableError, BlockchainTransactionError } = require("./blockchainService");
const { computePolicyHash } = require("./policyDefinition");

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
    return masked;
  }

  createRecord(record) {
    return this.db.createRecord(record);
  }

  updateRecord(record) {
    return this.db.updateRecord(record);
  }

  saveRecord(record) {
    return this.db.saveRecord(record);
  }

  getRecord(transactionId) {
    return this.db.getRecord(transactionId);
  }

  getRecordByAuditId(auditId) {
    return this.db.getRecordByAuditId(auditId);
  }

  getAllRecords(masked = false) {
    const records = this.db.getAllRecords();
    return masked ? records.map((r) => this.maskRecord(r)) : records;
  }

  clearAllRecords() {
    return this.db.clearAllRecords();
  }

  tamperRecord(transactionId, fieldToTamper, tamperedValue) {
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
    let record = this.getRecord(transactionId);
    if (!record) {
      // Check if transactionId might actually be an auditId
      record = this.getRecordByAuditId(transactionId);
      if (!record) {
        return {
          verified: false,
          status: "NOT_FOUND",
          message: `Audit record for '${transactionId}' was not found in off-chain database.`
        };
      }
    }

    // 1. Verify internal pseudonymous auditId derivation
    const expectedAuditId = hashService.computeAuditId(record.transactionId, record.salt);
    if (expectedAuditId.toLowerCase() !== record.auditId.toLowerCase()) {
      record.verificationStatus = "MODIFIED";
      record.lastVerifiedAt = new Date().toISOString();
      this.db.saveRecord(record);

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
      this.db.saveRecord(record);

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
    this.db.saveRecord(record);

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
   * Retries publishing an on-chain commitment for records that previously failed.
   */
  async retryBlockchainCommitment(transactionId) {
    const record = this.getRecord(transactionId);
    if (!record) {
      throw new Error(`Record ${transactionId} not found`);
    }

    // Reconciliation: check if it was already recorded on-chain
    const onChain = await blockchainService.getAudit(record.auditId);
    if (onChain.exists) {
      record.verificationStatus = onChain.isMock ? "VERIFIED_SIMULATED" : "RECORDED_ON_CHAIN";
      this.db.saveRecord(record);
      return {
        success: true,
        reconciled: true,
        message: "Record was already committed on-chain.",
        record: this.maskRecord(record)
      };
    }

    const blockchainRes = await blockchainService.recordAudit(
      record.auditId,
      record.recordHash,
      record.policyHash
    );

    record.blockchainTxId = blockchainRes.blockchainTxId;
    record.blockNumber = blockchainRes.blockNumber;
    record.verificationStatus = blockchainRes.isMock ? "VERIFIED_SIMULATED" : "RECORDED_ON_CHAIN";

    this.db.saveRecord(record);

    return {
      success: true,
      reconciled: false,
      message: "Successfully published commitment to smart contract.",
      blockchain: blockchainRes,
      record: this.maskRecord(record)
    };
  }
}

module.exports = new AuditService();
