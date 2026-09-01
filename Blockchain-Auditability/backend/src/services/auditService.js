const fs = require("fs");
const path = require("path");
const hashService = require("./hashService");
const blockchainService = require("./blockchainService");

/**
 * Off-Chain Audit Service & Data Repository
 * Handles persistent storage, verification logic, and tamper simulation.
 */
class AuditService {
  constructor() {
    this.dataDir = path.join(__dirname, "../../data");
    this.filePath = path.join(this.dataDir, "audits.json");
    this.memoryStore = new Map();
    this.initStorage();
  }

  initStorage() {
    try {
      if (!fs.existsSync(this.dataDir)) {
        fs.mkdirSync(this.dataDir, { recursive: true });
      }
      if (fs.existsSync(this.filePath)) {
        const rawData = fs.readFileSync(this.filePath, "utf8");
        const list = JSON.parse(rawData);
        list.forEach(record => this.memoryStore.set(record.transactionId, record));
      }
    } catch (err) {
      console.error("[AuditService] Error initializing storage:", err.message);
    }
  }

  persist() {
    try {
      const records = Array.from(this.memoryStore.values());
      fs.writeFileSync(this.filePath, JSON.stringify(records, null, 2), "utf8");
    } catch (err) {
      console.error("[AuditService] Error persisting audits:", err.message);
    }
  }

  saveRecord(record) {
    this.memoryStore.set(record.transactionId, record);
    this.persist();
    return record;
  }

  clearAllRecords() {
    this.memoryStore.clear();
    this.persist();
    return { success: true, message: "All off-chain audit records cleared successfully." };
  }

  getRecord(transactionId) {
    return this.memoryStore.get(transactionId) || null;
  }

  getAllRecords() {
    const list = Array.from(this.memoryStore.values());
    // Sort descending by timestamp
    return list.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
  }

  /**
   * Verification Logic:
   * 1. Retrieve the off-chain audit record.
   * 2. Reconstruct the canonical audit data structure.
   * 3. Calculate SHA-256 hash again.
   * 4. Retrieve stored hash from blockchain commitment.
   * 5. Compare calculated hash with blockchain hash.
   * 6. Return VERIFIED or INTEGRITY_FAILED.
   */
  async verifyRecord(transactionId) {
    const record = this.getRecord(transactionId);
    if (!record) {
      return {
        verified: false,
        status: "NOT_FOUND",
        message: `Audit record for transaction ${transactionId} was not found in off-chain database.`
      };
    }

    // 1. Reconstruct canonical representation & compute hash
    const { hash: calculatedHash, canonicalPayload } = hashService.generateCanonicalHash(record);

    // 2. Retrieve on-chain commitment from blockchain
    const onChainAudit = await blockchainService.getAudit(transactionId);

    const blockchainHash = onChainAudit.exists ? onChainAudit.recordHash : record.hash;
    const storedHash = record.hash;

    // 3. Integrity comparison
    const hashesMatch = calculatedHash === blockchainHash && calculatedHash === storedHash;

    const resultStatus = hashesMatch ? "VERIFIED" : "INTEGRITY_FAILED";
    
    // Update local record verification status
    record.verificationStatus = resultStatus;
    record.lastVerifiedAt = new Date().toISOString();
    this.saveRecord(record);

    return {
      verified: hashesMatch,
      status: resultStatus,
      message: hashesMatch
        ? "Audit record integrity successfully verified against blockchain commitment."
        : "ALERT: Cryptographic integrity failure detected! Calculated SHA-256 does not match blockchain commitment.",
      details: {
        transactionId,
        storedOffChainHash: storedHash,
        calculatedCurrentHash: calculatedHash,
        blockchainOnChainHash: blockchainHash,
        isTampered: record.isTampered || false,
        canonicalPayload,
        onChainData: onChainAudit.exists ? onChainAudit : null
      }
    };
  }

  /**
   * Simulate Tampering (Development Feature for Research Demo)
   * Modifies an off-chain audit field (e.g. riskScore) WITHOUT updating the blockchain hash.
   */
  tamperRecord(transactionId, fieldToTamper = "riskScore", tamperedValue = 20) {
    const record = this.getRecord(transactionId);
    if (!record) {
      throw new Error(`Record ${transactionId} not found.`);
    }

    const previousValue = record[fieldToTamper];
    record[fieldToTamper] = tamperedValue;
    record.isTampered = true;
    record.tamperedAt = new Date().toISOString();
    record.tamperedField = fieldToTamper;
    record.previousValue = previousValue;
    record.verificationStatus = "TAMPER_SUSPECTED";

    this.saveRecord(record);

    return {
      success: true,
      message: `Tampering simulated successfully on transaction ${transactionId}. Field '${fieldToTamper}' modified from ${previousValue} to ${tamperedValue}.`,
      tamperedRecord: record
    };
  }
}

module.exports = new AuditService();
