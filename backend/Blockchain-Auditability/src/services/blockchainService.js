const { ethers } = require("ethers");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

class BlockchainUnavailableError extends Error {
  constructor(message) {
    super(message);
    this.name = "BlockchainUnavailableError";
    this.statusCode = 503;
  }
}

class BlockchainTransactionError extends Error {
  constructor(message, details = null) {
    super(message);
    this.name = "BlockchainTransactionError";
    this.statusCode = 502;
    this.details = details;
  }
}

/**
 * Blockchain Service connecting to local Hardhat node via ethers.js.
 * Strictly operates in LIVE mode by default.
 * Mock mode is ONLY active when explicitly set via BLOCKCHAIN_MODE=mock.
 */
class BlockchainService {
  constructor() {
    this.configPath = process.env.CONTRACT_CONFIG_PATH || path.join(__dirname, "../config/contractConfig.json");
    this.provider = null;
    this.contract = null;
    this.signer = null;
    this.config = null;
    this.mockLedger = new Map();
    this.mockPolicies = new Set();
  }

  /**
   * The audit database, when it can persist the simulated chain (Postgres). In memory alone
   * the mock ledger forgets every commitment on restart, and a host that restarts idle
   * instances (Render's free tier) would then report every older record ON_CHAIN_MISSING.
   */
  persistentLedger() {
    // Required here, not at module load: database.js is loaded by the services that load this.
    const { getDatabase } = require("../db/database");
    const db = getDatabase();
    return typeof db.putLedgerEntry === "function" ? db : null;
  }

  isMockMode() {
    return process.env.BLOCKCHAIN_MODE === "mock";
  }

  getConfigPath() {
    return process.env.CONTRACT_CONFIG_PATH || path.join(__dirname, "../config/contractConfig.json");
  }

  async init() {
    if (this.isMockMode()) {
      return;
    }

    const configPath = this.getConfigPath();
    if (!fs.existsSync(configPath)) {
      throw new BlockchainUnavailableError(
        `Contract configuration not found at ${configPath}. Ensure the Hardhat node is running and deploy script has executed.`
      );
    }

    try {
      const configRaw = fs.readFileSync(configPath, "utf8");
      this.config = JSON.parse(configRaw);

      const rpcUrl = process.env.BLOCKCHAIN_RPC_URL || this.config.networkUrl || "http://127.0.0.1:8545";
      this.provider = new ethers.JsonRpcProvider(rpcUrl);

      // Verify node connectivity
      await this.provider.getBlockNumber();

      // Retrieve first local signer (or configured private key)
      if (process.env.SIGNER_PRIVATE_KEY) {
        this.signer = new ethers.Wallet(process.env.SIGNER_PRIVATE_KEY, this.provider);
      } else {
        this.signer = await this.provider.getSigner(0);
      }

      this.contract = new ethers.Contract(this.config.address, this.config.abi, this.signer);
    } catch (err) {
      this.contract = null;
      this.provider = null;
      this.signer = null;
      throw new BlockchainUnavailableError(
        `Failed to connect to local blockchain RPC: ${err.message}`
      );
    }
  }

  /**
   * Records an immutable audit commitment to the smart contract.
   * @param {string} auditId 0x-prefixed 32-byte hex string
   * @param {string} recordHash 0x-prefixed 32-byte hex string
   * @param {string} policyHash 0x-prefixed 32-byte hex string
   */
  async recordAudit(auditId, recordHash, policyHash) {
    if (this.isMockMode()) {
      const mockTxHash = "0x" + crypto.createHash("sha256").update(auditId + recordHash + Date.now()).digest("hex");
      const blockTimestamp = Math.floor(Date.now() / 1000);
      const mockBlockNumber = Math.floor(1000 + Math.random() * 9000);

      const entry = {
        auditId,
        recordHash,
        policyHash,
        timestamp: blockTimestamp,
        submitter: "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
        blockchainTxId: mockTxHash,
        blockNumber: mockBlockNumber
      };
      this.mockLedger.set(auditId, entry);
      const store = this.persistentLedger();
      if (store) {
        await store.putLedgerEntry(auditId, entry);
      }

      return {
        success: true,
        blockchainTxId: mockTxHash,
        blockNumber: mockBlockNumber,
        blockTimestamp,
        isMock: true,
        network: "Simulated Local Blockchain Adapter (BLOCKCHAIN_MODE=mock)"
      };
    }

    await this.init();

    try {
      const tx = await this.contract.recordAudit(auditId, recordHash, policyHash);
      const receipt = await tx.wait();

      const block = await this.provider.getBlock(receipt.blockNumber);
      const blockTimestamp = block ? block.timestamp : Math.floor(Date.now() / 1000);

      return {
        success: true,
        blockchainTxId: receipt.hash,
        blockNumber: receipt.blockNumber,
        blockTimestamp,
        isMock: false,
        network: `Hardhat Local Network (ChainId: ${this.config.chainId || 31337})`
      };
    } catch (err) {
      console.error("[BlockchainService] Transaction Error:", err.message);
      throw new BlockchainTransactionError(
        `Smart contract transaction failed: ${err.message}`,
        err.shortMessage || err.message
      );
    }
  }

  /**
   * Retrieves an on-chain commitment by auditId.
   * @param {string} auditId 0x-prefixed 32-byte hex string
   */
  async getAudit(auditId) {
    if (this.isMockMode()) {
      let item = this.mockLedger.get(auditId);
      const store = this.persistentLedger();
      if (!item && store) {
        item = await store.getLedgerEntry(auditId);
        if (item) this.mockLedger.set(auditId, item);
      }
      if (item) {
        return {
          exists: true,
          recordHash: item.recordHash,
          policyHash: item.policyHash,
          timestamp: item.timestamp,
          submitter: item.submitter,
          isMock: true
        };
      }
      return { exists: false, isMock: true };
    }

    await this.init();

    try {
      const result = await this.contract.getAudit(auditId);
      return {
        exists: Boolean(result[0]),
        recordHash: result[1],
        policyHash: result[2],
        timestamp: Number(result[3]),
        submitter: result[4],
        isMock: false
      };
    } catch (err) {
      console.error("[BlockchainService] getAudit error:", err.message);
      throw new BlockchainUnavailableError(`Failed to fetch audit from smart contract: ${err.message}`);
    }
  }

  /**
   * Checks whether a policy hash is active on-chain.
   * @param {string} policyHash 0x-prefixed 32-byte hex string
   */
  async isPolicyActive(policyHash) {
    if (this.isMockMode()) {
      return true;
    }

    await this.init();

    try {
      return await this.contract.isPolicyActive(policyHash);
    } catch (err) {
      throw new BlockchainUnavailableError(`Failed to query policy status from contract: ${err.message}`);
    }
  }

  getStatus() {
    const isMock = this.isMockMode();
    return {
      isMock,
      mode: isMock ? "Simulated Blockchain Adapter (Mock)" : "Live Hardhat Blockchain (Ethers.js)",
      rpcUrl: isMock ? null : (process.env.BLOCKCHAIN_RPC_URL || "http://127.0.0.1:8545"),
      contractAddress: this.config ? this.config.address : null,
      isConfigured: fs.existsSync(this.getConfigPath())
    };
  }
}

const blockchainService = new BlockchainService();

module.exports = {
  blockchainService,
  BlockchainService,
  BlockchainUnavailableError,
  BlockchainTransactionError
};
