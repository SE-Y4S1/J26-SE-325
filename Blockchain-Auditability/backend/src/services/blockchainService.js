const { ethers } = require("ethers");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

/**
 * Blockchain Service connecting to local Hardhat node via ethers.js.
 * Includes an automatic fallback mock adapter if local Hardhat RPC is unavailable.
 */
class BlockchainService {
  constructor() {
    this.configPath = path.join(__dirname, "../config/contractConfig.json");
    this.provider = null;
    this.contract = null;
    this.signer = null;
    this.isLiveNetwork = false;
    this.mockLedger = new Map();

    this.init();
  }

  async init() {
    try {
      if (fs.existsSync(this.configPath)) {
        const config = JSON.parse(fs.readFileSync(this.configPath, "utf8"));
        this.provider = new ethers.JsonRpcProvider(config.networkUrl || "http://127.0.0.1:8545");
        
        // Quick probe to test connection
        await this.provider.getBlockNumber();
        
        this.signer = await this.provider.getSigner(0);
        this.contract = new ethers.Contract(config.address, config.abi, this.signer);
        this.isLiveNetwork = true;
        console.log(`[BlockchainService] Connected to Live Local Blockchain at ${config.address}`);
      } else {
        console.log("[BlockchainService] No contractConfig.json found. Operating in Mock Blockchain mode.");
      }
    } catch (err) {
      console.log(`[BlockchainService] Live RPC unavailable (${err.message}). Using Mock Blockchain fallback.`);
      this.isLiveNetwork = false;
    }
  }

  async recordAudit(transactionId, decision, policyVersion, modelVersion, recordHash) {
    await this.init(); // Refresh connection if needed

    if (this.isLiveNetwork && this.contract) {
      try {
        const tx = await this.contract.recordAudit(
          transactionId,
          decision,
          policyVersion,
          modelVersion,
          recordHash
        );
        const receipt = await tx.wait();
        return {
          success: true,
          blockchainTxId: receipt.hash,
          blockNumber: receipt.blockNumber,
          isMock: false,
          network: "Hardhat Local Network (ChainId: 31337)"
        };
      } catch (err) {
        console.error("[BlockchainService] Smart contract execution error, falling back to mock:", err.message);
      }
    }

    // Fallback Mock Recording
    const mockTxHash = "0x" + crypto.createHash("sha256").update(transactionId + recordHash + Date.now()).digest("hex");
    this.mockLedger.set(transactionId, {
      transactionId,
      recordHash,
      decision,
      policyVersion,
      modelVersion,
      timestamp: Math.floor(Date.now() / 1000),
      blockchainTxId: mockTxHash
    });

    return {
      success: true,
      blockchainTxId: mockTxHash,
      blockNumber: Math.floor(1000 + Math.random() * 9000),
      isMock: true,
      network: "Simulated Local Blockchain Adapter"
    };
  }

  async getAudit(transactionId) {
    await this.init();

    if (this.isLiveNetwork && this.contract) {
      try {
        const exists = await this.contract.hasAudit(transactionId);
        if (exists) {
          const audit = await this.contract.getAudit(transactionId);
          return {
            exists: true,
            recordHash: audit[0],
            decision: audit[1],
            policyVersion: audit[2],
            modelVersion: audit[3],
            timestamp: Number(audit[4]),
            submitter: audit[5],
            isMock: false
          };
        }
      } catch (err) {
        console.error("[BlockchainService] Error fetching audit from contract:", err.message);
      }
    }

    if (this.mockLedger.has(transactionId)) {
      const mockRecord = this.mockLedger.get(transactionId);
      return {
        exists: true,
        recordHash: mockRecord.recordHash,
        decision: mockRecord.decision,
        policyVersion: mockRecord.policyVersion,
        modelVersion: mockRecord.modelVersion,
        timestamp: mockRecord.timestamp,
        submitter: "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
        isMock: true
      };
    }

    return { exists: false };
  }

  getStatus() {
    return {
      isLiveNetwork: this.isLiveNetwork,
      mode: this.isLiveNetwork ? "Hardhat Local Blockchain (Ethers.js)" : "Simulated Blockchain Adapter",
      rpcUrl: "http://127.0.0.1:8545"
    };
  }
}

module.exports = new BlockchainService();
