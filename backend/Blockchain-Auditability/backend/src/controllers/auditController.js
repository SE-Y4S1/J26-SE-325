const policyEngine = require("../services/policyEngine");
const hashService = require("../services/hashService");
const { blockchainService, BlockchainUnavailableError, BlockchainTransactionError } = require("../services/blockchainService");
const auditService = require("../services/auditService");
const { computePolicyHash } = require("../services/policyDefinition");
const { DuplicateTransactionError } = require("../db/database");

// Component 2's gateway actions mapped onto P001's. STEP_UP is a challenge the user must
// pass before the transaction proceeds, so it holds the record for review like HUMAN_REVIEW.
const UPSTREAM_ACTIONS = { ALLOW: "APPROVE", STEP_UP: "HUMAN_REVIEW", BLOCK: "REJECT" };
const ACTION_SEVERITY = { APPROVE: 0, HUMAN_REVIEW: 1, REJECT: 2 };

/**
 * Controller handling API requests for AI decision evaluations, human reviews,
 * audit storage, integrity verification, retries, and tamper simulation.
 */
class AuditController {
  validateInput(data) {
    const errors = [];

    if (!data.transactionId || typeof data.transactionId !== "string" || !data.transactionId.trim()) {
      errors.push("Transaction ID must exist and cannot be empty.");
    }
    if (typeof data.riskScore !== "number" || isNaN(data.riskScore) || data.riskScore < 0 || data.riskScore > 100) {
      errors.push("Risk score must be a number between 0 and 100.");
    }
    if (typeof data.confidence !== "number" || isNaN(data.confidence) || data.confidence < 0 || data.confidence > 100) {
      errors.push("Confidence score must be a number between 0 and 100.");
    }
    if (typeof data.amount !== "number" || isNaN(data.amount) || data.amount <= 0) {
      errors.push("Amount must be a positive number greater than 0.");
    }
    if (!data.transactionType || typeof data.transactionType !== "string" || !data.transactionType.trim()) {
      errors.push("Transaction type must exist.");
    }
    if (!data.modelVersion || typeof data.modelVersion !== "string" || !data.modelVersion.trim()) {
      errors.push("Model version must exist.");
    }
    if (data.upstreamDecision !== undefined && data.upstreamDecision !== null) {
      const upstream = data.upstreamDecision;
      if (typeof upstream !== "object" || !UPSTREAM_ACTIONS[upstream.decision]) {
        errors.push(`upstreamDecision.decision must be one of ${Object.keys(UPSTREAM_ACTIONS).join(", ")}.`);
      }
      if (!upstream || typeof upstream.source !== "string" || !upstream.source.trim()) {
        errors.push("upstreamDecision.source must name the component that enforced the decision.");
      }
    }
    if (data.planHash !== undefined && data.planHash !== null && !/^0x[0-9a-f]{64}$/.test(data.planHash)) {
      errors.push("planHash must be a 0x-prefixed lowercase SHA-256 hex digest.");
    }

    return errors;
  }

  /**
   * Combines P001 with a decision already enforced upstream (Component 2's gateway).
   *
   * The stricter action wins. Without this, P001 judges only riskScore and confidence, so a
   * transaction Component 2 BLOCKED on a rule (structuring, rate limit, reputation) at a
   * moderate score would be anchored on-chain as APPROVE -- a ledger that contradicts the
   * enforcement it is meant to prove.
   */
  combineWithUpstream(policyResult, upstream) {
    if (!upstream) {
      return policyResult;
    }
    const upstreamAction = UPSTREAM_ACTIONS[upstream.decision];
    if (ACTION_SEVERITY[upstreamAction] <= ACTION_SEVERITY[policyResult.action]) {
      return {
        ...policyResult,
        reason: `${policyResult.reason} ${upstream.source} enforced ${upstream.decision}.`
      };
    }
    return {
      ...policyResult,
      action: upstreamAction,
      reason: `${upstream.source} enforced ${upstream.decision}, which is stricter than ${policyResult.policyId} (${policyResult.action}). ${policyResult.reason}`
    };
  }

  /**
   * POST /api/audit/evaluate
   */
  async evaluateDecision(req, res) {
    try {
      const inputData = req.body;

      // 1. Input Validation
      const validationErrors = this.validateInput(inputData);
      if (validationErrors.length > 0) {
        return res.status(400).json({
          success: false,
          message: "Input validation failed",
          errors: validationErrors
        });
      }

      // 2. Policy Engine Evaluation
      const upstream = inputData.upstreamDecision || null;
      const policyResult = this.combineWithUpstream(
        policyEngine.evaluate({
          riskScore: inputData.riskScore,
          confidence: inputData.confidence
        }),
        upstream
      );
      const provenance = {
        upstreamSource: upstream ? upstream.source : null,
        upstreamDecision: upstream ? upstream.decision : null,
        upstreamModelVersion: upstream && upstream.modelVersion ? String(upstream.modelVersion) : null,
        planHash: inputData.planHash || null
      };

      const timestamp = new Date().toISOString();
      const salt = hashService.generateSalt();
      const auditId = hashService.computeAuditId(inputData.transactionId, salt);
      const policyHash = computePolicyHash();
      const reviewRequired = policyResult.action === "HUMAN_REVIEW";

      if (reviewRequired) {
        // Pending human review state
        const pendingRecord = {
          transactionId: inputData.transactionId,
          auditId,
          salt,
          policyHash,
          recordHash: "PENDING_REVIEW",
          riskScore: inputData.riskScore,
          confidence: inputData.confidence,
          amount: inputData.amount,
          transactionType: inputData.transactionType,
          aiDecision: policyResult.action,
          finalDecision: "PENDING_HUMAN_REVIEW",
          modelVersion: inputData.modelVersion,
          policyId: policyResult.policyId,
          policyVersion: policyResult.policyVersion,
          reason: policyResult.reason,
          timestamp,
          reviewRequired: true,
          reviewerDecision: null,
          verificationStatus: "PENDING_HUMAN_REVIEW",
          isTampered: false,
          ...provenance
        };

        try {
          auditService.saveRecord(pendingRecord);
        } catch (dbErr) {
          if (dbErr instanceof DuplicateTransactionError) {
            return res.status(409).json({ success: false, message: dbErr.message });
          }
          throw dbErr;
        }

        return res.status(200).json({
          success: true,
          status: "HUMAN_REVIEW_REQUIRED",
          policyResult,
          record: auditService.maskRecord(pendingRecord)
        });
      }

      // Automated decision (APPROVE or REJECT)
      const auditRecord = {
        transactionId: inputData.transactionId,
        auditId,
        salt,
        policyHash,
        riskScore: inputData.riskScore,
        confidence: inputData.confidence,
        amount: inputData.amount,
        transactionType: inputData.transactionType,
        aiDecision: policyResult.action,
        finalDecision: policyResult.action,
        modelVersion: inputData.modelVersion,
        policyId: policyResult.policyId,
        policyVersion: policyResult.policyVersion,
        reason: policyResult.reason,
        timestamp,
        reviewRequired: false,
        reviewerDecision: null,
        isTampered: false,
        ...provenance
      };

      // 3. Generate Canonical SHA-256 Hash
      const { hash: recordHash } = hashService.generateCanonicalHash(auditRecord);
      auditRecord.recordHash = recordHash;

      // 4. Commit to Smart Contract
      let blockchainRes = null;
      try {
        blockchainRes = await blockchainService.recordAudit(auditId, recordHash, policyHash);
        auditRecord.blockchainTxId = blockchainRes.blockchainTxId;
        auditRecord.blockNumber = blockchainRes.blockNumber;
        auditRecord.verificationStatus = blockchainRes.isMock ? "VERIFIED_SIMULATED" : "RECORDED_ON_CHAIN";

        auditService.createRecord(auditRecord);

        return res.status(200).json({
          success: true,
          status: "COMPLETED",
          policyResult,
          blockchain: blockchainRes,
          record: auditService.maskRecord(auditRecord)
        });
      } catch (chainErr) {
        if (chainErr instanceof BlockchainUnavailableError) {
          auditRecord.verificationStatus = "BLOCKCHAIN_FAILED";
          auditService.saveRecord(auditRecord);

          return res.status(503).json({
            success: false,
            status: "BLOCKCHAIN_UNAVAILABLE",
            message: `Record saved locally, but smart contract anchoring failed: ${chainErr.message}. Use /api/audit/${auditRecord.transactionId}/retry to publish once the blockchain is online.`,
            record: auditService.maskRecord(auditRecord)
          });
        }

        if (chainErr instanceof BlockchainTransactionError) {
          auditRecord.verificationStatus = "BLOCKCHAIN_FAILED";
          auditService.saveRecord(auditRecord);

          return res.status(502).json({
            success: false,
            status: "BLOCKCHAIN_ERROR",
            message: chainErr.message,
            record: auditService.maskRecord(auditRecord)
          });
        }

        throw chainErr;
      }
    } catch (err) {
      if (err instanceof DuplicateTransactionError) {
        return res.status(409).json({ success: false, message: err.message });
      }
      console.error("Evaluation Error:", err);
      return res.status(500).json({ success: false, message: err.message });
    }
  }

  /**
   * POST /api/audit/review
   */
  async submitHumanReview(req, res) {
    try {
      const { transactionId, reviewerDecision } = req.body;

      if (!transactionId || !["APPROVE", "REJECT"].includes(reviewerDecision)) {
        return res.status(400).json({
          success: false,
          message: "Invalid review decision. Must specify transactionId and decision (APPROVE or REJECT)."
        });
      }

      const existingRecord = auditService.getRecord(transactionId);
      if (!existingRecord) {
        return res.status(404).json({ success: false, message: `Record ${transactionId} not found.` });
      }

      if (!existingRecord.reviewRequired) {
        return res.status(400).json({
          success: false,
          message: "This decision did not require human review."
        });
      }

      // Update record with Human Reviewer Decision
      existingRecord.reviewerDecision = reviewerDecision;
      existingRecord.finalDecision = reviewerDecision === "APPROVE" ? "HUMAN_APPROVED" : "HUMAN_REJECTED";
      existingRecord.reviewedAt = new Date().toISOString();

      // Regenerate Canonical SHA-256 Hash including human decision
      const { hash: recordHash } = hashService.generateCanonicalHash(existingRecord);
      existingRecord.recordHash = recordHash;

      // Commit to Smart Contract Blockchain
      try {
        const blockchainRes = await blockchainService.recordAudit(
          existingRecord.auditId,
          recordHash,
          existingRecord.policyHash
        );

        existingRecord.blockchainTxId = blockchainRes.blockchainTxId;
        existingRecord.blockNumber = blockchainRes.blockNumber;
        existingRecord.verificationStatus = blockchainRes.isMock ? "VERIFIED_SIMULATED" : "RECORDED_ON_CHAIN";

        auditService.saveRecord(existingRecord);

        return res.status(200).json({
          success: true,
          message: "Human review recorded and on-chain commitment published successfully.",
          blockchain: blockchainRes,
          record: auditService.maskRecord(existingRecord)
        });
      } catch (chainErr) {
        if (chainErr instanceof BlockchainUnavailableError) {
          existingRecord.verificationStatus = "BLOCKCHAIN_FAILED";
          auditService.saveRecord(existingRecord);

          return res.status(503).json({
            success: false,
            status: "BLOCKCHAIN_UNAVAILABLE",
            message: `Human review saved locally, but smart contract anchoring failed: ${chainErr.message}. Retry via /api/audit/${existingRecord.transactionId}/retry.`,
            record: auditService.maskRecord(existingRecord)
          });
        }
        throw chainErr;
      }
    } catch (err) {
      console.error("Human Review Error:", err);
      return res.status(500).json({ success: false, message: err.message });
    }
  }

  /**
   * GET /api/audit/:transactionId/verify
   */
  async verifyAudit(req, res) {
    try {
      const { transactionId } = req.params;
      const verificationResult = await auditService.verifyRecord(transactionId);
      return res.status(200).json(verificationResult);
    } catch (err) {
      return res.status(500).json({ success: false, message: err.message });
    }
  }

  /**
   * GET /api/audit/by-audit-id/:auditId/verify
   */
  async verifyByAuditId(req, res) {
    try {
      const { auditId } = req.params;
      const verificationResult = await auditService.verifyRecord(auditId);
      return res.status(200).json(verificationResult);
    } catch (err) {
      return res.status(500).json({ success: false, message: err.message });
    }
  }

  /**
   * POST /api/audit/:transactionId/retry
   */
  async retryCommitment(req, res) {
    try {
      const { transactionId } = req.params;
      const result = await auditService.retryBlockchainCommitment(transactionId);
      return res.status(200).json(result);
    } catch (err) {
      if (err instanceof BlockchainUnavailableError) {
        return res.status(503).json({ success: false, message: err.message });
      }
      return res.status(400).json({ success: false, message: err.message });
    }
  }

  /**
   * POST /api/audit/:transactionId/tamper
   */
  async simulateTampering(req, res) {
    try {
      if (process.env.NODE_ENV === "production" && process.env.ALLOW_DEMO_TAMPER !== "true") {
        return res.status(403).json({
          success: false,
          message: "Tamper simulation is a development research feature disabled in production environments. Set ALLOW_DEMO_TAMPER=true to enable."
        });
      }

      const { transactionId } = req.params;
      const { field, tamperedValue } = req.body || {};

      const result = auditService.tamperRecord(
        transactionId,
        field || "riskScore",
        tamperedValue !== undefined ? tamperedValue : 20
      );

      return res.status(200).json({
        ...result,
        tamperedRecord: auditService.maskRecord(result.tamperedRecord)
      });
    } catch (err) {
      return res.status(400).json({ success: false, message: err.message });
    }
  }

  /**
   * DELETE /api/audit/records (or POST /api/audit/reset)
   */
  async clearRecords(req, res) {
    try {
      const result = auditService.clearAllRecords();
      return res.status(200).json(result);
    } catch (err) {
      return res.status(500).json({ success: false, message: err.message });
    }
  }

  /**
   * GET /api/audit/records
   */
  async getRecords(req, res) {
    try {
      const records = auditService.getAllRecords(true); // true = mask salts
      return res.status(200).json({ success: true, count: records.length, records });
    } catch (err) {
      return res.status(500).json({ success: false, message: err.message });
    }
  }

  /**
   * GET /api/audit/stats
   */
  async getStats(req, res) {
    try {
      const records = auditService.getAllRecords(false);
      const blockchainStatus = blockchainService.getStatus();

      let approved = 0;
      let rejected = 0;
      let humanReview = 0;
      let verified = 0;
      let integrityFailures = 0;
      let onChainMissing = 0;

      records.forEach((r) => {
        if (r.finalDecision === "APPROVE" || r.finalDecision === "HUMAN_APPROVED") approved++;
        if (r.finalDecision === "REJECT" || r.finalDecision === "HUMAN_REJECTED") rejected++;
        if (r.reviewRequired) humanReview++;
        if (r.verificationStatus === "VERIFIED_ON_CHAIN" || r.verificationStatus === "VERIFIED_SIMULATED") verified++;
        if (r.verificationStatus === "MODIFIED" || r.isTampered) integrityFailures++;
        if (r.verificationStatus === "ON_CHAIN_MISSING" || r.verificationStatus === "BLOCKCHAIN_FAILED") onChainMissing++;
      });

      return res.status(200).json({
        success: true,
        stats: {
          totalDecisions: records.length,
          approved,
          rejected,
          humanReview,
          verified,
          integrityFailures,
          onChainMissing,
          blockchainStatus
        }
      });
    } catch (err) {
      return res.status(500).json({ success: false, message: err.message });
    }
  }
}

const controller = new AuditController();

module.exports = {
  evaluateDecision: controller.evaluateDecision.bind(controller),
  submitHumanReview: controller.submitHumanReview.bind(controller),
  verifyAudit: controller.verifyAudit.bind(controller),
  verifyByAuditId: controller.verifyByAuditId.bind(controller),
  retryCommitment: controller.retryCommitment.bind(controller),
  simulateTampering: controller.simulateTampering.bind(controller),
  clearRecords: controller.clearRecords.bind(controller),
  getRecords: controller.getRecords.bind(controller),
  getStats: controller.getStats.bind(controller)
};
