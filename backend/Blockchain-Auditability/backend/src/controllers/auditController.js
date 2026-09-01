const policyEngine = require("../services/policyEngine");
const hashService = require("../services/hashService");
const blockchainService = require("../services/blockchainService");
const auditService = require("../services/auditService");

/**
 * Controller handling API requests for AI decision evaluations, human reviews,
 * audit storage, integrity verification, and tamper simulation.
 */
class AuditController {
  /**
   * Validate Mock AI Decision Input
   */
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

    return errors;
  }

  /**
   * POST /api/audit/evaluate
   * Input: transactionId, riskScore, confidence, amount, transactionType, modelVersion
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
      const policyResult = policyEngine.evaluate({
        riskScore: inputData.riskScore,
        confidence: inputData.confidence
      });

      const timestamp = new Date().toISOString();
      const reviewRequired = policyResult.action === "HUMAN_REVIEW";

      if (reviewRequired) {
        // If HUMAN_REVIEW is required, save intermediate state
        const pendingRecord = {
          transactionId: inputData.transactionId,
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
          hash: "PENDING_REVIEW",
          blockchainTransactionId: "PENDING",
          verificationStatus: "PENDING_HUMAN_REVIEW",
          isTampered: false
        };

        auditService.saveRecord(pendingRecord);

        return res.status(200).json({
          success: true,
          status: "HUMAN_REVIEW_REQUIRED",
          policyResult,
          record: pendingRecord
        });
      }

      // Automated decision (APPROVE or REJECT)
      const auditRecord = {
        transactionId: inputData.transactionId,
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
        isTampered: false
      };

      // 3. Generate SHA-256 Hash
      const { hash } = hashService.generateCanonicalHash(auditRecord);
      auditRecord.hash = hash;

      // 4. Commit Hash to Smart Contract Blockchain
      const blockchainRes = await blockchainService.recordAudit(
        auditRecord.transactionId,
        auditRecord.finalDecision,
        auditRecord.policyVersion,
        auditRecord.modelVersion,
        hash
      );

      auditRecord.blockchainTransactionId = blockchainRes.blockchainTxId;
      auditRecord.verificationStatus = "RECORDED_ON_CHAIN";

      // 5. Store Off-Chain Record
      auditService.saveRecord(auditRecord);

      return res.status(200).json({
        success: true,
        status: "COMPLETED",
        policyResult,
        blockchain: blockchainRes,
        record: auditRecord
      });
    } catch (err) {
      console.error("Evaluation Error:", err);
      return res.status(500).json({ success: false, message: err.message });
    }
  }

  /**
   * POST /api/audit/review
   * Body: { transactionId, reviewerDecision: "APPROVE" | "REJECT" }
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
      const { hash } = hashService.generateCanonicalHash(existingRecord);
      existingRecord.hash = hash;

      // Commit to Smart Contract Blockchain
      const blockchainRes = await blockchainService.recordAudit(
        existingRecord.transactionId,
        existingRecord.finalDecision,
        existingRecord.policyVersion,
        existingRecord.modelVersion,
        hash
      );

      existingRecord.blockchainTransactionId = blockchainRes.blockchainTxId;
      existingRecord.verificationStatus = "RECORDED_ON_CHAIN";

      auditService.saveRecord(existingRecord);

      return res.status(200).json({
        success: true,
        message: "Human review recorded and on-chain commitment published successfully.",
        blockchain: blockchainRes,
        record: existingRecord
      });
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
   * POST /api/audit/:transactionId/tamper
   */
  async simulateTampering(req, res) {
    try {
      // Dev/Demo Environment Gating Check
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

      return res.status(200).json(result);
    } catch (err) {
      return res.status(400).json({ success: false, message: err.message });
    }
  }

  /**
   * DELETE /api/audit/records (or POST /api/audit/reset)
   * Reseed/Clear demo records for clean evaluation runs
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
      const records = auditService.getAllRecords();
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
      const records = auditService.getAllRecords();
      const blockchainStatus = blockchainService.getStatus();

      let approved = 0;
      let rejected = 0;
      let humanReview = 0;
      let verified = 0;
      let integrityFailures = 0;

      records.forEach(r => {
        if (r.finalDecision === "APPROVE" || r.finalDecision === "HUMAN_APPROVED") approved++;
        if (r.finalDecision === "REJECT" || r.finalDecision === "HUMAN_REJECTED") rejected++;
        if (r.reviewRequired) humanReview++;
        if (r.verificationStatus === "VERIFIED") verified++;
        if (r.verificationStatus === "INTEGRITY_FAILED" || r.isTampered) integrityFailures++;
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
  simulateTampering: controller.simulateTampering.bind(controller),
  clearRecords: controller.clearRecords.bind(controller),
  getRecords: controller.getRecords.bind(controller),
  getStats: controller.getStats.bind(controller)
};
