const hashService = require("../services/hashService");
const { blockchainService, BlockchainUnavailableError, BlockchainTransactionError } = require("../services/blockchainService");
const auditService = require("../services/auditService");
const assuranceEngine = require("../services/assuranceEngine");
const { policyRegistry } = require("../services/policyRegistry");
const { DuplicateTransactionError } = require("../db/database");

const { authRequired, canSeeRecordContents } = require("../auth/auth");

const PENDING = "PENDING_HUMAN_REVIEW";
const SIGNAL_KEYS = new Set(["modelScores", "rules", "benchmarkModel", "sourceModelVersion", "subjectRef"]);
const MODEL_NAME = /^[A-Za-z0-9_.-]{1,32}$/;

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
    if (data.signals !== undefined) {
      errors.push(...this.validateSignals(data.signals));
    }

    return errors;
  }

  /**
   * Assurance signals are validated strictly and never coerced: a malformed signal must be
   * rejected, not quietly read as "missing" and routed on a degraded basis.
   */
  validateSignals(signals) {
    const errors = [];
    if (signals === null || typeof signals !== "object" || Array.isArray(signals)) {
      return ["signals must be an object."];
    }
    const unknown = Object.keys(signals).filter((k) => !SIGNAL_KEYS.has(k));
    if (unknown.length) {
      errors.push(`signals has unknown field(s): ${unknown.join(", ")}.`);
    }
    if (signals.modelScores !== undefined) {
      const scores = signals.modelScores;
      if (scores === null || typeof scores !== "object" || Array.isArray(scores)) {
        errors.push("signals.modelScores must be an object of model name to score.");
      } else {
        const entries = Object.entries(scores);
        if (entries.length > 10) errors.push("signals.modelScores accepts at most 10 models.");
        for (const [name, value] of entries) {
          if (!MODEL_NAME.test(name)) errors.push(`signals.modelScores has an invalid model name '${name}'.`);
          if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
            errors.push(`signals.modelScores.${name} must be a number between 0 and 1.`);
          }
        }
      }
    }
    if (signals.rules !== undefined) {
      if (!Array.isArray(signals.rules) || signals.rules.length > 50
          || !signals.rules.every((r) => typeof r === "string" && r.length > 0 && r.length <= 64)) {
        errors.push("signals.rules must be an array of at most 50 non-empty strings (max 64 chars).");
      }
    }
    if (signals.benchmarkModel !== undefined && typeof signals.benchmarkModel !== "boolean") {
      errors.push("signals.benchmarkModel must be a boolean.");
    }
    if (signals.sourceModelVersion !== undefined
        && (typeof signals.sourceModelVersion !== "string" || signals.sourceModelVersion.length > 128)) {
      errors.push("signals.sourceModelVersion must be a string of at most 128 characters.");
    }
    if (signals.subjectRef !== undefined && !/^[0-9a-f]{64}$/.test(String(signals.subjectRef))) {
      errors.push("signals.subjectRef must be a 64-character lowercase hex keyed reference.");
    }
    return errors;
  }

  /**
   * Which policy routes this request. Callers that send assurance signals get the active
   * assurance policy; callers that do not (every client written before the engine existed)
   * keep P001, so their outcomes do not change underneath them. ASSURANCE_REQUIRE_SIGNALS
   * routes everything through the active policy, on a degraded basis when signals are absent.
   */
  selectPolicy(inputData) {
    const requireSignals = String(process.env.ASSURANCE_REQUIRE_SIGNALS || "").toLowerCase() === "true";
    return inputData.signals !== undefined || requireSignals ? policyRegistry.getActive() : policyRegistry.getLegacy();
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

      // 2. Assurance assessment under the selected policy, then any decision already
      // enforced upstream (stricter wins). policyResult keeps its original shape.
      const upstream = inputData.upstreamDecision || null;
      const policy = this.selectPolicy(inputData);
      const assessed = assuranceEngine.assess(inputData, policy);
      const assurance = assuranceEngine.applyUpstream(assessed, upstream);
      const policyResult = this.combineWithUpstream(
        {
          action: assessed.legacyAction,
          reason: assessed.reasons.join(" "),
          policyId: policy.id,
          policyVersion: policy.version,
          policyHash: policy.policyHash,
          evaluatedAt: new Date().toISOString()
        },
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
      const policyHash = policy.policyHash;
      const reviewRequired = policyResult.action === "HUMAN_REVIEW";
      const outcome = reviewRequired ? PENDING : policyResult.action;

      // 3. Record v2: every decision -- including one held for verification or review -- is
      // committed now, so the AI's decision and its assurance result are tamper-evident from
      // the moment they are made. A later review is a second, linked commitment.
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
        finalDecision: outcome,
        decisionFinalDecision: outcome,
        modelVersion: inputData.modelVersion,
        policyId: policyResult.policyId,
        policyVersion: policyResult.policyVersion,
        reason: policyResult.reason,
        timestamp,
        reviewRequired,
        reviewerDecision: null,
        isTampered: false,
        recordVersion: 2,
        assurance,
        subjectRef: (inputData.signals && inputData.signals.subjectRef) || null,
        ...provenance
      };
      const { hash: recordHash } = hashService.decisionCommitment(auditRecord);
      auditRecord.recordHash = recordHash;

      // Stored before anchoring: a duplicate is refused before anything reaches the chain, and
      // a failed anchor leaves a record that /retry can publish.
      await auditService.createRecord({ ...auditRecord, verificationStatus: "ANCHOR_PENDING" });

      // 4. Commit to Smart Contract
      let blockchainRes = null;
      try {
        blockchainRes = await blockchainService.recordAudit(auditId, recordHash, policyHash);
        auditRecord.blockchainTxId = blockchainRes.blockchainTxId;
        auditRecord.blockNumber = blockchainRes.blockNumber;
        // A held decision keeps the status callers already read as "awaiting review".
        auditRecord.verificationStatus = reviewRequired
          ? PENDING
          : blockchainRes.isMock ? "VERIFIED_SIMULATED" : "RECORDED_ON_CHAIN";

        await auditService.saveRecord(auditRecord);

        return res.status(200).json({
          success: true,
          status: reviewRequired ? "HUMAN_REVIEW_REQUIRED" : "COMPLETED",
          policyResult,
          assurance,
          blockchain: blockchainRes,
          record: auditService.maskRecord(auditRecord)
        });
      } catch (chainErr) {
        if (chainErr instanceof BlockchainUnavailableError) {
          auditRecord.verificationStatus = "BLOCKCHAIN_FAILED";
          await auditService.saveRecord(auditRecord);

          return res.status(503).json({
            success: false,
            status: "BLOCKCHAIN_UNAVAILABLE",
            message: `Record saved locally, but smart contract anchoring failed: ${chainErr.message}. Use /api/audit/${auditRecord.transactionId}/retry to publish once the blockchain is online.`,
            record: auditService.maskRecord(auditRecord)
          });
        }

        if (chainErr instanceof BlockchainTransactionError) {
          auditRecord.verificationStatus = "BLOCKCHAIN_FAILED";
          await auditService.saveRecord(auditRecord);

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

      const existingRecord = await auditService.getRecord(transactionId);
      if (!existingRecord) {
        return res.status(404).json({ success: false, message: `Record ${transactionId} not found.` });
      }

      if (!existingRecord.reviewRequired) {
        return res.status(400).json({
          success: false,
          message: "This decision did not require human review."
        });
      }
      if (existingRecord.reviewerDecision) {
        return res.status(409).json({ success: false, message: `Record ${transactionId} has already been reviewed.` });
      }
      const reason = req.body.reason;
      if (reason !== undefined && reason !== null && (typeof reason !== "string" || reason.length > 500)) {
        return res.status(400).json({ success: false, message: "reason must be a string of at most 500 characters." });
      }

      if (Number(existingRecord.recordVersion) === 2) {
        return this.submitReviewV2(req, res, existingRecord, reviewerDecision, reason || null);
      }

      // Record v1 (held before decision-time anchoring existed): its single commitment is
      // written now, as it always was.
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

        await auditService.saveRecord(existingRecord);

        return res.status(200).json({
          success: true,
          message: "Human review recorded and on-chain commitment published successfully.",
          blockchain: blockchainRes,
          record: auditService.maskRecord(existingRecord)
        });
      } catch (chainErr) {
        if (chainErr instanceof BlockchainUnavailableError) {
          existingRecord.verificationStatus = "BLOCKCHAIN_FAILED";
          await auditService.saveRecord(existingRecord);

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
   * Review of a v2 record: anchored as a second commitment that links to the decision's.
   *
   * Who may resolve what (enforced when AUTH_REQUIRED is on):
   *   VERIFY        the platform service, after the account holder passes step-up, or an analyst;
   *   HUMAN_REVIEW  an analyst in AUDIT_REVIEWER_IDS only.
   * Separation of duties: whenever the reviewer is identifiable, a review of a decision about
   * that same person is refused, by comparing keyed references.
   */
  async submitReviewV2(req, res, record, reviewerDecision, reason) {
    const route = record.assurance ? record.assurance.route : "HUMAN_REVIEW";
    const caller = req.caller;

    if (authRequired() && route === "HUMAN_REVIEW" && !(caller && caller.isReviewer)) {
      return res.status(403).json({ success: false, message: "This decision was routed to HUMAN_REVIEW and must be reviewed by an authorised analyst." });
    }
    if (caller && !caller.isService && record.subjectRef && caller.ref && caller.ref === record.subjectRef) {
      return res.status(403).json({ success: false, message: "Separation of duties: a reviewer cannot review a decision about themselves." });
    }
    if (["ANCHOR_PENDING", "BLOCKCHAIN_FAILED"].includes(record.verificationStatus)) {
      return res.status(409).json({ success: false, message: `The decision for ${record.transactionId} is not anchored yet; publish it with /api/audit/${record.transactionId}/retry first.` });
    }

    const reviewed = {
      ...record,
      reviewerDecision,
      finalDecision: reviewerDecision === "APPROVE" ? "HUMAN_APPROVED" : "HUMAN_REJECTED",
      reviewedAt: new Date().toISOString(),
      reviewerKind: !caller ? "UNAUTHENTICATED" : caller.isService ? "ACCOUNT_HOLDER_STEP_UP" : "ANALYST",
      reviewerRef: caller ? caller.ref : null,
      reviewReason: reason,
      reviewAuditId: hashService.computeReviewAuditId(record.transactionId, record.salt),
      parentRecordHash: record.recordHash
    };
    const { hash: reviewHash } = hashService.reviewCommitment(reviewed);
    reviewed.reviewRecordHash = reviewHash;

    let blockchainRes;
    try {
      blockchainRes = await blockchainService.recordAudit(reviewed.reviewAuditId, reviewHash, reviewed.policyHash);
    } catch (chainErr) {
      // Nothing is saved: a review counts only once it is anchored, so the reviewer can submit
      // it again when the ledger is back.
      const status = chainErr instanceof BlockchainUnavailableError ? 503 : 502;
      return res.status(status).json({
        success: false,
        status: status === 503 ? "BLOCKCHAIN_UNAVAILABLE" : "BLOCKCHAIN_ERROR",
        message: `The review was not recorded: anchoring failed (${chainErr.message}). Submit it again once the ledger is available.`
      });
    }

    reviewed.reviewBlockchainTxId = blockchainRes.blockchainTxId;
    reviewed.reviewBlockNumber = blockchainRes.blockNumber;
    reviewed.verificationStatus = blockchainRes.isMock ? "VERIFIED_SIMULATED" : "RECORDED_ON_CHAIN";
    await auditService.saveRecord(reviewed);

    return res.status(200).json({
      success: true,
      message: "Human review recorded and on-chain commitment published successfully.",
      blockchain: blockchainRes,
      separationOfDuties: caller ? "ENFORCED" : "NOT_ENFORCED_ANONYMOUS_REVIEWER",
      ...(record.assurance ? { assurance: record.assurance } : {}),
      record: auditService.maskRecord(reviewed)
    });
  }

  /** Strips record contents from a verification result for callers not allowed to see them. */
  redactVerification(req, result) {
    if (canSeeRecordContents(req) || !result.details) return result;
    const details = { ...result.details };
    delete details.canonicalPayload; // amounts, scores, salt
    delete details.onChainData;
    return { ...result, details, redacted: true };
  }

  /**
   * GET /api/audit/:transactionId/verify
   */
  async verifyAudit(req, res) {
    try {
      const { transactionId } = req.params;
      const verificationResult = await auditService.verifyRecord(transactionId);
      return res.status(200).json(this.redactVerification(req, verificationResult));
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
      return res.status(200).json(this.redactVerification(req, verificationResult));
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

      const result = await auditService.tamperRecord(
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
      // Deleting the off-chain store is a demonstration reset, gated like tamper simulation.
      if (process.env.NODE_ENV === "production" && process.env.ALLOW_DEMO_TAMPER !== "true") {
        return res.status(403).json({
          success: false,
          message: "Resetting audit records is a demonstration feature disabled in production. Set ALLOW_DEMO_TAMPER=true to enable."
        });
      }
      const result = await auditService.clearAllRecords();
      return res.status(200).json(result);
    } catch (err) {
      return res.status(500).json({ success: false, message: err.message });
    }
  }

  /**
   * GET /api/audit/queue
   * Decisions held for step-up verification or analyst review, oldest first.
   */
  async getQueue(req, res) {
    try {
      const queue = await auditService.getQueue();
      return res.status(200).json({ success: true, count: queue.length, queue });
    } catch (err) {
      return res.status(500).json({ success: false, message: err.message });
    }
  }

  /**
   * GET /api/audit/records/:transactionId
   * One record with its assurance result and the policy that routed it.
   */
  async getRecordDetail(req, res) {
    try {
      const record = await auditService.getRecord(req.params.transactionId);
      if (!record) {
        return res.status(404).json({ success: false, message: `Record ${req.params.transactionId} not found.` });
      }
      const policy = policyRegistry.getByHash(record.policyHash);
      return res.status(200).json({
        success: true,
        record: auditService.maskRecord(record),
        policy: policy
          ? { id: policy.id, version: policy.version, engine: policy.engine, status: policy.status, hashScheme: policy.hashScheme, policyHash: policy.policyHash }
          : null,
        evidenceExports: await auditService.db.getEvidenceExports(record.transactionId)
      });
    } catch (err) {
      return res.status(500).json({ success: false, message: err.message });
    }
  }

  /**
   * GET /api/audit/:transactionId/evidence
   * The independently verifiable evidence bundle (scripts/verify-evidence.js checks it).
   * Restricted, never cached, and every export is logged with the caller's keyed reference.
   */
  async getEvidence(req, res) {
    try {
      const actorRef = req.caller ? req.caller.ref || req.caller.sub : "anonymous (AUTH_REQUIRED off)";
      const bundle = await auditService.buildEvidenceBundle(req.params.transactionId, actorRef);
      res.set("Cache-Control", "no-store");
      if (!bundle) {
        return res.status(404).json({ success: false, message: `Record ${req.params.transactionId} not found.` });
      }
      return res.status(200).json(bundle);
    } catch (err) {
      return res.status(500).json({ success: false, message: err.message });
    }
  }

  /**
   * GET /api/audit/records
   */
  async getRecords(req, res) {
    try {
      const records = await auditService.getAllRecords(true); // true = mask salts
      return res.status(200).json({ success: true, count: records.length, records });
    } catch (err) {
      return res.status(500).json({ success: false, message: err.message });
    }
  }

  /**
   * GET /api/audit/policies
   * Every policy document with the hash recomputed from it, and whether that hash is active
   * on-chain, so a reader can check the commitment themselves.
   */
  async getPolicies(req, res) {
    try {
      const activeId = policyRegistry.getActive().id;
      const policies = await Promise.all(policyRegistry.list().map(async (p) => {
        let onChain = { active: null, error: null };
        try {
          onChain.active = await blockchainService.isPolicyActive(p.policyHash);
        } catch (err) {
          onChain.error = err.message;
        }
        return {
          id: p.id,
          version: p.version,
          engine: p.engine,
          status: p.status,
          baseline: p.baseline,
          routesSignalRequests: p.id === activeId,
          hashScheme: p.hashScheme,
          policyHash: p.policyHash,
          description: p.description,
          document: p.document,
          onChain
        };
      }));
      return res.status(200).json({ success: true, activePolicyId: activeId, legacyPolicyId: "P001", policies });
    } catch (err) {
      return res.status(500).json({ success: false, message: err.message });
    }
  }

  /**
   * GET /api/audit/stats
   */
  async getStats(req, res) {
    try {
      const records = await auditService.getAllRecords(false);
      const blockchainStatus = blockchainService.getStatus();

      let approved = 0;
      let rejected = 0;
      let humanReview = 0;
      let verified = 0;
      let integrityFailures = 0;
      let onChainMissing = 0;
      let pendingReview = 0;
      // Routing as decided by the assurance engine; records anchored before it count as LEGACY.
      const routes = { AUTO: 0, VERIFY: 0, HUMAN_REVIEW: 0, LEGACY: 0 };
      const signalQuality = { FULL: 0, DEGRADED: 0, LEGACY: 0 };

      records.forEach((r) => {
        if (r.finalDecision === "APPROVE" || r.finalDecision === "HUMAN_APPROVED") approved++;
        if (r.finalDecision === "REJECT" || r.finalDecision === "HUMAN_REJECTED") rejected++;
        if (r.reviewRequired) humanReview++;
        if (r.finalDecision === PENDING) pendingReview++;
        if (r.verificationStatus === "VERIFIED_ON_CHAIN" || r.verificationStatus === "VERIFIED_SIMULATED") verified++;
        if (r.verificationStatus === "MODIFIED" || r.verificationStatus === "CHAIN_BROKEN" || r.isTampered) integrityFailures++;
        if (["ON_CHAIN_MISSING", "BLOCKCHAIN_FAILED", "ANCHOR_PENDING"].includes(r.verificationStatus)) onChainMissing++;
        const a = r.assurance;
        if (a && a.signalQuality !== "LEGACY") {
          routes[a.route] = (routes[a.route] || 0) + 1;
          signalQuality[a.signalQuality] = (signalQuality[a.signalQuality] || 0) + 1;
        } else {
          routes.LEGACY++;
          signalQuality.LEGACY++;
        }
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
          pendingReview,
          routes,
          signalQuality,
          activePolicyId: policyRegistry.getActive().id,
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
  getStats: controller.getStats.bind(controller),
  getPolicies: controller.getPolicies.bind(controller),
  getQueue: controller.getQueue.bind(controller),
  getRecordDetail: controller.getRecordDetail.bind(controller),
  getEvidence: controller.getEvidence.bind(controller)
};
