const express = require("express");
const router = express.Router();
const auditController = require("../controllers/auditController");
const { requireRole } = require("../auth/auth");

// Who may call what when AUTH_REQUIRED is on (with it off, every route is open as before):
//   service   -- the platform service anchoring decisions and recording step-up verifications;
//   reviewer  -- an analyst/auditor listed in AUDIT_REVIEWER_IDS.
const serviceOrReviewer = requireRole("service", "reviewer");
const reviewer = requireRole("reviewer");

// Decisions
router.post("/evaluate", serviceOrReviewer, auditController.evaluateDecision);
router.post("/review", serviceOrReviewer, auditController.submitHumanReview);
router.get("/queue", reviewer, auditController.getQueue);

// Records and evidence (contain amounts and scores)
router.get("/records", reviewer, auditController.getRecords);
router.get("/records/:transactionId", reviewer, auditController.getRecordDetail);
router.get("/stats", reviewer, auditController.getStats);
router.get("/:transactionId/evidence", reviewer, auditController.getEvidence);

// Public: policy documents and hashes, and verification (record contents redacted unless permitted)
router.get("/policies", auditController.getPolicies);
router.get("/:transactionId/verify", auditController.verifyAudit);
router.get("/by-audit-id/:auditId/verify", auditController.verifyByAuditId);

// Ledger maintenance
router.post("/:transactionId/retry", serviceOrReviewer, auditController.retryCommitment);

// Demonstration controls (also gated by NODE_ENV/ALLOW_DEMO_TAMPER)
router.delete("/records", reviewer, auditController.clearRecords);
router.post("/reset", reviewer, auditController.clearRecords);
router.post("/:transactionId/tamper", reviewer, auditController.simulateTampering);

module.exports = router;
