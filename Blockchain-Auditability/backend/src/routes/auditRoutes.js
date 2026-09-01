const express = require("express");
const router = express.Router();
const auditController = require("../controllers/auditController");

// API Routes
router.post("/evaluate", auditController.evaluateDecision);
router.post("/review", auditController.submitHumanReview);
router.get("/records", auditController.getRecords);
router.delete("/records", auditController.clearRecords);
router.post("/reset", auditController.clearRecords);
router.get("/stats", auditController.getStats);
router.get("/:transactionId/verify", auditController.verifyAudit);
router.post("/:transactionId/tamper", auditController.simulateTampering);

module.exports = router;
