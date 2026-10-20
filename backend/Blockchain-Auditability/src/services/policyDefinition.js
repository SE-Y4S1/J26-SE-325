const crypto = require("crypto");

/**
 * Single source of truth for Policy P001 v1.0 specification.
 * Pure module without external dependencies, safe for import in deploy scripts, backend, and tests.
 */
const POLICY_DEFINITION = {
  id: "P001",
  version: "1.0",
  thresholds: {
    riskThreshold: 80,
    confidenceThreshold: 70
  },
  rules: [
    "IF riskScore >= 80 AND confidence >= 70 THEN action = REJECT",
    "IF riskScore >= 80 AND confidence < 70 THEN action = HUMAN_REVIEW",
    "IF riskScore < 80 THEN action = APPROVE"
  ]
};

/**
 * Computes deterministic 0x-prefixed bytes32 SHA-256 hash of the policy specification.
 */
function computePolicyHash(policyDef = POLICY_DEFINITION) {
  const canonicalString = JSON.stringify(policyDef, Object.keys(policyDef).sort());
  const hash = crypto.createHash("sha256").update(canonicalString).digest("hex");
  return "0x" + hash;
}

module.exports = {
  POLICY_DEFINITION,
  computePolicyHash
};
