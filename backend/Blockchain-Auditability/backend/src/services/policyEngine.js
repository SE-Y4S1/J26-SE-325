const { POLICY_DEFINITION, computePolicyHash } = require("./policyDefinition");

/**
 * Policy Engine for AI Financial Risk Evaluation
 * Evaluates inputs against the single source of truth defined in policyDefinition.js.
 */
class PolicyEngine {
  constructor() {
    this.policyId = POLICY_DEFINITION.id;
    this.policyVersion = POLICY_DEFINITION.version;
    this.thresholds = POLICY_DEFINITION.thresholds;
    this.policyHash = computePolicyHash(POLICY_DEFINITION);
  }

  evaluate(input) {
    const { riskScore, confidence } = input;
    const { riskThreshold, confidenceThreshold } = this.thresholds;

    let action = "APPROVE";
    let reason = `Risk score is below threshold (riskScore < ${riskThreshold}). Transaction approved automatically.`;

    if (riskScore >= riskThreshold) {
      if (confidence >= confidenceThreshold) {
        action = "REJECT";
        reason = `High risk (riskScore >= ${riskThreshold}) and high confidence (confidence >= ${confidenceThreshold}). Automated rejection triggered.`;
      } else {
        action = "HUMAN_REVIEW";
        reason = `High risk (riskScore >= ${riskThreshold}) but low confidence (confidence < ${confidenceThreshold}). Requires human authorization.`;
      }
    }

    return {
      action,
      reason,
      policyId: this.policyId,
      policyVersion: this.policyVersion,
      policyHash: this.policyHash,
      evaluatedAt: new Date().toISOString()
    };
  }
}

module.exports = new PolicyEngine();
