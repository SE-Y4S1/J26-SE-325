/**
 * Policy Engine for AI Financial Risk Evaluation
 * Policy ID: P001, Version: 1.0
 * 
 * Rules:
 * - IF riskScore >= 80 AND confidence >= 70  => REJECT
 * - IF riskScore >= 80 AND confidence < 70   => HUMAN_REVIEW
 * - IF riskScore < 80                        => APPROVE
 */
class PolicyEngine {
  constructor() {
    this.policyId = "P001";
    this.policyVersion = "1.0";
  }

  evaluate(input) {
    const { riskScore, confidence } = input;
    
    let action = "APPROVE";
    let reason = "Risk score is below threshold (riskScore < 80). Transaction approved automatically.";

    if (riskScore >= 80) {
      if (confidence >= 70) {
        action = "REJECT";
        reason = "High risk (riskScore >= 80) and high confidence (confidence >= 70). Automated rejection triggered.";
      } else {
        action = "HUMAN_REVIEW";
        reason = "High risk (riskScore >= 80) but low confidence (confidence < 70). Requires human authorization.";
      }
    }

    return {
      action,
      reason,
      policyId: this.policyId,
      policyVersion: this.policyVersion,
      evaluatedAt: new Date().toISOString()
    };
  }
}

module.exports = new PolicyEngine();
