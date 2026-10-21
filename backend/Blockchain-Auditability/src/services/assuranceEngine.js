/**
 * AI Decision Assurance Engine.
 *
 * Decides how much an AI decision can be trusted to run unattended:
 *   AUTO          execute the AI decision and anchor it as final;
 *   VERIFY        hold it for step-up verification by the account holder;
 *   HUMAN_REVIEW  hold it for an authorised analyst who is not the subject.
 *
 * Three signals, each expressed as an uncertainty in [0,1]:
 *   confidence    how close the decision score sits to the decision boundary;
 *   disagreement  the spread between independent models' scores;
 *   context       amount, upstream rule hits, model validation status, transaction type.
 *
 *   assuranceScore = 1 - (w_c * u_confidence + w_d * u_disagreement + w_r * u_context)
 *
 * Conservative degradation: a missing signal is scored at its worst non-vetoed value, an
 * unvalidated model's confidence is floored, and any DEGRADED assessment is capped below
 * AUTO. Missing or unreliable data can therefore never make a decision more automatic.
 *
 * Pure: no I/O, no clock, no randomness. The same (input, policy) always yields the same
 * assessment, which is what lets research/ reproduce it exactly (golden vectors).
 */

const AUTOMATION_LEVEL = { HUMAN_REVIEW: 0, VERIFY: 1, AUTO: 2 };

const round6 = (x) => Math.round(x * 1e6) / 1e6;
const clamp01 = (x) => Math.min(1, Math.max(0, x));

function lessAutomated(a, b) {
  return AUTOMATION_LEVEL[a] <= AUTOMATION_LEVEL[b] ? a : b;
}

function assess(input, policy) {
  if (policy.engine === "threshold-router/1.0") return assessThreshold(input, policy);
  if (policy.engine === "assurance-engine/1.0") return assessAssurance(input, policy);
  throw new Error(`No engine for policy ${policy.id} (${policy.engine})`);
}

/** P001: the legacy router, expressed as an assessment so both paths share one shape. */
function assessThreshold(input, policy) {
  const { riskThreshold, confidenceThreshold } = policy.document.thresholds;
  let action = "APPROVE";
  let reason = `Risk score is below threshold (riskScore < ${riskThreshold}). Transaction approved automatically.`;
  if (input.riskScore >= riskThreshold) {
    if (input.confidence >= confidenceThreshold) {
      action = "REJECT";
      reason = `High risk (riskScore >= ${riskThreshold}) and high confidence (confidence >= ${confidenceThreshold}). Automated rejection triggered.`;
    } else {
      action = "HUMAN_REVIEW";
      reason = `High risk (riskScore >= ${riskThreshold}) but low confidence (confidence < ${confidenceThreshold}). Requires human authorization.`;
    }
  }
  const route = action === "HUMAN_REVIEW" ? "HUMAN_REVIEW" : "AUTO";
  return {
    engine: policy.engine,
    policyId: policy.id,
    policyVersion: policy.version,
    policyHash: policy.policyHash,
    policyStatus: policy.status,
    aiDecision: input.riskScore >= riskThreshold ? "REJECT" : "APPROVE",
    route,
    assessedRoute: route,
    legacyAction: action,
    assuranceScore: null,
    components: null,
    context: null,
    vetoes: [],
    signalQuality: "LEGACY",
    degradedSignals: [],
    unreliableSignals: [],
    reasons: [reason],
    escalation: null
  };
}

function modelValidation(signals, patterns) {
  const version = typeof signals.sourceModelVersion === "string" ? signals.sourceModelVersion.toLowerCase() : null;
  if (signals.benchmarkModel === false) return "unvalidated";
  if (version && patterns.some((p) => version.includes(p.toLowerCase()))) return "unvalidated";
  if (signals.benchmarkModel === true) return "validated";
  return "unknown";
}

function assessAssurance(input, policy) {
  const doc = policy.document;
  const signals = input.signals || {};
  const degraded = [];
  const unreliable = [];
  const reasons = [];
  const vetoes = [];

  // --- The AI decision being assured ---------------------------------------------------
  const t = doc.decision.rejectThreshold;
  const p = clamp01(input.riskScore / 100);
  const aiDecision = p >= t ? "REJECT" : "APPROVE";

  // --- Confidence: distance of the score from the decision boundary ----------------------
  // Derived from the score, not from a caller-asserted confidence field, so a caller cannot
  // buy automation by claiming certainty.
  const margin = p >= t ? (p - t) / (1 - t) : (t - p) / t;
  let uConfidence = clamp01(1 - margin);
  const validation = modelValidation(signals, doc.unreliable.modelVersionPatterns || []);
  if (validation !== "validated") {
    if (uConfidence < doc.unreliable.confidenceFloor) {
      reasons.push(`Model is ${validation === "unknown" ? "of unknown validation status" : "not benchmark-validated"}; confidence uncertainty raised to ${doc.unreliable.confidenceFloor}.`);
    }
    uConfidence = Math.max(uConfidence, doc.unreliable.confidenceFloor);
    unreliable.push("confidence");
  }

  // --- Disagreement: spread between independent model scores -----------------------------
  const scores = Object.values(signals.modelScores || {});
  let uDisagreement;
  if (scores.length >= 2) {
    uDisagreement = clamp01(Math.max(...scores) - Math.min(...scores));
    if (uDisagreement > doc.vetoes.maxDisagreement) {
      vetoes.push({ code: "MODEL_DISAGREEMENT", detail: `Models disagree by ${round6(uDisagreement)} (> ${doc.vetoes.maxDisagreement}).` });
    }
  } else {
    uDisagreement = doc.missing.disagreement;
    degraded.push("disagreement");
    reasons.push(`Fewer than two independent model scores; disagreement scored at its worst non-vetoed value (${doc.missing.disagreement}).`);
  }

  // --- Context ------------------------------------------------------------------------
  const ctx = doc.context;
  const { low, high } = ctx.amountScale;
  const amountRisk = clamp01(Math.log(Math.max(input.amount, low) / low) / Math.log(high / low));

  let ruleRisk;
  if (Array.isArray(signals.rules)) {
    ruleRisk = signals.rules.reduce((worst, rule) => {
      const severity = Object.prototype.hasOwnProperty.call(ctx.ruleSeverity, rule) ? ctx.ruleSeverity[rule] : ctx.unknownRuleSeverity;
      return Math.max(worst, severity);
    }, 0);
  } else {
    ruleRisk = 1;
    degraded.push("context.upstreamRules");
  }

  let validationRisk;
  if (validation === "validated") validationRisk = 0;
  else if (validation === "unvalidated") validationRisk = 1;
  else {
    validationRisk = 1;
    degraded.push("context.modelValidation");
  }

  const type = String(input.transactionType || "").toUpperCase();
  const typeRisk = Object.prototype.hasOwnProperty.call(ctx.transactionTypeRisk, type)
    ? ctx.transactionTypeRisk[type]
    : ctx.defaultTransactionTypeRisk;

  const cw = ctx.weights;
  const uContext = clamp01(cw.amount * amountRisk + cw.upstreamRules * ruleRisk + cw.modelValidation * validationRisk + cw.transactionType * typeRisk);

  if (input.amount >= doc.vetoes.reviewAmount) {
    vetoes.push({ code: "AMOUNT_REVIEW", detail: `Amount ${input.amount} is at or above the review amount ${doc.vetoes.reviewAmount}.` });
  }

  // --- Score and route -----------------------------------------------------------------
  const w = doc.weights;
  const assuranceScore = clamp01(1 - (w.confidence * uConfidence + w.disagreement * uDisagreement + w.context * uContext));
  const { autoMin, verifyMin, degradedCap } = doc.routing;

  let route = assuranceScore >= autoMin ? "AUTO" : assuranceScore >= verifyMin ? "VERIFY" : "HUMAN_REVIEW";
  reasons.unshift(`Assurance score ${round6(assuranceScore)} against thresholds VERIFY >= ${verifyMin}, AUTO >= ${autoMin}: ${route}.`);

  const signalQuality = degraded.length || unreliable.length ? "DEGRADED" : "FULL";
  if (signalQuality === "DEGRADED" && lessAutomated(route, degradedCap) !== route) {
    route = degradedCap;
    reasons.push(`Degraded or unreliable signals cap the route at ${degradedCap}.`);
  }
  if (vetoes.length) {
    route = "HUMAN_REVIEW";
    reasons.push(`Veto: ${vetoes.map((v) => v.code).join(", ")} requires human review.`);
  }

  return {
    engine: doc.engine,
    policyId: policy.id,
    policyVersion: policy.version,
    policyHash: policy.policyHash,
    policyStatus: policy.status,
    aiDecision,
    route,
    assessedRoute: route,
    legacyAction: route === "AUTO" ? aiDecision : "HUMAN_REVIEW",
    assuranceScore: round6(assuranceScore),
    components: {
      confidence: round6(uConfidence),
      disagreement: round6(uDisagreement),
      context: round6(uContext)
    },
    context: {
      amount: round6(amountRisk),
      upstreamRules: round6(ruleRisk),
      modelValidation: round6(validationRisk),
      transactionType: round6(typeRisk)
    },
    vetoes,
    signalQuality,
    degradedSignals: degraded,
    unreliableSignals: unreliable,
    reasons,
    escalation: null
  };
}

/**
 * Applies a decision another component has already enforced, mirroring the controller's
 * stricter-action-wins rule. It never relaxes the outcome: STEP_UP holds an automatic
 * APPROVE for verification (an automatic REJECT is already stricter), and BLOCK replaces the
 * outcome with a rejection enforced upstream -- automatic by definition, so the route reads
 * AUTO while the engine's own judgement is kept as assessedRoute.
 */
function applyUpstream(assessment, upstream) {
  if (!upstream) return assessment;
  if (upstream.decision === "STEP_UP" && assessment.route === "AUTO" && assessment.legacyAction === "APPROVE") {
    return {
      ...assessment,
      route: "VERIFY",
      legacyAction: "HUMAN_REVIEW",
      escalation: { source: upstream.source, decision: "STEP_UP", from: "AUTO", to: "VERIFY" }
    };
  }
  if (upstream.decision === "BLOCK") {
    return {
      ...assessment,
      route: "AUTO",
      legacyAction: "REJECT",
      escalation: { source: upstream.source, decision: "BLOCK", from: assessment.route, to: "AUTO", note: "Rejection enforced upstream." }
    };
  }
  return assessment;
}

module.exports = { assess, applyUpstream, AUTOMATION_LEVEL, ENGINE_VERSION: "assurance-engine/1.0" };
