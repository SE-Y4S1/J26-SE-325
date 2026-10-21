const fs = require("fs");
const path = require("path");
const { computePolicyHashForScheme } = require("./policyDefinition");

const POLICY_DIR = path.join(__dirname, "../policies");
const ROUTES = ["AUTO", "VERIFY", "HUMAN_REVIEW"];

class PolicyValidationError extends Error {}

function check(condition, file, message) {
  if (!condition) throw new PolicyValidationError(`${file}: ${message}`);
}

const isUnit = (x) => typeof x === "number" && Number.isFinite(x) && x >= 0 && x <= 1;

function sumIsOne(weights) {
  const values = Object.values(weights);
  return values.every(isUnit) && Math.abs(values.reduce((a, b) => a + b, 0) - 1) < 1e-9;
}

/**
 * Structural checks for an assurance-engine policy. A policy that loads is a policy the
 * engine can apply without guessing, so the checks that encode a guarantee are enforced
 * here rather than trusted to whoever edits the file.
 */
function validateAssurancePolicy(doc, file) {
  for (const key of ["id", "version", "engine", "decision", "weights", "routing", "unreliable", "missing", "vetoes", "context"]) {
    check(doc[key] !== undefined, file, `missing '${key}'`);
  }
  check(doc.engine === "assurance-engine/1.0", file, `unsupported engine '${doc.engine}'`);
  check(isUnit(doc.decision.rejectThreshold) && doc.decision.rejectThreshold > 0 && doc.decision.rejectThreshold < 1,
    file, "decision.rejectThreshold must be in (0,1)");
  check(sumIsOne(doc.weights) && ["confidence", "disagreement", "context"].every((k) => k in doc.weights),
    file, "weights must be confidence/disagreement/context and sum to 1");
  check(isUnit(doc.routing.autoMin) && isUnit(doc.routing.verifyMin) && doc.routing.autoMin > doc.routing.verifyMin,
    file, "routing requires 0 <= verifyMin < autoMin <= 1");
  check(["VERIFY", "HUMAN_REVIEW"].includes(doc.routing.degradedCap), file, "routing.degradedCap must be VERIFY or HUMAN_REVIEW");
  check(isUnit(doc.unreliable.confidenceFloor), file, "unreliable.confidenceFloor must be in [0,1]");
  check(isUnit(doc.vetoes.maxDisagreement), file, "vetoes.maxDisagreement must be in [0,1]");
  check(typeof doc.vetoes.reviewAmount === "number" && doc.vetoes.reviewAmount > 0, file, "vetoes.reviewAmount must be > 0");
  // The conservative-fallback guarantee: a missing disagreement signal is scored as the worst
  // value it could have taken without triggering the veto, so dropping it never makes a
  // decision look more certain than any non-vetoed observation would.
  check(isUnit(doc.missing.disagreement) && doc.missing.disagreement >= doc.vetoes.maxDisagreement,
    file, "missing.disagreement must be >= vetoes.maxDisagreement");
  const ctx = doc.context;
  check(sumIsOne(ctx.weights) && ["amount", "upstreamRules", "modelValidation", "transactionType"].every((k) => k in ctx.weights),
    file, "context.weights must cover amount/upstreamRules/modelValidation/transactionType and sum to 1");
  check(ctx.amountScale && ctx.amountScale.low > 0 && ctx.amountScale.high > ctx.amountScale.low,
    file, "context.amountScale requires 0 < low < high");
  check(Object.values(ctx.ruleSeverity || {}).every(isUnit) && isUnit(ctx.unknownRuleSeverity), file, "rule severities must be in [0,1]");
  check(Object.values(ctx.transactionTypeRisk || {}).every(isUnit) && isUnit(ctx.defaultTransactionTypeRisk),
    file, "transaction type risks must be in [0,1]");
}

function validateThresholdPolicy(doc, file) {
  check(doc.thresholds && typeof doc.thresholds.riskThreshold === "number" && typeof doc.thresholds.confidenceThreshold === "number",
    file, "threshold-router requires thresholds.riskThreshold and thresholds.confidenceThreshold");
}

function loadPolicies(dir = POLICY_DIR) {
  const entries = new Map();
  for (const file of fs.readdirSync(dir).filter((f) => f.endsWith(".json")).sort()) {
    const raw = JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
    check(raw.meta && raw.document, file, "expected { meta, document }");
    const doc = raw.document;
    const engine = doc.engine || raw.meta.engine;
    check(typeof doc.id === "string" && typeof doc.version === "string", file, "document.id and document.version are required");
    if (engine === "assurance-engine/1.0") validateAssurancePolicy(doc, file);
    else if (engine === "threshold-router/1.0") validateThresholdPolicy(doc, file);
    else check(false, file, `unknown engine '${engine}'`);

    const key = `${doc.id}@${doc.version}`;
    check(!entries.has(key), file, `duplicate policy ${key}`);
    entries.set(key, Object.freeze({
      id: doc.id,
      version: doc.version,
      engine,
      hashScheme: raw.meta.hashScheme,
      baseline: Boolean(raw.meta.baseline),
      status: doc.status || "active",
      description: raw.meta.description || "",
      policyHash: computePolicyHashForScheme(raw.meta.hashScheme, doc),
      document: deepFreeze(doc),
      file
    }));
  }
  return entries;
}

function deepFreeze(value) {
  if (value && typeof value === "object") {
    Object.values(value).forEach(deepFreeze);
    Object.freeze(value);
  }
  return value;
}

class PolicyRegistry {
  constructor(dir = POLICY_DIR) {
    this.entries = loadPolicies(dir);
  }

  list() {
    return [...this.entries.values()];
  }

  /** Latest version of a policy id (versions compared numerically, e.g. 1.10 > 1.9). */
  get(id) {
    const matches = this.list().filter((p) => p.id === id);
    if (!matches.length) return null;
    return matches.sort((a, b) => b.version.localeCompare(a.version, undefined, { numeric: true }))[0];
  }

  getByHash(policyHash) {
    const wanted = String(policyHash || "").toLowerCase();
    return this.list().find((p) => p.policyHash.toLowerCase() === wanted) || null;
  }

  /** The policy that routes requests carrying assurance signals. */
  getActive() {
    const id = process.env.ACTIVE_ASSURANCE_POLICY || "P002";
    const policy = this.get(id);
    if (!policy) throw new Error(`ACTIVE_ASSURANCE_POLICY '${id}' is not a registered policy`);
    return policy;
  }

  /** The legacy router, for callers that send no assurance signals. */
  getLegacy() {
    return this.get("P001");
  }
}

module.exports = { PolicyRegistry, policyRegistry: new PolicyRegistry(), PolicyValidationError, ROUTES };
