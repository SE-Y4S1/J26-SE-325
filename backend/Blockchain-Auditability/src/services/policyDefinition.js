const crypto = require("crypto");
const { canonicalJson } = require("./canonicalJson");

/**
 * Policy P001 v1.0, the legacy two-threshold router. The document lives in
 * src/policies/P001-v1.json, the single source for the backend, the deploy script and tests.
 */
const POLICY_DEFINITION = require("../policies/P001-v1.json").document;

function sha256Hex(text) {
  return "0x" + crypto.createHash("sha256").update(text).digest("hex");
}

/**
 * Policy hash scheme v1 -- LEGACY, kept only because P001's on-chain registration and every
 * record anchored so far use it.
 *
 * Known defect: JSON.stringify(obj, keyArray) applies keyArray as an allow-list at every
 * depth, so nested objects serialise as {} and the hash does NOT commit to
 * `thresholds.riskThreshold` / `thresholds.confidenceThreshold`. (The rules strings, which
 * restate the numbers, are committed.) New policies use computePolicyHashV2.
 */
function computePolicyHashV1(policyDef = POLICY_DEFINITION) {
  return sha256Hex(JSON.stringify(policyDef, Object.keys(policyDef).sort()));
}

/**
 * Policy hash scheme v2: SHA-256 over canonical JSON (keys sorted at every depth), so every
 * field of the policy document -- weights, thresholds, vetoes -- is committed.
 */
function computePolicyHashV2(policyDef) {
  return sha256Hex(canonicalJson(policyDef));
}

function computePolicyHashForScheme(scheme, policyDef) {
  if (scheme === "v1") return computePolicyHashV1(policyDef);
  if (scheme === "v2") return computePolicyHashV2(policyDef);
  throw new Error(`Unknown policy hash scheme '${scheme}'`);
}

module.exports = {
  POLICY_DEFINITION,
  // Unchanged name and behaviour for existing callers (deploy script, hashService defaults).
  computePolicyHash: computePolicyHashV1,
  computePolicyHashV1,
  computePolicyHashV2,
  computePolicyHashForScheme
};
