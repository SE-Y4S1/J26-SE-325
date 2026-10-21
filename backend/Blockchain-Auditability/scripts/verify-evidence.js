#!/usr/bin/env node
/**
 * Independent verifier for a Component 3 evidence bundle
 * (GET /api/audit/:transactionId/evidence).
 *
 * It recomputes every identifier and hash from the bundle itself and, given an RPC endpoint,
 * reads the AuditRegistry contract directly -- it does not call the audit service or import
 * any of its code, so a reviewer does not have to trust the service that produced the bundle.
 *
 *   node scripts/verify-evidence.js bundle.json [--rpc http://127.0.0.1:8545] [--contract 0x...]
 *
 * Exit code 0 when every check passes, 1 otherwise.
 */
const crypto = require("crypto");
const fs = require("fs");

const sha256 = (text) => "0x" + crypto.createHash("sha256").update(text).digest("hex");
const eq = (a, b) => typeof a === "string" && typeof b === "string" && a.toLowerCase() === b.toLowerCase();

/** Canonical JSON: keys sorted at every depth (record v2 and policy scheme v2). */
function canonicalJson(value) {
  if (value === null) return "null";
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (typeof value === "object") {
    return `{${Object.keys(value).sort().map((k) => `${JSON.stringify(k)}:${canonicalJson(value[k])}`).join(",")}}`;
  }
  if (typeof value === "number" && !Number.isFinite(value)) throw new TypeError("non-finite number");
  return JSON.stringify(value);
}

/** Legacy scheme v1: JSON.stringify with the sorted top-level keys as the replacer. */
const v1Json = (obj) => JSON.stringify(obj, Object.keys(obj).sort());

function recordHash(payload, recordVersion) {
  return sha256(recordVersion === 2 ? canonicalJson(payload) : v1Json(payload));
}

function policyHash(document, scheme) {
  return sha256(scheme === "v1" ? v1Json(document) : canonicalJson(document));
}

/** Offline checks: everything that follows from the bundle alone. */
function verifyBundleOffline(bundle) {
  const checks = [];
  const check = (name, ok, detail = "") => checks.push({ name, ok: Boolean(ok), detail });
  const { transactionId, salt, recordVersion } = bundle;
  const decision = bundle.commitments && bundle.commitments.decision;
  const review = bundle.commitments && bundle.commitments.review;

  check("bundle has a decision commitment", decision && decision.payload);
  if (!decision || !decision.payload) return checks;

  const auditId = sha256(`${transactionId}:${salt}`);
  check("auditId = sha256(transactionId:salt)", eq(auditId, decision.auditId), auditId);
  check("decision payload names the same auditId, transaction and salt",
    eq(decision.payload.auditId, decision.auditId) && decision.payload.transactionId === transactionId && decision.payload.salt === salt);

  const decisionHash = recordHash(decision.payload, recordVersion);
  check("decision recordHash recomputes from its payload", eq(decisionHash, decision.recordHash), decisionHash);

  if (bundle.policy && bundle.policy.document) {
    const ph = policyHash(bundle.policy.document, bundle.policy.hashScheme);
    check("policyHash recomputes from the policy document", eq(ph, bundle.policy.policyHash), ph);
    check("the decision commits to that policy", eq(decision.payload.policyHash, bundle.policy.policyHash));
  } else {
    check("policy document included", false, "the bundle does not include the policy document");
  }

  if (review) {
    const reviewId = sha256(`${transactionId}:${salt}:review`);
    check("reviewAuditId = sha256(transactionId:salt:review)", eq(reviewId, review.auditId), reviewId);
    const reviewHash = recordHash(review.payload, 2);
    check("review recordHash recomputes from its payload", eq(reviewHash, review.recordHash), reviewHash);
    check("review links to the decision (parentRecordHash)", eq(review.payload.parentRecordHash, decisionHash));
    check("review payload names the same reviewAuditId", eq(review.payload.reviewAuditId, review.auditId));
  }
  return checks;
}

const REGISTRY_ABI = [
  "function getAudit(bytes32) view returns (bool exists, bytes32 recordHash, bytes32 policyHash, uint256 timestamp, address submitter)",
  "function isPolicyActive(bytes32) view returns (bool)"
];

/** On-chain checks: read the commitments straight from the contract. */
async function verifyBundleOnChain(bundle, rpcUrl, contractAddress) {
  const checks = [];
  const check = (name, ok, detail = "") => checks.push({ name, ok: Boolean(ok), detail });
  const { ethers } = require("ethers");
  const provider = new ethers.JsonRpcProvider(rpcUrl);
  const registry = new ethers.Contract(contractAddress, REGISTRY_ABI, provider);
  const policy = bundle.commitments.decision.payload.policyHash;

  for (const [name, c] of [["decision", bundle.commitments.decision], ["review", bundle.commitments.review]]) {
    if (!c) continue;
    const onChain = await registry.getAudit(c.auditId);
    check(`${name} commitment exists on-chain`, onChain[0], c.auditId);
    if (onChain[0]) {
      check(`${name} on-chain recordHash matches`, eq(onChain[1], c.recordHash), onChain[1]);
      check(`${name} on-chain policyHash matches`, eq(onChain[2], policy), onChain[2]);
      check(`${name} anchored at`, true, new Date(Number(onChain[3]) * 1000).toISOString());
    }
  }
  check("policy is registered and active on-chain", await registry.isPolicyActive(policy), policy);
  return checks;
}

async function main(argv) {
  const args = argv.slice(2);
  const file = args.find((a) => !a.startsWith("--"));
  const opt = (name) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : null; };
  if (!file) {
    console.error("usage: node scripts/verify-evidence.js bundle.json [--rpc URL] [--contract 0x...]");
    return 2;
  }
  const bundle = JSON.parse(fs.readFileSync(file, "utf8"));
  let checks = verifyBundleOffline(bundle);

  const rpc = opt("--rpc");
  const contract = opt("--contract") || (bundle.ledger && bundle.ledger.contractAddress);
  if (bundle.ledger && bundle.ledger.isMock) {
    checks.push({ name: "on-chain checks", ok: true, detail: "SKIPPED: this record was anchored on the simulated ledger (BLOCKCHAIN_MODE=mock); there is no chain to read." });
  } else if (rpc && contract) {
    checks = checks.concat(await verifyBundleOnChain(bundle, rpc, contract));
  } else {
    checks.push({ name: "on-chain checks", ok: false, detail: "Pass --rpc (and --contract if the bundle has no address) to check the chain." });
  }

  for (const c of checks) console.log(`${c.ok ? "PASS" : "FAIL"}  ${c.name}${c.detail ? `  (${c.detail})` : ""}`);
  const passed = checks.every((c) => c.ok);
  console.log(passed ? "\nEvidence verified." : "\nEvidence did NOT verify.");
  return passed ? 0 : 1;
}

if (require.main === module) {
  main(process.argv).then((code) => { process.exitCode = code; }, (err) => { console.error(err.message); process.exitCode = 1; });
}

module.exports = { verifyBundleOffline, verifyBundleOnChain, canonicalJson, recordHash, policyHash };
