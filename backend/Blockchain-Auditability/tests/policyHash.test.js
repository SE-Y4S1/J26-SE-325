const fs = require("fs");
const os = require("os");
const path = require("path");
const { canonicalJson } = require("../src/services/canonicalJson");
const {
  POLICY_DEFINITION,
  computePolicyHash,
  computePolicyHashV1,
  computePolicyHashV2
} = require("../src/services/policyDefinition");
const { PolicyRegistry, PolicyValidationError, policyRegistry } = require("../src/services/policyRegistry");
const hashService = require("../src/services/hashService");

// The P001 hash registered on the live Hardhat chain and stored in every record so far.
const DEPLOYED_P001_HASH = "0x6011145b5a10bf85f8b82c8daf9aec9b0a1c0590bba7d2b8d5663e67055c6096";

/** Every path to a leaf value in a JSON document, e.g. ["context","weights","amount"]. */
function leafPaths(value, prefix = []) {
  if (value === null || typeof value !== "object") return [prefix];
  return Object.entries(value).flatMap(([k, v]) => leafPaths(v, [...prefix, k]));
}

function mutateAt(doc, keys) {
  const copy = JSON.parse(JSON.stringify(doc));
  let node = copy;
  for (const k of keys.slice(0, -1)) node = node[k];
  const last = keys[keys.length - 1];
  const v = node[last];
  node[last] = typeof v === "number" ? v + 0.001 : typeof v === "boolean" ? !v : v === null ? "x" : `${v}~`;
  return copy;
}

describe("canonicalJson", () => {
  test("sorts keys at every depth and keeps array order", () => {
    expect(canonicalJson({ b: { y: 1, x: [3, 1] }, a: true })).toBe('{"a":true,"b":{"x":[3,1],"y":1}}');
  });

  test("is insensitive to key insertion order", () => {
    expect(canonicalJson({ a: 1, b: { c: 2, d: 3 } })).toBe(canonicalJson({ b: { d: 3, c: 2 }, a: 1 }));
  });

  test.each([NaN, Infinity, undefined, () => 1])("rejects %p instead of coercing it", (bad) => {
    expect(() => canonicalJson({ a: bad })).toThrow(TypeError);
  });
});

describe("policy hash schemes", () => {
  test("P001's v1 hash is unchanged and equals the value registered on-chain", () => {
    expect(computePolicyHash()).toBe(DEPLOYED_P001_HASH);
    expect(computePolicyHashV1(POLICY_DEFINITION)).toBe(DEPLOYED_P001_HASH);
  });

  test("documents the v1 defect: it does not commit to the nested thresholds", () => {
    const altered = JSON.parse(JSON.stringify(POLICY_DEFINITION));
    altered.thresholds.riskThreshold = 10;
    expect(computePolicyHashV1(altered)).toBe(DEPLOYED_P001_HASH);
    expect(computePolicyHashV2(altered)).not.toBe(computePolicyHashV2(POLICY_DEFINITION));
  });

  test("v2 commits to every leaf of the assurance policy document", () => {
    const doc = policyRegistry.get("P002").document;
    const base = computePolicyHashV2(doc);
    const paths = leafPaths(doc);
    expect(paths.length).toBeGreaterThan(30);
    for (const keys of paths) {
      expect([keys.join("."), computePolicyHashV2(mutateAt(doc, keys))]).not.toEqual([keys.join("."), base]);
    }
  });

  test("records without a stored policyHash still default to P001's v1 hash", () => {
    const record = {
      transactionId: "TX-H", auditId: "0x" + "11".repeat(32), salt: "22".repeat(32), riskScore: 1,
      confidence: 1, amount: 1, transactionType: "T", aiDecision: "APPROVE", finalDecision: "APPROVE",
      modelVersion: "m", timestamp: "t", reviewRequired: false, reviewerDecision: null
    };
    expect(hashService.generateCanonicalHash(record).canonicalPayload.policyHash).toBe(DEPLOYED_P001_HASH);
  });
});

describe("legacy records still verify", () => {
  const { records } = require("./fixtures/legacy-records-v1.json");

  test.each(records.map((r) => [r.transactionId, r]))("%s recomputes to its stored commitment unless tampered", (_, record) => {
    const { hash } = hashService.generateCanonicalHash(record);
    expect(hashService.computeAuditId(record.transactionId, record.salt)).toBe(record.auditId);
    if (record.isTampered) {
      expect(hash).not.toBe(record.recordHash);
    } else {
      expect(hash).toBe(record.recordHash);
    }
    expect(record.policyHash).toBe(DEPLOYED_P001_HASH);
  });
});

describe("policy registry", () => {
  test("loads P001 as the baseline and P002 as the active assurance policy", () => {
    expect(policyRegistry.getLegacy().policyHash).toBe(DEPLOYED_P001_HASH);
    const active = policyRegistry.getActive();
    expect(active.id).toBe("P002");
    expect(active.engine).toBe("assurance-engine/1.0");
    expect(active.policyHash).toBe(computePolicyHashV2(active.document));
    expect(policyRegistry.getByHash(active.policyHash.toUpperCase().replace("0X", "0x"))).toBe(active);
  });

  test("policy documents are frozen so nothing can alter a policy after it is hashed", () => {
    const doc = policyRegistry.getActive().document;
    expect(() => { "use strict"; doc.weights.confidence = 1; }).toThrow();
  });

  test("ACTIVE_ASSURANCE_POLICY selects the active policy and rejects unknown ids", () => {
    process.env.ACTIVE_ASSURANCE_POLICY = "P001";
    expect(policyRegistry.getActive().id).toBe("P001");
    process.env.ACTIVE_ASSURANCE_POLICY = "P999";
    expect(() => policyRegistry.getActive()).toThrow(/not a registered policy/);
    delete process.env.ACTIVE_ASSURANCE_POLICY;
  });

  function registryWith(mutate) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "c3-policies-"));
    const p002 = JSON.parse(fs.readFileSync(path.join(__dirname, "../src/policies/P002-assurance-v1.json"), "utf8"));
    mutate(p002.document);
    fs.writeFileSync(path.join(dir, "P002.json"), JSON.stringify(p002));
    try {
      return new PolicyRegistry(dir);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  }

  test.each([
    ["a missing-disagreement prior below the veto (breaks the fallback guarantee)", (d) => { d.missing.disagreement = 0.2; }],
    ["weights that do not sum to 1", (d) => { d.weights.context = 0.5; }],
    ["verifyMin not below autoMin", (d) => { d.routing.verifyMin = 0.9; }],
    ["a degraded cap of AUTO", (d) => { d.routing.degradedCap = "AUTO"; }],
    ["an unknown engine", (d) => { d.engine = "magic/9"; }]
  ])("refuses to load a policy with %s", (_, mutate) => {
    expect(() => registryWith(mutate)).toThrow(PolicyValidationError);
  });
});
