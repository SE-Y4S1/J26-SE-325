/**
 * The research pipeline (Python) and the service (JavaScript) must route identically, or the
 * policy evaluated on IEEE-CIS would not be the policy enforced. research/assurance/golden.py
 * writes inputs with the Python engine's outputs; this replays every one through the
 * JavaScript engine.
 *
 * Regenerate after changing either engine or a policy:
 *   cd research && uv run python -m assurance.golden
 */

const { assess } = require("../src/services/assuranceEngine");
const { computePolicyHashV2 } = require("../src/services/policyDefinition");
const golden = require("./fixtures/golden-vectors.json");

const policy = {
  id: golden.policy.id,
  version: golden.policy.version,
  engine: golden.policy.engine,
  status: golden.policy.status,
  policyHash: golden.policyHash,
  document: golden.policy
};

describe("golden vectors: Python research engine == JavaScript service engine", () => {
  test("the policy hash computed in Python equals the JavaScript one", () => {
    expect(computePolicyHashV2(golden.policy)).toBe(golden.policyHash);
  });

  test(`all ${golden.cases.length} cases route identically`, () => {
    expect(golden.cases.length).toBeGreaterThanOrEqual(300);
    const mismatches = [];
    for (const [i, { input, expected }] of golden.cases.entries()) {
      const got = assess(input, policy);
      const same =
        got.route === expected.route &&
        got.aiDecision === expected.aiDecision &&
        got.signalQuality === expected.signalQuality &&
        JSON.stringify(got.vetoes.map((v) => v.code)) === JSON.stringify(expected.vetoes) &&
        JSON.stringify(got.degradedSignals) === JSON.stringify(expected.degradedSignals) &&
        JSON.stringify(got.unreliableSignals) === JSON.stringify(expected.unreliableSignals) &&
        Math.abs(got.assuranceScore - expected.assuranceScore) <= 1e-9 &&
        ["confidence", "disagreement", "context"].every((k) => Math.abs(got.components[k] - expected.components[k]) <= 1e-9);
      if (!same) mismatches.push({ i, input, expected, got: { route: got.route, score: got.assuranceScore, components: got.components } });
    }
    expect(mismatches.slice(0, 3)).toEqual([]);
  });

  test("the vectors exercise every route and both signal qualities", () => {
    const routes = new Set(golden.cases.map((c) => c.expected.route));
    const qualities = new Set(golden.cases.map((c) => c.expected.signalQuality));
    expect([...routes].sort()).toEqual(["AUTO", "HUMAN_REVIEW", "VERIFY"]);
    expect([...qualities].sort()).toEqual(["DEGRADED", "FULL"]);
  });
});
