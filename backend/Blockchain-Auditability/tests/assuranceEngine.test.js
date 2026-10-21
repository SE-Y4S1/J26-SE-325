const { assess, applyUpstream, AUTOMATION_LEVEL } = require("../src/services/assuranceEngine");
const { policyRegistry } = require("../src/services/policyRegistry");

const P001 = policyRegistry.get("P001");
// Mechanics tests use P002 v1.0, whose hand-set numbers their inputs were chosen for. The
// guarantees (property tests) are checked for every P002 version, including tuned ones.
const P002_VERSIONS = policyRegistry.list().filter((p) => p.id === "P002");
const P002 = P002_VERSIONS.find((p) => p.version === "1.0");
const doc = P002.document;
const EVERY_P002 = P002_VERSIONS.map((p) => [`P002 v${p.version}`, p]);

const VALIDATED = { benchmarkModel: true, sourceModelVersion: "fraud-bundle-2026-10" };

function input(overrides = {}) {
  return {
    riskScore: 10,
    confidence: 90,
    amount: 500,
    transactionType: "PAYMENT",
    signals: { modelScores: { lstm: 0.1, gnn: 0.12 }, rules: [], ...VALIDATED },
    ...overrides
  };
}

// Small seeded PRNG (mulberry32) so property tests are reproducible.
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const RULES = Object.keys(doc.context.ruleSeverity).concat(["some_new_rule"]);
const TYPES = Object.keys(doc.context.transactionTypeRisk).concat(["UNLISTED"]);

function randomInput(r) {
  const n = 2 + Math.floor(r() * 3);
  const modelScores = {};
  for (let i = 0; i < n; i++) modelScores[`m${i}`] = r();
  const rules = RULES.filter(() => r() < 0.15);
  const validation = r();
  const signals = { modelScores, rules };
  if (validation < 0.4) Object.assign(signals, VALIDATED);
  else if (validation < 0.7) Object.assign(signals, { benchmarkModel: false, sourceModelVersion: "heuristic-untrained" });
  return {
    riskScore: Math.round(r() * 10000) / 100,
    confidence: Math.round(r() * 100),
    amount: Math.round(Math.exp(r() * Math.log(2e6)) * 100) / 100 + 1,
    transactionType: TYPES[Math.floor(r() * TYPES.length)],
    signals
  };
}

const level = (route) => AUTOMATION_LEVEL[route];

describe("routes with full signals", () => {
  test("a confident, agreeing, low-context decision is AUTO", () => {
    const a = assess(input(), P002);
    expect(a.signalQuality).toBe("FULL");
    expect(a.aiDecision).toBe("APPROVE");
    expect(a.route).toBe("AUTO");
    expect(a.assuranceScore).toBeGreaterThanOrEqual(doc.routing.autoMin);
  });

  test("a score near the decision boundary drops to VERIFY", () => {
    const a = assess(input({ riskScore: 42, signals: { modelScores: { lstm: 0.38, gnn: 0.44 }, rules: [], ...VALIDATED } }), P002);
    expect(a.route).toBe("VERIFY");
  });

  test("strong model disagreement is vetoed to HUMAN_REVIEW", () => {
    const a = assess(input({ signals: { modelScores: { lstm: 0.05, gnn: 0.9 }, rules: [], ...VALIDATED } }), P002);
    expect(a.vetoes.map((v) => v.code)).toContain("MODEL_DISAGREEMENT");
    expect(a.route).toBe("HUMAN_REVIEW");
  });

  test("an amount at the review threshold is vetoed to HUMAN_REVIEW", () => {
    const a = assess(input({ amount: doc.vetoes.reviewAmount }), P002);
    expect(a.vetoes.map((v) => v.code)).toContain("AMOUNT_REVIEW");
    expect(a.route).toBe("HUMAN_REVIEW");
  });

  test("severe upstream rules raise contextual uncertainty", () => {
    const quiet = assess(input(), P002);
    const loud = assess(input({ signals: { ...input().signals, rules: ["rate_limit_block"] } }), P002);
    expect(loud.context.upstreamRules).toBe(1);
    expect(loud.components.context).toBeGreaterThan(quiet.components.context);
    expect(loud.assuranceScore).toBeLessThan(quiet.assuranceScore);
  });

  test("an unknown rule name counts, at the policy's unknown-rule severity", () => {
    const a = assess(input({ signals: { ...input().signals, rules: ["brand_new_rule"] } }), P002);
    expect(a.context.upstreamRules).toBe(doc.context.unknownRuleSeverity);
  });

  test("a caller-asserted confidence cannot buy automation", () => {
    const low = assess(input({ riskScore: 45, confidence: 0 }), P002);
    const high = assess(input({ riskScore: 45, confidence: 100 }), P002);
    expect(high).toEqual(low);
  });

  test("is deterministic", () => {
    expect(assess(input(), P002)).toEqual(assess(input(), P002));
  });
});

describe("conservative degradation", () => {
  test("one model score only: disagreement degraded and the route capped below AUTO", () => {
    const a = assess(input({ signals: { modelScores: { gnn: 0.1 }, rules: [], ...VALIDATED } }), P002);
    expect(a.degradedSignals).toContain("disagreement");
    expect(a.components.disagreement).toBe(doc.missing.disagreement);
    expect(a.signalQuality).toBe("DEGRADED");
    expect(a.route).not.toBe("AUTO");
  });

  test("an unvalidated model has its confidence floored and is never AUTO", () => {
    const a = assess(input({ signals: { modelScores: { lstm: 0.01, gnn: 0.02 }, rules: [], benchmarkModel: false } }), P002);
    expect(a.unreliableSignals).toContain("confidence");
    expect(a.components.confidence).toBeGreaterThanOrEqual(doc.unreliable.confidenceFloor);
    expect(a.route).not.toBe("AUTO");
  });

  test("a heuristic model version is unreliable even if benchmarkModel is claimed", () => {
    const a = assess(input({ signals: { modelScores: { a: 0.1, b: 0.1 }, rules: [], benchmarkModel: true, sourceModelVersion: "heuristic-untrained" } }), P002);
    expect(a.context.modelValidation).toBe(1);
    expect(a.unreliableSignals).toContain("confidence");
  });

  test("no signals at all under the assurance policy is DEGRADED and never AUTO", () => {
    const a = assess(input({ signals: undefined }), P002);
    expect(a.degradedSignals).toEqual(expect.arrayContaining(["disagreement", "context.upstreamRules", "context.modelValidation"]));
    expect(a.route).not.toBe("AUTO");
  });

  test.each(EVERY_P002)("property (%s): a DEGRADED assessment is never AUTO (2,000 random inputs)", (_, policy) => {
    const r = rng(20261020);
    for (let i = 0; i < 2000; i++) {
      const a = assess(randomInput(r), policy);
      if (a.signalQuality === "DEGRADED") expect(a.route).not.toBe("AUTO");
    }
  });

  test.each(EVERY_P002)("property (%s): removing a signal never makes the route more automated than any non-vetoed observation (2,000 random inputs)", (_, policy) => {
    const r = rng(42);
    const removals = {
      modelScores: (s) => { const c = { ...s }; delete c.modelScores; return c; },
      singleModel: (s) => ({ ...s, modelScores: { only: Object.values(s.modelScores)[0] } }),
      rules: (s) => { const c = { ...s }; delete c.rules; return c; },
      validation: (s) => { const c = { ...s }; delete c.benchmarkModel; delete c.sourceModelVersion; return c; }
    };
    let compared = 0;
    for (let i = 0; i < 2000; i++) {
      const present = randomInput(r);
      const full = assess(present, policy);
      for (const [name, remove] of Object.entries(removals)) {
        const reduced = assess({ ...present, signals: remove(present.signals) }, policy);
        // The one documented exception: a disagreement veto is evidence observed only when the
        // scores are present. Dropping the scores cannot reproduce it, but the reduced
        // assessment is still DEGRADED and capped below AUTO.
        const vetoedByRemovedSignal = (name === "modelScores" || name === "singleModel")
          && full.vetoes.some((v) => v.code === "MODEL_DISAGREEMENT");
        if (vetoedByRemovedSignal) {
          expect(reduced.route).not.toBe("AUTO");
          continue;
        }
        expect([name, level(reduced.route) <= level(full.route)]).toEqual([name, true]);
        compared++;
      }
    }
    // Coverage, not the guarantee: at least half of the 8,000 comparisons were not excused by a
    // disagreement veto (a tuned policy with a tighter veto excuses more of them).
    expect(compared).toBeGreaterThan(4000);
  });

  test.each(EVERY_P002)("property (%s): more disagreement or more context risk never increases automation", (_, policy) => {
    const r = rng(7);
    for (let i = 0; i < 1000; i++) {
      const base = randomInput(r);
      const a = assess(base, policy);
      const scores = Object.values(base.signals.modelScores);
      const wider = { ...base.signals.modelScores, extreme: scores.some((s) => s > 0.5) ? 0 : 1 };
      const b = assess({ ...base, signals: { ...base.signals, modelScores: wider } }, policy);
      expect(level(b.route)).toBeLessThanOrEqual(level(a.route));
      const c = assess({ ...base, signals: { ...base.signals, rules: [...base.signals.rules, "amount_hard_block"] } }, policy);
      expect(level(c.route)).toBeLessThanOrEqual(level(a.route));
    }
  });
});

describe("upstream decisions", () => {
  const upstream = (decision) => ({ source: "component2", decision });

  test("STEP_UP holds an automatic APPROVE for verification", () => {
    const a = applyUpstream(assess(input(), P002), upstream("STEP_UP"));
    expect(a.route).toBe("VERIFY");
    expect(a.legacyAction).toBe("HUMAN_REVIEW");
    expect(a.escalation).toMatchObject({ from: "AUTO", to: "VERIFY" });
  });

  test("STEP_UP never relaxes an automatic REJECT", () => {
    const rejected = assess(input({ riskScore: 99, signals: { modelScores: { a: 0.98, b: 0.99 }, rules: [], ...VALIDATED } }), P002);
    expect(rejected).toMatchObject({ route: "AUTO", legacyAction: "REJECT" });
    expect(applyUpstream(rejected, upstream("STEP_UP"))).toEqual(rejected);
  });

  test("BLOCK is an enforced rejection and keeps the engine's own route", () => {
    const held = assess(input({ amount: doc.vetoes.reviewAmount }), P002);
    const a = applyUpstream(held, upstream("BLOCK"));
    expect(a.legacyAction).toBe("REJECT");
    expect(a.assessedRoute).toBe("HUMAN_REVIEW");
    expect(a.escalation.decision).toBe("BLOCK");
  });

  test("ALLOW changes nothing", () => {
    const a = assess(input(), P002);
    expect(applyUpstream(a, upstream("ALLOW"))).toEqual(a);
  });
});

describe("P001 through the engine matches the legacy policy engine exactly", () => {
  const legacy = require("../src/services/policyEngine");

  test("for every risk and confidence on a 0..100 grid", () => {
    for (let risk = 0; risk <= 100; risk += 5) {
      for (let conf = 0; conf <= 100; conf += 5) {
        const a = assess({ riskScore: risk, confidence: conf, amount: 1, transactionType: "T" }, P001);
        const l = legacy.evaluate({ riskScore: risk, confidence: conf });
        expect(a.legacyAction).toBe(l.action);
        expect(a.reasons[0]).toBe(l.reason);
        expect(a.signalQuality).toBe("LEGACY");
      }
    }
  });
});
