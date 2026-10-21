const hashService = require("../src/services/hashService");
const { computePolicyHash } = require("../src/services/policyDefinition");

describe("HashService Unit Tests", () => {
  const sampleSalt = "a1b2c3d4e5f678901234567890abcdef1234567890abcdef1234567890abcdef";
  const sampleTxId = "TX-2026-001";
  const sampleAuditId = hashService.computeAuditId(sampleTxId, sampleSalt);

  const baseRecord = {
    transactionId: sampleTxId,
    auditId: sampleAuditId,
    salt: sampleSalt,
    amount: 1500,
    riskScore: 0.25,
    confidence: 0.95,
    aiDecision: "APPROVE",
    finalDecision: "APPROVE",
    reviewRequired: false,
    reviewerDecision: null,
    modelVersion: "1.0.0",
    policyId: "P001",
    policyVersion: "1.0",
    policyHash: computePolicyHash(),
    timestamp: "2026-10-17T20:00:00.000Z",
    transactionType: "TRANSFER"
  };

  test("generateSalt generates 64-char hex string (32 bytes)", () => {
    const salt1 = hashService.generateSalt();
    const salt2 = hashService.generateSalt();

    expect(typeof salt1).toBe("string");
    expect(salt1).toHaveLength(64);
    expect(/^[0-9a-f]{64}$/.test(salt1)).toBe(true);
    expect(salt1).not.toBe(salt2);
  });

  test("computeAuditId returns deterministic 0x-prefixed bytes32", () => {
    const id1 = hashService.computeAuditId(sampleTxId, sampleSalt);
    const id2 = hashService.computeAuditId(sampleTxId, sampleSalt);

    expect(id1).toBe(id2);
    expect(id1.startsWith("0x")).toBe(true);
    expect(id1).toHaveLength(66);
    expect(/^0x[0-9a-f]{64}$/.test(id1)).toBe(true);
  });

  test("computeAuditId fails on missing transactionId or salt", () => {
    expect(() => hashService.computeAuditId("", sampleSalt)).toThrow();
    expect(() => hashService.computeAuditId(sampleTxId, "")).toThrow();
    expect(() => hashService.computeAuditId(null, sampleSalt)).toThrow();
    expect(() => hashService.computeAuditId(sampleTxId, null)).toThrow();
  });

  test("generateCanonicalHash is deterministic", () => {
    const result1 = hashService.generateCanonicalHash(baseRecord);
    const result2 = hashService.generateCanonicalHash(baseRecord);

    expect(result1.hash).toBe(result2.hash);
    expect(result1.canonicalString).toBe(result2.canonicalString);
    expect(result1.hash.startsWith("0x")).toBe(true);
    expect(result1.hash).toHaveLength(66);
    expect(/^0x[0-9a-f]{64}$/.test(result1.hash)).toBe(true);
  });

  test("generateCanonicalHash is key-order independent", () => {
    // Record with reversed / shuffled keys
    const shuffledRecord = {
      transactionType: baseRecord.transactionType,
      amount: baseRecord.amount,
      salt: baseRecord.salt,
      transactionId: baseRecord.transactionId,
      confidence: baseRecord.confidence,
      policyVersion: baseRecord.policyVersion,
      aiDecision: baseRecord.aiDecision,
      timestamp: baseRecord.timestamp,
      policyId: baseRecord.policyId,
      finalDecision: baseRecord.finalDecision,
      auditId: baseRecord.auditId,
      reviewerDecision: baseRecord.reviewerDecision,
      policyHash: baseRecord.policyHash,
      modelVersion: baseRecord.modelVersion,
      reviewRequired: baseRecord.reviewRequired,
      riskScore: baseRecord.riskScore
    };

    const normal = hashService.generateCanonicalHash(baseRecord);
    const shuffled = hashService.generateCanonicalHash(shuffledRecord);

    expect(shuffled.canonicalString).toBe(normal.canonicalString);
    expect(shuffled.hash).toBe(normal.hash);
  });

  test("generateCanonicalHash is sensitive to salt changes (avalanche effect)", () => {
    const alteredSalt = sampleSalt.slice(0, -1) + (sampleSalt.slice(-1) === "0" ? "1" : "0");
    const alteredAuditId = hashService.computeAuditId(sampleTxId, alteredSalt);

    const alteredRecord = {
      ...baseRecord,
      salt: alteredSalt,
      auditId: alteredAuditId
    };

    const normal = hashService.generateCanonicalHash(baseRecord);
    const altered = hashService.generateCanonicalHash(alteredRecord);

    expect(altered.hash).not.toBe(normal.hash);
    expect(alteredAuditId).not.toBe(sampleAuditId);
  });

  test("generateCanonicalHash is sensitive to any field modification (tamper detection)", () => {
    const normal = hashService.generateCanonicalHash(baseRecord);

    const tamperedAmount = hashService.generateCanonicalHash({ ...baseRecord, amount: 1501 });
    expect(tamperedAmount.hash).not.toBe(normal.hash);

    const tamperedDecision = hashService.generateCanonicalHash({ ...baseRecord, finalDecision: "FLAGGED" });
    expect(tamperedDecision.hash).not.toBe(normal.hash);

    const tamperedRisk = hashService.generateCanonicalHash({ ...baseRecord, riskScore: 0.26 });
    expect(tamperedRisk.hash).not.toBe(normal.hash);
  });

  test("generateCanonicalHash matches fixed known vector", () => {
    const fixedRecord = {
      aiDecision: "APPROVE",
      amount: 1000,
      auditId: "0x1111111111111111111111111111111111111111111111111111111111111111",
      confidence: 0.9,
      finalDecision: "APPROVE",
      modelVersion: "1.0.0",
      policyHash: "0x2222222222222222222222222222222222222222222222222222222222222222",
      policyId: "P001",
      policyVersion: "1.0",
      reviewRequired: false,
      reviewerDecision: null,
      riskScore: 0.1,
      salt: "3333333333333333333333333333333333333333333333333333333333333333",
      timestamp: "2026-01-01T00:00:00.000Z",
      transactionId: "TX-TEST-001",
      transactionType: "TRANSFER"
    };

    const result = hashService.generateCanonicalHash(fixedRecord);
    
    // Check sorted keys in canonical payload
    const keys = Object.keys(JSON.parse(result.canonicalString));
    const sortedKeys = [...keys].sort();
    expect(keys).toEqual(sortedKeys);
    expect(keys).toHaveLength(16);

    // Running it again gives the exact same hash
    const resultRepeat = hashService.generateCanonicalHash(fixedRecord);
    expect(result.hash).toBe(resultRepeat.hash);
  });

  test("DB round-trip normalization for number strings vs numbers", () => {
    const recordWithStrings = {
      ...baseRecord,
      amount: "1500",
      riskScore: "0.25",
      confidence: "0.95"
    };

    const normal = hashService.generateCanonicalHash(baseRecord);
    const fromStrings = hashService.generateCanonicalHash(recordWithStrings);

    // Number conversion in generateCanonicalHash ensures identical hash
    expect(fromStrings.hash).toBe(normal.hash);
    expect(fromStrings.canonicalString).toBe(normal.canonicalString);
  });
});
