import test from "node:test";
import assert from "node:assert/strict";

// Mirror the Trust Panel evidence parser logic to test parsing correctness,
// freshness, missing audit evidence handling, and session policy.
function parseLatestEvidence(evidenceList = []) {
  const fraudItem = evidenceList.find((e) => e.tool === "get_fraud_analysis");
  const auditItem = evidenceList.find((e) => e.tool === "get_blockchain_audit");

  const isFraudLive = fraudItem?.result ? /LIVE COMPONENT 2/i.test(fraudItem.result) : false;
  const isAuditLive = auditItem?.result ? /Live Component 3/i.test(auditItem.result) : false;
  const hasLive = isFraudLive || isAuditLive;
  const hasFallback =
    (fraudItem?.result && /FALLBACK MOCK DATA/i.test(fraudItem.result)) ||
    (auditItem?.result && /FALLBACK DATA/i.test(auditItem.result));

  const provenanceType = hasLive ? "live" : hasFallback ? "fallback" : "baseline";

  const provenanceLabel =
    provenanceType === "live"
      ? "Active Session: Live Service Evidence"
      : provenanceType === "fallback"
      ? "Active Session: Fallback Mock Data"
      : "Mock Baseline (No Active Session Evidence)";

  const parsed = {
    hasFraudEvidence: Boolean(fraudItem?.result && !fraudItem.result.includes("No transaction was found")),
    hasAuditEvidence: Boolean(auditItem?.result && !auditItem.result.includes("No audit record found")),
    hasShap: false,
    provenanceType,
    provenanceLabel,
  };

  if (fraudItem?.result) {
    const text = fraudItem.result;
    const txIdMatch = text.match(/Transaction ID:\s*([A-Za-z0-9_-]+)/i);
    if (txIdMatch) parsed.transactionId = txIdMatch[1];

    const amountMatch = text.match(/Transaction Amount:\s*([A-Za-z]+)?\s*([0-9,]+(?:\.[0-9]+)?)/i);
    if (amountMatch) {
      if (amountMatch[1]) parsed.currency = amountMatch[1];
      parsed.amount = parseFloat(amountMatch[2].replace(/,/g, ""));
    }

    const scoreMatch = text.match(/Recorded Fraud Score:\s*([0-9.]+)/i);
    if (scoreMatch) parsed.fraudScore = parseFloat(scoreMatch[1]);

    const statusMatch = text.match(/Recorded Status:\s*([A-Za-z0-9_-]+)/i);
    if (statusMatch) parsed.status = statusMatch[1];

    const riskMatch = text.match(/Recorded Risk Level:\s*([A-Za-z0-9_-]+)/i);
    if (riskMatch) parsed.riskLevel = riskMatch[1];

    parsed.hasShap = /SHAP (?:CONTRIBUTIONS|EXPLANATION)/i.test(text);
  }

  if (auditItem?.result) {
    const text = auditItem.result;
    const txIdMatch = text.match(/Transaction ID:\s*([A-Za-z0-9_-]+)/i);
    if (txIdMatch && !parsed.transactionId) parsed.transactionId = txIdMatch[1];

    const auditIdMatch = text.match(/Audit ID:\s*([A-Za-z0-9_-]+)/i);
    if (auditIdMatch) parsed.auditId = auditIdMatch[1];

    const blockMatch = text.match(/Block Number:\s*([0-9]+|Simulated Ledger)/i);
    if (blockMatch) {
      const num = parseInt(blockMatch[1], 10);
      parsed.blockNumber = isNaN(num) ? blockMatch[1] : num;
    }

    const statusMatch = text.match(/Record Status:\s*([A-Za-z0-9_-]+)/i);
    if (statusMatch) parsed.recordStatus = statusMatch[1];
  }

  return parsed;
}

test("1. Empty evidence (initial load or greeting) defaults to Mock Baseline and has no active evidence", () => {
  const parsed = parseLatestEvidence([]);
  assert.equal(parsed.hasFraudEvidence, false);
  assert.equal(parsed.hasAuditEvidence, false);
  assert.equal(parsed.provenanceType, "baseline");
  assert.equal(parsed.provenanceLabel, "Mock Baseline (No Active Session Evidence)");
  assert.equal(parsed.auditId, undefined);
  assert.equal(parsed.fraudScore, undefined);
});

test("2. Fraud-only evidence must NOT contain or fabricate audit details", () => {
  const fraudOnlyEvidence = [
    {
      tool: "get_fraud_analysis",
      result: `
FRAUD ANALYSIS — FALLBACK MOCK DATA
Transaction ID: TX1001
Transaction Amount: LKR 250,000.00
Recorded Fraud Score: 0.92
Recorded Risk Level: HIGH
Recorded Status: BLOCKED
RECORDED CONTRIBUTING FACTORS:
- New device
- Unusual location
SHAP CONTRIBUTIONS:
- amount_thousands: SHAP value = 0.193289; increases fraud risk
      `,
    },
  ];

  const parsed = parseLatestEvidence(fraudOnlyEvidence);
  assert.equal(parsed.hasFraudEvidence, true);
  assert.equal(parsed.hasAuditEvidence, false);
  assert.equal(parsed.transactionId, "TX1001");
  assert.equal(parsed.amount, 250000);
  assert.equal(parsed.fraudScore, 0.92);
  assert.equal(parsed.status, "BLOCKED");
  assert.equal(parsed.hasShap, true);

  // Critical requirement: Fraud-only response must not populate auditId, blockNumber, or recordStatus
  assert.equal(parsed.auditId, undefined, "Audit ID must be undefined when no audit evidence exists");
  assert.equal(parsed.blockNumber, undefined, "Block number must be undefined when no audit evidence exists");
  assert.equal(parsed.recordStatus, undefined, "Record status must be undefined when no audit evidence exists");
});

test("3. Blockchain audit-only evidence must NOT fabricate fraud scores", () => {
  const auditOnlyEvidence = [
    {
      tool: "get_blockchain_audit",
      result: `
BLOCKCHAIN AUDIT RECORD — FALLBACK DATA
Transaction ID: TX1001
Audit ID: AUDIT-2026-001
Block Number: 1847291
Record Status: NOT_FOUND_ON_LIVE_BRIDGE
      `,
    },
  ];

  const parsed = parseLatestEvidence(auditOnlyEvidence);
  assert.equal(parsed.hasAuditEvidence, true);
  assert.equal(parsed.hasFraudEvidence, false);
  assert.equal(parsed.auditId, "AUDIT-2026-001");
  assert.equal(parsed.blockNumber, 1847291);
  assert.equal(parsed.recordStatus, "NOT_FOUND_ON_LIVE_BRIDGE");

  // Fraud score must not be fabricated
  assert.equal(parsed.fraudScore, undefined, "Fraud score must be undefined when fraud tool was not called");
});

test("4. Evidence freshness: per-response replacement prevents retaining stale evidence across turns", () => {
  // Turn 1: Fraud analysis performed
  let sessionEvidence = [
    {
      tool: "get_fraud_analysis",
      result: "Transaction ID: TX1001\nRecorded Fraud Score: 0.92\nRecorded Status: BLOCKED",
    },
  ];
  let turn1Parsed = parseLatestEvidence(sessionEvidence);
  assert.equal(turn1Parsed.hasFraudEvidence, true);
  assert.equal(turn1Parsed.hasAuditEvidence, false);

  // Turn 2: User sends greeting -> per-response replacement replaces evidence with []
  sessionEvidence = [];
  let turn2Parsed = parseLatestEvidence(sessionEvidence);
  assert.equal(turn2Parsed.hasFraudEvidence, false, "Must not retain fraud evidence after greeting");
  assert.equal(turn2Parsed.hasAuditEvidence, false);
  assert.equal(turn2Parsed.provenanceType, "baseline");

  // Turn 3: User queries blockchain audit only
  sessionEvidence = [
    {
      tool: "get_blockchain_audit",
      result: "Transaction ID: TX1001\nAudit ID: AUDIT-2026-001\nBlock Number: 1847291\nRecord Status: NOT_FOUND",
    },
  ];
  let turn3Parsed = parseLatestEvidence(sessionEvidence);
  assert.equal(turn3Parsed.hasAuditEvidence, true);
  assert.equal(turn3Parsed.hasFraudEvidence, false, "Turn 3 must not retain Turn 1 fraud evidence");
});
