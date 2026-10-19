import test from "node:test";
import assert from "node:assert/strict";

// Mirror the Trust Panel evidence parser logic to test parsing correctness,
// freshness, missing audit evidence handling, and session policy.
function parseLatestEvidence(evidenceList = []) {
  const fraudItem = evidenceList.find((e) => e.tool === "get_fraud_analysis");
  const auditItem = evidenceList.find((e) => e.tool === "get_blockchain_audit");
  const portfolioItem = evidenceList.find((e) => e.tool === "get_portfolio_analysis");

  const isFraudFallback = fraudItem?.result
    ? /FALLBACK MOCK DATA|Local mock baseline|FALLBACK NOTICE|NOT a live lookup result/i.test(fraudItem.result)
    : false;
  const isFraudLive =
    !isFraudFallback &&
    Boolean(
      fraudItem?.result &&
      (/LIVE COMPONENT 2 AUDIT RECORD/i.test(fraudItem.result) ||
       /Live Component 2 Gateway Audit Log/i.test(fraudItem.result))
    );

  const isAuditFallback = auditItem?.result
    ? /FALLBACK DATA|Local mock baseline|NOT_FOUND_ON_LIVE_BRIDGE/i.test(auditItem.result)
    : false;
  const isAuditLive =
    !isAuditFallback &&
    Boolean(
      auditItem?.result &&
      (/LIVE SERVICE VERIFICATION|Live Component 3|CONFIGURED BLOCKCHAIN|SIMULATED LEDGER/i.test(auditItem.result))
    );

  const isPortfolioFallback = portfolioItem?.result
    ? /FALLBACK|Local mock baseline|NO LIVE PORTFOLIO/i.test(portfolioItem.result)
    : false;
  const isPortfolioLive =
    !isPortfolioFallback &&
    Boolean(portfolioItem?.result && (/LIVE PLATFORM RECORD/i.test(portfolioItem.result)));

  let provenanceType = "baseline";
  if (fraudItem && auditItem) {
    provenanceType = isFraudLive && isAuditLive ? "live" : "fallback";
  } else if (fraudItem) {
    provenanceType = isFraudLive ? "live" : "fallback";
  } else if (auditItem) {
    provenanceType = isAuditLive ? "live" : "fallback";
  } else if (portfolioItem) {
    provenanceType = isPortfolioLive ? "live" : "fallback";
  } else {
    provenanceType = "baseline";
  }

  const provenanceLabel =
    provenanceType === "live"
      ? "Active Session: Live Service Evidence"
      : provenanceType === "fallback"
      ? portfolioItem && !fraudItem && !auditItem
        ? "Active Session: Tool Evidence — Synthetic Demo Data"
        : "Active Session: Fallback Mock Data"
      : "Mock Baseline (No Active Session Evidence)";

  const parsed = {
    hasFraudEvidence: Boolean(fraudItem?.result && !fraudItem.result.includes("No transaction was found")),
    hasAuditEvidence: Boolean(auditItem?.result && !auditItem.result.includes("No audit record found")),
    hasPortfolioEvidence: Boolean(portfolioItem?.result && portfolioItem.result.trim().length > 0),
    isPortfolioFallback,
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

    const protoProbMatch = text.match(/Prototype fraud probability:\s*([0-9.]+)/i);
    if (protoProbMatch) parsed.prototypeProbability = parseFloat(protoProbMatch[1]);

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

  if (portfolioItem?.result) {
    const text = portfolioItem.result;
    const pIdMatch =
      text.match(/(?:Portfolio ID|User ID|USER \/ PORTFOLIO IDENTIFIER):\s*([A-Za-z0-9_-]+)/i) ||
      text.match(/user ID ['"]?([A-Za-z0-9_-]+)['"]?/i);
    if (pIdMatch) parsed.portfolioId = pIdMatch[1];

    const valMatch = text.match(/(?:Total Portfolio Value|Illustrative Baseline Valuation):\s*([A-Za-z]+)?\s*([0-9,]+(?:\.[0-9]+)?)/i);
    if (valMatch) {
      parsed.portfolioCurrency = valMatch[1] || (isPortfolioLive ? "USD" : "LKR");
      parsed.portfolioTotalValue = valMatch[2];
    }

    const retMatch = text.match(/(?:Illustrative\s+)?Expected Annual Return:\s*([0-9.]+)%/i);
    if (retMatch) parsed.portfolioExpectedReturn = parseFloat(retMatch[1]);

    const riskMatch = text.match(/(?:Illustrative\s+)?Risk Score:\s*([0-9]+)/i);
    if (riskMatch) parsed.portfolioRiskScore = parseInt(riskMatch[1], 10);

    const levelMatch = text.match(/(?:Illustrative\s+)?Risk Level:\s*([A-Za-z]+)/i);
    if (levelMatch) parsed.portfolioRiskLevel = levelMatch[1];

    const divMatch = text.match(/(?:Illustrative\s+)?Diversification Score:\s*([0-9]+)/i);
    if (divMatch) parsed.portfolioDiversificationScore = parseInt(divMatch[1], 10);

    const liqMatch = text.match(/(?:Illustrative\s+)?Liquidity:\s*([A-Za-z]+)/i);
    if (liqMatch) parsed.portfolioLiquidity = liqMatch[1];

    const countMatch = text.match(/Holding Count:\s*([0-9]+)/i);
    parsed.portfolioHoldingCount = countMatch ? parseInt(countMatch[1], 10) : 0;
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

test("5. Fallback fraud evidence with live service notice must NEVER produce a live service badge", () => {
  const fallbackEvidence = [
    {
      tool: "get_fraud_analysis",
      result: `
FRAUD ANALYSIS — FALLBACK MOCK DATA

Data Source:
Local mock baseline (mock_financial_data.json).
Live service status (http://localhost:8001): Component 2 service at http://localhost:8001 timed out.

FALLBACK NOTICE:
This is FALLBACK MOCK DATA because the live Component 2 service did not provide a record for TX1001.
Reason: Component 2 service at http://localhost:8001 timed out.
This record is NOT a live lookup result from Component 2.

Transaction ID:
TX1001

Transaction Amount:
LKR 250,000.00

Recorded Fraud Score:
0.92

Recorded Risk Level:
HIGH

Recorded Status:
BLOCKED

RECORDED CONTRIBUTING FACTORS:
- New device
- Unusual location

LOCAL SHAP PROTOTYPE — NOT THE TEAM MODEL

Prototype prediction:
1

Prototype fraud probability:
1.0000

Prototype input features:
- New device: 1
- Unusual location: 1
- Amount (LKR): 250,000.00

SHAP CONTRIBUTIONS:
- amount_thousands: SHAP value = 0.193289; increases fraud risk
- unusual_location: SHAP value = 0.153450; increases fraud risk
- new_device: SHAP value = 0.130761; increases fraud risk

LIMITATION:
The SHAP values explain a locally trained synthetic Random Forest prototype,
not the production model or Component 2 engine.
Device and location inputs were inferred from mock reason labels.
These results are prototype evidence, not verified production output.
Probability calibration not established.
      `,
    },
  ];

  const parsed = parseLatestEvidence(fallbackEvidence);
  assert.equal(parsed.hasFraudEvidence, true);
  assert.equal(parsed.provenanceType, "fallback", "Fallback evidence must NEVER be classified as live");
  assert.notEqual(parsed.provenanceType, "live");
  assert.equal(parsed.provenanceLabel, "Active Session: Fallback Mock Data");
  assert.notEqual(parsed.provenanceLabel, "Active Session: Live Service Evidence");
  assert.equal(parsed.transactionId, "TX1001");
  assert.equal(parsed.fraudScore, 0.92);
  assert.equal(parsed.prototypeProbability, 1.0);
  assert.equal(parsed.status, "BLOCKED");
  assert.equal(parsed.riskLevel, "HIGH");
  assert.equal(parsed.hasShap, true);
});

test("6. Truly live Component 2 gateway evidence produces live badge", () => {
  const liveEvidence = [
    {
      tool: "get_fraud_analysis",
      result: `
FRAUD ANALYSIS — LIVE COMPONENT 2 AUDIT RECORD

Data Source:
Live Component 2 Gateway Audit Log (http://localhost:8001/audit).
This record is live governance evidence from Component 2's real-time fraud engine.

Transaction ID:
TX2002

Recorded Fraud Score:
0.9412

Recorded Risk Level:
HIGH

Recorded Status:
BLOCK

RECORDED CONTRIBUTING FACTORS:
- velocity_anomaly
      `,
    },
  ];

  const parsed = parseLatestEvidence(liveEvidence);
  assert.equal(parsed.hasFraudEvidence, true);
  assert.equal(parsed.provenanceType, "live");
  assert.equal(parsed.provenanceLabel, "Active Session: Live Service Evidence");
  assert.equal(parsed.transactionId, "TX2002");
  assert.equal(parsed.fraudScore, 0.9412);
  assert.equal(parsed.status, "BLOCK");
});

test("7. Mixed fallback fraud and live audit evidence does NOT label session as live service evidence", () => {
  const mixedEvidence = [
    {
      tool: "get_fraud_analysis",
      result: "FRAUD ANALYSIS — FALLBACK MOCK DATA\nTransaction ID: TX1001\nRecorded Fraud Score: 0.92\nRecorded Status: BLOCKED",
    },
    {
      tool: "get_blockchain_audit",
      result: "BLOCKCHAIN AUDIT RECORD — LIVE SERVICE VERIFICATION\nTransaction ID: TX1001\nAudit ID: AUDIT-001\nRecord Status: VERIFIED",
    },
  ];

  const parsed = parseLatestEvidence(mixedEvidence);
  assert.equal(parsed.provenanceType, "fallback", "Mixed evidence with fallback fraud must not be labeled live");
  assert.equal(parsed.provenanceLabel, "Active Session: Fallback Mock Data");
});

test("8. Portfolio fallback evidence: mode is fallback, provenance is tool evidence synthetic demo data, no fabrication", () => {
  const fallbackPortfolioEvidence = [
    {
      tool: "get_portfolio_analysis",
      result: `
PORTFOLIO ANALYSIS — FALLBACK DATA (NO LIVE RECORD RETRIEVED)
STATUS: NO LIVE PORTFOLIO RECORD RETRIEVED for user ID 'USER001'.
FALLBACK NOTICE: No live portfolio was retrieved for user 'USER001' from the platform service.
USER / PORTFOLIO IDENTIFIER: USER001
LIVE HOLDING STATUS: Live holdings: 0 verified assets on platform.
ILLUSTRATIVE MOCK BASELINE ESTIMATES (NOT ACTUAL USER PORTFOLIO):
- Illustrative Baseline Valuation: LKR 1,000,000 (Mock Baseline Demonstration Only)
- Expected Annual Return: 11.8% (Synthetic Estimate)
- Risk Score: 38/100 (Moderate Risk Baseline)
- Diversification Score: 74/100 (Moderate Diversification Baseline)
- Liquidity: High (Synthetic Estimate)
      `,
    },
  ];

  const parsed = parseLatestEvidence(fallbackPortfolioEvidence);
  assert.equal(parsed.hasPortfolioEvidence, true);
  assert.equal(parsed.hasFraudEvidence, false);
  assert.equal(parsed.hasAuditEvidence, false);
  assert.equal(parsed.isPortfolioFallback, true);
  assert.equal(parsed.provenanceType, "fallback");
  assert.equal(parsed.provenanceLabel, "Active Session: Tool Evidence — Synthetic Demo Data");
  assert.equal(parsed.portfolioId, "USER001");
  assert.equal(parsed.portfolioTotalValue, "1,000,000");
  assert.equal(parsed.portfolioCurrency, "LKR");
  assert.equal(parsed.portfolioRiskScore, 38);
  assert.equal(parsed.portfolioExpectedReturn, 11.8);
  assert.equal(parsed.portfolioDiversificationScore, 74);
  assert.equal(parsed.portfolioLiquidity, "High");
});

test("9. Live portfolio evidence: mode is live, live service evidence label preserved", () => {
  const livePortfolioEvidence = [
    {
      tool: "get_portfolio_analysis",
      result: `
PORTFOLIO ANALYSIS — LIVE PLATFORM RECORD
Data Source: Live platform portfolio service at http://localhost:8100/portfolios/USER001.
Portfolio ID: USER001
Total Portfolio Value: USD 45,250.00
Holding Count: 5
      `,
    },
  ];

  const parsed = parseLatestEvidence(livePortfolioEvidence);
  assert.equal(parsed.hasPortfolioEvidence, true);
  assert.equal(parsed.isPortfolioFallback, false);
  assert.equal(parsed.provenanceType, "live");
  assert.equal(parsed.provenanceLabel, "Active Session: Live Service Evidence");
  assert.equal(parsed.portfolioId, "USER001");
  assert.equal(parsed.portfolioTotalValue, "45,250.00");
  assert.equal(parsed.portfolioCurrency, "USD");
  assert.equal(parsed.portfolioHoldingCount, 5);
});

test("10. Sequential conversation regression: Portfolio query followed by TX1001 fraud query", () => {
  // Turn 1: Portfolio query returns fallback portfolio evidence
  const turn1PortfolioEvidence = [
    {
      tool: "get_portfolio_analysis",
      result: `
PORTFOLIO ANALYSIS — FALLBACK DATA (NO LIVE RECORD RETRIEVED)
STATUS: NO LIVE PORTFOLIO RECORD RETRIEVED for user ID 'USER001'.
Platform service note: HTTP 404: Portfolio not found.
FALLBACK NOTICE: No live portfolio was retrieved for user 'USER001' from the platform service.
USER / PORTFOLIO IDENTIFIER: USER001
LIVE HOLDING STATUS: Live holdings: 0 verified assets on platform.
ILLUSTRATIVE MOCK BASELINE ESTIMATES (NOT ACTUAL USER PORTFOLIO):
- Illustrative Baseline Valuation: LKR 1,000,000 (Mock Baseline Demonstration Only)
- Expected Annual Return: 11.8% (Synthetic Estimate)
- Risk Score: 38/100 (Moderate Risk Baseline)
- Diversification Score: 74/100 (Moderate Diversification Baseline)
- Liquidity: High (Synthetic Estimate)
      `,
    },
  ];

  const parsed1 = parseLatestEvidence(turn1PortfolioEvidence);
  assert.equal(parsed1.hasPortfolioEvidence, true);
  assert.equal(parsed1.hasFraudEvidence, false);
  assert.equal(parsed1.isPortfolioFallback, true);
  assert.equal(parsed1.provenanceType, "fallback");
  assert.equal(parsed1.provenanceLabel, "Active Session: Tool Evidence — Synthetic Demo Data");
  assert.equal(parsed1.portfolioTotalValue, "1,000,000");

  // Turn 2: User asks "Why was TX1001 blocked?", returning get_fraud_analysis evidence for TX1001
  const turn2FraudEvidence = [
    {
      tool: "get_fraud_analysis",
      result: `
FRAUD ANALYSIS — FALLBACK MOCK DATA
Data Source: Local mock baseline (mock_financial_data.json).
Live service status (http://localhost:8001): Component 2 service at http://localhost:8001 timed out.
FALLBACK NOTICE: This is FALLBACK MOCK DATA because the live Component 2 service did not provide a record for TX1001.
Transaction ID: TX1001
Transaction Amount: LKR 250,000.00
Recorded Fraud Score: 0.92
Recorded Risk Level: HIGH
Recorded Status: BLOCKED
RECORDED CONTRIBUTING FACTORS:
- New device
- Unusual location
- High transfer amount
LOCAL SHAP PROTOTYPE — NOT THE TEAM MODEL
Prototype fraud probability: 1.0000
SHAP CONTRIBUTIONS:
- amount_thousands: SHAP value = 0.193289; increases fraud risk
      `,
    },
  ];

  const parsed2 = parseLatestEvidence(turn2FraudEvidence);
  // Second response must display fraud evidence only, with correct mock provenance disclosure
  assert.equal(parsed2.hasFraudEvidence, true);
  assert.equal(parsed2.hasPortfolioEvidence, false, "Must not retain portfolio evidence in fraud turn");
  assert.equal(parsed2.hasAuditEvidence, false);
  assert.equal(parsed2.provenanceType, "fallback");
  assert.equal(parsed2.provenanceLabel, "Active Session: Fallback Mock Data");
  assert.equal(parsed2.transactionId, "TX1001");
  assert.equal(parsed2.amount, 250000);
  assert.equal(parsed2.currency, "LKR");
  assert.equal(parsed2.fraudScore, 0.92);
  assert.equal(parsed2.status, "BLOCKED");
  assert.equal(parsed2.riskLevel, "HIGH");

  // No portfolio balance, allocation, or portfolio fallback fields may survive into second turn
  assert.equal(parsed2.portfolioId, undefined, "Portfolio ID must be undefined in fraud turn");
  assert.equal(parsed2.portfolioTotalValue, undefined, "Portfolio valuation must be undefined in fraud turn");
  assert.equal(parsed2.portfolioExpectedReturn, undefined, "Portfolio return must be undefined in fraud turn");
  assert.equal(parsed2.portfolioRiskScore, undefined, "Portfolio risk score must be undefined in fraud turn");
  assert.equal(parsed2.portfolioDiversificationScore, undefined, "Diversification score must be undefined in fraud turn");
  assert.equal(parsed2.portfolioLiquidity, undefined, "Liquidity must be undefined in fraud turn");

  // Verify UI decision mode
  const isPortfolio2 = Boolean(parsed2.hasPortfolioEvidence && !parsed2.hasFraudEvidence && !parsed2.hasAuditEvidence);
  assert.equal(isPortfolio2, false, "UI must not be in portfolio mode during fraud turn");
});


