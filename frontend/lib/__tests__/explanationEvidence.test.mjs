import test from "node:test";
import assert from "node:assert/strict";

const MOCK_TRANSACTION = {
  id: "TX1001",
  amount: 250000,
  currency: "LKR",
  status: "BLOCKED",
  riskLevel: "HIGH",
  fraudScore: 0.92,
  factors: ["High transfer amount", "Unusual location", "New device"],
  shapContributions: [
    {
      feature: "amount_thousands",
      label: "High transaction amount",
      shapValue: 0.1933,
      impact: "Increased fraud risk",
      strength: "High",
    },
    {
      feature: "unusual_location",
      label: "Unusual location",
      shapValue: 0.1535,
      impact: "Increased fraud risk",
      strength: "Medium",
    },
    {
      feature: "new_device",
      label: "New device",
      shapValue: 0.1308,
      impact: "Increased fraud risk",
      strength: "Medium",
    },
  ],
};

function parseFraudResult(text) {
  const isFallback =
    /FALLBACK MOCK DATA|Local mock baseline|FALLBACK NOTICE|NOT a live lookup result/i.test(text);

  const isLive =
    !isFallback &&
    Boolean(
      /LIVE COMPONENT 2 AUDIT RECORD/i.test(text) ||
      /Live Component 2 Gateway Audit Log/i.test(text)
    );

  const provenanceType = isLive ? "live" : isFallback ? "fallback" : "baseline";
  const provenanceLabel = isLive
    ? "Live Service Evidence (Component 2 Gateway Audit Log)"
    : "Fallback Mock Data (Component 4 Local Prototype)";

  const txIdMatch = text.match(/Transaction ID:\s*([A-Za-z0-9_-]+)/i);
  const txId = txIdMatch ? txIdMatch[1] : MOCK_TRANSACTION.id;

  const amountMatch = text.match(/Transaction Amount:\s*([A-Za-z]+)?\s*([0-9,]+(?:\.[0-9]+)?)/i);
  const amount = amountMatch
    ? parseFloat(amountMatch[2].replace(/,/g, ""))
    : isLive
    ? undefined
    : txId === MOCK_TRANSACTION.id
    ? MOCK_TRANSACTION.amount
    : undefined;
  const currency = amountMatch?.[1] || "LKR";

  const scoreMatch = text.match(/Recorded Fraud Score:\s*([0-9.]+)/i);
  const fraudScore = scoreMatch ? parseFloat(scoreMatch[1]) : MOCK_TRANSACTION.fraudScore;

  const protoProbMatch = text.match(/Prototype fraud probability:\s*([0-9.]+)/i);
  const prototypeProbability = protoProbMatch ? parseFloat(protoProbMatch[1]) : undefined;

  const protoPredMatch = text.match(/Prototype prediction:\s*([^\n\r]+)/i);
  const prototypePrediction = protoPredMatch ? protoPredMatch[1].trim() : undefined;

  const statusMatch = text.match(/Recorded Status:\s*([A-Za-z0-9_-]+)/i);
  const status = statusMatch ? statusMatch[1] : MOCK_TRANSACTION.status;

  const riskMatch = text.match(/Recorded Risk Level:\s*([A-Za-z0-9_-]+)/i);
  const riskLevel = riskMatch ? riskMatch[1] : MOCK_TRANSACTION.riskLevel;

  const factorsSection = text.match(
    /RECORDED CONTRIBUTING FACTORS:\s*([\s\S]*?)(?=(?:COMPONENT 2|LOCAL SHAP|TRANSACTION TELEMETRY|LIMITATION|$))/i
  );
  const factors = [];
  if (factorsSection?.[1]) {
    const lines = factorsSection[1].split("\n");
    for (const l of lines) {
      const m = l.match(/^\s*-\s*(.+)/);
      if (m && m[1].trim()) {
        factors.push(m[1].trim());
      }
    }
  }

  const shapSection = text.match(/SHAP CONTRIBUTIONS:\s*([\s\S]*?)(?=(?:LIMITATION|$))/i);
  const shapContributions = [];
  if (shapSection?.[1]) {
    const lines = shapSection[1].split("\n");
    for (const l of lines) {
      const m = l.match(/-\s*([A-Za-z0-9_-]+):\s*SHAP value\s*=\s*([0-9.-]+)(?:;\s*(.*))?/i);
      if (m) {
        const feature = m[1];
        const val = parseFloat(m[2]);
        const rawImpact = m[3]?.trim() || "increases fraud risk";
        let label = feature.replace(/_/g, " ");
        if (feature === "amount_thousands") label = "High transaction amount";
        if (feature === "unusual_location") label = "Unusual location";
        if (feature === "new_device") label = "New device";

        const strength = Math.abs(val) >= 0.18 ? "High" : Math.abs(val) >= 0.10 ? "Medium" : "Low";
        shapContributions.push({
          feature,
          label,
          shapValue: Math.abs(val),
          impact: rawImpact.toLowerCase().includes("increase") ? "Increased fraud risk" : "Decreased fraud risk",
          strength,
        });
      }
    }
  }

  let streamSignals;
  if (isLive) {
    const behMatch = text.match(/Behavioral Stream Anomaly Score[^:]*:\s*([0-9.]+)/i);
    const relMatch = text.match(/Relational Stream Risk Score[^:]*:\s*([0-9.]+)/i);
    const fusedMatch = text.match(/Fused Final Risk Score:\s*([0-9.]+)/i);
    const timeMatch = text.match(/Audit Timestamp:\s*([^\n\r]+)/i);

    streamSignals = {
      behavioralScore: behMatch ? parseFloat(behMatch[1]) : undefined,
      graphScore: relMatch ? parseFloat(relMatch[1]) : undefined,
      riskScore: fusedMatch ? parseFloat(fusedMatch[1]) : undefined,
      timestamp: timeMatch ? timeMatch[1].trim() : undefined,
    };
  }

  return {
    hasEvidence: true,
    provenanceType,
    provenanceLabel,
    transactionId: txId,
    amount,
    currency,
    status,
    riskLevel,
    fraudScore,
    prototypeProbability,
    prototypePrediction,
    reason: factors.join("; ") || "No specific factors recorded",
    factors: factors.length > 0 ? factors : txId === MOCK_TRANSACTION.id ? MOCK_TRANSACTION.factors : [],
    shapContributions,
    streamSignals,
  };
}

function parseAuditResult(text) {
  const isLive =
    /LIVE SERVICE VERIFICATION|Live Component 3|CONFIGURED BLOCKCHAIN|SIMULATED LEDGER/i.test(text) &&
    !/FALLBACK DATA/i.test(text);
  const isFallback = /FALLBACK DATA|Local mock baseline/i.test(text);
  const provenanceType = isLive ? "live" : "fallback";
  const provenanceLabel = isLive
    ? "Live Service Evidence (Component 3 Blockchain Bridge)"
    : "Fallback Mock Data (Audit Bridge)";

  const txIdMatch = text.match(/Transaction ID:\s*([A-Za-z0-9_-]+)/i);
  const transactionId = txIdMatch ? txIdMatch[1] : "TX1001";

  const auditIdMatch = text.match(/Audit ID:\s*([A-Za-z0-9_-]+)/i);
  const auditId = auditIdMatch ? auditIdMatch[1] : undefined;

  const decisionMatch = text.match(/Decision:\s*([A-Za-z0-9_-]+)/i);
  const decision = decisionMatch ? decisionMatch[1] : "RECORDED";

  const recordedAtMatch = text.match(/Recorded At:\s*([^\n\r]+)/i);
  const recordedAt = recordedAtMatch ? recordedAtMatch[1].trim() : undefined;

  const blockMatch = text.match(/Block Number:\s*([0-9]+|Simulated Ledger)/i);
  let blockNumber;
  if (blockMatch) {
    const num = parseInt(blockMatch[1], 10);
    blockNumber = isNaN(num) ? blockMatch[1] : num;
  }

  const statusMatch = text.match(/Record Status:\s*([A-Za-z0-9_-]+)/i);
  const recordStatus = statusMatch ? statusMatch[1] : isFallback ? "NOT_FOUND_ON_LIVE_BRIDGE" : "RECORDED";

  const isMockLedger = /Simulated Mock Ledger|Simulated Ledger|mock/i.test(text);

  let explanationText = "";
  if (recordStatus === "NOT_FOUND_ON_LIVE_BRIDGE" || isFallback) {
    explanationText = `The live audit bridge at http://localhost:8002 was queried for transaction ${transactionId}, but no audit record was found on the live bridge. The retrieved record was supplied from local fallback mock data (Audit ID: ${auditId || "AUDIT-2026-001"}, Block: #${blockNumber ?? 1847291}, Status: ${recordStatus}). Independent blockchain verification is not established.`;
  } else if (recordStatus === "INTEGRITY_FAILED") {
    explanationText = `Cryptographic integrity verification failed for transaction ${transactionId}. The calculated SHA-256 hash does not match the blockchain commitment in the ledger (Audit ID: ${auditId || "N/A"}, Block: #${blockNumber ?? "N/A"}).`;
  } else if (recordStatus === "VERIFIED") {
    explanationText = `Blockchain audit record for transaction ${transactionId} was successfully verified in Component 3's ${isMockLedger ? "simulated ledger adapter" : "blockchain ledger"} (Audit ID: ${auditId || "N/A"}, Block: #${blockNumber ?? 1}, Status: VERIFIED). Record hash matches the ledger commitment. Independent external network verification remains unproven.`;
  } else {
    explanationText = `Blockchain audit record for transaction ${transactionId} was retrieved from Component 3. Audit ID: ${auditId || "N/A"}, Block: #${blockNumber ?? "N/A"}, Record Status: ${recordStatus}. Independent blockchain verification is not established.`;
  }

  return {
    hasEvidence: true,
    provenanceType,
    provenanceLabel,
    transactionId,
    auditId,
    decision,
    recordedAt,
    blockNumber,
    recordStatus,
    isMockLedger,
    isFallback,
    explanationText,
  };
}

function parsePortfolioResult(text) {
  const isFallback = /FALLBACK|Local mock baseline|NO LIVE PORTFOLIO/i.test(text);
  const isLive = !isFallback && /LIVE PLATFORM RECORD/i.test(text);
  const provenanceType = isLive ? "live" : "fallback";
  const provenanceLabel = isLive
    ? "Live Service Evidence (Platform Portfolio)"
    : "Fallback Mock Data (No Live Portfolio)";

  const pIdMatch = text.match(/(?:Portfolio ID|User ID):\s*([A-Za-z0-9_-]+)/i);
  const portfolioId = pIdMatch ? pIdMatch[1] : "USER001";

  let fallbackReason;
  const reasonMatch =
    text.match(/Platform service note:\s*([^\n\r]+)/i) ||
    text.match(/live platform service at [^:]*:\s*([^\n\r\)]+)/i);
  if (reasonMatch) {
    fallbackReason = reasonMatch[1].trim();
  }

  let totalValue = "N/A";
  let currency = isLive ? "USD" : "LKR";
  let holdingCount = 0;
  let expectedAnnualReturn;
  let riskScore;
  let riskLevel;
  let diversificationScore;
  let liquidity;
  let allocation;

  if (isLive) {
    const valMatch = text.match(/Total Portfolio Value:\s*([A-Za-z]+)?\s*([0-9,]+(?:\.[0-9]+)?)/i);
    if (valMatch) {
      currency = valMatch[1] || "USD";
      totalValue = valMatch[2];
    }
    const countMatch = text.match(/Holding Count:\s*([0-9]+)/i);
    holdingCount = countMatch ? parseInt(countMatch[1], 10) : 0;
  } else {
    const valMatch = text.match(
      /(?:Total Portfolio Value|Illustrative Baseline Valuation):\s*([A-Za-z]+)?\s*([0-9,]+(?:\.[0-9]+)?)/i
    );
    if (valMatch) {
      currency = valMatch[1] || "LKR";
      totalValue = valMatch[2];
    } else {
      totalValue = "1,000,000";
    }

    const retMatch = text.match(/(?:Illustrative\s+)?Expected Annual Return:\s*([0-9.]+)%/i);
    expectedAnnualReturn = retMatch ? parseFloat(retMatch[1]) : 11.8;

    const riskMatch = text.match(/(?:Illustrative\s+)?Risk Score:\s*([0-9]+)/i);
    riskScore = riskMatch ? parseInt(riskMatch[1], 10) : 38;

    const levelMatch = text.match(/(?:Illustrative\s+)?Risk Level:\s*([A-Za-z]+)/i);
    riskLevel = levelMatch ? levelMatch[1] : "Moderate";

    const divMatch = text.match(/(?:Illustrative\s+)?Diversification Score:\s*([0-9]+)/i);
    diversificationScore = divMatch ? parseInt(divMatch[1], 10) : 74;

    const liqMatch = text.match(/(?:Illustrative\s+)?Liquidity:\s*([A-Za-z]+)/i);
    liquidity = liqMatch ? liqMatch[1] : "High";

    const allocSection = text.match(
      /(?:ILLUSTRATIVE|MOCK|CURRENT)?\s*(?:CURRENT\s+)?ALLOCATION[^:]*:\s*([\s\S]*?)(?=(?:OBSERVATIONS|ILLUSTRATIVE OBSERVATIONS|IMPORTANT DISCLAIMER|FALLBACK NOTICE|$))/i
    );
    if (allocSection?.[1]) {
      allocation = {};
      for (const line of allocSection[1].split("\n")) {
        const m = line.match(/[-*]?\s*([A-Za-z\s]+):\s*([0-9.]+)%/);
        if (m) {
          allocation[m[1].trim()] = parseFloat(m[2]);
        }
      }
    }
    holdingCount = 0;
  }

  const explanationText = isLive
    ? `Live portfolio record retrieved from the platform portfolio store. Total recorded market value: ${currency} ${totalValue} across ${holdingCount} verified holdings. Optimization metrics require Component 1 analytical modeling.`
    : `Notice: No live portfolio was retrieved for user ${portfolioId} from the platform service (${fallbackReason || "portfolio not found"}). The user has 0 verified live holdings on record. The displayed valuation (${currency} ${totalValue}) and synthetic metrics (${expectedAnnualReturn}% expected return, ${riskScore}/100 risk score, ${diversificationScore}/100 diversification, ${liquidity} liquidity) are illustrative mock baseline examples for demonstration only and do not reflect actual user holdings or a verified account balance.`;

  return {
    hasEvidence: true,
    provenanceType,
    provenanceLabel,
    portfolioId,
    totalValue,
    currency,
    holdingCount,
    isFallback,
    fallbackReason,
    expectedAnnualReturn,
    riskScore,
    riskLevel,
    diversificationScore,
    liquidity,
    allocation,
    explanationText,
  };
}

function getBaselineFraud() {
  return {
    hasEvidence: false,
    provenanceType: "baseline",
    provenanceLabel: "Mock Baseline — No Active Session Evidence",
    transactionId: MOCK_TRANSACTION.id,
    amount: MOCK_TRANSACTION.amount,
    currency: MOCK_TRANSACTION.currency,
    status: MOCK_TRANSACTION.status,
    riskLevel: MOCK_TRANSACTION.riskLevel,
    fraudScore: MOCK_TRANSACTION.fraudScore,
    prototypeProbability: 1.0,
    prototypePrediction: "1",
    reason: MOCK_TRANSACTION.factors.join(", "),
    factors: MOCK_TRANSACTION.factors,
    shapContributions: MOCK_TRANSACTION.shapContributions.map((c) => ({
      feature: c.feature,
      label: c.label,
      shapValue: c.shapValue,
      impact: c.impact,
      strength: c.strength,
    })),
  };
}

function getExplanationSummary(session) {
  const fraud = session.fraud;
  const audit = session.audit;
  const portfolio = session.portfolio;

  if (session.mode === "audit" && audit) {
    return audit.explanationText;
  } else if (session.mode === "portfolio" && portfolio) {
    return portfolio.explanationText;
  } else if (session.mode === "fraud_and_audit" && fraud && audit) {
    const fraudSummary =
      fraud.provenanceType === "live"
        ? `Transaction ${fraud.transactionId} evaluated in Component 2's live gateway audit trail with decision ${fraud.status} (Recorded Fraud Score: ${fraud.fraudScore.toFixed(4)}).`
        : `Mock transaction record ${fraud.transactionId} lists a recorded fraud score of ${fraud.fraudScore.toFixed(2)}, ${fraud.riskLevel} risk level, and ${fraud.status} status (${fraud.factors.join(", ")}). Separately, the local synthetic Random Forest prototype provides illustrative SHAP attributions (uncalibrated probability ${fraud.prototypeProbability?.toFixed(4) ?? "1.0000"}, distinct from recorded score ${fraud.fraudScore.toFixed(2)}). Device and location inputs were inferred from mock labels and do not explain Component 2's production model. Component 2 live gateway service was not reached.`;
    const auditSummary =
      audit.provenanceType === "live"
        ? `In Component 3's blockchain ledger, audit ID ${audit.auditId || "RECORDED"} is registered at block #${audit.blockNumber ?? "N/A"} with status ${audit.recordStatus}.`
        : `In the blockchain ledger fallback data, audit ID ${audit.auditId || "RECORDED"} is registered at block #${audit.blockNumber ?? "N/A"} with status ${audit.recordStatus}. Independent blockchain verification is not established.`;
    return `${fraudSummary} ${auditSummary}`;
  } else if (fraud) {
    if (fraud.provenanceType === "live") {
      const streamDetails = fraud.streamSignals
        ? `Dual-stream anomaly detection recorded a behavioral anomaly score of ${fraud.streamSignals.behavioralScore?.toFixed(4) ?? "N/A"} and relational risk score of ${fraud.streamSignals.graphScore?.toFixed(4) ?? "N/A"}.`
        : "";
      return `Recorded in Component 2's live gateway audit trail with decision ${fraud.status} (Recorded Fraud Score: ${fraud.fraudScore.toFixed(4)}). Contributing factor: "${fraud.reason}". ${streamDetails} Component 4's local synthetic prototype Random Forest was not used for this live decision.`;
    } else if (fraud.provenanceType === "fallback") {
      const factorList =
        fraud.factors.length > 0 ? fraud.factors.join(", ") : "high transfer amount, unusual location, and new device";
      const protoProb = fraud.prototypeProbability != null ? fraud.prototypeProbability.toFixed(4) : "1.0000";
      return `The available mock transaction record lists a recorded fraud score of ${fraud.fraudScore.toFixed(2)}, a ${fraud.riskLevel} risk level, and ${fraud.status} status based on stated risk factors: ${factorList}. Separately, the local synthetic Random Forest prototype outputs an uncalibrated fraud probability of ${protoProb} (distinct from the recorded score ${fraud.fraudScore.toFixed(2)}) and provides illustrative SHAP attributions. Device and location inputs were inferred from mock reason labels and do not explain Component 2's production fraud model. Component 2's live gateway audit service was not reached for this decision.`;
    }
  } else if (session.mode === "baseline") {
    return "No active session evidence available.";
  }
  return "";
}

function parseSessionEvidence(evidenceList = []) {
  const fraudItem = evidenceList.find((e) => e.tool === "get_fraud_analysis");
  const auditItem = evidenceList.find((e) => e.tool === "get_blockchain_audit");
  const portfolioItem = evidenceList.find((e) => e.tool === "get_portfolio_analysis");

  const hasFraud = Boolean(fraudItem?.result && !fraudItem.result.includes("No transaction was found"));
  const hasAudit = Boolean(auditItem?.result && !auditItem.result.includes("No audit record found"));
  const hasPortfolio = Boolean(portfolioItem?.result && !portfolioItem.result.includes("No portfolio found"));

  if (hasFraud && hasAudit) {
    return {
      mode: "fraud_and_audit",
      hasActiveEvidence: true,
      fraud: parseFraudResult(fraudItem.result),
      audit: parseAuditResult(auditItem.result),
    };
  }

  if (hasAudit) {
    return {
      mode: "audit",
      hasActiveEvidence: true,
      audit: parseAuditResult(auditItem.result),
    };
  }

  if (hasFraud) {
    return {
      mode: "fraud",
      hasActiveEvidence: true,
      fraud: parseFraudResult(fraudItem.result),
    };
  }

  if (hasPortfolio) {
    return {
      mode: "portfolio",
      hasActiveEvidence: true,
      portfolio: parsePortfolioResult(portfolioItem.result),
    };
  }

  return {
    mode: "baseline",
    hasActiveEvidence: false,
    baseline: getBaselineFraud(),
  };
}

test("1. Audit-only evidence: mode is 'audit', fraud score and SHAP bars are NOT present", () => {
  const auditEvidence = [
    {
      tool: "get_blockchain_audit",
      result: `
BLOCKCHAIN AUDIT RECORD — FALLBACK DATA

Data Source:
Local mock baseline (live audit service responded: record not found).

Transaction ID:
TX1001

Audit ID:
AUDIT-2026-001

Decision:
TRANSACTION_BLOCKED

Recorded At:
2026-08-27 09:42:18

Block Number:
1847291

Record Status:
NOT_FOUND_ON_LIVE_BRIDGE

FALLBACK NOTICE:
The live audit bridge at http://localhost:8002 was queried, but no audit record was found for TX1001.
The record above is local mock baseline data and is NOT a live lookup result. Independent blockchain verification is not established.
      `,
    },
  ];

  const session = parseSessionEvidence(auditEvidence);

  assert.equal(session.mode, "audit");
  assert.equal(session.hasActiveEvidence, true);
  assert.equal(session.fraud, undefined, "Fraud object must be undefined for audit-only response");

  assert.ok(session.audit, "Audit object must be defined");
  assert.equal(session.audit.transactionId, "TX1001");
  assert.equal(session.audit.auditId, "AUDIT-2026-001");
  assert.equal(session.audit.blockNumber, 1847291);
  assert.equal(session.audit.recordStatus, "NOT_FOUND_ON_LIVE_BRIDGE");
  assert.equal(session.audit.isFallback, true);
  assert.equal(session.audit.provenanceType, "fallback");
  assert.match(session.audit.explanationText, /fallback mock data/i);
  assert.match(session.audit.explanationText, /independent blockchain verification is not established/i);
});

test("2. Fraud-only evidence: mode is 'fraud', audit is undefined, SHAP present only when returned", () => {
  const fraudEvidenceWithShap = [
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
- unusual_location: SHAP value = 0.153450; increases fraud risk
- new_device: SHAP value = 0.130761; increases fraud risk
      `,
    },
  ];

  const session = parseSessionEvidence(fraudEvidenceWithShap);

  assert.equal(session.mode, "fraud");
  assert.equal(session.hasActiveEvidence, true);
  assert.equal(session.audit, undefined, "Audit object must be undefined for fraud-only response");
  assert.ok(session.fraud);
  assert.equal(session.fraud.transactionId, "TX1001");
  assert.equal(session.fraud.fraudScore, 0.92);
  assert.equal(session.fraud.status, "BLOCKED");
  assert.equal(session.fraud.shapContributions.length, 3);
  assert.equal(session.fraud.shapContributions[0].feature, "amount_thousands");

  // When SHAP is NOT in fraud evidence (e.g. Component 2 live record without SHAP), do not fabricate SHAP
  const fraudEvidenceWithoutShap = [
    {
      tool: "get_fraud_analysis",
      result: `
FRAUD ANALYSIS — LIVE COMPONENT 2 AUDIT RECORD
Transaction ID: TX2002
Recorded Fraud Score: 0.85
Recorded Status: BLOCK
Recorded Risk Level: HIGH
RECORDED CONTRIBUTING FACTORS:
- Velocity limit exceeded
      `,
    },
  ];

  const sessionNoShap = parseSessionEvidence(fraudEvidenceWithoutShap);
  assert.equal(sessionNoShap.mode, "fraud");
  assert.ok(sessionNoShap.fraud);
  assert.equal(sessionNoShap.fraud.transactionId, "TX2002");
  assert.equal(sessionNoShap.fraud.shapContributions.length, 0, "Must not fabricate SHAP values if not in evidence");
});

test("3. No-tool greeting: mode is 'baseline', active fraud and audit are undefined, baseline is labeled", () => {
  const session = parseSessionEvidence([]);

  assert.equal(session.mode, "baseline");
  assert.equal(session.hasActiveEvidence, false);
  assert.equal(session.fraud, undefined, "Active fraud evidence must be undefined in greeting/baseline state");
  assert.equal(session.audit, undefined, "Active audit evidence must be undefined in greeting/baseline state");
  assert.equal(session.portfolio, undefined, "Active portfolio evidence must be undefined in greeting/baseline state");

  assert.ok(session.baseline, "Baseline illustrative object is available");
  assert.equal(session.baseline.provenanceType, "baseline");
  assert.equal(session.baseline.provenanceLabel, "Mock Baseline — No Active Session Evidence");
  assert.equal(session.baseline.transactionId, "TX1001");
});

test("4. Stale evidence prevention across multi-turn interactions", () => {
  // Turn 1: Fraud query
  const turn1 = parseSessionEvidence([
    {
      tool: "get_fraud_analysis",
      result: "Transaction ID: TX1001\nRecorded Fraud Score: 0.92\nRecorded Status: BLOCKED",
    },
  ]);
  assert.equal(turn1.mode, "fraud");
  assert.ok(turn1.fraud);
  assert.equal(turn1.audit, undefined);

  // Turn 2: User asks "Check the blockchain audit record for transaction TX1001"
  const turn2 = parseSessionEvidence([
    {
      tool: "get_blockchain_audit",
      result: "Transaction ID: TX1001\nAudit ID: AUDIT-2026-001\nBlock Number: 1847291\nRecord Status: NOT_FOUND_ON_LIVE_BRIDGE",
    },
  ]);
  assert.equal(turn2.mode, "audit", "Turn 2 must switch to audit mode");
  assert.ok(turn2.audit);
  assert.equal(turn2.fraud, undefined, "Turn 2 must NOT retain fraud evidence from Turn 1");

  // Turn 3: User sends greeting ("Hello, what can you help me with?") -> 0 tools
  const turn3 = parseSessionEvidence([]);
  assert.equal(turn3.mode, "baseline", "Turn 3 must revert to baseline mode");
  assert.equal(turn3.hasActiveEvidence, false);
  assert.equal(turn3.audit, undefined, "Turn 3 must NOT retain audit evidence from Turn 2");
  assert.equal(turn3.fraud, undefined, "Turn 3 must NOT retain fraud evidence from Turn 1");
  assert.equal(turn3.baseline?.provenanceLabel, "Mock Baseline — No Active Session Evidence");
});

test("5. Greeting state regression: greeting state must not present active TX1001 fraud or unlabeled SHAP values", () => {
  // Greeting query after audit: latestEvidence is empty
  const session = parseSessionEvidence([]);

  // Mode and active flags
  assert.equal(session.mode, "baseline");
  assert.equal(session.hasActiveEvidence, false, "Must not flag active session evidence for greeting");

  // No active fraud or audit records
  assert.equal(session.fraud, undefined, "Active fraud details for TX1001 must NOT be exposed as session.fraud");
  assert.equal(session.audit, undefined, "Active audit details must NOT be exposed as session.audit");

  // Baseline illustrative demonstration must be explicitly labeled
  assert.ok(session.baseline, "Illustrative baseline exists for demonstration preview");
  assert.equal(session.baseline.provenanceType, "baseline");
  assert.equal(session.baseline.provenanceLabel, "Mock Baseline — No Active Session Evidence");

  // Active SHAP contributions must be empty
  const activeShapContributions = session.fraud?.shapContributions ?? [];
  assert.equal(activeShapContributions.length, 0, "No unlabeled or active SHAP contributions allowed in greeting state");
});

test("6. Fallback fraud evidence regression: cannot produce live-service badge or live-gateway explanation", () => {
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

  const session = parseSessionEvidence(fallbackEvidence);
  assert.equal(session.mode, "fraud");
  assert.equal(session.hasActiveEvidence, true);
  assert.ok(session.fraud);

  // Requirement 2: Classify as fallback/mock evidence and never label it live service evidence
  assert.equal(session.fraud.provenanceType, "fallback");
  assert.notEqual(session.fraud.provenanceType, "live");
  assert.equal(session.fraud.provenanceLabel, "Fallback Mock Data (Component 4 Local Prototype)");
  assert.notEqual(session.fraud.provenanceLabel, "Live Service Evidence (Component 2 Gateway Audit Log)");

  // Requirement 5: Keep recorded score 0.92 distinct from prototype probability 1.0000
  assert.equal(session.fraud.fraudScore, 0.92);
  assert.equal(session.fraud.prototypeProbability, 1.0);
  assert.notEqual(session.fraud.fraudScore, session.fraud.prototypeProbability);

  // Generate explanation summary as rendered on Explanation screen
  const summary = getExplanationSummary(session);

  // Requirement 4: Do not claim Component 2's live gateway made or returned this specific decision
  assert.ok(!summary.includes("Component 2's live gateway audit trail"), "Must NOT claim decision came from Component 2's live gateway");
  assert.ok(!summary.includes("recorded gateway score"), "Must NOT refer to recorded gateway score for fallback data");
  assert.ok(summary.includes("Component 2's live gateway audit service was not reached for this decision"));

  // Requirement 3: Explain recorded mock decision lists stated factors and score, while local prototype provides separate SHAP attributions
  assert.ok(summary.includes("recorded fraud score of 0.92"));
  assert.ok(summary.includes("HIGH risk level"));
  assert.ok(summary.includes("BLOCKED status"));
  assert.ok(summary.includes("uncalibrated fraud probability of 1.0000 (distinct from the recorded score 0.92)"));

  // Requirement 6: Disclosures that device/location features were inferred and SHAP does not explain production model
  assert.ok(summary.includes("Device and location inputs were inferred from mock reason labels and do not explain Component 2's production fraud model"));
});

test("7. Combined fallback fraud and audit evidence regression: keeps provenances and disclosures distinct", () => {
  const combinedEvidence = [
    {
      tool: "get_fraud_analysis",
      result: `
FRAUD ANALYSIS — FALLBACK MOCK DATA
Transaction ID: TX1001
Recorded Fraud Score: 0.92
Recorded Risk Level: HIGH
Recorded Status: BLOCKED
RECORDED CONTRIBUTING FACTORS:
- New device
- Unusual location
LOCAL SHAP PROTOTYPE — NOT THE TEAM MODEL
Prototype fraud probability: 1.0000
SHAP CONTRIBUTIONS:
- amount_thousands: SHAP value = 0.193289; increases fraud risk
      `,
    },
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

  const session = parseSessionEvidence(combinedEvidence);
  assert.equal(session.mode, "fraud_and_audit");
  assert.ok(session.fraud);
  assert.ok(session.audit);

  assert.equal(session.fraud.provenanceType, "fallback");
  assert.equal(session.audit.provenanceType, "fallback");

  const summary = getExplanationSummary(session);
  assert.ok(summary.includes("Component 2 live gateway service was not reached"));
  assert.ok(summary.includes("uncalibrated probability 1.0000, distinct from recorded score 0.92"));
  assert.ok(summary.includes("Independent blockchain verification is not established"));
});

test("8. Truly live Component 2 gateway evidence produces live gateway explanation", () => {
  const liveEvidence = [
    {
      tool: "get_fraud_analysis",
      result: `
FRAUD ANALYSIS — LIVE COMPONENT 2 AUDIT RECORD
Data Source:
Live Component 2 Gateway Audit Log (http://localhost:8001/audit).
Transaction ID: TX2002
Recorded Fraud Score: 0.8920
Recorded Risk Level: HIGH
Recorded Status: BLOCK
RECORDED CONTRIBUTING FACTORS:
- dual_stream_velocity
COMPONENT 2 DUAL-STREAM MODEL SIGNALS:
- Behavioral Stream Anomaly Score (Simulated LSTM Autoencoder): 0.7400
- Relational Stream Risk Score (Simulated GNN): 0.8100
- Fused Final Risk Score: 0.8920
      `,
    },
  ];

  const session = parseSessionEvidence(liveEvidence);
  assert.equal(session.mode, "fraud");
  assert.ok(session.fraud);
  assert.equal(session.fraud.provenanceType, "live");
  assert.equal(session.fraud.provenanceLabel, "Live Service Evidence (Component 2 Gateway Audit Log)");

  const summary = getExplanationSummary(session);
  assert.ok(summary.includes("Component 2's live gateway audit trail"));
  assert.ok(summary.includes("0.8920"));
});

test("9. Portfolio fallback: mode is 'portfolio', provenance is fallback, synthetic metrics disclosed as mock baseline", () => {
  const fallbackPortfolioEvidence = [
    {
      tool: "get_portfolio_analysis",
      result: `
PORTFOLIO ANALYSIS — FALLBACK DATA (NO LIVE RECORD RETRIEVED)

STATUS: NO LIVE PORTFOLIO RECORD RETRIEVED for user ID 'USER001'.
Platform service note: HTTP 404: Portfolio not found.

FALLBACK NOTICE:
No live portfolio was retrieved for user 'USER001' from the platform service (HTTP 404: Portfolio not found). This record is NOT the user's actual portfolio and does NOT reflect verified customer holdings. All values, returns, and allocations below are illustrative mock baseline estimates provided solely for system capability demonstration.

USER / PORTFOLIO IDENTIFIER:
USER001

LIVE HOLDING STATUS:
Live holdings: 0 verified assets on platform.
Live market value: Unverified (No live portfolio record).

ILLUSTRATIVE MOCK BASELINE ESTIMATES (NOT ACTUAL USER PORTFOLIO):
- Illustrative Baseline Valuation: LKR 1,000,000 (Mock Baseline Demonstration Only)
- Expected Annual Return: 11.8% (Synthetic Estimate)
- Risk Score: 38/100 (Moderate Risk Baseline)
- Diversification Score: 74/100 (Moderate Diversification Baseline)
- Liquidity: High (Synthetic Estimate)

MOCK ALLOCATION BREAKDOWN (DEMONSTRATION SAMPLE):
- Technology: 35.0%
- Banking: 25.0%
- Healthcare: 15.0%
- Government Bonds: 15.0%
- Cash: 10.0%
      `,
    },
  ];

  const session = parseSessionEvidence(fallbackPortfolioEvidence);
  assert.equal(session.mode, "portfolio");
  assert.ok(session.portfolio);
  assert.equal(session.portfolio.provenanceType, "fallback");
  assert.equal(session.portfolio.isFallback, true);
  assert.equal(session.portfolio.holdingCount, 0);
  assert.equal(session.portfolio.portfolioId, "USER001");
  assert.equal(session.portfolio.totalValue, "1,000,000");
  assert.equal(session.portfolio.expectedAnnualReturn, 11.8);
  assert.equal(session.portfolio.riskScore, 38);
  assert.equal(session.portfolio.diversificationScore, 74);
  assert.equal(session.portfolio.liquidity, "High");
  assert.equal(session.portfolio.allocation?.["Technology"], 35.0);

  const summary = getExplanationSummary(session);
  assert.ok(summary.startsWith("Notice: No live portfolio was retrieved for user USER001"));
  assert.ok(summary.includes("0 verified live holdings on record"));
  assert.ok(summary.includes("illustrative mock baseline examples for demonstration only"));
  assert.ok(summary.includes("do not reflect actual user holdings or a verified account balance"));
});

test("10. Live portfolio success: mode is 'portfolio', provenance is live, holding count and valuation preserved", () => {
  const livePortfolioEvidence = [
    {
      tool: "get_portfolio_analysis",
      result: `
PORTFOLIO ANALYSIS — LIVE PLATFORM RECORD

Data Source:
Live platform portfolio service at http://localhost:8100/portfolios/USER001.

Portfolio ID:
USER001

Total Portfolio Value:
USD 45,250.00

Holding Count:
5

Holdings:
- AAPL: 50 shares @ USD 180.00 (Value: USD 9,000.00)
- MSFT: 40 shares @ USD 400.00 (Value: USD 16,000.00)
      `,
    },
  ];

  const session = parseSessionEvidence(livePortfolioEvidence);
  assert.equal(session.mode, "portfolio");
  assert.ok(session.portfolio);
  assert.equal(session.portfolio.provenanceType, "live");
  assert.equal(session.portfolio.isFallback, false);
  assert.equal(session.portfolio.portfolioId, "USER001");
  assert.equal(session.portfolio.totalValue, "45,250.00");
  assert.equal(session.portfolio.currency, "USD");
  assert.equal(session.portfolio.holdingCount, 5);

  const summary = getExplanationSummary(session);
  assert.ok(summary.includes("Live portfolio record retrieved from the platform portfolio store"));
  assert.ok(summary.includes("USD 45,250.00 across 5 verified holdings"));
});



