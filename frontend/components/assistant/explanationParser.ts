import { MOCK_TRANSACTION } from "./mockData";

export interface ParsedShapItem {
  feature: string;
  label: string;
  shapValue: number;
  impact: string;
  strength: "High" | "Medium" | "Low";
}

export interface ParsedFraudEvidence {
  hasEvidence: boolean;
  provenanceType: "live" | "fallback" | "baseline";
  provenanceLabel: string;
  transactionId: string;
  amount?: number;
  currency: string;
  status: string;
  riskLevel: string;
  fraudScore: number;
  prototypeProbability?: number;
  prototypePrediction?: string;
  reason: string;
  factors: string[];
  shapContributions: ParsedShapItem[];
  streamSignals?: {
    behavioralScore?: number;
    graphScore?: number;
    riskScore?: number;
    timestamp?: string;
  };
}

export interface ParsedAuditEvidence {
  hasEvidence: boolean;
  provenanceType: "live" | "fallback";
  provenanceLabel: string;
  transactionId: string;
  auditId?: string;
  decision: string;
  recordedAt?: string;
  blockNumber?: number | string;
  recordStatus: string;
  isMockLedger: boolean;
  isFallback: boolean;
  explanationText: string;
}

export interface ParsedPortfolioEvidence {
  hasEvidence: boolean;
  provenanceType: "live" | "fallback";
  provenanceLabel: string;
  portfolioId: string;
  totalValue: string;
  currency: string;
  holdingCount: number;
  isFallback: boolean;
  fallbackReason?: string;
  expectedAnnualReturn?: number;
  riskScore?: number;
  riskLevel?: string;
  diversificationScore?: number;
  liquidity?: string;
  allocation?: Record<string, number>;
  explanationText: string;
}

export interface ParsedSessionEvidence {
  mode: "fraud" | "audit" | "fraud_and_audit" | "portfolio" | "baseline";
  hasActiveEvidence: boolean;
  fraud?: ParsedFraudEvidence;
  audit?: ParsedAuditEvidence;
  portfolio?: ParsedPortfolioEvidence;
  baseline?: ParsedFraudEvidence;
}

export function parseFraudResult(text: string): ParsedFraudEvidence {
  // Safe provenance derivation from actual text content:
  // If the text contains explicit fallback indicators, it must NEVER be classified as live.
  const isFallback =
    /FALLBACK MOCK DATA|Local mock baseline|FALLBACK NOTICE|NOT a live lookup result/i.test(text);

  const isLive =
    !isFallback &&
    Boolean(
      /LIVE COMPONENT 2 AUDIT RECORD/i.test(text) ||
      /Live Component 2 Gateway Audit Log/i.test(text)
    );

  const provenanceType: "live" | "fallback" | "baseline" = isLive
    ? "live"
    : isFallback
    ? "fallback"
    : "baseline";

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
  const factors: string[] = [];
  if (factorsSection?.[1]) {
    const lines = factorsSection[1].split("\n");
    for (const l of lines) {
      const m = l.match(/^\s*-\s*(.+)/);
      if (m && m[1].trim()) {
        factors.push(m[1].trim());
      }
    }
  }

  // Extract SHAP contributions ONLY when actually present in this evidence
  const shapSection = text.match(/SHAP CONTRIBUTIONS:\s*([\s\S]*?)(?=(?:LIMITATION|$))/i);
  const shapContributions: ParsedShapItem[] = [];
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

        const strength: "High" | "Medium" | "Low" =
          Math.abs(val) >= 0.18 ? "High" : Math.abs(val) >= 0.10 ? "Medium" : "Low";
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

  let streamSignals: ParsedFraudEvidence["streamSignals"];
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

export function parseAuditResult(text: string): ParsedAuditEvidence {
  const isLive =
    /LIVE SERVICE VERIFICATION|Live Component 3|CONFIGURED BLOCKCHAIN|SIMULATED LEDGER/i.test(text) &&
    !/FALLBACK DATA/i.test(text);
  const isFallback = /FALLBACK DATA|Local mock baseline/i.test(text);
  const provenanceType: "live" | "fallback" = isLive ? "live" : "fallback";
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
  let blockNumber: number | string | undefined;
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

export function parsePortfolioResult(text: string): ParsedPortfolioEvidence {
  const isPersonalizationDisabled = /PERSONALIZED CONTEXT DISABLED|Personalized financial context is disabled/i.test(text);

  if (isPersonalizationDisabled) {
    return {
      hasEvidence: false,
      provenanceType: "fallback",
      provenanceLabel: "Personalized Context Disabled",
      portfolioId: "N/A",
      totalValue: "Disabled",
      currency: "USD",
      holdingCount: 0,
      isFallback: false,
      fallbackReason: "Personalized financial context is disabled by user preference.",
      explanationText: "Notice: Personalized financial context is disabled by user preference. Account-specific portfolio queries are blocked without fabricating anonymous holdings.",
    };
  }

  const isFallback = /FALLBACK|Local mock baseline|NO LIVE PORTFOLIO/i.test(text);
  const isLive = !isFallback && /LIVE PLATFORM RECORD/i.test(text);
  const provenanceType: "live" | "fallback" = isLive ? "live" : "fallback";
  const provenanceLabel = isLive
    ? "Live Service Evidence (Platform Portfolio)"
    : "Fallback Mock Data (No Live Portfolio)";

  const pIdMatch = text.match(/(?:Portfolio ID|User ID):\s*([A-Za-z0-9_-]+)/i);
  const portfolioId = pIdMatch ? pIdMatch[1] : "USER001";

  let fallbackReason: string | undefined;
  const reasonMatch =
    text.match(/Platform service note:\s*([^\n\r]+)/i) ||
    text.match(/live platform service at [^:]*:\s*([^\n\r\)]+)/i);
  if (reasonMatch) {
    fallbackReason = reasonMatch[1].trim();
  }

  let totalValue = "N/A";
  let currency = isLive ? "USD" : "LKR";
  let holdingCount = 0;
  let expectedAnnualReturn: number | undefined;
  let riskScore: number | undefined;
  let riskLevel: string | undefined;
  let diversificationScore: number | undefined;
  let liquidity: string | undefined;
  let allocation: Record<string, number> | undefined;

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
      totalValue = "N/A";
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

export function getBaselineFraud(): ParsedFraudEvidence {
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

export function parseSessionEvidence(evidenceList: Array<{ tool: string; result: string }> = []): ParsedSessionEvidence {
  const fraudItem = evidenceList.find((e) => e.tool === "get_fraud_analysis");
  const auditItem = evidenceList.find((e) => e.tool === "get_blockchain_audit");
  const portfolioItem = evidenceList.find((e) => e.tool === "get_portfolio_analysis");

  const hasFraud = Boolean(fraudItem?.result && !fraudItem.result.includes("No transaction was found"));
  const hasAudit = Boolean(auditItem?.result && !auditItem.result.includes("No audit record found"));
  const isPortfolioDisabled = Boolean(
    portfolioItem?.result &&
    /PERSONALIZED CONTEXT DISABLED|Personalized financial context is disabled/i.test(portfolioItem.result)
  );
  const hasPortfolio = Boolean(
    portfolioItem?.result &&
    !portfolioItem.result.includes("No portfolio found") &&
    !isPortfolioDisabled
  );

  if (hasFraud && hasAudit) {
    return {
      mode: "fraud_and_audit",
      hasActiveEvidence: true,
      fraud: parseFraudResult(fraudItem!.result),
      audit: parseAuditResult(auditItem!.result),
    };
  }

  if (hasAudit) {
    return {
      mode: "audit",
      hasActiveEvidence: true,
      audit: parseAuditResult(auditItem!.result),
    };
  }

  if (hasFraud) {
    return {
      mode: "fraud",
      hasActiveEvidence: true,
      fraud: parseFraudResult(fraudItem!.result),
    };
  }

  if (hasPortfolio) {
    return {
      mode: "portfolio",
      hasActiveEvidence: true,
      portfolio: parsePortfolioResult(portfolioItem!.result),
    };
  }

  return {
    mode: "baseline",
    hasActiveEvidence: false,
    baseline: getBaselineFraud(),
  };
}
