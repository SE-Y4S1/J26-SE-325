"use client";

import React, { useState } from "react";
import { MOCK_TRANSACTION } from "./mockData";
import { TrustActionState } from "./types";
import { ShieldIcon, CheckCircleIcon, LockIcon, AlertTriangleIcon, UndoIcon, SparklesIcon } from "./icons";

interface TrustPanelScreenProps {
  latestEvidence?: Array<{ tool: string; result: string }>;
}

interface ParsedEvidenceData {
  transactionId?: string;
  amount?: number;
  currency?: string;
  fraudScore?: number;
  prototypeProbability?: number;
  status?: string;
  riskLevel?: string;
  factors?: string[];
  auditId?: string;
  blockNumber?: number | string;
  recordStatus?: string;
  auditIsMockLedger?: boolean;
  hasFraudEvidence: boolean;
  hasAuditEvidence: boolean;
  hasPortfolioEvidence?: boolean;
  portfolioId?: string;
  isPortfolioFallback?: boolean;
  portfolioTotalValue?: string;
  portfolioCurrency?: string;
  portfolioRiskScore?: number;
  portfolioRiskLevel?: string;
  portfolioDiversificationScore?: number;
  portfolioLiquidity?: string;
  portfolioExpectedReturn?: number;
  portfolioHoldingCount?: number;
  hasShap: boolean;
  provenanceType: "live" | "fallback" | "baseline";
  provenanceLabel: string;
}

function parseLatestEvidence(evidenceList: Array<{ tool: string; result: string }> = []): ParsedEvidenceData {
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

  let provenanceType: "live" | "fallback" | "baseline" = "baseline";
  if (fraudItem && auditItem && portfolioItem) {
    provenanceType = isFraudLive && isAuditLive && isPortfolioLive ? "live" : "fallback";
  } else if (fraudItem && auditItem) {
    provenanceType = isFraudLive && isAuditLive ? "live" : "fallback";
  } else if (fraudItem && portfolioItem) {
    provenanceType = isFraudLive && isPortfolioLive ? "live" : "fallback";
  } else if (auditItem && portfolioItem) {
    provenanceType = isAuditLive && isPortfolioLive ? "live" : "fallback";
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

  const parsed: ParsedEvidenceData = {
    hasFraudEvidence: !!(fraudItem?.result && !fraudItem.result.includes("No transaction was found")),
    hasAuditEvidence: !!(auditItem?.result && !auditItem.result.includes("No audit record found")),
    hasPortfolioEvidence: !!(portfolioItem?.result && portfolioItem.result.trim().length > 0),
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

    const factorsMatch = text.match(
      /RECORDED CONTRIBUTING FACTORS:\s*([\s\S]*?)(?=(?:COMPONENT 2|LOCAL SHAP|TRANSACTION TELEMETRY|LIMITATION|$))/i
    );
    if (factorsMatch?.[1]) {
      const items: string[] = [];
      for (const line of factorsMatch[1].split("\n")) {
        const m = line.match(/^\s*-\s*(.+)/);
        if (m && m[1].trim()) items.push(m[1].trim());
      }
      if (items.length > 0) parsed.factors = items;
    }

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

    parsed.auditIsMockLedger = /Simulated Ledger|mock/i.test(text);
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

export function TrustPanelScreen({ latestEvidence = [] }: TrustPanelScreenProps) {
  const [actionState, setActionState] = useState<TrustActionState>("pending");
  const [showMoreExplanation, setShowMoreExplanation] = useState(false);
  const tx = MOCK_TRANSACTION;

  const parsed = parseLatestEvidence(latestEvidence);
  // Session policy: Active session evidence is valid only when tools returned results in this response.
  const hasLiveEvidence = latestEvidence.length > 0 && (parsed.hasFraudEvidence || parsed.hasAuditEvidence || Boolean(parsed.hasPortfolioEvidence));
  const isPortfolio = Boolean(parsed.hasPortfolioEvidence && !parsed.hasFraudEvidence && !parsed.hasAuditEvidence);

  const displayTxId = parsed.transactionId ?? (hasLiveEvidence ? "TX1001" : tx.id);
  const displayAmount = parsed.amount ?? (hasLiveEvidence ? undefined : tx.amount);
  const displayCurrency = parsed.currency ?? (hasLiveEvidence ? "LKR" : tx.currency);
  const displayStatus = isPortfolio
    ? (parsed.isPortfolioFallback ? "DEMO BASELINE" : "LIVE PORTFOLIO")
    : (parsed.status ?? (hasLiveEvidence ? (parsed.recordStatus ?? "RECORDED") : tx.status));

  return (
    <div className="flex flex-col gap-6">
      {/* Title & Subtitle Header */}
      <div className="rounded-xl border border-black/10 bg-white p-5 shadow-xs dark:border-white/15 dark:bg-neutral-900">
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-blue-600 dark:text-blue-400">
          <ShieldIcon className="size-4" />
          <span>Human-in-the-Loop Oversight</span>
        </div>
        <h2 className="mt-1 text-2xl font-bold text-neutral-900 dark:text-white">
          Trust Panel
        </h2>
        <p className="mt-1 text-sm text-neutral-500">
          Understand and control AI-assisted decisions
        </p>
      </div>

      {/* AI Decision Card */}
      <div className="rounded-xl border border-black/10 bg-white p-5 shadow-xs dark:border-white/15 dark:bg-neutral-900">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <h3 className="text-base font-semibold text-neutral-900 dark:text-white">
              AI Decision Summary
            </h3>
            {hasLiveEvidence ? (
              <span
                className={`rounded-full px-2.5 py-0.5 text-[11px] font-semibold ${parsed.provenanceType === "live"
                    ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300"
                    : isPortfolio && parsed.isPortfolioFallback
                      ? "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300"
                      : "bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300"
                  }`}
              >
                {parsed.provenanceLabel}
              </span>
            ) : (
              <span className="rounded-full bg-neutral-100 px-2.5 py-0.5 text-[11px] font-medium text-neutral-600 dark:bg-neutral-800 dark:text-neutral-400">
                Mock Baseline (No Active Session Evidence)
              </span>
            )}
          </div>
          <span
            className={`rounded-full px-3 py-1 text-xs font-bold ${isPortfolio
                ? parsed.isPortfolioFallback
                  ? "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300"
                  : "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300"
                : "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300"
              }`}
          >
            {displayStatus}
          </span>
        </div>

        <div className="rounded-lg border border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-800 dark:bg-neutral-800/60">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-xs font-semibold text-neutral-500">
                {isPortfolio ? "Target Account / Portfolio" : "Target Transaction"}
              </p>
              <p className="text-lg font-bold text-neutral-900 dark:text-white">
                {isPortfolio
                  ? `Portfolio ${parsed.portfolioId || "USER001"}${parsed.portfolioTotalValue ? ` — ${parsed.portfolioCurrency || "LKR"} ${parsed.portfolioTotalValue} (Illustrative)` : ""}`
                  : `Transaction ${displayTxId}${displayAmount != null ? ` — ${displayCurrency} ${displayAmount.toLocaleString()}` : ""}`}
              </p>
            </div>
            <div className="text-right">
              <span className="text-xs text-neutral-500">Status</span>
              <p
                className={`text-base font-bold ${isPortfolio
                    ? parsed.isPortfolioFallback
                      ? "text-amber-600 dark:text-amber-400"
                      : "text-emerald-600 dark:text-emerald-400"
                    : "text-red-600 dark:text-red-400"
                  }`}
              >
                {displayStatus}
              </p>
            </div>
          </div>

          <p className="mt-3 text-sm text-neutral-700 dark:text-neutral-300">
            <span className="font-semibold text-neutral-900 dark:text-white">Explanation:</span>{" "}
            {hasLiveEvidence ? (
              isPortfolio ? (
                parsed.isPortfolioFallback ? (
                  <>Notice: No live portfolio record was retrieved for user <span className="font-semibold text-neutral-900 dark:text-white">{parsed.portfolioId || "USER001"}</span> from the platform service (0 verified holdings). The displayed metrics and valuation are synthetic baseline demonstrations shown for system demonstration purposes and do not represent verified holdings, actual account balances, or personalized investment advice.</>
                ) : (
                  <>Live portfolio record retrieved from the platform portfolio store. Recorded market value: <span className="font-semibold text-neutral-900 dark:text-white">{parsed.portfolioCurrency || "USD"} {parsed.portfolioTotalValue}</span> across <span className="font-semibold text-neutral-900 dark:text-white">{parsed.portfolioHoldingCount}</span> verified holdings.</>
                )
              ) : parsed.hasFraudEvidence ? (
                parsed.provenanceType === "live" ? (
                  <>Component 2 gateway audit trail reports status <span className="font-semibold text-neutral-900 dark:text-white">{parsed.status || "BLOCKED"}</span> with risk score <span className="font-semibold text-neutral-900 dark:text-white">{parsed.fraudScore?.toFixed(4) ?? "N/A"}</span>. The recorded decision factor is {parsed.factors?.[0] || "recorded dual-stream signals"}. The production model&apos;s internal reasoning has not been independently established.</>
                ) : (
                  <>The available mock transaction record lists a recorded fraud score of <span className="font-semibold text-neutral-900 dark:text-white">{parsed.fraudScore != null ? parsed.fraudScore.toFixed(2) : "0.92"}</span>, <span className="font-semibold text-neutral-900 dark:text-white">{parsed.riskLevel || "HIGH"}</span> risk level, and <span className="font-semibold text-neutral-900 dark:text-white">{parsed.status || "BLOCKED"}</span> status based on stated risk factors ({parsed.factors?.join(", ") || "high transfer amount, unusual location, and new device"}). Separately, the local synthetic Random Forest prototype provides illustrative SHAP attributions (uncalibrated probability {parsed.prototypeProbability != null ? parsed.prototypeProbability.toFixed(4) : "1.0000"}, distinct from recorded score {parsed.fraudScore != null ? parsed.fraudScore.toFixed(2) : "0.92"}). Device and location features were inferred from mock labels and do not explain Component 2&apos;s production model. Component 2&apos;s live gateway was not reached for this decision.</>
                )
              ) : (
                <>Blockchain audit record for transaction {displayTxId} was retrieved from Component 3 bridge. Status: <span className="font-semibold text-neutral-900 dark:text-white">{parsed.recordStatus || "RECORDED"}</span>.</>
              )
            ) : (
              <>The static mock baseline record reports a HIGH risk level and BLOCKED status for illustration. Active session backend tools have not been invoked yet.</>
            )}
          </p>

          <div className="mt-4 flex flex-col gap-2 border-t border-neutral-200/80 pt-3 dark:border-neutral-700">
            <p className="text-xs font-bold uppercase tracking-wider text-neutral-500">
              Available Evidence Checklist
            </p>
            <div className="flex flex-col gap-1.5 text-xs text-neutral-800 dark:text-neutral-200">
              {isPortfolio ? (
                <>
                  <div className="flex items-center gap-2">
                    <CheckCircleIcon className="size-4 shrink-0 text-emerald-500" />
                    <span>
                      {parsed.isPortfolioFallback
                        ? "Platform service query: No live portfolio record found on platform service (0 verified holdings on record)"
                        : `Platform service query: Live portfolio retrieved (${parsed.portfolioCurrency || "USD"} ${parsed.portfolioTotalValue}, ${parsed.portfolioHoldingCount} verified holdings)`}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircleIcon className="size-4 shrink-0 text-emerald-500" />
                    <span>
                      {parsed.isPortfolioFallback
                        ? `Illustrative demo metrics: Sample return (${parsed.portfolioExpectedReturn ?? 11.8}%), risk score (${parsed.portfolioRiskScore ?? 38}/100), and allocation (calibration not established; not actual user assets)`
                        : "Live asset allocation and holdings distribution verified"}
                    </span>
                  </div>
                  {parsed.isPortfolioFallback ? (
                    <div className="flex items-center gap-2 text-neutral-500 dark:text-neutral-400">
                      <svg className="size-4 shrink-0 text-neutral-400 dark:text-neutral-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <circle cx="12" cy="12" r="8" />
                      </svg>
                      <span>Actual portfolio risk and verified account balance: Unassessed (0 verified live holdings on platform)</span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2">
                      <CheckCircleIcon className="size-4 shrink-0 text-emerald-500" />
                      <span>Verified portfolio balance on platform record</span>
                    </div>
                  )}
                </>
              ) : (
                <>
                  {/* Row 1: Fraud score evidence */}
                  {hasLiveEvidence ? (
                    parsed.hasFraudEvidence ? (
                      <div className="flex items-center gap-2">
                        <CheckCircleIcon className="size-4 shrink-0 text-emerald-500" />
                        <span>
                          {parsed.provenanceType === "live"
                            ? `Live gateway fraud score: ${parsed.fraudScore?.toFixed(4) ?? "N/A"} (verification and probability calibration not established)`
                            : `Recorded mock fraud score: ${parsed.fraudScore != null ? parsed.fraudScore.toFixed(2) : "0.92"} (distinct from prototype prob ${parsed.prototypeProbability != null ? parsed.prototypeProbability.toFixed(4) : "1.0000"}; calibration not established)`}
                        </span>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 text-neutral-500 dark:text-neutral-400">
                        <svg className="size-4 shrink-0 text-neutral-400 dark:text-neutral-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <circle cx="12" cy="12" r="8" />
                        </svg>
                        <span>Recorded fraud score: Not evaluated in this response</span>
                      </div>
                    )
                  ) : (
                    <div className="flex items-center gap-2 text-neutral-500 dark:text-neutral-400">
                      <svg className="size-4 shrink-0 text-neutral-400 dark:text-neutral-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <circle cx="12" cy="12" r="8" />
                      </svg>
                      <span>Mock baseline fraud score: {tx.fraudScore} (verification and probability calibration not established)</span>
                    </div>
                  )}

                  {/* Row 2: Blockchain audit evidence */}
                  {hasLiveEvidence ? (
                    parsed.hasAuditEvidence ? (
                      <div className="flex items-center gap-2">
                        <CheckCircleIcon className="size-4 shrink-0 text-emerald-500" />
                        <span>
                          Blockchain audit record returned (Audit ID:{" "}
                          <code className="rounded bg-neutral-200 px-1 py-0.5 font-mono text-[11px] dark:bg-neutral-700">
                            {parsed.auditId || "RECORDED"}
                          </code>
                          {parsed.blockNumber != null && `, Block: #${parsed.blockNumber}`}
                          {parsed.recordStatus && `, Status: ${parsed.recordStatus}`}
                          {parsed.auditIsMockLedger && " — simulated ledger"}
                          )
                        </span>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 text-neutral-500 dark:text-neutral-400">
                        <svg className="size-4 shrink-0 text-neutral-400 dark:text-neutral-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <circle cx="12" cy="12" r="8" />
                        </svg>
                        <span>
                          Blockchain audit record: Not requested or provided in this response (audit status unverified)
                        </span>
                      </div>
                    )
                  ) : (
                    <div className="flex items-center gap-2 text-neutral-500 dark:text-neutral-400">
                      <svg className="size-4 shrink-0 text-neutral-400 dark:text-neutral-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <circle cx="12" cy="12" r="8" />
                      </svg>
                      <span>
                        Mock baseline audit record (Audit ID:{" "}
                        <code className="rounded bg-neutral-200 px-1 py-0.5 font-mono text-[11px] dark:bg-neutral-700">
                          {tx.blockchainAudit.auditId}
                        </code>
                        , Block: #{tx.blockchainAudit.blockNumber}, Status: {tx.blockchainAudit.recordStatus} — unverified baseline)
                      </span>
                    </div>
                  )}

                  {/* Row 3: Model attribution evidence */}
                  {hasLiveEvidence ? (
                    parsed.hasShap ? (
                      <div className="flex items-center gap-2">
                        <CheckCircleIcon className="size-4 shrink-0 text-emerald-500" />
                        <span>
                          {parsed.provenanceType === "live"
                            ? "SHAP explanation: Not computed for live Component 2 gateway decision (telemetry not retained)"
                            : `SHAP explanation available (local synthetic Random Forest prototype, prob ${parsed.prototypeProbability != null ? parsed.prototypeProbability.toFixed(4) : "1.0000"}; does not explain production model)`}
                        </span>
                      </div>
                    ) : parsed.provenanceType === "live" ? (
                      <div className="flex items-center gap-2 text-neutral-500 dark:text-neutral-400">
                        <svg className="size-4 shrink-0 text-neutral-400 dark:text-neutral-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <circle cx="12" cy="12" r="8" />
                        </svg>
                        <span>SHAP explanation: Not computed for live Component 2 gateway decision (telemetry not retained)</span>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 text-neutral-500 dark:text-neutral-400">
                        <svg className="size-4 shrink-0 text-neutral-400 dark:text-neutral-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <circle cx="12" cy="12" r="8" />
                        </svg>
                        <span>SHAP explanation: Not evaluated in this response</span>
                      </div>
                    )
                  ) : (
                    <div className="flex items-center gap-2 text-neutral-500 dark:text-neutral-400">
                      <svg className="size-4 shrink-0 text-neutral-400 dark:text-neutral-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <circle cx="12" cy="12" r="8" />
                      </svg>
                      <span>Mock baseline SHAP explanation (3 primary features computed on local prototype)</span>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>

          {hasLiveEvidence && latestEvidence.length > 0 && (
            <details className="mt-4 rounded-lg border border-neutral-200 bg-white p-3 dark:border-neutral-700 dark:bg-neutral-900">
              <summary className="cursor-pointer text-xs font-semibold text-blue-700 dark:text-blue-300">
                View raw backend tool evidence from chat session ({latestEvidence.length})
              </summary>
              <div className="mt-3 space-y-3">
                {latestEvidence.map((item, index) => (
                  <div key={`${item.tool}-${index}`}>
                    <p className="mb-1 text-xs font-bold text-neutral-700 dark:text-neutral-200">
                      {item.tool}
                    </p>
                    <pre className="whitespace-pre-wrap break-words rounded bg-neutral-50 p-2 font-mono text-xs leading-relaxed text-neutral-700 dark:bg-neutral-800 dark:text-neutral-300">
                      {item.result}
                    </pre>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
      </div>

      {/* AI Transparency & Limitations */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="rounded-xl border border-blue-200/80 bg-blue-50/40 p-4 shadow-xs dark:border-blue-900/40 dark:bg-neutral-900">
          <h4 className="text-xs font-bold uppercase tracking-wider text-blue-900 dark:text-blue-300">
            How was this decision explained?
          </h4>
          <p className="mt-2 text-xs leading-relaxed text-neutral-700 dark:text-neutral-300">
            Explanation uses available tool evidence and model-attribution outputs. When fallback mock data is returned, recorded decisions list mock risk factors, while the local synthetic Random Forest provides separate illustrative SHAP attributions that do not explain Component 2&apos;s production model.
          </p>
        </div>

        <div className="rounded-xl border border-amber-200/80 bg-amber-50/40 p-4 shadow-xs dark:border-amber-900/40 dark:bg-neutral-900">
          <h4 className="text-xs font-bold uppercase tracking-wider text-amber-900 dark:text-amber-300">
            AI Limitations
          </h4>
          <p className="mt-2 text-xs leading-relaxed text-neutral-700 dark:text-neutral-300">
            The assistant provides explanations based on available evidence. It does not guarantee future financial outcomes.
          </p>
        </div>
      </div>

      {/* Responsible AI Status Checks */}
      <div className="rounded-xl border border-black/10 bg-white p-4 shadow-xs dark:border-white/15 dark:bg-neutral-900">
        <h4 className="mb-3 text-xs font-bold uppercase tracking-wider text-neutral-500">
          Responsible AI Governance Checklist
        </h4>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div className="flex items-center gap-2 rounded-lg border border-neutral-200 bg-neutral-50/60 p-2.5 text-xs font-medium text-neutral-700 dark:border-neutral-800 dark:bg-neutral-800/50 dark:text-neutral-300">
            <svg className="size-4 shrink-0 text-neutral-400 dark:text-neutral-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <circle cx="12" cy="12" r="8" />
            </svg>
            <span>Privacy review</span>
          </div>
          <div className="flex items-center gap-2 rounded-lg border border-neutral-200 bg-neutral-50/60 p-2.5 text-xs font-medium text-neutral-700 dark:border-neutral-800 dark:bg-neutral-800/50 dark:text-neutral-300">
            <svg className="size-4 shrink-0 text-neutral-400 dark:text-neutral-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <circle cx="12" cy="12" r="8" />
            </svg>
            <span>Safety review</span>
          </div>
          <div className="flex items-center gap-2 rounded-lg border border-neutral-200 bg-neutral-50/60 p-2.5 text-xs font-medium text-neutral-700 dark:border-neutral-800 dark:bg-neutral-800/50 dark:text-neutral-300">
            <svg className="size-4 shrink-0 text-neutral-400 dark:text-neutral-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <circle cx="12" cy="12" r="8" />
            </svg>
            <span>Evidence grounding review</span>
          </div>
          <div className="flex items-center gap-2 rounded-lg border border-neutral-200 bg-neutral-50/60 p-2.5 text-xs font-medium text-neutral-700 dark:border-neutral-800 dark:bg-neutral-800/50 dark:text-neutral-300">
            <svg className="size-4 shrink-0 text-neutral-400 dark:text-neutral-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <circle cx="12" cy="12" r="8" />
            </svg>
            <span>AI disclosure review</span>
          </div>
        </div>
      </div>

      {/* User Control Interface */}
      <div className="rounded-xl border border-blue-200/90 bg-white p-5 shadow-sm dark:border-blue-900/60 dark:bg-neutral-900">
        <div className="mb-4">
          <h3 className="text-base font-semibold text-neutral-900 dark:text-white">
            User Control & Human Oversight
          </h3>
          <p className="text-xs text-neutral-500">
            Confirm, reject, or request further clarification regarding the AI recommendation.
          </p>
        </div>

        {/* Action Status Notification */}
        {actionState === "confirmed" && (
          <div className="mb-4 flex items-center gap-2 rounded-lg bg-emerald-100 p-3 text-xs font-bold text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200">
            <CheckCircleIcon className="size-4" />
            <span>Decision acknowledged locally. No governance log entry was created.</span>
          </div>
        )}

        {actionState === "rejected" && (
          <div className="mb-4 flex items-center gap-2 rounded-lg bg-red-100 p-3 text-xs font-bold text-red-800 dark:bg-red-950 dark:text-red-200">
            <AlertTriangleIcon className="size-4" />
            <span>Rejection recorded in this session only. No secondary review was requested.</span>
          </div>
        )}

        {showMoreExplanation && (
          <div className="mb-4 rounded-lg border border-blue-200 bg-blue-50/80 p-4 text-xs dark:border-blue-900 dark:bg-blue-950/40">
            <div className="flex items-center gap-2 font-bold text-blue-900 dark:text-blue-200">
              <SparklesIcon className="size-4 text-blue-600" />
              <span>Expanded Model Attribution & Governance Disclosures</span>
            </div>
            <p className="mt-2 text-neutral-700 dark:text-neutral-300 leading-relaxed">
              {parsed.provenanceType === "live" ? (
                <>
                  This decision was evaluated by Component 2&apos;s real-time dual-stream fraud engine. Component 4&apos;s local synthetic Random Forest prototype was NOT used for this live decision, and prototype SHAP values do not explain Component 2&apos;s production model. The local prototype&apos;s predicted probability is an output of a local synthetic prototype, not a calibrated probability or independent confirmation of the original decision. Probability calibration has not been established.
                </>
              ) : (
                <>
                  The local synthetic Random Forest prototype produced the following SHAP attributions: transaction amount (+0.193289), unusual location (+0.153450), and new device (+0.130761). These values explain the local prototype&apos;s output, not Component 2&apos;s production model. The local prototype&apos;s predicted probability is not a calibrated probability or independent confirmation of the original fraud decision. Device and location inputs were inferred from mock reason labels and are not independently verified real-world observations. Probability calibration has not been established.
                </>
              )}
            </p>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={() => setActionState("confirmed")}
            className="rounded-lg bg-emerald-600 px-4 py-2 text-xs font-bold text-white shadow-xs transition hover:bg-emerald-700"
          >
            Confirm Decision
          </button>
          <button
            onClick={() => setActionState("rejected")}
            className="rounded-lg bg-red-600 px-4 py-2 text-xs font-bold text-white shadow-xs transition hover:bg-red-700"
          >
            Reject Action
          </button>
          <button
            onClick={() => setShowMoreExplanation(!showMoreExplanation)}
            className="rounded-lg border border-neutral-300 bg-white px-4 py-2 text-xs font-semibold text-neutral-700 shadow-2xs hover:bg-neutral-100 dark:border-neutral-700 dark:bg-neutral-800 dark:text-neutral-200 dark:hover:bg-neutral-700"
          >
            {showMoreExplanation ? "Hide Detailed Breakdown" : "Request More Explanation"}
          </button>

          {/* Rollback / Undo Button */}
          <button
            disabled
            className="flex items-center gap-1.5 rounded-lg border border-neutral-200 bg-neutral-100 px-4 py-2 text-xs font-medium text-neutral-400 opacity-60 cursor-not-allowed dark:border-neutral-800 dark:bg-neutral-800 dark:text-neutral-500"
            title="Rollback is available only for completed transaction states"
          >
            <UndoIcon className="size-3.5" />
            <span>Rollback / Undo (Inactive)</span>
          </button>
        </div>

        {/* Responsible AI Governance Notice */}
        <div className="mt-4 flex items-center gap-2 rounded-md bg-neutral-100 p-2.5 text-[11px] font-medium text-neutral-600 dark:bg-neutral-800 dark:text-neutral-400">
          <LockIcon className="size-4 shrink-0 text-blue-600" />
          <span>
            Human Governance Policy: The AI assistant does not automatically execute irreversible financial decisions without explicit user review and confirmation.
          </span>
        </div>
      </div>
    </div>
  );
}
