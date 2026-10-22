"use client";

import React, { useState, useEffect } from "react";
import { MOCK_TRANSACTION } from "./mockData";
import { TrustActionState } from "./types";
import { ShieldIcon, CheckCircleIcon, LockIcon, AlertTriangleIcon, UndoIcon, SparklesIcon } from "./icons";
import {
  recordGovernanceDecision,
  getGovernanceDecision,
  type GovernanceDecisionRecord,
} from "@/lib/api/assistant";
import { translateSecurityRulesToEnglish } from "./explanationParser";

interface TrustPanelScreenProps {
  latestEvidence?: Array<{ tool: string; result: string }>;
  selectedTransactionId?: string;
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
  modelVersion?: string;
  benchmarkModel?: boolean;
  unavailableSignals?: string[];
  retrievalTimestamp?: string;
  auditId?: string;
  blockNumber?: number | string;
  recordStatus?: string;
  auditIsMockLedger?: boolean;
  hasFraudEvidence: boolean;
  hasAuditEvidence: boolean;
  hasPortfolioEvidence?: boolean;
  portfolioId?: string;
  isPortfolioFallback?: boolean;
  portfolioStatusType?: "live" | "fallback" | "service_error" | "empty" | "disabled";
  portfolioIsPartial?: boolean;
  portfolioTotalValue?: string;
  portfolioCurrency?: string;
  portfolioRiskScore?: number;
  portfolioRiskLevel?: string;
  portfolioDiversificationScore?: number;
  portfolioLiquidity?: string;
  portfolioExpectedReturn?: number;
  portfolioHoldingCount?: number;
  hasShap: boolean;
  provenanceType: "live" | "fallback" | "baseline" | "demo" | "not_found" | "unavailable";
  provenanceLabel: string;
}

function parseLatestEvidence(evidenceList: Array<{ tool: string; result: string }> = []): ParsedEvidenceData {
  const fraudItem = evidenceList.find((e) => e.tool === "get_fraud_analysis");
  const auditItem = evidenceList.find((e) => e.tool === "get_blockchain_audit");
  const portfolioItem = evidenceList.find((e) => e.tool === "get_portfolio_analysis");

  const isFraudNotFound = fraudItem?.result
    ? /RECORD NOT FOUND|SERVICE UNAVAILABLE|RECORD_NOT_FOUND|SERVICE_UNAVAILABLE|No verified live fraud record exists/i.test(fraudItem.result)
    : false;
  const isFraudDemo = fraudItem?.result
    ? /DEMO \/ SYNTHETIC|DEMO SYNTHETIC|EXPLICIT DEMO MODE/i.test(fraudItem.result)
    : false;
  const isFraudFallback = !isFraudNotFound && !isFraudDemo && (fraudItem?.result
    ? /FALLBACK MOCK DATA|Local mock baseline|FALLBACK NOTICE|NOT a live lookup result/i.test(fraudItem.result)
    : false);
  const isFraudLive =
    !isFraudFallback && !isFraudNotFound && !isFraudDemo &&
    Boolean(
      fraudItem?.result &&
      (/LIVE COMPONENT 2 AUDIT RECORD/i.test(fraudItem.result) ||
        /Live Component 2 Gateway Audit Log/i.test(fraudItem.result) ||
        /Live Component 2 Gateway Decision Store/i.test(fraudItem.result))
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

  const isPortfolioDisabled = Boolean(portfolioItem?.result && /PERSONALIZED CONTEXT DISABLED|Personalized financial context is disabled/i.test(portfolioItem.result));

  const portfolioFallbackReason = portfolioItem?.result
    ? (portfolioItem.result.match(/Platform service note:\s*([^\n\r]+)/i)?.[1]?.trim() ||
       portfolioItem.result.match(/live platform service at [^:]*:\s*([^\n\r\)]+)/i)?.[1]?.trim())
    : undefined;

  const isPortfolioServiceError = Boolean(
    portfolioItem?.result &&
    (/SERVICE UNAVAILABLE|service unreachable|Connection refused|timed out|timeout|HTTP 50[0-9]|Service unavailable/i.test(portfolioItem.result) ||
     Boolean(portfolioFallbackReason && /unreachable|timed out|timeout|connection refused|500|Service unavailable/i.test(portfolioFallbackReason))) &&
    !/Empty portfolio list|0 active positions|No portfolios registered/i.test(portfolioItem.result)
  );

  const isPortfolioEmpty = Boolean(
    portfolioItem?.result &&
    /Empty portfolio list|no registered portfolios|0 active positions|NO LIVE HOLDINGS FOUND|No holdings recorded/i.test(portfolioItem.result)
  );

  const isPortfolioFallback = portfolioItem?.result
    ? /FALLBACK|Local mock baseline|NO LIVE PORTFOLIO/i.test(portfolioItem.result)
    : false;
  const isPortfolioLive =
    !isPortfolioFallback &&
    Boolean(portfolioItem?.result && (/LIVE PLATFORM RECORD/i.test(portfolioItem.result)));

  const isPortfolioPartial = Boolean(
    isPortfolioLive &&
    portfolioItem?.result &&
    /Unavailable|Partial data notice/i.test(portfolioItem.result)
  );

  const portfolioStatusType: "live" | "fallback" | "service_error" | "empty" | "disabled" = isPortfolioDisabled
    ? "disabled"
    : isPortfolioServiceError
    ? "service_error"
    : isPortfolioLive
    ? "live"
    : isPortfolioEmpty
    ? "empty"
    : "fallback";

  let provenanceType: "live" | "fallback" | "baseline" | "demo" | "not_found" | "unavailable" = "baseline";
  if (fraudItem && auditItem && portfolioItem) {
    provenanceType = isFraudLive && isAuditLive && isPortfolioLive ? "live" : "fallback";
  } else if (fraudItem && auditItem) {
    provenanceType = isFraudLive && isAuditLive ? "live" : "fallback";
  } else if (fraudItem && portfolioItem) {
    provenanceType = isFraudLive && isPortfolioLive ? "live" : "fallback";
  } else if (auditItem && portfolioItem) {
    provenanceType = isAuditLive && isPortfolioLive ? "live" : "fallback";
  } else if (fraudItem) {
    if (isFraudLive) provenanceType = "live";
    else if (isFraudDemo) provenanceType = "demo";
    else if (isFraudNotFound) {
      const isServiceError =
        /Lookup Status:\s*SERVICE_UNAVAILABLE/i.test(fraudItem.result) ||
        (!/Lookup Status:\s*RECORD_NOT_FOUND/i.test(fraudItem.result) &&
          /Connection refused|Cannot connect|timed out/i.test(fraudItem.result));
      provenanceType = isServiceError ? "unavailable" : "not_found";
    }
    else provenanceType = "fallback";
  } else if (auditItem) {
    provenanceType = isAuditLive ? "live" : "fallback";
  } else if (portfolioItem) {
    provenanceType = isPortfolioLive ? "live" : "fallback";
  } else {
    provenanceType = "baseline";
  }

  let provenanceLabel =
    provenanceType === "live"
      ? "Active Session: Live Service Evidence"
      : provenanceType === "demo"
        ? "Active Session: Synthetic Demo Data (Explicit Demo Mode)"
        : provenanceType === "not_found"
          ? "Active Session: Component 2 Record Not Found"
          : provenanceType === "unavailable"
            ? "Active Session: Component 2 Service Unavailable"
            : provenanceType === "fallback"
              ? portfolioItem && !fraudItem && !auditItem
                ? "Active Session: Tool Evidence — Synthetic Demo Data"
                : "Active Session: Fallback Mock Data"
              : "Mock Baseline (No Active Session Evidence)";

  if (portfolioItem && !fraudItem && !auditItem) {
    if (portfolioStatusType === "service_error") {
      provenanceLabel = "Active Session: Service Unavailable (Platform Outage)";
    } else if (portfolioStatusType === "disabled") {
      provenanceLabel = "Active Session: Context Disabled (Responsible AI)";
    } else if (isPortfolioPartial) {
      provenanceLabel = "Active Session: Live Service Evidence (Partial Data)";
    }
  }

  const parsed: ParsedEvidenceData = {
    hasFraudEvidence: !!(fraudItem?.result && !fraudItem.result.includes("No transaction was found") && !isFraudNotFound),
    hasAuditEvidence: !!(auditItem?.result && !auditItem.result.includes("No audit record found")),
    hasPortfolioEvidence: !!(portfolioItem?.result && portfolioItem.result.trim().length > 0),
    isPortfolioFallback,
    portfolioStatusType,
    portfolioIsPartial: isPortfolioPartial,
    hasShap: false,
    provenanceType,
    provenanceLabel,
  };

  if (fraudItem?.result) {
    const text = fraudItem.result;
    const txIdMatch = text.match(/Transaction ID:\s*([A-Za-z0-9_-]+)/i);
    if (txIdMatch) parsed.transactionId = txIdMatch[1];

    if (isFraudNotFound) {
      parsed.status = provenanceType === "unavailable" ? "SERVICE_UNAVAILABLE" : "RECORD_NOT_FOUND";
      parsed.riskLevel = "UNAVAILABLE";
      parsed.fraudScore = undefined;
      parsed.unavailableSignals = [
        "Fraud Score (Live lookup failed)",
        "Decision Status (Unavailable)",
        "Device Signals (Unavailable)",
        "Location Signals (Unavailable)",
        "Amount Risk Assessment (Unavailable)",
      ];
      parsed.factors = [];
      return parsed;
    }

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

    const verMatch = text.match(/Model Version:\s*([^\n\r]+)/i);
    if (verMatch) parsed.modelVersion = verMatch[1].trim();
    parsed.benchmarkModel = /Benchmark Status:[^\n\r]*benchmark/i.test(text) && !/NOT a benchmark/i.test(text);

    const unavailableSigs: string[] = [];
    if (/Device Signals:[^\n\r]*Unavailable/i.test(text)) {
      unavailableSigs.push("Device Telemetry (Not retained)");
    }
    if (/Location Signals:[^\n\r]*Unavailable/i.test(text)) {
      unavailableSigs.push("Location Telemetry (Not retained)");
    }
    if (/Amount Risk Assessment:[^\n\r]*Unavailable/i.test(text)) {
      unavailableSigs.push("Amount Risk Threshold (Not retained)");
    }
    parsed.unavailableSignals = unavailableSigs;

    const timeMatch = text.match(/(?:Audit Timestamp|Decision Timestamp):\s*([^\n\r]+)/i);
    if (timeMatch) parsed.retrievalTimestamp = timeMatch[1].trim();

    const factorsMatch = text.match(
      /(?:RECORDED CONTRIBUTING FACTORS|DECISION RULES & REASONING|FEATURE ATTRIBUTIONS & CONTRIBUTING FACTORS):\s*([\s\S]*?)(?=(?:COMPONENT 2|LOCAL SHAP|TRANSACTION TELEMETRY|SIGNALS|LIMITATION|$))/i
    );
    if (factorsMatch?.[1]) {
      const items: string[] = [];
      for (const line of factorsMatch[1].split("\n")) {
        const m = line.match(/^\s*-\s*(.+)/);
        if (m && m[1].trim() && !m[1].startsWith("None returned")) items.push(m[1].trim());
      }
      if (items.length > 0) parsed.factors = items;
    }

    parsed.hasShap = /SHAP (?:CONTRIBUTIONS|EXPLANATION)/i.test(text) && isFraudDemo;
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

export function TrustPanelScreen({ latestEvidence = [], selectedTransactionId }: TrustPanelScreenProps) {
  const [actionState, setActionState] = useState<TrustActionState>("pending");
  const [persistedRecord, setPersistedRecord] = useState<GovernanceDecisionRecord | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [showMoreExplanation, setShowMoreExplanation] = useState(false);
  const tx = MOCK_TRANSACTION;

  const parsed = parseLatestEvidence(latestEvidence);
  // Session policy: Active session evidence is valid only when tools returned results in this response.
  const hasLiveEvidence =
    latestEvidence.length > 0 &&
    (parsed.hasFraudEvidence ||
      parsed.hasAuditEvidence ||
      Boolean(parsed.hasPortfolioEvidence) ||
      parsed.provenanceType === "not_found" ||
      parsed.provenanceType === "unavailable");
  const isPortfolio = Boolean(parsed.hasPortfolioEvidence && !parsed.hasFraudEvidence && !parsed.hasAuditEvidence);

  const displayTxId =
    parsed.transactionId ??
    selectedTransactionId ??
    (hasLiveEvidence ? "Unknown" : tx.id);
  const displayAmount = parsed.amount ?? (hasLiveEvidence ? undefined : tx.amount);
  const displayCurrency = parsed.currency ?? (hasLiveEvidence ? "LKR" : tx.currency);
  const displayStatus = isPortfolio
    ? (parsed.portfolioStatusType === "disabled"
        ? "CONTEXT DISABLED"
        : parsed.portfolioStatusType === "service_error"
          ? "SERVICE UNAVAILABLE"
          : parsed.portfolioStatusType === "empty"
            ? (parsed.isPortfolioFallback ? "DEMO BASELINE" : "0 HOLDINGS")
            : parsed.portfolioIsPartial
              ? "LIVE (PARTIAL DATA)"
              : (parsed.isPortfolioFallback ? "DEMO BASELINE" : "LIVE PORTFOLIO"))
    : (parsed.status ?? (hasLiveEvidence ? (parsed.recordStatus ?? "RECORDED") : tx.status));

  // Load and display saved decision state when returning to the same recommendation / on page refresh
  useEffect(() => {
    let active = true;
    async function loadSaved() {
      if (!displayTxId) return;
      try {
        const res = await getGovernanceDecision(displayTxId);
        if (!active) return;
        if (res && res.found && res.decision) {
          setActionState(res.decision.decision_type);
          setPersistedRecord(res.decision);
        }
      } catch {
        // Safe fallback if service is unreachable or not yet recorded
      }
    }
    loadSaved();
    return () => {
      active = false;
    };
  }, [displayTxId]);

  const handleDecision = async (type: "confirmed" | "rejected") => {
    if (isSubmitting) return;

    // Prevent accidental duplicate submission if already recorded in this exact state
    if (persistedRecord && persistedRecord.decision_type === type && actionState === type) {
      return;
    }

    setIsSubmitting(true);
    setActionError(null);

    try {
      const res = await recordGovernanceDecision({
        decision_type: type,
        reference_id: displayTxId,
        reference_type: isPortfolio ? "portfolio" : "transaction",
        evidence_reference: {
          auditId: parsed.auditId,
          blockNumber: parsed.blockNumber,
          fraudScore: parsed.fraudScore,
          prototypeProbability: parsed.prototypeProbability,
          provenanceType: parsed.provenanceType,
          status: displayStatus,
        },
        rationale: type === "confirmed" ? "Human operator confirmed decision" : "Human operator rejected action",
      });

      // ONLY display success after persistence succeeds
      if (res && res.decision) {
        setActionState(res.decision.decision_type);
        setPersistedRecord(res.decision);
      }
    } catch (err: unknown) {
      // On failure, show an error and DO NOT pretend the decision was recorded
      const msg = err instanceof Error ? err.message : "Failed to record governance decision";
      setActionError(msg);
    } finally {
      setIsSubmitting(false);
    }
  };

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
              What the system decided <span className="text-xs font-normal text-neutral-500">(AI Decision Summary)</span>
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
          {actionState === "rejected" ? (
            <span className="rounded-full bg-red-600 px-3 py-1 text-xs font-bold text-white shadow-xs">
              REJECTED BY HUMAN OVERSIGHT
            </span>
          ) : actionState === "confirmed" ? (
            <span className="rounded-full bg-emerald-600 px-3 py-1 text-xs font-bold text-white shadow-xs">
              CONFIRMED BY HUMAN REVIEW
            </span>
          ) : (
            <span
              className={`rounded-full px-3 py-1 text-xs font-bold ${
                isPortfolio
                  ? parsed.portfolioStatusType === "service_error"
                    ? "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300"
                    : parsed.portfolioStatusType === "disabled"
                      ? "bg-neutral-100 text-neutral-700 dark:bg-neutral-800 dark:text-neutral-300"
                      : parsed.portfolioIsPartial
                        ? "bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300"
                        : parsed.isPortfolioFallback
                          ? "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300"
                          : "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300"
                  : "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300"
              }`}
            >
              {displayStatus}
            </span>
          )}
        </div>

        <div className="rounded-lg border border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-800 dark:bg-neutral-800/60">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-xs font-semibold text-neutral-500">
                {isPortfolio ? "Target Account / Portfolio" : "Target Transaction"}
              </p>
              <p className="text-lg font-bold text-neutral-900 dark:text-white">
                {isPortfolio
                  ? `Portfolio ${parsed.portfolioId || "USER001"}${parsed.portfolioTotalValue && !parsed.portfolioTotalValue.includes("Unavailable") ? ` — ${parsed.portfolioCurrency || "LKR"} ${parsed.portfolioTotalValue}` : ""}`
                  : `Transaction ${displayTxId}${displayAmount != null ? ` — ${displayCurrency} ${displayAmount.toLocaleString()}` : ""}`}
              </p>
            </div>
            <div className="text-right">
              <span className="text-xs text-neutral-500">Recorded result</span>
              <p
                className={`text-base font-bold ${
                  actionState === "rejected"
                    ? "text-red-600 dark:text-red-400"
                    : actionState === "confirmed"
                      ? "text-emerald-600 dark:text-emerald-400"
                      : isPortfolio
                        ? parsed.portfolioStatusType === "service_error"
                          ? "text-red-600 dark:text-red-400"
                          : parsed.portfolioStatusType === "disabled"
                            ? "text-neutral-600 dark:text-neutral-400"
                            : parsed.portfolioIsPartial
                              ? "text-blue-600 dark:text-blue-400"
                              : parsed.isPortfolioFallback
                                ? "text-amber-600 dark:text-amber-400"
                                : "text-emerald-600 dark:text-emerald-400"
                        : "text-red-600 dark:text-red-400"
                }`}
              >
                {actionState === "rejected"
                  ? "Execution Halted (Human Rejected)"
                  : actionState === "confirmed"
                    ? "Approved by Human Review"
                    : displayStatus}
              </p>
            </div>
          </div>

          <p className="mt-3 text-sm text-neutral-700 dark:text-neutral-300">
            <span className="font-semibold text-neutral-900 dark:text-white">What the system recorded:</span>{" "}
            {hasLiveEvidence ? (
              isPortfolio ? (
                parsed.portfolioStatusType === "service_error" ? (
                  <>Notice: The live platform portfolio service was unreachable or timed out for user <span className="font-semibold text-neutral-900 dark:text-white">{parsed.portfolioId || "USER001"}</span>. Live holdings remain unverified due to platform service unavailability; this does NOT indicate an empty portfolio or zero holdings.</>
                ) : parsed.portfolioStatusType === "disabled" ? (
                  <>Notice: Personalized financial context is disabled by user preference. Account-specific portfolio queries are blocked without fabricating anonymous holdings.</>
                ) : parsed.portfolioIsPartial ? (
                  <>Live portfolio record retrieved with partial data for <span className="font-semibold text-neutral-900 dark:text-white">{parsed.portfolioId || "USER001"}</span> ({parsed.portfolioHoldingCount} recorded positions). Verified fields are displayed; unrecorded valuation/cost fields are marked as unavailable rather than estimated.</>
                ) : parsed.isPortfolioFallback ? (
                  <>Notice: No live portfolio record was retrieved for user <span className="font-semibold text-neutral-900 dark:text-white">{parsed.portfolioId || "USER001"}</span> from the platform service (0 verified holdings). The displayed metrics and valuation are synthetic baseline demonstrations shown for system demonstration purposes and do not represent verified holdings, actual account balances, or personalized investment advice.</>
                ) : (
                  <>Live portfolio record retrieved from the platform portfolio store. Recorded market value: <span className="font-semibold text-neutral-900 dark:text-white">{parsed.portfolioCurrency || "USD"} {parsed.portfolioTotalValue}</span> across <span className="font-semibold text-neutral-900 dark:text-white">{parsed.portfolioHoldingCount}</span> verified holdings.</>
                )
              ) : parsed.hasFraudEvidence || parsed.provenanceType === "not_found" || parsed.provenanceType === "unavailable" ? (
                parsed.provenanceType === "not_found" || parsed.provenanceType === "unavailable" ? (
                  <>
                    Notice: No verified fraud decision was found for transaction <span className="font-semibold text-neutral-900 dark:text-white">{displayTxId}</span> in Component 2&apos;s live decision store. Live lookup failed and mock fallback data is not substituted under FinTrust Responsible AI rules.
                  </>
                ) : parsed.provenanceType === "live" ? (
                  <>
                    The system recorded a decision of <span className="font-semibold text-neutral-900 dark:text-white">{parsed.status || "BLOCKED"}</span> with a risk score of <span className="font-semibold text-neutral-900 dark:text-white">{parsed.fraudScore?.toFixed(2) ?? "1.00"}</span> ({parsed.riskLevel || "HIGH"} risk) for transaction <span className="font-semibold text-neutral-900 dark:text-white">{displayTxId}</span>. What contributed to the result: {parsed.factors && parsed.factors.length > 0 ? translateSecurityRulesToEnglish(parsed.factors)[0] : "automated security gateway rules"}. This score has not been validated as a probability of fraud, and device and location information were not collected in the recorded evidence.
                  </>
                ) : (
                  <>
                    In this mock demonstration record for transaction <span className="font-semibold text-neutral-900 dark:text-white">{displayTxId}</span>, the system recorded a simulated score of <span className="font-semibold text-neutral-900 dark:text-white">{parsed.fraudScore != null ? parsed.fraudScore.toFixed(2) : "0.92"}</span>, <span className="font-semibold text-neutral-900 dark:text-white">{parsed.riskLevel || "HIGH"}</span> risk level, and <span className="font-semibold text-neutral-900 dark:text-white">{parsed.status || "BLOCKED"}</span> status based on sample risk factors ({parsed.factors?.join(", ") || "high transfer amount, unusual location, and new device"}). This score has not been validated as a probability of fraud, and device and location details were simulated rather than observed from live telemetry.
                  </>
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
              Evidence Checklist (Available vs. Missing Information)
            </p>
            <div className="flex flex-col gap-1.5 text-xs text-neutral-800 dark:text-neutral-200">
              {isPortfolio ? (
                parsed.portfolioStatusType === "service_error" ? (
                  <>
                    <div className="flex items-center gap-2 text-amber-700 dark:text-amber-400">
                      <AlertTriangleIcon className="size-4 shrink-0" />
                      <span>Platform service query: Service unreachable or timed out — live holdings could not be retrieved</span>
                    </div>
                    <div className="flex items-center gap-2 text-neutral-500 dark:text-neutral-400">
                      <svg className="size-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <circle cx="12" cy="12" r="8" />
                      </svg>
                      <span>Holdings verification: Unverified due to platform outage (Not an empty portfolio)</span>
                    </div>
                    <div className="flex items-center gap-2 text-neutral-500 dark:text-neutral-400">
                      <svg className="size-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <circle cx="12" cy="12" r="8" />
                      </svg>
                      <span>Synthetic baseline values: Suppressed to prevent misrepresenting service outage</span>
                    </div>
                  </>
                ) : parsed.portfolioStatusType === "disabled" ? (
                  <>
                    <div className="flex items-center gap-2 text-neutral-600 dark:text-neutral-400">
                      <LockIcon className="size-4 shrink-0 text-blue-500" />
                      <span>Personalized financial context: Blocked by user preference in Responsible AI settings</span>
                    </div>
                    <div className="flex items-center gap-2 text-neutral-500 dark:text-neutral-400">
                      <svg className="size-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <circle cx="12" cy="12" r="8" />
                      </svg>
                      <span>Holdings lookup: Suppressed (No mock holdings substituted)</span>
                    </div>
                  </>
                ) : parsed.portfolioIsPartial ? (
                  <>
                    <div className="flex items-center gap-2">
                      <CheckCircleIcon className="size-4 shrink-0 text-emerald-500" />
                      <span>Platform service query: Live portfolio retrieved ({parsed.portfolioHoldingCount} positions recorded)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <CheckCircleIcon className="size-4 shrink-0 text-emerald-500" />
                      <span>Verified fields displayed; unrecorded valuation/price fields marked Unavailable</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <CheckCircleIcon className="size-4 shrink-0 text-emerald-500" />
                      <span>Synthetic baseline values: Strictly omitted from live record</span>
                    </div>
                  </>
                ) : (
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
                )
              ) : (
                <>
                  {/* Row 1: Fraud score evidence */}
                  {hasLiveEvidence ? (
                    parsed.provenanceType === "not_found" || parsed.provenanceType === "unavailable" ? (
                      <div className="flex items-center gap-2 text-amber-700 dark:text-amber-400">
                        <AlertTriangleIcon className="size-4 shrink-0" />
                        <span>
                          Component 2 live decision store: {parsed.provenanceType === "unavailable" ? "Service Unavailable" : "Record Not Found"} for transaction {displayTxId} (Fraud score: Unavailable; no mock score substituted)
                        </span>
                      </div>
                    ) : parsed.hasFraudEvidence ? (
                      <div className="flex flex-col gap-1">
                        <div className="flex items-center gap-2">
                          <CheckCircleIcon className="size-4 shrink-0 text-emerald-500" />
                          <span>
                            {parsed.provenanceType === "live"
                              ? `Live Component 2 recorded result: ${parsed.status || "UNKNOWN"} (Risk score: ${parsed.fraudScore?.toFixed(2) ?? "1.00"}, ${parsed.riskLevel || "N/A"} risk)`
                              : `Recorded mock fraud score: ${parsed.fraudScore != null ? parsed.fraudScore.toFixed(2) : "0.92"} (Synthetic Demo Mode)`}
                          </span>
                        </div>
                        {parsed.modelVersion && (
                          <div className="ml-6 text-[11px] text-neutral-600 dark:text-neutral-400">
                            Model Version: {parsed.modelVersion} (
                            {parsed.benchmarkModel
                              ? "Validated benchmark model"
                              : "Untrained heuristic rule set, not a validated benchmark model"}
                            )
                          </div>
                        )}
                        {parsed.unavailableSignals && parsed.unavailableSignals.length > 0 && (
                          <div className="ml-6 text-[11px] text-amber-700 dark:text-amber-400">
                            Information the system did not collect: {parsed.unavailableSignals.join(", ")}
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 text-neutral-500 dark:text-neutral-400">
                        <svg className="size-4 shrink-0 text-neutral-400 dark:text-neutral-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <circle cx="12" cy="12" r="8" />
                        </svg>
                        <span>Recorded risk score: Not evaluated in this response</span>
                      </div>
                    )
                  ) : (
                    <div className="flex items-center gap-2 text-neutral-500 dark:text-neutral-400">
                      <svg className="size-4 shrink-0 text-neutral-400 dark:text-neutral-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <circle cx="12" cy="12" r="8" />
                      </svg>
                      <span>Mock baseline risk score: {tx.fraudScore} (this score has not been validated as a probability of fraud)</span>
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
                          {parsed.auditIsMockLedger && " — simulated ledger (independent blockchain verification not established)"}
                          )
                        </span>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 text-neutral-500 dark:text-neutral-400">
                        <svg className="size-4 shrink-0 text-neutral-400 dark:text-neutral-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <circle cx="12" cy="12" r="8" />
                        </svg>
                        <span>
                          Blockchain audit record: Not requested or evaluated in this response (remains unverified)
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
                            ? "A SHAP explanation was not generated for this transaction (raw telemetry not retained in gateway)"
                            : `SHAP explanation available (local synthetic Random Forest prototype, prob ${parsed.prototypeProbability != null ? parsed.prototypeProbability.toFixed(4) : "1.0000"}; does not explain production model)`}
                        </span>
                      </div>
                    ) : parsed.provenanceType === "live" ? (
                      <div className="flex items-center gap-2 text-neutral-500 dark:text-neutral-400">
                        <svg className="size-4 shrink-0 text-neutral-400 dark:text-neutral-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <circle cx="12" cy="12" r="8" />
                        </svg>
                        <span>A SHAP explanation was not generated for this transaction (raw telemetry not retained in gateway)</span>
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
                Technical details & raw tool evidence ({latestEvidence.length})
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

        {/* Error Notification */}
        {actionError && (
          <div className="mb-4 flex items-start gap-2 rounded-lg bg-red-100 p-3 text-xs font-medium text-red-900 dark:bg-red-950/80 dark:text-red-200 border border-red-300 dark:border-red-900">
            <AlertTriangleIcon className="size-4 shrink-0 text-red-600 dark:text-red-400 mt-0.5" />
            <div>
              <p className="font-bold">Persistence Failed</p>
              <p className="mt-0.5">{actionError}. The decision was NOT recorded in the governance log. Please verify credentials and network status.</p>
            </div>
          </div>
        )}

        {/* Action Status Notification */}
        {actionState === "confirmed" && (
          <div className="mb-4 rounded-lg bg-emerald-100 p-3.5 text-xs text-emerald-900 dark:bg-emerald-950 dark:text-emerald-200 border border-emerald-300 dark:border-emerald-800">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 font-bold">
                <CheckCircleIcon className="size-4 text-emerald-600 dark:text-emerald-400" />
                <span>Decision confirmed and permanently anchored in Component 4 governance log.</span>
              </div>
              {persistedRecord?.is_duplicate && (
                <span className="rounded-full bg-emerald-200 px-2 py-0.5 text-[10px] font-bold text-emerald-800 dark:bg-emerald-900 dark:text-emerald-300">
                  Already Recorded
                </span>
              )}
            </div>
            {persistedRecord && (
              <div className="mt-2 flex flex-wrap items-center gap-3 text-[11px] text-emerald-800/90 dark:text-emerald-300/90 font-mono">
                <span>Record ID: {persistedRecord.decision_id}</span>
                <span>•</span>
                <span>Timestamp: {new Date(persistedRecord.timestamp).toLocaleString()}</span>
                {persistedRecord.user_id && (
                  <>
                    <span>•</span>
                    <span>Reviewer: #{persistedRecord.user_id}</span>
                  </>
                )}
              </div>
            )}
          </div>
        )}

        {actionState === "rejected" && (
          <div className="mb-4 rounded-lg bg-red-100 p-3.5 text-xs text-red-900 dark:bg-red-950 dark:text-red-200 border border-red-300 dark:border-red-800">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 font-bold">
                <AlertTriangleIcon className="size-4 text-red-600 dark:text-red-400" />
                <span>Action REJECTED by human operator and anchored in governance log. Financial execution is strictly blocked.</span>
              </div>
              <span className="rounded-full bg-red-200 px-2 py-0.5 text-[10px] font-bold text-red-900 dark:bg-red-900 dark:text-red-300 uppercase">
                Execution Prohibited
              </span>
            </div>
            {persistedRecord && (
              <div className="mt-2 flex flex-wrap items-center gap-3 text-[11px] text-red-800/90 dark:text-red-300/90 font-mono">
                <span>Record ID: {persistedRecord.decision_id}</span>
                <span>•</span>
                <span>Timestamp: {new Date(persistedRecord.timestamp).toLocaleString()}</span>
                {persistedRecord.user_id && (
                  <>
                    <span>•</span>
                    <span>Reviewer: #{persistedRecord.user_id}</span>
                  </>
                )}
              </div>
            )}
            <p className="mt-2 text-[11px] font-semibold text-red-950 dark:text-red-200">
              Downstream Policy: Subsequent financial transactions for this recommendation will not be executed.
            </p>
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
            onClick={() => handleDecision("confirmed")}
            disabled={isSubmitting || (actionState === "confirmed" && persistedRecord?.decision_type === "confirmed")}
            className="rounded-lg bg-emerald-600 px-4 py-2 text-xs font-bold text-white shadow-xs transition hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isSubmitting ? "Recording..." : actionState === "confirmed" ? "Decision Confirmed" : "Confirm Decision"}
          </button>
          <button
            onClick={() => handleDecision("rejected")}
            disabled={isSubmitting || (actionState === "rejected" && persistedRecord?.decision_type === "rejected")}
            className="rounded-lg bg-red-600 px-4 py-2 text-xs font-bold text-white shadow-xs transition hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isSubmitting ? "Recording..." : actionState === "rejected" ? "Action Rejected" : "Reject Action"}
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
            title="Rollback is disabled: external financial streams and blockchain state commitments are irreversible."
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
