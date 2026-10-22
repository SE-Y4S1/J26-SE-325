"use client";

import React, { useState, useEffect } from "react";
import { BarChartIcon, CheckCircleIcon, SparklesIcon, ChevronRightIcon, ShieldIcon } from "./icons";
import {
  parseSessionEvidence,
  ParsedShapItem,
  ParsedFraudEvidence,
  ParsedAuditEvidence,
  ParsedPortfolioEvidence,
  ParsedSessionEvidence,
  parseFraudResult,
  parseAuditResult,
  parsePortfolioResult,
  getBaselineFraud,
} from "./explanationParser";

export type {
  ParsedShapItem,
  ParsedFraudEvidence,
  ParsedAuditEvidence,
  ParsedPortfolioEvidence,
  ParsedSessionEvidence,
};
export { parseSessionEvidence, parseFraudResult, parseAuditResult, parsePortfolioResult, getBaselineFraud };

interface Props {
  onNavigateToTrustPanel: () => void;
  latestEvidence?: Array<{ tool: string; result: string }>;
}

export function ExplanationScreen({ onNavigateToTrustPanel, latestEvidence = [] }: Props) {
  const [feedbackGiven, setFeedbackGiven] = useState<"yes" | "no" | null>(null);
  const [showIllustrativeBaseline, setShowIllustrativeBaseline] = useState(false);
  const [personalizedContext, setPersonalizedContext] = useState<boolean>(true);

  useEffect(() => {
    try {
      const stored =
        localStorage.getItem("finagent_responsible_ai_preferences") ||
        localStorage.getItem("finagent_responsible_ai_prefs");
      if (stored) {
        const parsed = JSON.parse(stored);
        if (typeof parsed?.personalizedContext === "boolean") {
          setPersonalizedContext(parsed.personalizedContext);
        }
      }
    } catch {
      // Fall back to default
    }
  }, []);

  const session = parseSessionEvidence(latestEvidence);
  const fraud = session.fraud;
  const audit = session.audit;
  const portfolio = session.portfolio;
  const baseline = session.baseline;

  // Active SHAP list for proportional visual bar calculation
  const activeShapList =
    fraud && fraud.shapContributions.length > 0
      ? fraud.shapContributions
      : showIllustrativeBaseline && baseline && baseline.shapContributions.length > 0
      ? baseline.shapContributions
      : [];

  const maxShap =
    activeShapList.length > 0 ? Math.max(...activeShapList.map((c) => c.shapValue), 0.001) : 1.0;

  // Construct explanation text based on session mode and provenance
  let explanationSummary = "";
  if (session.mode === "audit" && audit) {
    explanationSummary = audit.explanationText;
  } else if (session.mode === "portfolio" && portfolio) {
    explanationSummary = portfolio.explanationText;
  } else if (session.mode === "fraud_and_audit" && fraud && audit) {
    const fraudSummary =
      fraud.provenanceType === "live"
        ? `Transaction ${fraud.transactionId} evaluated in Component 2's live gateway audit trail with decision ${fraud.status} (Recorded Fraud Score: ${fraud.fraudScore?.toFixed(4) ?? "N/A"}).`
        : fraud.provenanceType === "not_found" || fraud.provenanceType === "unavailable"
        ? `No verified live fraud decision was found for transaction ${fraud.transactionId} in Component 2's live decision store.`
        : `Mock transaction record ${fraud.transactionId} lists a recorded fraud score of ${fraud.fraudScore?.toFixed(2) ?? "N/A"}, ${fraud.riskLevel} risk level, and ${fraud.status} status (${fraud.factors.join(", ")}). Separately, the local synthetic Random Forest prototype provides illustrative SHAP attributions (uncalibrated probability ${fraud.prototypeProbability?.toFixed(4) ?? "1.0000"}). Component 2 live gateway service was not reached.`;
    const auditSummary =
      audit.provenanceType === "live"
        ? `In Component 3's blockchain ledger, audit ID ${audit.auditId || "RECORDED"} is registered at block #${audit.blockNumber ?? "N/A"} with status ${audit.recordStatus}.`
        : `In the blockchain ledger fallback data, audit ID ${audit.auditId || "RECORDED"} is registered at block #${audit.blockNumber ?? "N/A"} with status ${audit.recordStatus}. Independent blockchain verification is not established.`;
    explanationSummary = `${fraudSummary} ${auditSummary}`;
  } else if (fraud) {
    if (fraud.whyBlockedPlainEnglish) {
      explanationSummary = `${fraud.whyBlockedPlainEnglish} ${fraud.resultMeaningPlainEnglish ?? ""}`;
    } else if (fraud.provenanceType === "live") {
      explanationSummary = `Your payment was blocked because it triggered the system's security rules. The recorded decision shows that the risk score crossed the blocking threshold and an amount-related rule also contributed to the decision. The system assigned a risk score of ${fraud.fraudScore?.toFixed(2) ?? "1.00"}, which it classified as high risk. This does not prove that the payment was fraudulent. The score has not been validated as a probability of fraud. Device and location information were not available in the recorded evidence.`;
    } else if (fraud.provenanceType === "not_found" || fraud.provenanceType === "unavailable") {
      explanationSummary = `No verified live fraud decision was found for transaction ${fraud.transactionId} in Component 2's live decision store. Fraud score, status, and contributing factors are unavailable.`;
    } else if (fraud.provenanceType === "fallback") {
      explanationSummary = `In this mock demonstration record for transaction ${fraud.transactionId}, the system simulated a ${fraud.riskLevel.toLowerCase()}-risk result with a score of ${fraud.fraudScore?.toFixed(2) ?? "0.92"} and status ${fraud.status} based on sample risk factors. This does not prove that the payment was fraudulent. The score has not been validated as a probability of fraud, and device and location details were simulated rather than observed from live telemetry.`;
    }
  } else if (session.mode === "baseline") {
    if (showIllustrativeBaseline && baseline) {
      explanationSummary = `Illustrative Baseline Demonstration: In this mock baseline record, recorded contributing factors are high transfer amount, unusual location, and new device. In the local synthetic prototype, transaction amount is the strongest contributing feature. Active session backend tools have not evaluated a transaction for this response.`;
    } else {
      explanationSummary = `No active transaction or audit evidence was returned in the latest assistant response. Ask a question such as "Analyze transaction TX1001" or "Check the blockchain audit record for transaction TX1001" to inspect explainable AI attributions.`;
    }
  }

  return (
    <div className="flex flex-col gap-6">
      {/* Header Section */}
      <div className="rounded-xl border border-black/10 bg-white p-5 shadow-xs dark:border-white/15 dark:bg-neutral-900">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-blue-600 dark:text-blue-400">
            {session.mode === "audit" ? (
              <>
                <ShieldIcon className="size-4" />
                <span>Blockchain Audit Verification</span>
              </>
            ) : session.mode === "portfolio" ? (
              <>
                <BarChartIcon className="size-4" />
                <span>Portfolio Analysis</span>
              </>
            ) : (
              <>
                <BarChartIcon className="size-4" />
                <span>Explainable AI (XAI) Attribution</span>
              </>
            )}
          </div>

          {/* Provenance Badge */}
          {session.mode === "audit" && audit && (
            <span
              className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold ${
                audit.provenanceType === "live"
                  ? "border-emerald-300 bg-emerald-50 text-emerald-800 dark:border-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300"
                  : "border-amber-300 bg-amber-50 text-amber-800 dark:border-amber-800 dark:bg-amber-950/60 dark:text-amber-300"
              }`}
            >
              <span
                className={`size-2 rounded-full ${audit.provenanceType === "live" ? "bg-emerald-500 animate-pulse" : "bg-amber-500"}`}
              ></span>
              {audit.provenanceLabel}
            </span>
          )}

          {session.mode === "fraud" && fraud && (
            <span
              className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold ${
                fraud.provenanceType === "live"
                  ? "border-emerald-300 bg-emerald-50 text-emerald-800 dark:border-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300"
                  : "border-amber-300 bg-amber-50 text-amber-800 dark:border-amber-800 dark:bg-amber-950/60 dark:text-amber-300"
              }`}
            >
              <span
                className={`size-2 rounded-full ${fraud.provenanceType === "live" ? "bg-emerald-500 animate-pulse" : "bg-amber-500"}`}
              ></span>
              {fraud.provenanceLabel}
            </span>
          )}

          {session.mode === "fraud_and_audit" && (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-blue-300 bg-blue-50 px-3 py-1 text-xs font-semibold text-blue-800 dark:border-blue-800 dark:bg-blue-950/60 dark:text-blue-300">
              <span className="size-2 rounded-full bg-blue-500"></span>
              Active Session Backend Evidence
            </span>
          )}

          {session.mode === "portfolio" && portfolio && (
            <span
              className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold ${
                portfolio.provenanceType === "live"
                  ? "border-emerald-300 bg-emerald-50 text-emerald-800 dark:border-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300"
                  : "border-amber-300 bg-amber-50 text-amber-800 dark:border-amber-800 dark:bg-amber-950/60 dark:text-amber-300"
              }`}
            >
              <span
                className={`size-2 rounded-full ${portfolio.provenanceType === "live" ? "bg-emerald-500 animate-pulse" : "bg-amber-500"}`}
              ></span>
              {portfolio.provenanceLabel}
            </span>
          )}

          {session.mode === "baseline" && (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-neutral-300 bg-neutral-100 px-3 py-1 text-xs font-semibold text-neutral-700 dark:border-neutral-700 dark:bg-neutral-800 dark:text-neutral-300">
              <span className="size-2 rounded-full bg-neutral-400"></span>
              Mock Baseline — No Active Session Evidence
            </span>
          )}
        </div>

        <h2 className="mt-2 text-2xl font-bold text-neutral-900 dark:text-white">
          {session.mode === "audit" && audit
            ? `Blockchain Audit Record: Transaction ${audit.transactionId}`
            : session.mode === "fraud_and_audit" && fraud
            ? `Analysis & Audit Record: Transaction ${fraud.transactionId}`
            : session.mode === "portfolio" && portfolio
            ? portfolio.isFallback
              ? `Mock Baseline Portfolio: ${portfolio.portfolioId} (Demonstration)`
              : `Portfolio Overview: ${portfolio.portfolioId}`
            : session.mode === "fraud" && fraud
            ? `Why was ${fraud.transactionId} ${fraud.status.toLowerCase()}?`
            : showIllustrativeBaseline && baseline
            ? `Illustrative Mock Baseline: ${baseline.transactionId} (Demonstration)`
            : "Explainable AI (XAI) Attribution"}
        </h2>
        <p className="mt-1 text-sm text-neutral-500">
          {session.mode === "audit"
            ? "Blockchain ledger audit trail & smart-contract bridge verification status"
            : session.mode === "portfolio"
            ? portfolio?.isFallback
              ? "Illustrative mock baseline estimates — No live portfolio record retrieved from platform"
              : "Asset allocation, market valuation, and portfolio risk breakdown"
            : session.mode === "baseline"
            ? showIllustrativeBaseline
              ? "Illustrative Baseline Example — Sample data for demonstration. No active session evidence evaluated."
              : "No active session evidence available. Submit a transaction or audit query in chat to inspect live explanations."
            : "AI explanation based on model evidence & SHAP feature contributions"}
        </p>
      </div>

      {/* Top Summary Card */}
      {session.mode === "audit" && audit ? (
        <div className="rounded-xl border border-blue-200/80 bg-blue-50/40 p-5 shadow-xs overflow-hidden dark:border-blue-900/40 dark:bg-neutral-900">
          <div className="mb-3 text-xs font-bold uppercase tracking-wider text-blue-700 dark:text-blue-400">
            Blockchain Audit Overview
          </div>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-5">
            <div className="min-w-0">
              <span className="text-xs text-neutral-500">Transaction ID</span>
              <p className="font-mono text-base font-bold text-neutral-900 dark:text-white break-words [overflow-wrap:anywhere]">
                {audit.transactionId}
              </p>
            </div>
            <div className="min-w-0">
              <span className="text-xs text-neutral-500">Audit ID</span>
              <p className="font-mono text-base font-bold text-neutral-900 dark:text-white break-words [overflow-wrap:anywhere]">
                {audit.auditId || "RECORDED"}
              </p>
            </div>
            <div className="min-w-0">
              <span className="text-xs text-neutral-500">Recorded Decision</span>
              <div className="mt-1">
                <span className="inline-block max-w-full rounded-md bg-blue-600 px-2.5 py-1 text-xs font-bold text-white break-words [overflow-wrap:anywhere] whitespace-normal">
                  {audit.decision}
                </span>
              </div>
            </div>
            <div className="min-w-0">
              <span className="text-xs text-neutral-500">Block Number</span>
              <p className="font-mono text-base font-bold text-neutral-900 dark:text-white break-words [overflow-wrap:anywhere]">
                {audit.blockNumber != null ? `#${audit.blockNumber}` : "Simulated Ledger"}
              </p>
            </div>
            <div className="min-w-0 sm:col-span-2 md:col-span-2 xl:col-span-1">
              <span className="text-xs text-neutral-500">Record Status</span>
              <p className="text-base font-bold text-neutral-900 dark:text-white break-words [overflow-wrap:anywhere]">
                {audit.recordStatus}
              </p>
              <p className="mt-0.5 text-[11px] text-neutral-400 break-words [overflow-wrap:anywhere]">
                Independent blockchain verification not established.
              </p>
            </div>
          </div>
        </div>
      ) : session.mode === "portfolio" && portfolio ? (
        portfolio.isFallback ? (
          <div className="rounded-xl border border-amber-300/80 bg-amber-50/50 p-5 shadow-xs overflow-hidden dark:border-amber-900/40 dark:bg-neutral-900">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <span className="text-xs font-bold uppercase tracking-wider text-amber-800 dark:text-amber-400">
                Mock Baseline Portfolio Overview (No Live Record)
              </span>
              <span className="rounded-full bg-amber-100 px-2.5 py-0.5 text-[11px] font-bold text-amber-900 dark:bg-amber-950 dark:text-amber-300">
                Fallback Mock Baseline
              </span>
            </div>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <div className="min-w-0">
                <span className="text-xs text-neutral-500">Target User / Portfolio ID</span>
                <p className="font-mono text-base font-bold text-neutral-900 dark:text-white break-words [overflow-wrap:anywhere]">{portfolio.portfolioId}</p>
                <p className="mt-0.5 text-[11px] text-amber-700 dark:text-amber-400">Platform lookup: Not found</p>
              </div>
              <div className="min-w-0">
                <span className="text-xs text-neutral-500">Live Market Valuation</span>
                <p className="text-base font-bold text-neutral-500 dark:text-neutral-400 break-words [overflow-wrap:anywhere]">
                  Unverified (No Record)
                </p>
                <p className="mt-0.5 text-[11px] text-neutral-500">Sample baseline: {portfolio.currency} {portfolio.totalValue}</p>
              </div>
              <div className="min-w-0">
                <span className="text-xs text-neutral-500">Verified Live Holdings</span>
                <p className="text-base font-bold text-neutral-900 dark:text-white break-words [overflow-wrap:anywhere]">
                  0 live assets
                </p>
                <p className="mt-0.5 text-[11px] text-neutral-500">No verified holdings on platform</p>
              </div>
              <div className="min-w-0">
                <span className="text-xs text-neutral-500">Record Status</span>
                <div className="mt-1">
                  <span className="inline-block rounded-md bg-amber-600 px-2.5 py-0.5 text-xs font-bold text-white">
                    FALLBACK / NOT FOUND
                  </span>
                </div>
                <p className="mt-0.5 text-[11px] text-neutral-500 break-words [overflow-wrap:anywhere]">{portfolio.fallbackReason || "No live portfolio retrieved"}</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="rounded-xl border border-blue-200/80 bg-blue-50/40 p-5 shadow-xs overflow-hidden dark:border-blue-900/40 dark:bg-neutral-900">
            <div className="mb-3 text-xs font-bold uppercase tracking-wider text-blue-700 dark:text-blue-400">
              Live Portfolio Overview
            </div>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <div className="min-w-0">
                <span className="text-xs text-neutral-500">Portfolio ID</span>
                <p className="font-mono text-base font-bold text-neutral-900 dark:text-white break-words [overflow-wrap:anywhere]">{portfolio.portfolioId}</p>
              </div>
              <div className="min-w-0">
                <span className="text-xs text-neutral-500">Total Valuation</span>
                <p className="text-base font-bold text-neutral-900 dark:text-white break-words [overflow-wrap:anywhere]">
                  {portfolio.currency} {portfolio.totalValue}
                </p>
              </div>
              <div className="min-w-0">
                <span className="text-xs text-neutral-500">Holdings Count</span>
                <p className="text-base font-bold text-blue-600 dark:text-blue-400 break-words [overflow-wrap:anywhere]">{portfolio.holdingCount} assets</p>
              </div>
              <div className="min-w-0">
                <span className="text-xs text-neutral-500">Service Status</span>
                <div className="mt-1">
                  <span className="inline-block rounded-md bg-emerald-600 px-2.5 py-0.5 text-xs font-bold text-white">
                    ACTIVE
                  </span>
                </div>
              </div>
            </div>
          </div>
        )
      ) : fraud ? (
        <div className="flex flex-col gap-6">
          {/* Top Transaction Overview */}
          <div className="rounded-xl border border-red-200/80 bg-red-50/40 p-5 shadow-xs overflow-hidden dark:border-red-900/40 dark:bg-neutral-900">
            <div className="mb-3 text-xs font-bold uppercase tracking-wider text-red-700 dark:text-red-400">
              Transaction Overview
            </div>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-5">
              <div className="min-w-0">
                <span className="text-xs text-neutral-500">Transaction ID</span>
                <p className="font-mono text-base font-bold text-neutral-900 dark:text-white break-words [overflow-wrap:anywhere]">
                  {fraud.transactionId}
                </p>
              </div>
              <div className="min-w-0">
                <span className="text-xs text-neutral-500">Amount</span>
                <p className="text-base font-bold text-neutral-900 dark:text-white break-words [overflow-wrap:anywhere]">
                  {fraud.amount != null ? `${fraud.currency} ${fraud.amount.toLocaleString()}` : "Not in audit trail"}
                </p>
              </div>
              <div className="min-w-0">
                <span className="text-xs text-neutral-500">Recorded Result</span>
                <div className="mt-1">
                  <span className="inline-block max-w-full rounded-md bg-red-600 px-2.5 py-0.5 text-xs font-bold text-white break-words [overflow-wrap:anywhere] whitespace-normal">
                    {fraud.status}
                  </span>
                </div>
              </div>
              <div className="min-w-0">
                <span className="text-xs text-neutral-500">Risk Level</span>
                <p className="text-base font-bold text-red-600 dark:text-red-400 break-words [overflow-wrap:anywhere]">
                  {fraud.riskLevel}
                </p>
              </div>
              <div className="min-w-0">
                <span className="text-xs text-neutral-500">Recorded Risk Score</span>
                <p className="text-base font-bold text-red-600 dark:text-red-400 break-words [overflow-wrap:anywhere]">
                  {typeof fraud.fraudScore === "number" ? fraud.fraudScore.toFixed(2) : fraud.fraudScore ?? "N/A"}
                </p>
                <p className="mt-0.5 text-[11px] text-neutral-400 break-words [overflow-wrap:anywhere]">
                  This score has not been validated as a probability of fraud.
                </p>
              </div>
            </div>
          </div>

          {/* Preferred Plain-English Explanation Card */}
          <div className="rounded-xl border border-black/10 bg-white p-6 shadow-xs dark:border-white/15 dark:bg-neutral-900">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-neutral-100 pb-3 dark:border-neutral-800">
              <div className="flex items-center gap-2">
                <SparklesIcon className="size-5 text-blue-600" />
                <h3 className="text-base font-bold text-neutral-900 dark:text-white">
                  Payment Decision Explanation
                </h3>
              </div>
              <span
                className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold ${
                  fraud.provenanceType === "live"
                    ? "border-emerald-300 bg-emerald-50 text-emerald-800 dark:border-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300"
                    : "border-amber-300 bg-amber-50 text-amber-800 dark:border-amber-800 dark:bg-amber-950/60 dark:text-amber-300"
                }`}
              >
                <span
                  className={`size-2 rounded-full ${fraud.provenanceType === "live" ? "bg-emerald-500 animate-pulse" : "bg-amber-500"}`}
                ></span>
                {fraud.provenanceLabel}
              </span>
            </div>

            <div className="mt-5 space-y-6">
              {/* Section 1: Why was your payment blocked? */}
              <div>
                <h4 className="text-sm font-bold text-neutral-900 dark:text-white">
                  {fraud.status.toUpperCase() === "BLOCK" || fraud.status.toUpperCase() === "BLOCKED"
                    ? "Why was your payment blocked?"
                    : `Why was your payment evaluated as ${fraud.status.toLowerCase()}?`}
                </h4>
                <p className="mt-2 text-sm leading-relaxed text-neutral-700 dark:text-neutral-300">
                  {fraud.whyBlockedPlainEnglish ||
                    "Your payment was blocked because it triggered the system's security rules. The recorded decision shows that the risk score crossed the blocking threshold and an amount-related rule also contributed to the decision."}
                </p>
              </div>

              {/* Section 2: What contributed to the decision? */}
              <div className="border-t border-neutral-100 pt-4 dark:border-neutral-800">
                <h4 className="text-sm font-bold text-neutral-900 dark:text-white">
                  What contributed to the decision?
                </h4>
                <ul className="mt-2 space-y-2 text-sm leading-relaxed text-neutral-700 dark:text-neutral-300">
                  {fraud.rulesPlainEnglish && fraud.rulesPlainEnglish.map((rule, idx) => (
                    <li key={`rule-${idx}`} className="flex items-start gap-2">
                      <span className="mt-1 text-blue-500">•</span>
                      <span>{rule}</span>
                    </li>
                  ))}
                  {fraud.factorsPlainEnglish && fraud.factorsPlainEnglish.map((factor, idx) => (
                    <li key={`factor-${idx}`} className="flex items-start gap-2">
                      <span className="mt-1 text-blue-500">•</span>
                      <span>{factor}</span>
                    </li>
                  ))}
                  <li className="flex items-start gap-2">
                    <span className="mt-1 text-blue-500">•</span>
                    <span>
                      The system assigned a risk score of{" "}
                      <strong className="text-neutral-900 dark:text-white">
                        {typeof fraud.fraudScore === "number" ? fraud.fraudScore.toFixed(2) : fraud.fraudScore ?? "1.00"}
                      </strong>
                      , which it classified as {fraud.riskLevel.toLowerCase()} risk.
                      {fraud.streamSignals
                        ? ` The behavioral pattern stream (LSTM) reported ${(fraud.streamSignals.behavioralScore ?? 1.0).toFixed(2)}, and the entity relationship stream (GNN) reported ${(fraud.streamSignals.graphScore ?? 0.0).toFixed(2)}.`
                        : ""}
                    </span>
                  </li>
                </ul>
              </div>

              {/* Section 3: What information was unavailable? */}
              <div className="border-t border-neutral-100 pt-4 dark:border-neutral-800">
                <h4 className="text-sm font-bold text-neutral-900 dark:text-white">
                  What information was unavailable?
                </h4>
                <ul className="mt-2 space-y-1.5 text-sm leading-relaxed text-neutral-600 dark:text-neutral-400">
                  {fraud.missingInformationPlainEnglish && fraud.missingInformationPlainEnglish.map((info, idx) => (
                    <li key={`missing-${idx}`} className="flex items-start gap-2">
                      <span className="mt-1 text-neutral-400">•</span>
                      <span>{info}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Section 4: What does this result mean? */}
              <div className="border-t border-neutral-100 pt-4 dark:border-neutral-800">
                <h4 className="text-sm font-bold text-neutral-900 dark:text-white">
                  What does this result mean?
                </h4>
                <p className="mt-2 text-sm leading-relaxed text-neutral-700 dark:text-neutral-300">
                  {fraud.resultMeaningPlainEnglish ||
                    "This does not prove that the payment was fraudulent. The score has not been validated as a probability of fraud. Device and location information were not available in the recorded evidence."}
                </p>
              </div>

              {/* Section 5: Technical details — expandable section */}
              <details className="mt-4 rounded-lg border border-neutral-200 bg-neutral-50/70 p-4 transition-all dark:border-neutral-800 dark:bg-neutral-800/40">
                <summary className="cursor-pointer text-xs font-bold uppercase tracking-wider text-neutral-700 hover:text-blue-600 dark:text-neutral-300 dark:hover:text-blue-400">
                  Technical details (original rule identifiers, model outputs, engine version)
                </summary>
                <div className="mt-4 space-y-4 text-xs">
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 md:grid-cols-3">
                    <div className="rounded-md border border-neutral-200 bg-white p-3 dark:border-neutral-700 dark:bg-neutral-900">
                      <span className="text-neutral-500">Original Rule Identifiers</span>
                      <p className="mt-1 font-mono font-bold text-neutral-900 dark:text-white break-words">
                        {fraud.factors.length > 0 ? fraud.factors.join("; ") : "amount_hard_block; score_above_block_threshold"}
                      </p>
                    </div>
                    <div className="rounded-md border border-neutral-200 bg-white p-3 dark:border-neutral-700 dark:bg-neutral-900">
                      <span className="text-neutral-500">Model Version & Benchmark Status</span>
                      <p className="mt-1 font-mono font-bold text-neutral-900 dark:text-white break-words">
                        {fraud.modelVersion || "heuristic-untrained"}{" "}
                        <span className="font-normal text-neutral-500">
                          ({fraud.benchmarkModel ? "Trained benchmark model" : "Untrained heuristic rule set, not a benchmark model"})
                        </span>
                      </p>
                    </div>
                    <div className="rounded-md border border-neutral-200 bg-white p-3 dark:border-neutral-700 dark:bg-neutral-900">
                      <span className="text-neutral-500">Dual-Stream Model Outputs</span>
                      <p className="mt-1 font-mono text-neutral-900 dark:text-white">
                        LSTM (behavioral stream): {fraud.streamSignals?.behavioralScore?.toFixed(4) ?? "N/A"}<br />
                        GNN (relational stream): {fraud.streamSignals?.graphScore?.toFixed(4) ?? "N/A"}
                      </p>
                    </div>
                  </div>

                  {fraud.shapContributions.length > 0 && (
                    <div>
                      <p className="font-bold text-neutral-700 dark:text-neutral-300">SHAP Attributions (Local Prototype)</p>
                      <div className="mt-2 space-y-2">
                        {fraud.shapContributions.map((item) => (
                          <div key={item.feature} className="rounded border border-neutral-200 bg-white p-2.5 dark:border-neutral-700 dark:bg-neutral-900">
                            <div className="flex justify-between">
                              <span className="font-medium text-neutral-800 dark:text-neutral-200">{item.label}</span>
                              <span className="font-mono text-neutral-600 dark:text-neutral-400">+{item.shapValue.toFixed(4)} ({item.strength})</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {fraud.rawText && (
                    <div>
                      <p className="font-bold text-neutral-700 dark:text-neutral-300">Raw Tool Evidence</p>
                      <pre className="mt-1.5 max-h-48 overflow-auto rounded bg-neutral-100 p-2.5 font-mono text-[11px] leading-relaxed text-neutral-800 dark:bg-neutral-950 dark:text-neutral-300">
                        {fraud.rawText}
                      </pre>
                    </div>
                  )}
                </div>
              </details>
            </div>
          </div>
        </div>
      ) : session.mode === "baseline" ? (
        showIllustrativeBaseline && baseline ? (
          <div className="rounded-xl border border-neutral-300 bg-neutral-50/80 p-5 shadow-xs dark:border-neutral-700 dark:bg-neutral-900">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs font-bold uppercase tracking-wider text-neutral-600 dark:text-neutral-400">
                  Illustrative Baseline Example — Transaction Overview
                </span>
                <span className="rounded-full bg-amber-100 px-2.5 py-0.5 text-[11px] font-bold text-amber-800 dark:bg-amber-950 dark:text-amber-300">
                  Sample Demonstration (Not Active Evidence)
                </span>
              </div>
              <button
                onClick={() => setShowIllustrativeBaseline(false)}
                className="text-xs font-semibold text-blue-600 hover:underline dark:text-blue-400"
              >
                Hide Baseline Example
              </button>
            </div>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-5">
              <div className="min-w-0">
                <span className="text-xs text-neutral-500">Sample Transaction</span>
                <p className="font-mono text-base font-bold text-neutral-900 dark:text-white break-words [overflow-wrap:anywhere]">
                  {baseline.transactionId} <span className="text-xs font-normal text-neutral-500">(Sample)</span>
                </p>
              </div>
              <div className="min-w-0">
                <span className="text-xs text-neutral-500">Illustrative Amount</span>
                <p className="text-base font-bold text-neutral-900 dark:text-white break-words [overflow-wrap:anywhere]">
                  {baseline.amount != null ? `${baseline.currency} ${baseline.amount.toLocaleString()}` : "N/A"}
                </p>
              </div>
              <div className="min-w-0">
                <span className="text-xs text-neutral-500">Illustrative Decision</span>
                <div className="mt-1">
                  <span className="inline-block max-w-full rounded-md bg-neutral-600 px-2.5 py-0.5 text-xs font-bold text-white break-words [overflow-wrap:anywhere] whitespace-normal">
                    {baseline.status} (Sample)
                  </span>
                </div>
              </div>
              <div className="min-w-0">
                <span className="text-xs text-neutral-500">Illustrative Risk Level</span>
                <p className="text-base font-bold text-neutral-700 dark:text-neutral-300 break-words [overflow-wrap:anywhere]">{baseline.riskLevel}</p>
              </div>
              <div className="min-w-0">
                <span className="text-xs text-neutral-500">Illustrative Score</span>
                <p className="text-base font-bold text-neutral-700 dark:text-neutral-300 break-words [overflow-wrap:anywhere]">
                  {baseline.fraudScore}
                </p>
                <p className="mt-0.5 text-[11px] text-neutral-400 break-words [overflow-wrap:anywhere]">Demonstration score; probability calibration not established.</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="rounded-xl border border-neutral-200/80 bg-neutral-50/60 p-6 shadow-xs dark:border-neutral-800 dark:bg-neutral-900">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:gap-5">
              <div className="flex size-12 shrink-0 items-center justify-center rounded-xl bg-blue-100 text-blue-600 dark:bg-blue-950/80 dark:text-blue-400">
                <SparklesIcon className="size-6" />
              </div>
              <div className="flex-1">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <h3 className="text-base font-semibold text-neutral-900 dark:text-white">
                      No Active Session Evidence
                    </h3>
                    <span className="rounded-full bg-neutral-200/80 px-2.5 py-0.5 text-[11px] font-semibold text-neutral-700 dark:bg-neutral-800 dark:text-neutral-300">
                      General Response (No Tools Invoked)
                    </span>
                  </div>
                  <button
                    onClick={() => setShowIllustrativeBaseline(true)}
                    className="rounded-lg border border-neutral-300 bg-white px-3 py-1 text-xs font-semibold text-neutral-700 shadow-2xs hover:bg-neutral-50 dark:border-neutral-700 dark:bg-neutral-800 dark:text-neutral-200 dark:hover:bg-neutral-700"
                  >
                    Preview Mock Baseline Example
                  </button>
                </div>
                <p className="mt-2 text-sm leading-relaxed text-neutral-600 dark:text-neutral-400">
                  The latest assistant response was a general message (such as a greeting) and did not invoke financial analysis tools. Active transaction explanations, fraud risk scores, and SHAP feature attributions will appear here when a specific transaction or audit query is executed.
                </p>
                <div className="mt-4 flex flex-col gap-2 rounded-lg border border-neutral-200 bg-white p-3.5 dark:border-neutral-800 dark:bg-neutral-800/40">
                  <span className="text-xs font-bold uppercase tracking-wider text-neutral-500">
                    Try asking in chat:
                  </span>
                  <div className="flex flex-wrap gap-2 text-xs">
                    <span className="rounded-md bg-neutral-100 px-2.5 py-1 font-mono text-neutral-800 dark:bg-neutral-800 dark:text-neutral-200">
                      &ldquo;Analyze transaction TX1001&rdquo;
                    </span>
                    <span className="rounded-md bg-neutral-100 px-2.5 py-1 font-mono text-neutral-800 dark:bg-neutral-800 dark:text-neutral-200">
                      &ldquo;Check the blockchain audit record for transaction TX1001&rdquo;
                    </span>
                    <span className="rounded-md bg-neutral-100 px-2.5 py-1 font-mono text-neutral-800 dark:bg-neutral-800 dark:text-neutral-200">
                      &ldquo;Analyze my portfolio USER001&rdquo;
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )
      ) : null}

      {/* Middle Section: Audit Details (for Audit-Only) OR Model Attributions (for Fraud/Baseline) */}
      {session.mode === "audit" && audit ? (
        <div className="rounded-xl border border-black/10 bg-white p-5 shadow-xs overflow-hidden dark:border-white/15 dark:bg-neutral-900">
          <div className="mb-4 flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
            <div className="min-w-0">
              <h3 className="text-base font-semibold text-neutral-900 dark:text-white break-words">
                Blockchain Ledger Verification Details
              </h3>
              <p className="text-xs text-neutral-500">
                Smart-contract audit trail retrieved from Component 3 bridge
              </p>
            </div>
            <span className="w-fit shrink-0 rounded-full bg-blue-100 px-3 py-1 text-xs font-bold text-blue-700 dark:bg-blue-950 dark:text-blue-300">
              Blockchain Audit Trail
            </span>
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            <div className="min-w-0 rounded-lg border border-neutral-200 bg-neutral-50/80 p-4 overflow-hidden dark:border-neutral-800 dark:bg-neutral-800/60">
              <span className="text-xs text-neutral-500">Audit Reference</span>
              <p className="mt-1 font-mono text-base font-bold text-neutral-900 dark:text-white break-words [overflow-wrap:anywhere]">
                {audit.auditId || "N/A"}
              </p>
              <p className="mt-1 text-[11px] text-neutral-500 break-words [overflow-wrap:anywhere]">
                Unique bridge audit record identifier
              </p>
            </div>

            <div className="min-w-0 rounded-lg border border-neutral-200 bg-neutral-50/80 p-4 overflow-hidden dark:border-neutral-800 dark:bg-neutral-800/60">
              <span className="text-xs text-neutral-500">Ledger Block & Timestamp</span>
              <p className="mt-1 font-mono text-base font-bold text-neutral-900 dark:text-white break-words [overflow-wrap:anywhere]">
                {audit.blockNumber != null ? `Block #${audit.blockNumber}` : "Simulated Ledger"}
              </p>
              <p className="mt-1 text-[11px] text-neutral-500 break-words [overflow-wrap:anywhere]">
                {audit.recordedAt || "Timestamp in ledger"}
              </p>
            </div>

            <div className="min-w-0 rounded-lg border border-neutral-200 bg-neutral-50/80 p-4 overflow-hidden sm:col-span-2 lg:col-span-1 dark:border-neutral-800 dark:bg-neutral-800/60">
              <span className="text-xs text-neutral-500">Record Status & Ledger Type</span>
              <p className="mt-1 font-mono text-base font-bold text-blue-600 dark:text-blue-400 break-words [overflow-wrap:anywhere]">
                {audit.recordStatus}
              </p>
              <p className="mt-1 text-[11px] text-neutral-500 break-words [overflow-wrap:anywhere]">
                {audit.isFallback
                  ? "Local mock baseline fallback"
                  : audit.isMockLedger
                  ? "Simulated mock ledger adapter"
                  : "Verified ledger record"}
              </p>
            </div>
          </div>

          <p className="mt-3 text-xs leading-relaxed text-neutral-500 break-words [overflow-wrap:anywhere] dark:text-neutral-400">
            The blockchain audit record was retrieved from the Component 3 audit bridge.{" "}
            {audit.isFallback
              ? "The live bridge service at http://localhost:8002 was queried, but no audit record was found on the live bridge for this transaction ID; local mock baseline data was returned. "
              : ""}
            Independent blockchain verification is not established. This audit record reflects ledger bridge
            commitments and does not evaluate fraud risk scores or SHAP feature attributions.
          </p>
        </div>
      ) : session.mode === "portfolio" && portfolio ? (
        !personalizedContext ? (
          <div className="rounded-xl border border-amber-300 bg-amber-50/80 p-5 shadow-xs dark:border-amber-900/60 dark:bg-amber-950/30">
            <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-amber-800 dark:text-amber-300">
              <ShieldIcon className="size-4" />
              <span>Personalized Context Disabled</span>
            </div>
            <h3 className="mt-2 text-base font-semibold text-neutral-900 dark:text-white">
              Portfolio Metrics & Allocations Suppressed
            </h3>
            <p className="mt-1 text-xs leading-relaxed text-neutral-700 dark:text-neutral-300">
              Personalized financial context is disabled by user preference. Account-specific portfolio lookup, illustrative mock valuations, synthetic risk scores, and asset allocation breakdowns are blocked for this request, and no personalized portfolio evidence was accessed.
            </p>
            <p className="mt-2 text-[11px] text-neutral-500 dark:text-neutral-400">
              You can re-enable personalized context anytime in the Responsible AI settings tab.
            </p>
          </div>
        ) : (
          <div className="rounded-xl border border-black/10 bg-white p-5 shadow-xs overflow-hidden dark:border-white/15 dark:bg-neutral-900">
            <div className="mb-4 flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
              <div className="min-w-0">
                <h3 className="text-base font-semibold text-neutral-900 dark:text-white break-words">
                  {portfolio.isFallback
                    ? "Illustrative Mock Baseline Portfolio Breakdown"
                    : "Live Portfolio Allocation & Breakdown"}
                </h3>
                <p className="text-xs text-neutral-500">
                  {portfolio.isFallback
                    ? "Synthetic mock estimates shown for system demonstration — Not verified user assets"
                    : "Live platform portfolio asset distribution"}
                </p>
              </div>
              <span
                className={`w-fit shrink-0 rounded-full px-3 py-1 text-xs font-bold ${
                  portfolio.isFallback
                    ? "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300"
                    : "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300"
                }`}
              >
                {portfolio.isFallback ? "Tool Evidence — Synthetic Demo Data" : "Live Portfolio Data"}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
              {/* 1. Valuation Card */}
              <div className="min-w-0 rounded-lg border border-neutral-200 bg-neutral-50/80 p-3.5 dark:border-neutral-800 dark:bg-neutral-800/60">
                <span className="text-xs text-neutral-500">
                  {portfolio.isFallback ? "Illustrative Mock Valuation" : "Recorded Valuation"}
                </span>
                <p className="mt-1 text-base font-bold text-neutral-900 dark:text-white break-words [overflow-wrap:anywhere]">
                  {portfolio.totalValue !== "N/A" ? `${portfolio.currency} ${portfolio.totalValue}` : "Unverified"}
                </p>
                <p className="mt-0.5 text-[11px] text-neutral-400 break-words [overflow-wrap:anywhere]">
                  {portfolio.isFallback ? "Sample demo valuation (Not actual balance)" : `${portfolio.holdingCount} verified holdings`}
                </p>
              </div>

              {/* 2. Expected Annual Return Card */}
              <div className="min-w-0 rounded-lg border border-neutral-200 bg-neutral-50/80 p-3.5 dark:border-neutral-800 dark:bg-neutral-800/60">
                <span className="text-xs text-neutral-500">Expected Annual Return</span>
                <p className="mt-1 text-base font-bold text-neutral-900 dark:text-white break-words [overflow-wrap:anywhere]">
                  {portfolio.expectedAnnualReturn != null ? `${portfolio.expectedAnnualReturn}%` : "N/A"}
                </p>
                <p className="mt-0.5 text-[11px] text-neutral-400 break-words [overflow-wrap:anywhere]">
                  {portfolio.isFallback ? "Illustrative mock baseline" : "Projected annual return"}
                </p>
              </div>

              {/* 3. Risk Score Card */}
              <div className="min-w-0 rounded-lg border border-neutral-200 bg-neutral-50/80 p-3.5 dark:border-neutral-800 dark:bg-neutral-800/60">
                <span className="text-xs text-neutral-500">
                  {portfolio.isFallback ? "Illustrative Risk Score" : "Risk Score"}
                </span>
                <p className="mt-1 text-base font-bold text-neutral-900 dark:text-white break-words [overflow-wrap:anywhere]">
                  {portfolio.riskScore != null ? `${portfolio.riskScore}/100` : "N/A"}
                </p>
                <p className="mt-0.5 text-[11px] text-neutral-400 break-words [overflow-wrap:anywhere]">
                  {portfolio.isFallback ? `Illustrative mock (${portfolio.riskLevel ?? "Moderate"}) — Not user risk` : portfolio.riskLevel ?? "Evaluated"}
                </p>
              </div>

              {/* 4. Diversification Score Card */}
              <div className="min-w-0 rounded-lg border border-neutral-200 bg-neutral-50/80 p-3.5 dark:border-neutral-800 dark:bg-neutral-800/60">
                <span className="text-xs text-neutral-500">Diversification Score</span>
                <p className="mt-1 text-base font-bold text-neutral-900 dark:text-white break-words [overflow-wrap:anywhere]">
                  {portfolio.diversificationScore != null ? `${portfolio.diversificationScore}/100` : "N/A"}
                </p>
                <p className="mt-0.5 text-[11px] text-neutral-400 break-words [overflow-wrap:anywhere]">
                  {portfolio.isFallback ? "Illustrative mock baseline" : "Cross-sector spread"}
                </p>
              </div>

              {/* 5. Liquidity Card */}
              <div className="min-w-0 rounded-lg border border-neutral-200 bg-neutral-50/80 p-3.5 dark:border-neutral-800 dark:bg-neutral-800/60">
                <span className="text-xs text-neutral-500">Liquidity</span>
                <p className="mt-1 text-base font-bold text-neutral-900 dark:text-white break-words [overflow-wrap:anywhere]">
                  {portfolio.liquidity ?? "N/A"}
                </p>
                <p className="mt-0.5 text-[11px] text-neutral-400 break-words [overflow-wrap:anywhere]">
                  {portfolio.isFallback ? "Illustrative mock baseline" : "Cash & near-cash buffer"}
                </p>
              </div>
            </div>

            {portfolio.allocation && Object.keys(portfolio.allocation).length > 0 && (
              <div className="mt-4">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-xs font-semibold uppercase tracking-wider text-neutral-600 dark:text-neutral-400">
                    {portfolio.isFallback ? "Illustrative Mock Asset Allocation (Sample Only)" : "Asset Allocation Breakdown"}
                  </span>
                  {portfolio.isFallback && (
                    <span className="text-[11px] text-amber-700 dark:text-amber-400">
                      Not actual user holdings
                    </span>
                  )}
                </div>
                <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 md:grid-cols-3">
                  {Object.entries(portfolio.allocation).map(([sector, pct]) => (
                    <div key={sector} className="rounded-lg border border-neutral-200 bg-neutral-50/60 p-2.5 dark:border-neutral-800 dark:bg-neutral-800/40">
                      <div className="flex justify-between text-xs font-medium text-neutral-800 dark:text-neutral-200">
                        <span>{sector}</span>
                        <span className="font-mono font-bold">{pct}%</span>
                      </div>
                      <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-neutral-200 dark:bg-neutral-700">
                        <div
                          className="h-full rounded-full bg-blue-500"
                          style={{ width: `${Math.min(pct, 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {portfolio.isFallback && (
              <p className="mt-4 text-xs leading-relaxed text-amber-800 dark:text-amber-400/90 break-words [overflow-wrap:anywhere]">
                <strong>Notice:</strong> The live platform service was queried, but no portfolio record was found for this user ID. The metrics, risk score, and asset allocations displayed above are synthetic baseline estimates shown strictly for system demonstration purposes. They do not represent verified holdings, actual account balances, or personalized investment advice.
              </p>
            )}

            {latestEvidence.length > 0 && (
              <details className="mt-4 rounded-lg border border-neutral-200 bg-neutral-50/60 p-3 text-xs dark:border-neutral-800 dark:bg-neutral-800/40">
                <summary className="cursor-pointer font-semibold text-blue-700 dark:text-blue-400">
                  View raw portfolio tool evidence ({latestEvidence.filter(e => e.tool === "get_portfolio_analysis").length || latestEvidence.length})
                </summary>
                <div className="mt-2 space-y-2">
                  {latestEvidence
                    .filter((e) => e.tool === "get_portfolio_analysis")
                    .map((ev, idx) => (
                      <div key={idx} className="rounded border border-neutral-200 bg-white p-2.5 font-mono text-[11px] dark:border-neutral-700 dark:bg-neutral-900">
                        <div className="mb-1 font-bold text-neutral-700 dark:text-neutral-300">Tool: {ev.tool}</div>
                        <pre className="whitespace-pre-wrap break-all text-neutral-600 dark:text-neutral-400">{ev.result}</pre>
                      </div>
                    ))}
                </div>
              </details>
            )}
          </div>
        )
      ) : session.mode === "baseline" && showIllustrativeBaseline && baseline && baseline.shapContributions.length > 0 ? (
        <div className="rounded-xl border border-neutral-300 bg-neutral-50/80 p-5 shadow-xs dark:border-neutral-700 dark:bg-neutral-900">
          <div className="mb-4 flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h3 className="text-base font-semibold text-neutral-900 dark:text-white">
                Illustrative Prototype Attributions (Sample Demonstration)
              </h3>
              <p className="text-xs text-neutral-500">
                Local synthetic Random Forest prototype — Sample demonstration baseline
              </p>
            </div>
            <span className="w-fit rounded-full bg-amber-100 px-3 py-1 text-xs font-bold text-amber-800 dark:bg-amber-950 dark:text-amber-300">
              Sample Demonstration (Not Active Evidence)
            </span>
          </div>

          {/* Horizontal Contribution Cards */}
          <div className="flex flex-col gap-3">
            {baseline.shapContributions.map((item, idx) => {
              const barWidthPercent = (item.shapValue / maxShap) * 100;
              return (
                <div
                  key={item.feature}
                  className="rounded-lg border border-neutral-200 bg-white p-4 transition-all hover:border-neutral-400 dark:border-neutral-800 dark:bg-neutral-800/60"
                >
                  <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                    <div className="flex items-center gap-2.5">
                      <span className="flex size-6 items-center justify-center rounded-full bg-neutral-200 text-xs font-bold text-neutral-700 dark:bg-neutral-700 dark:text-neutral-300">
                        {idx + 1}
                      </span>
                      <span className="font-semibold text-neutral-900 dark:text-white">
                        {item.label} <span className="text-xs font-normal text-neutral-500">(Sample)</span>
                      </span>
                    </div>

                    <div className="flex items-center gap-3 text-xs">
                      <span className="font-mono font-bold text-neutral-700 dark:text-neutral-300">
                        Illustrative SHAP: +{item.shapValue.toFixed(4)}
                      </span>
                      <span className="rounded-md bg-neutral-200/80 px-2 py-0.5 text-[11px] text-neutral-700 dark:bg-neutral-700 dark:text-neutral-200">
                        Impact: {item.impact}
                      </span>
                      <span className="rounded-md bg-neutral-100 px-2 py-0.5 text-[11px] font-bold text-neutral-700 dark:bg-neutral-800 dark:text-neutral-300">
                        Strength: {item.strength}
                      </span>
                    </div>
                  </div>

                  {/* Progress bar visualizer */}
                  <div className="mt-3 h-2.5 w-full overflow-hidden rounded-full bg-neutral-200 dark:bg-neutral-700">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-amber-400 to-amber-600"
                      style={{ width: `${barWidthPercent}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>

          <p className="mt-3 text-xs leading-relaxed text-neutral-500 dark:text-neutral-400">
            Notice: These SHAP values explain the local synthetic prototype for illustration only. Active session
            backend tools have not evaluated a transaction for this response.
          </p>
        </div>
      ) : null}

      {/* Simple Natural Language Explanation */}
      <div className="rounded-xl border border-blue-200/80 bg-blue-50/50 p-5 shadow-xs dark:border-blue-900/40 dark:bg-blue-950/20">
        <div className="mb-2 flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-blue-900 dark:text-blue-300">
            <SparklesIcon className="size-4 text-blue-600" />
            <span>Simple explanation</span>
          </div>
          <span
            className={`rounded-full px-2.5 py-0.5 text-[11px] font-bold shadow-2xs ${
              session.mode === "portfolio" && portfolio?.isFallback
                ? "bg-amber-600 text-white dark:bg-amber-700"
                : session.mode === "fraud" && fraud?.provenanceType === "fallback"
                ? "bg-amber-600 text-white dark:bg-amber-700"
                : session.mode === "audit" && audit?.isFallback
                ? "bg-amber-600 text-white dark:bg-amber-700"
                : session.mode === "baseline"
                ? "bg-neutral-600 text-white dark:bg-neutral-700"
                : "bg-blue-600 text-white"
            }`}
          >
            {session.mode === "portfolio" && portfolio
              ? portfolio.isFallback
                ? "Tool evidence — synthetic demo data"
                : "Grounded in live tool evidence"
              : session.mode === "fraud" && fraud
              ? fraud.provenanceType === "fallback"
                ? "Tool evidence — synthetic demo data"
                : "Grounded in live tool evidence"
              : session.mode === "audit" && audit
              ? audit.isFallback
                ? "Tool evidence — synthetic demo data"
                : "Grounded in live tool evidence"
              : session.mode === "baseline"
              ? showIllustrativeBaseline
                ? "Illustrative Demonstration"
                : "No Active Evidence"
              : "Grounded in tool evidence"}
          </span>
        </div>
        <p className="text-sm font-medium leading-relaxed text-neutral-800 dark:text-neutral-200">
          {!personalizedContext && session.mode === "portfolio"
            ? "Notice: Personalized financial context is disabled by user preference. Account-specific portfolio lookup, illustrative mock valuations, and asset holdings analysis are blocked for this request, and no personalized portfolio evidence was accessed."
            : explanationSummary}
        </p>

        {/* Evidence Sources & Next Action */}
        <div className="mt-4 flex flex-col gap-4 border-t border-blue-200/60 pt-4 dark:border-blue-900/60 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="mb-1.5 text-xs font-semibold text-blue-950 dark:text-blue-200">Evidence sources:</p>
            <div className="flex flex-wrap items-center gap-3 text-xs font-medium text-blue-800 dark:text-blue-300">
              {/* Portfolio analysis badge */}
              <span className="flex items-center gap-1">
                {session.mode === "portfolio" && portfolio && portfolio.hasEvidence ? (
                  <>
                    <CheckCircleIcon className="size-4 text-emerald-500" />
                    <span>
                      {portfolio.isFallback
                        ? "Portfolio analysis (synthetic demo fallback)"
                        : "Portfolio analysis (live platform)"}
                    </span>
                  </>
                ) : (
                  <>
                    <svg className="size-4 shrink-0 text-neutral-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <circle cx="12" cy="12" r="8" />
                    </svg>
                    <span className="text-neutral-500 dark:text-neutral-400">Portfolio analysis (not requested)</span>
                  </>
                )}
              </span>

              {/* Fraud analysis badge */}
              <span className="flex items-center gap-1">
                {fraud && fraud.hasEvidence ? (
                  <>
                    <CheckCircleIcon className="size-4 text-emerald-500" />
                    <span>Fraud analysis ({fraud.provenanceType})</span>
                  </>
                ) : session.mode === "baseline" ? (
                  <>
                    <svg className="size-4 shrink-0 text-neutral-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <circle cx="12" cy="12" r="8" />
                    </svg>
                    <span className="text-neutral-500 dark:text-neutral-400">
                      {showIllustrativeBaseline ? "Fraud analysis (baseline demonstration)" : "Fraud analysis (not requested)"}
                    </span>
                  </>
                ) : (
                  <>
                    <svg className="size-4 shrink-0 text-neutral-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <circle cx="12" cy="12" r="8" />
                    </svg>
                    <span className="text-neutral-500 dark:text-neutral-400">Fraud analysis (not requested)</span>
                  </>
                )}
              </span>

              {/* Blockchain audit badge */}
              <span className="flex items-center gap-1">
                {audit && audit.hasEvidence ? (
                  <>
                    <CheckCircleIcon className="size-4 text-emerald-500" />
                    <span>Blockchain audit ({audit.provenanceType})</span>
                  </>
                ) : (
                  <>
                    <svg className="size-4 shrink-0 text-neutral-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <circle cx="12" cy="12" r="8" />
                    </svg>
                    <span className="text-neutral-500 dark:text-neutral-400">Blockchain audit (not requested)</span>
                  </>
                )}
              </span>

              {/* Model attribution badge */}
              <span className="flex items-center gap-1">
                {session.mode === "audit" ? (
                  <>
                    <svg className="size-4 shrink-0 text-neutral-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <circle cx="12" cy="12" r="8" />
                    </svg>
                    <span className="text-neutral-500 dark:text-neutral-400">Model attribution (not applicable for audit)</span>
                  </>
                ) : fraud && (fraud.shapContributions.length > 0 || fraud.streamSignals) ? (
                  <>
                    <CheckCircleIcon className="size-4 text-emerald-500" />
                    <span>Model attribution</span>
                  </>
                ) : session.mode === "baseline" ? (
                  <>
                    <svg className="size-4 shrink-0 text-neutral-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <circle cx="12" cy="12" r="8" />
                    </svg>
                    <span className="text-neutral-500 dark:text-neutral-400">
                      {showIllustrativeBaseline ? "Model attribution (baseline demonstration)" : "Model attribution (not evaluated)"}
                    </span>
                  </>
                ) : (
                  <>
                    <svg className="size-4 shrink-0 text-neutral-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <circle cx="12" cy="12" r="8" />
                    </svg>
                    <span className="text-neutral-500 dark:text-neutral-400">Model attribution (not evaluated)</span>
                  </>
                )}
              </span>
            </div>
          </div>

          <button
            onClick={onNavigateToTrustPanel}
            className="flex items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-xs font-bold text-white shadow-sm transition hover:bg-blue-700"
          >
            <span>View Evidence</span>
            <ChevronRightIcon className="size-4" />
          </button>
        </div>
      </div>

      {/* Helpful Feedback Widget */}
      <div className="flex items-center justify-between rounded-xl border border-black/10 bg-white p-4 shadow-xs dark:border-white/15 dark:bg-neutral-900">
        <span className="text-xs font-medium text-neutral-600 dark:text-neutral-400">
          Was this explanation helpful?
        </span>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setFeedbackGiven("yes")}
            className={`rounded-lg border px-3 py-1.5 text-xs font-semibold transition ${
              feedbackGiven === "yes"
                ? "bg-emerald-600 text-white border-emerald-600"
                : "border-neutral-300 hover:bg-neutral-100 dark:border-neutral-700 dark:hover:bg-neutral-800"
            }`}
          >
            ✓ Yes
          </button>
          <button
            onClick={() => setFeedbackGiven("no")}
            className={`rounded-lg border px-3 py-1.5 text-xs font-semibold transition ${
              feedbackGiven === "no"
                ? "bg-red-600 text-white border-red-600"
                : "border-neutral-300 hover:bg-neutral-100 dark:border-neutral-700 dark:hover:bg-neutral-800"
            }`}
          >
            ✕ No
          </button>
        </div>
      </div>
    </div>
  );
}
