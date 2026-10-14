"use client";

import React, { useState } from "react";
import { MOCK_TRANSACTION } from "./mockData";
import { BarChartIcon, CheckCircleIcon, SparklesIcon, ChevronRightIcon } from "./icons";

interface Props {
  onNavigateToTrustPanel: () => void;
  latestEvidence?: Array<{ tool: string; result: string }>;
}

interface ParsedShapItem {
  feature: string;
  label: string;
  shapValue: number;
  impact: string;
  strength: "High" | "Medium" | "Low";
}

interface ParsedFraudEvidence {
  hasEvidence: boolean;
  provenanceType: "live" | "fallback" | "baseline";
  provenanceLabel: string;
  transactionId: string;
  amount?: number;
  currency: string;
  status: string;
  riskLevel: string;
  fraudScore: number;
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

function parseFraudEvidence(evidenceList: Array<{ tool: string; result: string }> = []): ParsedFraudEvidence {
  const fraudItem = evidenceList.find((e) => e.tool === "get_fraud_analysis");

  if (!fraudItem || !fraudItem.result || !fraudItem.result.trim()) {
    return {
      hasEvidence: false,
      provenanceType: "baseline",
      provenanceLabel: "Mock Baseline (No Active Session Evidence)",
      transactionId: MOCK_TRANSACTION.id,
      amount: MOCK_TRANSACTION.amount,
      currency: MOCK_TRANSACTION.currency,
      status: MOCK_TRANSACTION.status,
      riskLevel: MOCK_TRANSACTION.riskLevel,
      fraudScore: MOCK_TRANSACTION.fraudScore,
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

  const text = fraudItem.result;
  const isLive = /LIVE COMPONENT 2/i.test(text);
  const isFallback = /FALLBACK MOCK DATA/i.test(text);
  const provenanceType: "live" | "fallback" | "baseline" = isLive ? "live" : isFallback ? "fallback" : "baseline";
  const provenanceLabel = isLive
    ? "Live Service Evidence (Component 2 Gateway Audit Log)"
    : "Fallback Mock Data (Component 4 Local Prototype)";

  // Extract transaction ID
  const txIdMatch = text.match(/Transaction ID:\s*([A-Za-z0-9_-]+)/i);
  const txId = txIdMatch ? txIdMatch[1] : MOCK_TRANSACTION.id;

  // Extract amount and currency
  const amountMatch = text.match(/Transaction Amount:\s*([A-Za-z]+)?\s*([0-9,]+(?:\.[0-9]+)?)/i);
  const amount = amountMatch
    ? parseFloat(amountMatch[2].replace(/,/g, ""))
    : isLive
    ? undefined
    : txId === MOCK_TRANSACTION.id
    ? MOCK_TRANSACTION.amount
    : undefined;
  const currency = amountMatch?.[1] || "LKR";

  // Extract fraud score
  const scoreMatch = text.match(/Recorded Fraud Score:\s*([0-9.]+)/i);
  const fraudScore = scoreMatch ? parseFloat(scoreMatch[1]) : MOCK_TRANSACTION.fraudScore;

  // Extract status and risk level
  const statusMatch = text.match(/Recorded Status:\s*([A-Za-z0-9_-]+)/i);
  const status = statusMatch ? statusMatch[1] : MOCK_TRANSACTION.status;

  const riskMatch = text.match(/Recorded Risk Level:\s*([A-Za-z0-9_-]+)/i);
  const riskLevel = riskMatch ? riskMatch[1] : MOCK_TRANSACTION.riskLevel;

  // Extract recorded contributing factors
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

  // Extract SHAP contributions if present
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
  } else if (!isLive && txId === MOCK_TRANSACTION.id) {
    // If not live and matching default TX1001, retain baseline SHAP contributions
    shapContributions.push(
      ...MOCK_TRANSACTION.shapContributions.map((c) => ({
        feature: c.feature,
        label: c.label,
        shapValue: c.shapValue,
        impact: c.impact,
        strength: c.strength,
      }))
    );
  }

  // Extract live stream signals if live
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
    reason: factors.join("; ") || "No specific factors recorded",
    factors: factors.length > 0 ? factors : txId === MOCK_TRANSACTION.id ? MOCK_TRANSACTION.factors : [],
    shapContributions,
    streamSignals,
  };
}

export function ExplanationScreen({ onNavigateToTrustPanel, latestEvidence = [] }: Props) {
  const [feedbackGiven, setFeedbackGiven] = useState<"yes" | "no" | null>(null);

  const parsed = parseFraudEvidence(latestEvidence);

  // Find max SHAP value for proportional visual bar calculation
  const maxShap = parsed.shapContributions.length > 0
    ? Math.max(...parsed.shapContributions.map((c) => c.shapValue), 0.001)
    : 1.0;

  // Construct explanation text based on provenance
  let explanationSummary = "";
  if (parsed.provenanceType === "live") {
    const streamDetails = parsed.streamSignals
      ? `Dual-stream anomaly detection recorded a behavioral anomaly score of ${parsed.streamSignals.behavioralScore?.toFixed(4) ?? "N/A"} and relational risk score of ${parsed.streamSignals.graphScore?.toFixed(4) ?? "N/A"}.`
      : "";
    explanationSummary = `Recorded in Component 2's live gateway audit trail with decision ${parsed.status} (Recorded Fraud Score: ${parsed.fraudScore.toFixed(4)}). Contributing factor: "${parsed.reason}". ${streamDetails} Component 4's local synthetic prototype Random Forest was not used for this live decision.`;
  } else if (parsed.provenanceType === "fallback") {
    const factorList = parsed.factors.length > 0 ? parsed.factors.join(", ") : "high transfer amount, unusual location, and new device";
    explanationSummary = `The available mock transaction record reports a ${parsed.riskLevel} risk level and ${parsed.status} status. Recorded contributing factors: ${factorList}. In the local synthetic prototype, transaction amount is the strongest contributing feature, followed by unusual location and new device inputs inferred from mock reason labels. The reasoning of the production model responsible for the recorded fraud score has not been independently established.`;
  } else {
    explanationSummary = `Recorded contributing factors in the mock baseline transaction record are high transfer amount, unusual location, and new device. In the local synthetic prototype, transaction amount is the strongest contributing feature, followed by unusual location and new device inputs inferred from mock reason labels. The reasoning of the production model responsible for the recorded fraud score has not been independently established.`;
  }

  return (
    <div className="flex flex-col gap-6">
      {/* Header Section */}
      <div className="rounded-xl border border-black/10 bg-white p-5 shadow-xs dark:border-white/15 dark:bg-neutral-900">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-blue-600 dark:text-blue-400">
            <BarChartIcon className="size-4" />
            <span>Explainable AI (XAI) Attribution</span>
          </div>

          {/* Provenance Badge */}
          {parsed.provenanceType === "live" && (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-300 bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-800 dark:border-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300">
              <span className="size-2 rounded-full bg-emerald-500 animate-pulse"></span>
              Live Service Evidence (Component 2)
            </span>
          )}
          {parsed.provenanceType === "fallback" && (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-300 bg-amber-50 px-3 py-1 text-xs font-semibold text-amber-800 dark:border-amber-800 dark:bg-amber-950/60 dark:text-amber-300">
              <span className="size-2 rounded-full bg-amber-500"></span>
              Fallback Mock Data
            </span>
          )}
          {parsed.provenanceType === "baseline" && (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-neutral-300 bg-neutral-100 px-3 py-1 text-xs font-semibold text-neutral-700 dark:border-neutral-700 dark:bg-neutral-800 dark:text-neutral-300">
              <span className="size-2 rounded-full bg-neutral-400"></span>
              Mock Baseline
            </span>
          )}
        </div>

        <h2 className="mt-2 text-2xl font-bold text-neutral-900 dark:text-white">
          Why was {parsed.transactionId} {parsed.status.toLowerCase()}?
        </h2>
        <p className="mt-1 text-sm text-neutral-500">
          AI explanation based on model evidence & SHAP feature contributions
        </p>
      </div>

      {/* Top Summary Card */}
      <div className="rounded-xl border border-red-200/80 bg-red-50/40 p-5 shadow-xs dark:border-red-900/40 dark:bg-neutral-900">
        <div className="mb-3 text-xs font-bold uppercase tracking-wider text-red-700 dark:text-red-400">
          Transaction Overview
        </div>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-5">
          <div>
            <span className="text-xs text-neutral-500">Transaction ID</span>
            <p className="font-mono text-base font-bold text-neutral-900 dark:text-white">{parsed.transactionId}</p>
          </div>
          <div>
            <span className="text-xs text-neutral-500">Amount</span>
            <p className="text-base font-bold text-neutral-900 dark:text-white">
              {parsed.amount != null ? `${parsed.currency} ${parsed.amount.toLocaleString()}` : "Not in audit trail"}
            </p>
          </div>
          <div>
            <span className="text-xs text-neutral-500">Decision</span>
            <span className="mt-1 inline-block rounded-md bg-red-600 px-2.5 py-0.5 text-xs font-bold text-white">
              {parsed.status}
            </span>
          </div>
          <div>
            <span className="text-xs text-neutral-500">Risk Level</span>
            <p className="text-base font-bold text-red-600 dark:text-red-400">{parsed.riskLevel}</p>
          </div>
          <div>
            <span className="text-xs text-neutral-500">Recorded Fraud Score</span>
            <p className="text-base font-bold text-red-600 dark:text-red-400">
              {typeof parsed.fraudScore === "number" ? parsed.fraudScore : parsed.fraudScore}
            </p>
            <p className="mt-0.5 text-[11px] text-neutral-400">
              Probability calibration not established.
            </p>
          </div>
        </div>
      </div>

      {/* SHAP Feature Contributions Section (when SHAP values are available) */}
      {parsed.shapContributions.length > 0 && (
        <div className="rounded-xl border border-black/10 bg-white p-5 shadow-xs dark:border-white/15 dark:bg-neutral-900">
          <div className="mb-4 flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h3 className="text-base font-semibold text-neutral-900 dark:text-white">
                Why did the model consider this high risk?
              </h3>
              <p className="text-xs text-neutral-500">
                Local synthetic Random Forest prototype — SHAP attributions
              </p>
            </div>
            <span className="w-fit rounded-full bg-blue-100 px-3 py-1 text-xs font-bold text-blue-700 dark:bg-blue-950 dark:text-blue-300">
              SHAP feature contributions
            </span>
          </div>

          {/* Horizontal Contribution Cards */}
          <div className="flex flex-col gap-3">
            {parsed.shapContributions.map((item, idx) => {
              const barWidthPercent = (item.shapValue / maxShap) * 100;
              return (
                <div
                  key={item.feature}
                  className="rounded-lg border border-neutral-200 bg-neutral-50/80 p-4 transition-all hover:border-blue-300 dark:border-neutral-800 dark:bg-neutral-800/60"
                >
                  <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                    <div className="flex items-center gap-2.5">
                      <span className="flex size-6 items-center justify-center rounded-full bg-red-100 text-xs font-bold text-red-700 dark:bg-red-950 dark:text-red-300">
                        {idx + 1}
                      </span>
                      <span className="font-semibold text-neutral-900 dark:text-white">
                        {item.label}
                      </span>
                    </div>

                    <div className="flex items-center gap-3 text-xs">
                      <span className="font-mono font-bold text-red-600 dark:text-red-400">
                        SHAP contribution: +{item.shapValue.toFixed(4)}
                      </span>
                      <span className="rounded-md bg-neutral-200/80 px-2 py-0.5 text-[11px] text-neutral-700 dark:bg-neutral-700 dark:text-neutral-200">
                        Impact: {item.impact}
                      </span>
                      <span className="rounded-md bg-red-100 px-2 py-0.5 text-[11px] font-bold text-red-800 dark:bg-red-950 dark:text-red-300">
                        Strength: {item.strength}
                      </span>
                    </div>
                  </div>

                  {/* Progress bar visualizer */}
                  <div className="mt-3 h-2.5 w-full overflow-hidden rounded-full bg-neutral-200 dark:bg-neutral-700">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-amber-500 to-red-600"
                      style={{ width: `${barWidthPercent}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>

          <p className="mt-3 text-xs leading-relaxed text-neutral-500 dark:text-neutral-400">
            These SHAP values explain the local synthetic prototype, not necessarily the model responsible for the recorded fraud score. Device and location inputs were inferred from mock reason labels and are not independently verified real-world observations.
          </p>
        </div>
      )}

      {/* Component 2 Dual-Stream Model Signals (Live Service Record) */}
      {parsed.provenanceType === "live" && parsed.streamSignals && (
        <div className="rounded-xl border border-black/10 bg-white p-5 shadow-xs dark:border-white/15 dark:bg-neutral-900">
          <div className="mb-4 flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h3 className="text-base font-semibold text-neutral-900 dark:text-white">
                Live Component 2 Dual-Stream Model Signals
              </h3>
              <p className="text-xs text-neutral-500">
                Simulated LSTM autoencoder & GNN relational graph stream outputs
              </p>
            </div>
            <span className="w-fit rounded-full bg-emerald-100 px-3 py-1 text-xs font-bold text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300">
              Live Gateway Output
            </span>
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <div className="rounded-lg border border-neutral-200 bg-neutral-50/80 p-4 dark:border-neutral-800 dark:bg-neutral-800/60">
              <span className="text-xs text-neutral-500">Behavioral Stream (LSTM)</span>
              <p className="mt-1 font-mono text-xl font-bold text-neutral-900 dark:text-white">
                {parsed.streamSignals.behavioralScore?.toFixed(4) ?? "N/A"}
              </p>
              <p className="mt-1 text-[11px] text-neutral-500">Sequence anomaly score</p>
            </div>

            <div className="rounded-lg border border-neutral-200 bg-neutral-50/80 p-4 dark:border-neutral-800 dark:bg-neutral-800/60">
              <span className="text-xs text-neutral-500">Relational Stream (GNN)</span>
              <p className="mt-1 font-mono text-xl font-bold text-neutral-900 dark:text-white">
                {parsed.streamSignals.graphScore?.toFixed(4) ?? "N/A"}
              </p>
              <p className="mt-1 text-[11px] text-neutral-500">Entity graph relational risk</p>
            </div>

            <div className="rounded-lg border border-neutral-200 bg-neutral-50/80 p-4 dark:border-neutral-800 dark:bg-neutral-800/60">
              <span className="text-xs text-neutral-500">Fused Final Risk Score</span>
              <p className="mt-1 font-mono text-xl font-bold text-red-600 dark:text-red-400">
                {parsed.streamSignals.riskScore?.toFixed(4) ?? parsed.fraudScore.toFixed(4)}
              </p>
              <p className="mt-1 text-[11px] text-neutral-500">Late fusion gateway input</p>
            </div>
          </div>

          <p className="mt-3 text-xs leading-relaxed text-neutral-500 dark:text-neutral-400">
            The live decision was produced by Component 2&apos;s dual-stream engine and deterministic security gateway. Component 4&apos;s local synthetic Random Forest prototype was not used for this live decision, and raw tabular telemetry is not retained in the gateway audit trail.
          </p>
        </div>
      )}

      {/* Simple Natural Language Explanation */}
      <div className="rounded-xl border border-blue-200/80 bg-blue-50/50 p-5 shadow-xs dark:border-blue-900/40 dark:bg-blue-950/20">
        <div className="mb-2 flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-blue-900 dark:text-blue-300">
            <SparklesIcon className="size-4 text-blue-600" />
            <span>Simple explanation</span>
          </div>
          <span className="rounded-full bg-blue-600 px-2.5 py-0.5 text-[11px] font-bold text-white shadow-2xs">
            Grounded in tool evidence
          </span>
        </div>
        <p className="text-sm font-medium leading-relaxed text-neutral-800 dark:text-neutral-200">
          {explanationSummary}
        </p>

        {/* Evidence Sources & Next Action */}
        <div className="mt-4 flex flex-col gap-4 border-t border-blue-200/60 pt-4 dark:border-blue-900/60 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="mb-1.5 text-xs font-semibold text-blue-950 dark:text-blue-200">
              Evidence sources:
            </p>
            <div className="flex flex-wrap items-center gap-3 text-xs font-medium text-blue-800 dark:text-blue-300">
              <span className="flex items-center gap-1">
                {parsed.hasEvidence ? (
                  <>
                    <CheckCircleIcon className="size-4 text-emerald-500" />
                    <span>Fraud analysis ({parsed.provenanceType})</span>
                  </>
                ) : (
                  <>
                    <svg className="size-4 shrink-0 text-neutral-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <circle cx="12" cy="12" r="8" />
                    </svg>
                    <span className="text-neutral-500 dark:text-neutral-400">Fraud analysis (baseline)</span>
                  </>
                )}
              </span>

              <span className="flex items-center gap-1">
                {(latestEvidence || []).some((e) => e.tool === "get_blockchain_audit") ? (
                  <>
                    <CheckCircleIcon className="size-4 text-emerald-500" />
                    <span>Blockchain audit</span>
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

              <span className="flex items-center gap-1">
                {parsed.shapContributions.length > 0 || parsed.streamSignals ? (
                  <>
                    <CheckCircleIcon className="size-4 text-emerald-500" />
                    <span>Model attribution</span>
                  </>
                ) : (
                  <>
                    <svg className="size-4 shrink-0 text-neutral-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <circle cx="12" cy="12" r="8" />
                    </svg>
                    <span className="text-neutral-500 dark:text-neutral-400">Model attribution (baseline)</span>
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

