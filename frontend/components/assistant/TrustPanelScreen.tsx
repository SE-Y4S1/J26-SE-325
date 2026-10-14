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
  status?: string;
  riskLevel?: string;
  factors?: string[];
  auditId?: string;
  blockNumber?: number | string;
  recordStatus?: string;
  auditIsMockLedger?: boolean;
  hasFraudEvidence: boolean;
  hasAuditEvidence: boolean;
  hasShap: boolean;
  provenanceType: "live" | "fallback" | "baseline";
  provenanceLabel: string;
}

function parseLatestEvidence(evidenceList: Array<{ tool: string; result: string }> = []): ParsedEvidenceData {
  const fraudItem = evidenceList.find((e) => e.tool === "get_fraud_analysis");
  const auditItem = evidenceList.find((e) => e.tool === "get_blockchain_audit");

  const isFraudLive = fraudItem?.result ? /LIVE COMPONENT 2/i.test(fraudItem.result) : false;
  const isAuditLive = auditItem?.result ? /Live Component 3/i.test(auditItem.result) : false;
  const hasLive = isFraudLive || isAuditLive;
  const hasFallback =
    (fraudItem?.result && /FALLBACK MOCK DATA/i.test(fraudItem.result)) ||
    (auditItem?.result && /FALLBACK DATA/i.test(auditItem.result));

  const provenanceType: "live" | "fallback" | "baseline" = hasLive
    ? "live"
    : hasFallback
    ? "fallback"
    : "baseline";

  const provenanceLabel =
    provenanceType === "live"
      ? "Active Session: Live Service Evidence"
      : provenanceType === "fallback"
      ? "Active Session: Fallback Mock Data"
      : "Mock Baseline (No Active Session Evidence)";

  const parsed: ParsedEvidenceData = {
    hasFraudEvidence: !!(fraudItem?.result && !fraudItem.result.includes("No transaction was found")),
    hasAuditEvidence: !!(auditItem?.result && !auditItem.result.includes("No audit record found")),
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

  return parsed;
}

export function TrustPanelScreen({ latestEvidence = [] }: TrustPanelScreenProps) {
  const [actionState, setActionState] = useState<TrustActionState>("pending");
  const [showMoreExplanation, setShowMoreExplanation] = useState(false);
  const tx = MOCK_TRANSACTION;

  const parsed = parseLatestEvidence(latestEvidence);
  // Session policy: Active session evidence is valid only when tools returned results in this response.
  const hasLiveEvidence = latestEvidence.length > 0 && (parsed.hasFraudEvidence || parsed.hasAuditEvidence);

  const displayTxId = parsed.transactionId ?? (hasLiveEvidence ? "TX1001" : tx.id);
  const displayAmount = parsed.amount ?? (hasLiveEvidence ? undefined : tx.amount);
  const displayCurrency = parsed.currency ?? (hasLiveEvidence ? "LKR" : tx.currency);
  const displayStatus = parsed.status ?? (hasLiveEvidence ? (parsed.recordStatus ?? "RECORDED") : tx.status);

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
                className={`rounded-full px-2.5 py-0.5 text-[11px] font-semibold ${
                  parsed.provenanceType === "live"
                    ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300"
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
          <span className="rounded-full bg-red-100 px-3 py-1 text-xs font-bold text-red-700 dark:bg-red-950 dark:text-red-300">
            {displayStatus}
          </span>
        </div>

        <div className="rounded-lg border border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-800 dark:bg-neutral-800/60">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-xs font-semibold text-neutral-500">Target Transaction</p>
              <p className="text-lg font-bold text-neutral-900 dark:text-white">
                Transaction {displayTxId}
                {displayAmount != null && ` — ${displayCurrency} ${displayAmount.toLocaleString()}`}
              </p>
            </div>
            <div className="text-right">
              <span className="text-xs text-neutral-500">Decision</span>
              <p className="text-base font-bold text-red-600 dark:text-red-400">{displayStatus}</p>
            </div>
          </div>

          <p className="mt-3 text-sm text-neutral-700 dark:text-neutral-300">
            <span className="font-semibold text-neutral-900 dark:text-white">Explanation:</span>{" "}
            {hasLiveEvidence ? (
              parsed.hasFraudEvidence ? (
                parsed.provenanceType === "live" ? (
                  <>Component 2 gateway audit trail reports status <span className="font-semibold text-neutral-900 dark:text-white">{parsed.status || "BLOCKED"}</span> with risk score <span className="font-semibold text-neutral-900 dark:text-white">{parsed.fraudScore?.toFixed(4) ?? "N/A"}</span>. The recorded decision factor is {parsed.factors?.[0] || "recorded dual-stream signals"}. The production model&apos;s internal reasoning has not been independently established.</>
                ) : (
                  <>The available mock transaction record reports a <span className="font-semibold text-neutral-900 dark:text-white">{parsed.riskLevel || "HIGH"}</span> risk level and <span className="font-semibold text-neutral-900 dark:text-white">{parsed.status || "BLOCKED"}</span> status. The recorded risk factors are {parsed.factors?.join(", ") || "high transfer amount, unusual location, and new device"}. The production model&apos;s reasoning has not been independently established.</>
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
              {/* Row 1: Fraud score evidence */}
              {hasLiveEvidence ? (
                parsed.hasFraudEvidence ? (
                  <div className="flex items-center gap-2">
                    <CheckCircleIcon className="size-4 shrink-0 text-emerald-500" />
                    <span>
                      Recorded fraud score: {parsed.fraudScore ?? "N/A"} (verification and probability calibration not established)
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
                    <span>SHAP explanation available (local synthetic prototype model attributions)</span>
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
            Explanation uses available tool evidence and model-attribution outputs. Some evidence is mock data, and the local prototype explanation may not represent the model behind the recorded fraud score.
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
