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
  auditId?: string;
  blockNumber?: number;
  recordStatus?: string;
  hasFraudEvidence: boolean;
  hasAuditEvidence: boolean;
}

function parseLatestEvidence(evidenceList: Array<{ tool: string; result: string }> = []): ParsedEvidenceData {
  const fraudItem = evidenceList.find((e) => e.tool === "get_fraud_analysis");
  const auditItem = evidenceList.find((e) => e.tool === "get_blockchain_audit");

  const parsed: ParsedEvidenceData = {
    hasFraudEvidence: !!fraudItem,
    hasAuditEvidence: !!auditItem,
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
  }

  if (auditItem?.result) {
    const text = auditItem.result;
    const auditIdMatch = text.match(/Audit ID:\s*([A-Za-z0-9_-]+)/i);
    if (auditIdMatch) parsed.auditId = auditIdMatch[1];

    const blockMatch = text.match(/Block Number:\s*([0-9]+)/i);
    if (blockMatch) parsed.blockNumber = parseInt(blockMatch[1], 10);

    const statusMatch = text.match(/Record Status:\s*([A-Za-z0-9_-]+)/i);
    if (statusMatch) parsed.recordStatus = statusMatch[1];
  }

  return parsed;
}

export function TrustPanelScreen({ latestEvidence = [] }: TrustPanelScreenProps) {
  const [actionState, setActionState] = useState<TrustActionState>("pending");
  const [showMoreExplanation, setShowMoreExplanation] = useState(false);
  const tx = MOCK_TRANSACTION;

  const parsed = parseLatestEvidence(latestEvidence);
  const hasLiveEvidence = parsed.hasFraudEvidence || parsed.hasAuditEvidence;

  const displayTxId = parsed.transactionId ?? tx.id;
  const displayAmount = parsed.amount ?? tx.amount;
  const displayCurrency = parsed.currency ?? tx.currency;
  const displayStatus = parsed.status ?? tx.status;
  const displayFraudScore = parsed.fraudScore ?? tx.fraudScore;
  const displayAuditId = parsed.auditId ?? tx.blockchainAudit.auditId;
  const displayBlockNumber = parsed.blockNumber ?? tx.blockchainAudit.blockNumber;
  const displayRecordStatus = parsed.recordStatus ?? tx.blockchainAudit.recordStatus;

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
              <span className="rounded-full bg-blue-100 px-2.5 py-0.5 text-[11px] font-semibold text-blue-800 dark:bg-blue-950 dark:text-blue-300">
                Active Session Backend Evidence
              </span>
            ) : (
              <span className="rounded-full bg-neutral-100 px-2.5 py-0.5 text-[11px] font-medium text-neutral-600 dark:bg-neutral-800 dark:text-neutral-400">
                Mock Baseline
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
                Transaction {displayTxId} — {displayCurrency} {displayAmount.toLocaleString()}
              </p>
            </div>
            <div className="text-right">
              <span className="text-xs text-neutral-500">Decision</span>
              <p className="text-base font-bold text-red-600 dark:text-red-400">{displayStatus}</p>
            </div>
          </div>

          <p className="mt-3 text-sm text-neutral-700 dark:text-neutral-300">
            <span className="font-semibold text-neutral-900 dark:text-white">Explanation:</span>{" "}
            The available mock transaction record reports a HIGH risk level and BLOCKED status. The recorded risk factors are high transfer amount, unusual location, and new device. The production model&apos;s reasoning has not been independently established.
          </p>

          <div className="mt-4 flex flex-col gap-2 border-t border-neutral-200/80 pt-3 dark:border-neutral-700">
            <p className="text-xs font-bold uppercase tracking-wider text-neutral-500">
              Available Evidence Checklist
            </p>
            <div className="flex flex-col gap-1.5 text-xs text-neutral-800 dark:text-neutral-200">
              <div className="flex items-center gap-2">
                <svg className="size-4 shrink-0 text-neutral-400 dark:text-neutral-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <circle cx="12" cy="12" r="8" />
                </svg>
                <span>
                  Recorded fraud score: {displayFraudScore} (verification and probability calibration not established)
                </span>
              </div>
              <div className="flex items-center gap-2">
                <svg className="size-4 shrink-0 text-neutral-400 dark:text-neutral-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <circle cx="12" cy="12" r="8" />
                </svg>
                <span>
                  Blockchain audit record returned (Audit ID:{" "}
                  <code className="rounded bg-neutral-200 px-1 py-0.5 font-mono text-[11px] dark:bg-neutral-700">
                    {displayAuditId}
                  </code>
                  , Block: #{displayBlockNumber}, Status: {displayRecordStatus})
                </span>
              </div>
              <div className="flex items-center gap-2">
                <svg className="size-4 shrink-0 text-neutral-400 dark:text-neutral-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <circle cx="12" cy="12" r="8" />
                </svg>
                <span>SHAP explanation available (3 primary features computed)</span>
              </div>
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
              <span>Expanded Model Attribution Breakdown</span>
            </div>
            <p className="mt-2 text-neutral-700 dark:text-neutral-300 leading-relaxed">
              The local synthetic Random Forest prototype produced the following SHAP attributions: transaction amount (+0.193289), unusual location (+0.153450), and new device (+0.130761). These values explain the local prototype&apos;s output, not necessarily the model responsible for the recorded fraud score. The device and location inputs were inferred from mock reason labels and are not independently verified real-world observations.
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
