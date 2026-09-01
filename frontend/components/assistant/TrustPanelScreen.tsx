"use client";

import React, { useState } from "react";
import { MOCK_TRANSACTION } from "./mockData";
import { TrustActionState } from "./types";
import { ShieldIcon, CheckCircleIcon, LockIcon, AlertTriangleIcon, UndoIcon, SparklesIcon } from "./icons";

export function TrustPanelScreen() {
  const [actionState, setActionState] = useState<TrustActionState>("pending");
  const [showMoreExplanation, setShowMoreExplanation] = useState(false);
  const tx = MOCK_TRANSACTION;

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
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-base font-semibold text-neutral-900 dark:text-white">
            AI Decision Summary
          </h3>
          <span className="rounded-full bg-red-100 px-3 py-1 text-xs font-bold text-red-700 dark:bg-red-950 dark:text-red-300">
            {tx.status}
          </span>
        </div>

        <div className="rounded-lg border border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-800 dark:bg-neutral-800/60">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-xs font-semibold text-neutral-500">Target Transaction</p>
              <p className="text-lg font-bold text-neutral-900 dark:text-white">
                Transaction {tx.id} — {tx.currency} {tx.amount.toLocaleString()}
              </p>
            </div>
            <div className="text-right">
              <span className="text-xs text-neutral-500">Decision</span>
              <p className="text-base font-bold text-red-600 dark:text-red-400">BLOCKED</p>
            </div>
          </div>

          <p className="mt-3 text-sm text-neutral-700 dark:text-neutral-300">
            <span className="font-semibold text-neutral-900 dark:text-white">Explanation:</span>{" "}
            "The transaction was classified as high risk based on the model evidence."
          </p>

          <div className="mt-4 flex flex-col gap-2 border-t border-neutral-200/80 pt-3 dark:border-neutral-700">
            <p className="text-xs font-bold uppercase tracking-wider text-neutral-500">
              Verified Evidence Checklist
            </p>
            <div className="flex flex-col gap-1.5 text-xs text-neutral-800 dark:text-neutral-200">
              <div className="flex items-center gap-2">
                <CheckCircleIcon className="size-4 text-emerald-500" />
                <span>Fraud analysis verified (Model Score: {(tx.fraudScore * 100).toFixed(0)}%)</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircleIcon className="size-4 text-emerald-500" />
                <span>
                  Blockchain audit verified (Audit ID:{" "}
                  <code className="rounded bg-neutral-200 px-1 py-0.5 font-mono text-[11px] dark:bg-neutral-700">
                    {tx.blockchainAudit.auditId}
                  </code>
                  , Block: #{tx.blockchainAudit.blockNumber}, Status: {tx.blockchainAudit.recordStatus})
                </span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircleIcon className="size-4 text-emerald-500" />
                <span>SHAP explanation available (3 primary features computed)</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* AI Transparency & Limitations */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="rounded-xl border border-blue-200/80 bg-blue-50/40 p-4 shadow-xs dark:border-blue-900/40 dark:bg-neutral-900">
          <h4 className="text-xs font-bold uppercase tracking-wider text-blue-900 dark:text-blue-300">
            How was this decision explained?
          </h4>
          <p className="mt-2 text-xs leading-relaxed text-neutral-700 dark:text-neutral-300">
            Using model attribution evidence and verified transaction context aggregated through LangGraph tool execution.
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
          Responsible AI Automated Governance Status
        </h4>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div className="flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50/60 p-2.5 text-xs font-semibold text-emerald-800 dark:border-emerald-900/50 dark:bg-emerald-950/40 dark:text-emerald-300">
            <CheckCircleIcon className="size-4 text-emerald-600" />
            <span>✓ Privacy check</span>
          </div>
          <div className="flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50/60 p-2.5 text-xs font-semibold text-emerald-800 dark:border-emerald-900/50 dark:bg-emerald-950/40 dark:text-emerald-300">
            <CheckCircleIcon className="size-4 text-emerald-600" />
            <span>✓ Safety check</span>
          </div>
          <div className="flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50/60 p-2.5 text-xs font-semibold text-emerald-800 dark:border-emerald-900/50 dark:bg-emerald-950/40 dark:text-emerald-300">
            <CheckCircleIcon className="size-4 text-emerald-600" />
            <span>✓ Evidence grounding</span>
          </div>
          <div className="flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50/60 p-2.5 text-xs font-semibold text-emerald-800 dark:border-emerald-900/50 dark:bg-emerald-950/40 dark:text-emerald-300">
            <CheckCircleIcon className="size-4 text-emerald-600" />
            <span>✓ AI disclosure</span>
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
            <span>Decision acknowledged — Action recorded in governance log</span>
          </div>
        )}

        {actionState === "rejected" && (
          <div className="mb-4 flex items-center gap-2 rounded-lg bg-red-100 p-3 text-xs font-bold text-red-800 dark:bg-red-950 dark:text-red-200">
            <AlertTriangleIcon className="size-4" />
            <span>Action rejected by user — Override requested for secondary review</span>
          </div>
        )}

        {showMoreExplanation && (
          <div className="mb-4 rounded-lg border border-blue-200 bg-blue-50/80 p-4 text-xs dark:border-blue-900 dark:bg-blue-950/40">
            <div className="flex items-center gap-2 font-bold text-blue-900 dark:text-blue-200">
              <SparklesIcon className="size-4 text-blue-600" />
              <span>Expanded Model Attribution Breakdown</span>
            </div>
            <p className="mt-2 text-neutral-700 dark:text-neutral-300 leading-relaxed">
              The model evaluated transfer velocity (+$0.1933 SHAP), IP geo-location deviation (+$0.1535 SHAP), and hardware token fingerprint mismatch (+$0.1308 SHAP). Blockchain ledger audit record AUDIT-2026-001 confirms zero prior trust score in this target subnet.
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
