"use client";

import React, { useState } from "react";
import { MOCK_TRANSACTION } from "./mockData";
import { BarChartIcon, CheckCircleIcon, ShieldIcon, SparklesIcon, ChevronRightIcon } from "./icons";

interface Props {
  onNavigateToTrustPanel: () => void;
}

export function ExplanationScreen({ onNavigateToTrustPanel }: Props) {
  const [feedbackGiven, setFeedbackGiven] = useState<"yes" | "no" | null>(null);
  const tx = MOCK_TRANSACTION;

  // Find max SHAP value for proportional visual bar calculation
  const maxShap = Math.max(...tx.shapContributions.map((c) => c.shapValue));

  return (
    <div className="flex flex-col gap-6">
      {/* Header Section */}
      <div className="rounded-xl border border-black/10 bg-white p-5 shadow-xs dark:border-white/15 dark:bg-neutral-900">
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-blue-600 dark:text-blue-400">
          <BarChartIcon className="size-4" />
          <span>Explainable AI (XAI) Attribution</span>
        </div>
        <h2 className="mt-1 text-2xl font-bold text-neutral-900 dark:text-white">
          Why was {tx.id} blocked?
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
            <p className="font-mono text-base font-bold text-neutral-900 dark:text-white">{tx.id}</p>
          </div>
          <div>
            <span className="text-xs text-neutral-500">Amount</span>
            <p className="text-base font-bold text-neutral-900 dark:text-white">
              {tx.currency} {tx.amount.toLocaleString()}
            </p>
          </div>
          <div>
            <span className="text-xs text-neutral-500">Decision</span>
            <span className="mt-1 inline-block rounded-md bg-red-600 px-2.5 py-0.5 text-xs font-bold text-white">
              {tx.status}
            </span>
          </div>
          <div>
            <span className="text-xs text-neutral-500">Risk Level</span>
            <p className="text-base font-bold text-red-600 dark:text-red-400">{tx.riskLevel}</p>
          </div>
          <div>
            <span className="text-xs text-neutral-500">Fraud Score</span>
            <p className="text-base font-bold text-red-600 dark:text-red-400">
              {(tx.fraudScore * 100).toFixed(0)}%
            </p>
          </div>
        </div>
      </div>

      {/* SHAP Feature Contributions Section */}
      <div className="rounded-xl border border-black/10 bg-white p-5 shadow-xs dark:border-white/15 dark:bg-neutral-900">
        <div className="mb-4 flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h3 className="text-base font-semibold text-neutral-900 dark:text-white">
              Why did the model consider this high risk?
            </h3>
            <p className="text-xs text-neutral-500">
              Quantified quantitative contributions computed via Kernel SHAP attributions
            </p>
          </div>
          <span className="w-fit rounded-full bg-blue-100 px-3 py-1 text-xs font-bold text-blue-700 dark:bg-blue-950 dark:text-blue-300">
            SHAP feature contributions
          </span>
        </div>

        {/* 3 Horizontal Contribution Cards */}
        <div className="flex flex-col gap-3">
          {tx.shapContributions.map((item, idx) => {
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

        <p className="mt-3 text-xs italic text-neutral-400">
          * SHAP values represent individual feature attributions relative to the baseline expectation score.
        </p>
      </div>

      {/* Simple Natural Language Explanation */}
      <div className="rounded-xl border border-blue-200/80 bg-blue-50/50 p-5 shadow-xs dark:border-blue-900/40 dark:bg-blue-950/20">
        <div className="mb-2 flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-blue-900 dark:text-blue-300">
            <SparklesIcon className="size-4 text-blue-600" />
            <span>Simple explanation</span>
          </div>
          <span className="rounded-full bg-blue-600 px-2.5 py-0.5 text-[11px] font-bold text-white shadow-2xs">
            Grounded in SHAP evidence
          </span>
        </div>
        <p className="text-sm font-medium leading-relaxed text-neutral-800 dark:text-neutral-200">
          "The transaction was classified as high risk mainly because of the unusually high transfer amount. The unusual location and new device also increased the estimated risk."
        </p>

        {/* Evidence Sources & Next Action */}
        <div className="mt-4 flex flex-col gap-4 border-t border-blue-200/60 pt-4 dark:border-blue-900/60 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="mb-1.5 text-xs font-semibold text-blue-950 dark:text-blue-200">
              Evidence sources:
            </p>
            <div className="flex flex-wrap items-center gap-3 text-xs font-medium text-blue-800 dark:text-blue-300">
              <span className="flex items-center gap-1">
                <CheckCircleIcon className="size-4 text-emerald-500" /> Fraud analysis
              </span>
              <span className="flex items-center gap-1">
                <CheckCircleIcon className="size-4 text-emerald-500" /> Blockchain audit
              </span>
              <span className="flex items-center gap-1">
                <CheckCircleIcon className="size-4 text-emerald-500" /> Model attribution
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
