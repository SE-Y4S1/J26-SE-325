"use client";

import React, { useState } from "react";
import { MOCK_RESPONSIBLE_METRICS } from "./mockData";
import { CpuIcon, LockIcon, ShieldIcon, CheckCircleIcon, AlertTriangleIcon } from "./icons";

export function ResponsibleAIScreen() {
  const [explanationEnabled, setExplanationEnabled] = useState(true);
  const [personalizedContext, setPersonalizedContext] = useState(true);
  const [dataSharingConsent, setDataSharingConsent] = useState(true);

  return (
    <div className="flex flex-col gap-6">
      {/* Title & Subtitle Header */}
      <div className="rounded-xl border border-black/10 bg-white p-5 shadow-xs dark:border-white/15 dark:bg-neutral-900">
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-blue-600 dark:text-blue-400">
          <CpuIcon className="size-4" />
          <span>Governance & Ethics Framework</span>
        </div>
        <h2 className="mt-1 text-2xl font-bold text-neutral-900 dark:text-white">
          Responsible AI & Safety
        </h2>
        <p className="mt-1 text-sm text-neutral-500">
          Platform alignment with TRiSM, privacy protection, and transparent user controls
        </p>
      </div>

      {/* 6 Governance Cards Grid */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {MOCK_RESPONSIBLE_METRICS.map((metric) => (
          <div
            key={metric.id}
            className="flex flex-col justify-between rounded-xl border border-black/10 bg-white p-5 shadow-2xs dark:border-white/15 dark:bg-neutral-900"
          >
            <div>
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-neutral-500">
                  {metric.title}
                </span>
                <span
                  className={`rounded-full px-2.5 py-0.5 text-[10px] font-bold ${
                    metric.status === "Protected" || metric.status === "Active" || metric.status === "Enabled"
                      ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300"
                      : "bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300"
                  }`}
                >
                  Status: {metric.status}
                </span>
              </div>

              <p className="mt-3 text-xs leading-relaxed text-neutral-600 dark:text-neutral-300">
                {metric.description}
              </p>
            </div>

            <div className="mt-4 flex items-center gap-1.5 text-[11px] font-semibold text-emerald-600 dark:text-emerald-400">
              <CheckCircleIcon className="size-3.5" />
              <span>Compliant with Research Framework</span>
            </div>
          </div>
        ))}
      </div>

      {/* Interactive Controls & Toggles */}
      <div className="rounded-xl border border-black/10 bg-white p-5 shadow-xs dark:border-white/15 dark:bg-neutral-900">
        <h3 className="text-base font-semibold text-neutral-900 dark:text-white">
          User Privacy & AI Preference Controls
        </h3>
        <p className="text-xs text-neutral-500">
          Customize how AI agents parse your context and deliver explanations.
        </p>

        <div className="mt-4 flex flex-col gap-4 border-t border-neutral-200/80 pt-4 dark:border-neutral-800">
          {/* Toggle 1 */}
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold text-neutral-900 dark:text-white">
                AI explanation enabled
              </p>
              <p className="text-[11px] text-neutral-500">
                Generate SHAP natural language explanations for flagged decisions
              </p>
            </div>
            <button
              onClick={() => setExplanationEnabled(!explanationEnabled)}
              className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full transition-colors ${
                explanationEnabled ? "bg-blue-600" : "bg-neutral-300 dark:bg-neutral-700"
              }`}
            >
              <span
                className={`inline-block size-5 transform rounded-full bg-white transition-transform ${
                  explanationEnabled ? "translate-x-5.5" : "translate-x-0.5"
                } my-0.5`}
              />
            </button>
          </div>

          {/* Toggle 2 */}
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold text-neutral-900 dark:text-white">
                Personalized financial context
              </p>
              <p className="text-[11px] text-neutral-500">
                Allow assistant to analyze localized telemetry for deeper explanation groundings
              </p>
            </div>
            <button
              onClick={() => setPersonalizedContext(!personalizedContext)}
              className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full transition-colors ${
                personalizedContext ? "bg-blue-600" : "bg-neutral-300 dark:bg-neutral-700"
              }`}
            >
              <span
                className={`inline-block size-5 transform rounded-full bg-white transition-transform ${
                  personalizedContext ? "translate-x-5.5" : "translate-x-0.5"
                } my-0.5`}
              />
            </button>
          </div>

          {/* Toggle 3 */}
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold text-neutral-900 dark:text-white">
                Data sharing consent
              </p>
              <p className="text-[11px] text-neutral-500">
                Consent to anonymized audit logging for university empirical research evaluation
              </p>
            </div>
            <button
              onClick={() => setDataSharingConsent(!dataSharingConsent)}
              className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full transition-colors ${
                dataSharingConsent ? "bg-blue-600" : "bg-neutral-300 dark:bg-neutral-700"
              }`}
            >
              <span
                className={`inline-block size-5 transform rounded-full bg-white transition-transform ${
                  dataSharingConsent ? "translate-x-5.5" : "translate-x-0.5"
                } my-0.5`}
              />
            </button>
          </div>
        </div>
      </div>

      {/* Mandatory AI Disclosure Box */}
      <div className="flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50/70 p-4 shadow-2xs dark:border-amber-900/40 dark:bg-amber-950/30">
        <AlertTriangleIcon className="size-5 shrink-0 text-amber-600 dark:text-amber-400" />
        <div>
          <h4 className="text-xs font-bold uppercase tracking-wider text-amber-900 dark:text-amber-200">
            AI Disclosure
          </h4>
          <p className="mt-1 text-xs leading-relaxed text-amber-950 dark:text-amber-300">
            This assistant uses AI to generate explanations. AI outputs may contain errors and should be reviewed before making important financial decisions.
          </p>
        </div>
      </div>
    </div>
  );
}
