"use client";

import React from "react";
import { ChevronRightIcon } from "./icons";

export function ResearchBanner() {
  const steps = [
    { label: "LLM Orchestration", desc: "Intent & Planning" },
    { label: "LangGraph Agent", desc: "State Workflow" },
    { label: "Tool Calling", desc: "Fraud & Audit APIs" },
    { label: "Evidence", desc: "Raw Data" },
    { label: "SHAP/LIME", desc: "Attribution" },
    { label: "SLM Explanation", desc: "Natural Language" },
    { label: "Responsible AI", desc: "Guardrails Check" },
    { label: "Trust Panel", desc: "User Control" },
  ];

  return (
    <section className="mb-6 rounded-xl border border-blue-200/80 bg-white p-4 shadow-xs dark:border-blue-900/50 dark:bg-neutral-900">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="flex size-2 rounded-full bg-blue-600"></span>
          <h3 className="text-xs font-bold uppercase tracking-wider text-blue-900 dark:text-blue-300">
            Research Architecture Pipeline
          </h3>
        </div>
        <span className="text-[11px] text-neutral-500">Observable Agentic Loop</span>
      </div>

      <div className="flex items-center gap-1 overflow-x-auto pb-1 text-xs">
        {steps.map((step, idx) => (
          <React.Fragment key={step.label}>
            <div className="flex shrink-0 flex-col rounded-lg border border-neutral-200 bg-neutral-50 px-2.5 py-1.5 dark:border-neutral-800 dark:bg-neutral-800/60">
              <span className="font-semibold text-neutral-900 dark:text-neutral-100">{step.label}</span>
              <span className="text-[10px] text-neutral-500">{step.desc}</span>
            </div>
            {idx < steps.length - 1 && (
              <ChevronRightIcon className="size-3.5 shrink-0 text-blue-500 opacity-60" />
            )}
          </React.Fragment>
        ))}
      </div>
    </section>
  );
}
