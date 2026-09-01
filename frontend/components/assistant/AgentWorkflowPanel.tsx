"use client";

import React, { useState } from "react";
import { MOCK_WORKFLOW_STEPS } from "./mockData";
import { CpuIcon, CheckCircleIcon, ChevronRightIcon, WrenchIcon } from "./icons";

export function AgentWorkflowPanel() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="mb-4 rounded-xl border border-neutral-200 bg-neutral-50/80 transition dark:border-neutral-800 dark:bg-neutral-800/40">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex w-full items-center justify-between p-3.5 text-left text-xs font-semibold text-neutral-700 hover:text-neutral-900 dark:text-neutral-300 dark:hover:text-white"
      >
        <div className="flex items-center gap-2.5">
          <CpuIcon className="size-4 text-blue-600 dark:text-blue-400" />
          <span>Agent Activity & Pipeline Trace</span>
          <span className="rounded-full bg-blue-100 px-2 py-0.5 text-[10px] font-bold text-blue-700 dark:bg-blue-950 dark:text-blue-300">
            2 tools called
          </span>
        </div>
        <div className="flex items-center gap-2 text-[11px] font-medium text-neutral-500">
          <span>{isOpen ? "Hide pipeline trace" : "View pipeline trace"}</span>
          <span className={`transform transition-transform ${isOpen ? "rotate-90" : ""}`}>
            <ChevronRightIcon className="size-4" />
          </span>
        </div>
      </button>

      {isOpen && (
        <div className="border-t border-neutral-200/80 p-4 dark:border-neutral-800">
          <div className="mb-3 flex items-center justify-between text-[11px] text-neutral-500">
            <span>LangGraph Agent Execution Workflow</span>
            <span className="flex items-center gap-1 font-mono text-neutral-600 dark:text-neutral-400">
              <WrenchIcon className="size-3 text-blue-600" /> get_fraud_analysis, get_blockchain_audit
            </span>
          </div>

          {/* Workflow Steps Grid */}
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-5">
            {MOCK_WORKFLOW_STEPS.map((step, idx) => (
              <div
                key={step.id}
                className="relative rounded-lg border border-neutral-200 bg-white p-2.5 shadow-2xs dark:border-neutral-700 dark:bg-neutral-800"
              >
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-bold uppercase text-blue-600 dark:text-blue-400">
                    Step {idx + 1}
                  </span>
                  <CheckCircleIcon className="size-3.5 text-emerald-500" />
                </div>
                <p className="mt-1 text-xs font-semibold text-neutral-900 dark:text-white">
                  {step.name}
                </p>
                <p className="mt-0.5 text-[10px] leading-tight text-neutral-500">
                  {step.description}
                </p>
              </div>
            ))}
          </div>

          <p className="mt-3 text-[11px] italic text-neutral-400">
            Note: Displays observable tool calls and evidence checks. Private chain-of-thought is hidden for safety and compliance.
          </p>
        </div>
      )}
    </div>
  );
}
