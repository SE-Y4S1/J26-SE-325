"use client";

import React from "react";
import { ChevronRightIcon, ShieldIcon, CpuIcon, CheckCircleIcon, SparklesIcon, AlertTriangleIcon } from "./icons";

interface Props {
  isOpen?: boolean;
  onClose?: () => void;
}

export function ResearchBanner({ isOpen = true, onClose }: Props) {
  if (!isOpen) return null;

  const pipelineStages = [
    {
      id: "1",
      label: "LLM Orchestration",
      tech: "Ollama (qwen2.5:3b / llama3.2)",
      status: "Active (PP1)",
      statusType: "active",
      desc: "Intent parsing, tool planning & deterministic fast-path routing.",
    },
    {
      id: "2",
      label: "LangGraph Agent",
      tech: "LangChain ReAct Graph",
      status: "Active (PP1)",
      statusType: "active",
      desc: "Cyclic state machine coordinating tool execution and evidence accumulation.",
    },
    {
      id: "3",
      label: "Tool Calling",
      tech: "Ports 8001, 8002, 8100",
      status: "Active + Fallback",
      statusType: "gateway",
      desc: "Live REST gateways to Fraud Detection, Blockchain Ledger & Portfolio APIs.",
    },
    {
      id: "4",
      label: "Evidence Validation",
      tech: "evidence_validator.py",
      status: "Active (PP1)",
      statusType: "active",
      desc: "Rule-based factual consistency checker verifying claims against tool outputs.",
    },
    {
      id: "5",
      label: "Feature Attribution",
      tech: "Component 2 Tree SHAP",
      status: "Active (PP1)",
      statusType: "active",
      desc: "Feature importance contributions (amount, location, device); LIME is inactive.",
    },
    {
      id: "6",
      label: "Natural Language SLM",
      tech: "DistilBERT Explainer",
      status: "PP2 Planned",
      statusType: "planned",
      desc: "DistilBERT fine-tuning is reserved for PP2; PP1 uses evidence-grounded Ollama LLM.",
    },
    {
      id: "7",
      label: "Responsible AI",
      tech: "Guardrail Policies",
      status: "Active (PP1)",
      statusType: "active",
      desc: "Privacy consent toggles, personalized context gating & empirical telemetry controls.",
    },
    {
      id: "8",
      label: "Trust Panel",
      tech: "governance_store.py",
      status: "Active (PP1)",
      statusType: "active",
      desc: "Persistent human-in-the-loop audit log with Confirm & Reject decision controls.",
    },
  ];

  return (
    <section
      aria-label="System Architecture & Evaluator Diagnostics"
      className="mb-6 rounded-2xl border border-blue-200/90 bg-gradient-to-br from-white via-blue-50/20 to-neutral-50 p-5 shadow-xs dark:border-blue-900/60 dark:from-neutral-900 dark:via-neutral-900 dark:to-neutral-900 transition-all"
    >
      {/* Header bar */}
      <div className="flex flex-col gap-2 border-b border-neutral-200/80 pb-4 dark:border-neutral-800 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <span className="flex size-2 rounded-full bg-blue-600"></span>
            <span className="text-[11px] font-bold uppercase tracking-wider text-blue-900 dark:text-blue-300">
              System Architecture & Evaluator Diagnostics
            </span>
            <span className="rounded-full bg-blue-100 px-2 py-0.5 text-[10px] font-bold text-blue-700 dark:bg-blue-950 dark:text-blue-300">
              PP1 Component 4
            </span>
          </div>
          <h3 className="mt-1 text-base font-bold text-neutral-900 dark:text-white">
            FinTrust AI Observable Agentic Architecture
          </h3>
          <p className="text-xs text-neutral-500">
            End-to-end technical overview of implemented reasoning loops, tool gateways, grounding validators, and governance logs.
          </p>
        </div>

        {onClose && (
          <button
            onClick={onClose}
            className="self-start rounded-lg border border-neutral-300 bg-white px-3 py-1.5 text-xs font-semibold text-neutral-700 shadow-2xs hover:bg-neutral-100 dark:border-neutral-700 dark:bg-neutral-800 dark:text-neutral-200 dark:hover:bg-neutral-700 sm:self-center transition"
          >
            Hide Diagnostics
          </button>
        )}
      </div>

      {/* 8-Stage Architecture Pipeline */}
      <div className="mt-4">
        <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-neutral-400">
          Component 4 Pipeline Flow & Component Classification
        </p>
        <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2 lg:grid-cols-4">
          {pipelineStages.map((stage) => {
            const isPlanned = stage.statusType === "planned";
            const isGateway = stage.statusType === "gateway";
            return (
              <div
                key={stage.id}
                className={`relative flex flex-col justify-between rounded-xl border p-3 shadow-2xs transition ${
                  isPlanned
                    ? "border-amber-300/80 bg-amber-50/50 dark:border-amber-900/60 dark:bg-amber-950/20"
                    : isGateway
                    ? "border-blue-200 bg-white dark:border-blue-950 dark:bg-neutral-800/80"
                    : "border-neutral-200 bg-white dark:border-neutral-800 dark:bg-neutral-800/70"
                }`}
              >
                <div>
                  <div className="flex items-center justify-between gap-1">
                    <span className="text-[10px] font-mono font-bold text-neutral-400">
                      STAGE {stage.id}
                    </span>
                    <span
                      className={`rounded-full px-2 py-0.5 text-[9px] font-bold ${
                        isPlanned
                          ? "bg-amber-100 text-amber-800 dark:bg-amber-900/80 dark:text-amber-200"
                          : isGateway
                          ? "bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-200"
                          : "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200"
                      }`}
                    >
                      {stage.status}
                    </span>
                  </div>
                  <h4 className="mt-1 text-xs font-bold text-neutral-900 dark:text-white">
                    {stage.label}
                  </h4>
                  <p className="font-mono text-[10px] text-blue-600 dark:text-blue-400">
                    {stage.tech}
                  </p>
                  <p className="mt-1.5 text-[11px] leading-relaxed text-neutral-600 dark:text-neutral-400">
                    {stage.desc}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Relocated Research Focus & Academic Contribution Card */}
      <div className="mt-4 rounded-xl border border-blue-200/70 bg-blue-50/60 p-3.5 dark:border-blue-900/40 dark:bg-blue-950/30">
        <div className="flex items-start gap-3">
          <ShieldIcon className="size-5 shrink-0 text-blue-600 dark:text-blue-400 mt-0.5" />
          <div className="text-xs">
            <span className="font-bold text-blue-950 dark:text-blue-200">
              Academic & Research Focus (PP1 Evaluation)
            </span>
            <p className="mt-1 leading-relaxed text-neutral-700 dark:text-neutral-300">
              Component 4 research investigates whether transparent multi-service agentic reasoning combined with local explainability (SHAP) and persistent human-in-the-loop oversight improves trust in automated fintech decisions. All explanations are grounded strictly in retrieved tool telemetry. High risk scores and missing blockchain records represent heuristic anomaly detections, not verified criminal convictions.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

export const SystemArchitecturePanel = ResearchBanner;
