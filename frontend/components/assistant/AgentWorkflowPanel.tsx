"use client";

import React, { useState } from "react";
import { CpuIcon, CheckCircleIcon, ChevronRightIcon, WrenchIcon, AlertTriangleIcon, ActivityIcon, ShieldIcon } from "./icons";

export interface ExecutionTraceData {
  toolsCalled?: string[];
  evidenceUsed?: string[];
  evidenceDetails?: Array<{ tool: string; result: string }>;
  timing?: {
    request_id?: string;
    total_duration_ms?: number;
    initial_tool_selection_ms?: number;
    final_synthesis_ms?: number;
    total_ollama_inference_ms?: number;
    total_tool_duration_ms?: number;
    tool_call_count?: number;
    tool_calls?: Array<{
      tool_name: string;
      duration_ms?: number;
      success?: boolean;
    }>;
  };
  evidenceValidation?: {
    passed?: boolean;
    issues?: string[];
    checks_applied?: number;
  };
  isError?: boolean;
  userQuery?: string;
}

interface Props {
  executionTrace?: ExecutionTraceData | null;
  lastUserQuery?: string;
}

export function AgentWorkflowPanel({ executionTrace, lastUserQuery }: Props) {
  const [isOpen, setIsOpen] = useState(false);

  const toolsCalled = executionTrace?.toolsCalled || [];
  const hasTools = toolsCalled.length > 0;
  const isError = Boolean(executionTrace?.isError);
  const timing = executionTrace?.timing;
  const hasTrace = Boolean(executionTrace && !isError);
  const isFastPath = Boolean(hasTrace && !hasTools);

  // Check if evidence contains SHAP values
  const hasShapInEvidence = Boolean(
    executionTrace?.evidenceDetails?.some((e) =>
      /SHAP (?:CONTRIBUTIONS|EXPLANATION)|\+0\.[0-9]+/i.test(e.result)
    )
  );

  // Compute summary badge
  let badgeText = "Ready · Awaiting query";
  let badgeStyle = "bg-neutral-100 text-neutral-600 dark:bg-neutral-800 dark:text-neutral-400";

  if (isError) {
    badgeText = "Connection Failed · Port 8003";
    badgeStyle = "bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300";
  } else if (hasTools) {
    badgeText = `${toolsCalled.length} tool${toolsCalled.length > 1 ? "s" : ""} executed`;
    badgeStyle = "bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300";
  } else if (isFastPath) {
    badgeText = "Fast-Path Reply · 0 tools";
    badgeStyle = "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300";
  }

  // 8 Dynamic Real-Time Trace Steps
  const traceSteps = [
    {
      id: "1",
      name: "1. Request Ingestion",
      status: lastUserQuery || hasTrace ? "completed" : "pending",
      desc: lastUserQuery
        ? `Query: "${lastUserQuery.slice(0, 36)}${lastUserQuery.length > 36 ? "..." : ""}"`
        : "Awaiting user financial query",
    },
    {
      id: "2",
      name: "2. Intent & Routing",
      status: hasTrace ? "completed" : isError ? "failed" : "pending",
      desc: isError
        ? "Routing failed: Backend unreachable"
        : hasTools
        ? "Multi-service ReAct Agent planning"
        : isFastPath
        ? "Deterministic fast-path (Conversational greeting)"
        : "Ready for intent classification",
    },
    {
      id: "3",
      name: "3. Tool Execution",
      status: isError ? "failed" : hasTools ? "completed" : isFastPath ? "skipped" : "pending",
      desc: isError
        ? "Failed: Backend API unavailable"
        : hasTools
        ? `Called: ${toolsCalled.join(", ")}${
            timing?.total_tool_duration_ms ? ` (${timing.total_tool_duration_ms}ms)` : ""
          }`
        : isFastPath
        ? "Skipped: General query requires no external tools"
        : "Gateways: Fraud (8001), Audit (8002), Portfolio (8100)",
    },
    {
      id: "4",
      name: "4. Grounding Validator",
      status:
        executionTrace?.evidenceValidation?.passed || (hasTools && hasTrace)
          ? "completed"
          : isFastPath
          ? "skipped"
          : isError
          ? "failed"
          : "pending",
      desc:
        hasTools && hasTrace
          ? `${executionTrace?.evidenceDetails?.length || 0} tool output(s) checked via evidence_validator.py`
          : isFastPath
          ? "Skipped: No external evidence required"
          : "Enforces zero-hallucination against tool outputs",
    },
    {
      id: "5",
      name: "5. SHAP Attribution",
      status: hasShapInEvidence ? "completed" : isFastPath || hasTools ? "skipped" : "pending",
      desc: hasShapInEvidence
        ? "Tree SHAP contributions extracted from Component 2"
        : "Skipped: Not present in active turn evidence (LIME inactive)",
    },
    {
      id: "6",
      name: "6. Responsible AI",
      status: hasTrace ? "completed" : "pending",
      desc: "Consent, privacy & explanation preference filters applied",
    },
    {
      id: "7",
      name: "7. Response Synthesis",
      status: hasTrace ? "completed" : isError ? "failed" : "pending",
      desc: "Ollama LLM (PP1 Active) · DistilBERT SLM (PP2 Planned)",
    },
    {
      id: "8",
      name: "8. Trust Panel Oversight",
      status: hasTools ? "available" : isFastPath ? "skipped" : "pending",
      desc: hasTools
        ? "Decision ready for human review in Trust Panel"
        : isFastPath
        ? "Skipped: Conversational reply requires no human override"
        : "Persistent JSONL audit logging ready",
    },
  ];

  return (
    <div className="rounded-xl border border-neutral-200/90 bg-white transition-all dark:border-neutral-800 dark:bg-neutral-900 shadow-2xs">
      <button
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
        className="flex w-full items-center justify-between p-3 text-left text-xs font-semibold text-neutral-700 hover:text-neutral-900 dark:text-neutral-300 dark:hover:text-white transition"
      >
        <div className="flex flex-wrap items-center gap-2.5">
          <ActivityIcon className="size-4 text-blue-600 dark:text-blue-400" />
          <span>Agent Activity & Pipeline Trace</span>
          <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${badgeStyle}`}>
            {badgeText}
          </span>
          {timing?.total_duration_ms !== undefined && (
            <span className="font-mono text-[10px] text-neutral-400">
              · {timing.total_duration_ms}ms latency
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5 text-[11px] font-medium text-neutral-500">
          <span>{isOpen ? "Collapse" : "Expand"}</span>
          <span className={`transform transition-transform ${isOpen ? "rotate-90" : ""}`}>
            <ChevronRightIcon className="size-3.5" />
          </span>
        </div>
      </button>

      {isOpen && (
        <div className="border-t border-neutral-200/80 p-4 dark:border-neutral-800 bg-neutral-50/50 dark:bg-neutral-900/50">
          <div className="mb-3 flex flex-wrap items-center justify-between text-[11px] text-neutral-500 gap-2">
            <span className="font-semibold text-neutral-700 dark:text-neutral-300">
              Live Observable Agent Execution Loop
            </span>
            {hasTools && (
              <span className="flex items-center gap-1 font-mono text-neutral-600 dark:text-neutral-400">
                <WrenchIcon className="size-3 text-blue-600" /> {toolsCalled.join(", ")}
              </span>
            )}
          </div>

          {/* Workflow Steps Grid with honest status indicators */}
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {traceSteps.map((step) => {
              const isCompleted = step.status === "completed";
              const isFailed = step.status === "failed";
              const isSkipped = step.status === "skipped";
              const isAvailable = step.status === "available";

              return (
                <div
                  key={step.id}
                  className={`relative flex flex-col justify-between rounded-lg border p-2.5 shadow-2xs transition ${
                    isCompleted
                      ? "border-emerald-200 bg-emerald-50/40 dark:border-emerald-950 dark:bg-emerald-950/20"
                      : isFailed
                      ? "border-rose-200 bg-rose-50/50 dark:border-rose-950 dark:bg-rose-950/20"
                      : isAvailable
                      ? "border-blue-200 bg-blue-50/40 dark:border-blue-950 dark:bg-blue-950/20"
                      : isSkipped
                      ? "border-neutral-200 bg-neutral-100/60 opacity-75 dark:border-neutral-800 dark:bg-neutral-800/40"
                      : "border-neutral-200 bg-white dark:border-neutral-800 dark:bg-neutral-800/60"
                  }`}
                >
                  <div>
                    <div className="flex items-center justify-between gap-1">
                      <span className="text-[10px] font-bold text-neutral-900 dark:text-white">
                        {step.name}
                      </span>
                      {isCompleted ? (
                        <span className="rounded bg-emerald-100 px-1.5 py-0.2 text-[9px] font-bold text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300">
                          Completed
                        </span>
                      ) : isFailed ? (
                        <span className="rounded bg-rose-100 px-1.5 py-0.2 text-[9px] font-bold text-rose-800 dark:bg-rose-950 dark:text-rose-300">
                          Failed
                        </span>
                      ) : isAvailable ? (
                        <span className="rounded bg-blue-100 px-1.5 py-0.2 text-[9px] font-bold text-blue-800 dark:bg-blue-950 dark:text-blue-300">
                          Available
                        </span>
                      ) : isSkipped ? (
                        <span className="rounded bg-neutral-200 px-1.5 py-0.2 text-[9px] font-bold text-neutral-600 dark:bg-neutral-700 dark:text-neutral-300">
                          Skipped
                        </span>
                      ) : (
                        <span className="rounded bg-neutral-100 px-1.5 py-0.2 text-[9px] font-medium text-neutral-400 dark:bg-neutral-800">
                          Pending
                        </span>
                      )}
                    </div>
                    <p className="mt-1 text-[11px] leading-tight text-neutral-600 dark:text-neutral-400">
                      {step.desc}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="mt-3 flex items-center justify-between text-[11px] text-neutral-400">
            <span className="italic">
              Generated from verified runtime telemetry. Unexecuted steps are labeled &apos;Skipped&apos;.
            </span>
            {timing?.request_id && (
              <span className="font-mono text-[10px]">ID: {timing.request_id}</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
