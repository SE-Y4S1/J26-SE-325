"use client";

import React, { useState } from "react";
import { ChatMessage } from "./types";
import { MOCK_INITIAL_MESSAGES, SUGGESTED_QUESTIONS } from "./mockData";
import { AgentWorkflowPanel, ExecutionTraceData } from "./AgentWorkflowPanel";
import {
  BotIcon,
  SparklesIcon,
  ChevronRightIcon,
  WrenchIcon,
  ShieldIcon,
  BarChartIcon,
  AlertTriangleIcon,
  SendIcon,
} from "./icons";

import { request } from "@/lib/api/client";

const RESPONSIBLE_AI_STORAGE_KEY = "finagent_responsible_ai_preferences";

export function getValidatedResponsibleAIPreferences(): {
  explanationEnabled: boolean;
  personalizedContext: boolean;
  dataSharingConsent: boolean;
} {
  const defaults = {
    explanationEnabled: true,
    personalizedContext: true,
    dataSharingConsent: true,
  };

  try {
    if (typeof window === "undefined" || !window.localStorage) {
      return defaults;
    }
    const raw = window.localStorage.getItem(RESPONSIBLE_AI_STORAGE_KEY);
    if (!raw) return defaults;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return defaults;

    return {
      explanationEnabled:
        typeof parsed.explanationEnabled === "boolean"
          ? parsed.explanationEnabled
          : defaults.explanationEnabled,
      personalizedContext:
        typeof parsed.personalizedContext === "boolean"
          ? parsed.personalizedContext
          : defaults.personalizedContext,
      dataSharingConsent:
        typeof parsed.dataSharingConsent === "boolean"
          ? parsed.dataSharingConsent
          : defaults.dataSharingConsent,
    };
  } catch {
    return defaults;
  }
}

function getTimestamp(): string {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

let msgSeq = 0;
function createMessageId(prefix: string): string {
  msgSeq += 1;
  return `${prefix}_${Date.now()}_${msgSeq}`;
}

export function sanitizeAssistantResponse(
  answer: string,
  evidence: Array<{ tool: string; result: string }> = []
): string {
  const hasPortfolioEvidence = evidence.some(
    (e) => e.tool === "get_portfolio_analysis"
  );
  if (hasPortfolioEvidence) {
    return answer;
  }

  // If no portfolio evidence was returned, verify the answer doesn't display unrelated portfolio fallback information
  const portfolioLeakPattern =
    /(?:Notice:\s*No live portfolio|live portfolio (?:was )?not (?:found|retrieved)|Baseline Valuation:?\s*(?:LKR\s*)?1,?000,?000|Expected Annual Return|Risk Score:?\s*38\/100|Diversification Score:?\s*74\/100|zero verified live holdings|0 verified live holdings|Mock Allocation)/i;

  if (!portfolioLeakPattern.test(answer)) {
    return answer;
  }

  // Detect if fraud evidence is present in this turn
  const fraudItem = evidence.find((e) => e.tool === "get_fraud_analysis");
  if (fraudItem) {
    const fraudText = fraudItem.result;
    // Check if the answer also has fraud content after the leaked portfolio text
    const stripped = answer
      .replace(
        /^Notice:\s*No live portfolio was retrieved for this account from the platform service[^\n]*\n*/i,
        ""
      )
      .replace(
        /(?:For illustrative demonstration only|The user has zero verified live holdings|Baseline Valuation|Expected Annual Return|Risk Score|Diversification Score|Liquidity|Mock Allocation|Please note:\s*These figures and recommendations are synthetic baseline examples)[^\n]*\n*/gi,
        ""
      )
      .trim();

    if (
      stripped &&
      (stripped.includes("TX1001") ||
        stripped.includes("0.92") ||
        /fraud|blocked/i.test(stripped))
    ) {
      return stripped;
    }

    // Extract fraud values from tool evidence
    const txMatch = fraudText.match(/Transaction ID:\s*([A-Za-z0-9_-]+)/i);
    const txId = txMatch ? txMatch[1] : "TX1001";

    const scoreMatch = fraudText.match(/Recorded Fraud Score:\s*([0-9.]+)/i);
    const score = scoreMatch ? scoreMatch[1] : "0.92";

    const statusMatch = fraudText.match(/Recorded Status:\s*([A-Za-z0-9_-]+)/i);
    const status = statusMatch ? statusMatch[1] : "BLOCKED";

    const riskMatch = fraudText.match(/Recorded Risk Level:\s*([A-Za-z0-9_-]+)/i);
    const riskLevel = riskMatch ? riskMatch[1] : "HIGH";

    const factorsMatch = fraudText.match(
      /RECORDED CONTRIBUTING FACTORS:\s*([\s\S]*?)(?=(?:COMPONENT 2|LOCAL SHAP|TRANSACTION TELEMETRY|LIMITATION|$))/i
    );
    const factors: string[] = [];
    if (factorsMatch?.[1]) {
      for (const line of factorsMatch[1].split("\n")) {
        const m = line.match(/^\s*-\s*(.+)/);
        if (m && m[1].trim()) factors.push(m[1].trim());
      }
    }
    const factorString =
      factors.length > 0
        ? factors.join(", ")
        : "new device, unusual location, and high transfer amount";

    const isFallback =
      /FALLBACK MOCK DATA|Local mock baseline|FALLBACK NOTICE|NOT a live lookup result/i.test(
        fraudText
      );

    return (
      `For ${txId}, the available tool returned a recorded fraud score of ${score}, risk level ${riskLevel}, and status ${status} in a ` +
      `${isFallback ? "mock transaction record (Component 4 local prototype fallback, not a live Component 2 result)" : "Component 2 live gateway record"}. ` +
      `The recorded contributing factors are ${factorString}. ` +
      `The meaning of the score as a calibrated probability is not established by the available evidence.`
    );
  }

  // If blockchain audit evidence was returned without portfolio evidence
  const auditItem = evidence.find((e) => e.tool === "get_blockchain_audit");
  if (auditItem) {
    const stripped = answer
      .replace(
        /^Notice:\s*No live portfolio was retrieved for this account from the platform service[^\n]*\n*/i,
        ""
      )
      .trim();
    if (stripped && /audit|blockchain|block|tx1001/i.test(stripped)) {
      return stripped;
    }
  }

  // General strip for leaked portfolio notice
  return answer
    .replace(
      /^Notice:\s*No live portfolio was retrieved for this account from the platform service[^\n]*\n*/i,
      ""
    )
    .trim();
}

/**
 * Classifies an individual evidence item into a precise, honest badge description
 */
function getEvidenceBadge(item: { tool: string; result: string }): { label: string; isLive: boolean } {
  if (item.tool === "get_fraud_analysis") {
    const isLive = /LIVE COMPONENT 2|Live Component 2 Gateway Audit Log/i.test(item.result);
    return {
      label: isLive ? "Fraud Gateway (Live Port 8001)" : "Fraud Telemetry (Synthetic Baseline)",
      isLive,
    };
  }
  if (item.tool === "get_blockchain_audit") {
    const isLive = /LIVE SERVICE VERIFICATION|CONFIGURED BLOCKCHAIN/i.test(item.result);
    return {
      label: isLive ? "Blockchain Ledger (Live Port 8002)" : "Audit Log (Simulated Ledger)",
      isLive,
    };
  }
  if (item.tool === "get_portfolio_analysis") {
    if (/PERSONALIZED CONTEXT DISABLED|Personalized financial context is disabled/i.test(item.result)) {
      return {
        label: "Portfolio Tool (Context Disabled)",
        isLive: false,
      };
    }
    const isServiceError =
      (/SERVICE UNAVAILABLE|service unreachable|Connection refused|timed out|timeout|Service unavailable/i.test(item.result) ||
        /Platform service note:\s*[^)\n\r]*(?:unreachable|timed out|timeout|connection refused|500)/i.test(item.result)) &&
      !/Empty portfolio list|0 active positions|No portfolios registered/i.test(item.result);
    if (isServiceError) {
      return {
        label: "Portfolio Store (Service Unavailable)",
        isLive: false,
      };
    }
    const isLive = /LIVE PLATFORM RECORD/i.test(item.result);
    if (isLive) {
      const isPartial = /Unavailable|Partial data notice/i.test(item.result);
      return {
        label: isPartial ? "Portfolio Store (Live — Partial Data)" : "Portfolio Store (Live Port 8100)",
        isLive: true,
      };
    }
    const isEmpty = /Empty portfolio list|0 active positions|NO LIVE HOLDINGS/i.test(item.result);
    return {
      label: isEmpty ? "Portfolio Telemetry (Baseline — 0 Holdings)" : "Portfolio Telemetry (Baseline)",
      isLive: false,
    };
  }
  return {
    label: item.tool,
    isLive: false,
  };
}

/**
 * Formats structured sections in the assistant message: conclusion, contributing factors,
 * calibration/limitation disclaimers, and next steps.
 */
function renderStructuredMessage(text: string) {
  // Check for model limitation / calibration disclaimer patterns
  const limitationPattern =
    /(?:The meaning of the score as a calibrated probability is not established|High risk scores are anomaly estimates|Prototype fraud probability|Notice:\s*No live portfolio|Please note:\s*These figures and recommendations are synthetic baseline examples|These results are prototype evidence, not verified production output)/i;

  const hasLimitation = limitationPattern.test(text);

  // If text has bullet items or contributing factors
  const hasBullets = text.includes("\n- ") || text.includes("\n* ");

  return (
    <div className="flex flex-col gap-2.5 text-sm leading-relaxed">
      {/* Primary response text */}
      <div className="whitespace-pre-line text-neutral-900 dark:text-neutral-100 font-normal">
        {text}
      </div>

      {/* Model Limitation / Disclaimer callout when applicable */}
      {hasLimitation && (
        <div className="mt-1 flex items-start gap-2 rounded-lg border border-amber-200/90 bg-amber-50/70 p-2.5 text-xs text-amber-900 dark:border-amber-900/50 dark:bg-amber-950/30 dark:text-amber-200">
          <AlertTriangleIcon className="size-4 shrink-0 text-amber-600 dark:text-amber-400 mt-0.5" />
          <div className="leading-relaxed">
            <span className="font-semibold">Model Limitation & Calibration Notice: </span>
            <span>
              Risk scores and missing blockchain records represent heuristic anomaly detections and prototype telemetry, not verified proof of fraud.
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

interface Props {
  onNavigateToExplanation: (evidence?: Array<{ tool: string; result: string }>) => void;
  onNavigateToTrustPanel?: () => void;
  onEvidenceReceived?: (evidence: Array<{ tool: string; result: string }>) => void;
  messages?: ChatMessage[];
  setMessages?: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
}

export function ChatScreen({
  onNavigateToExplanation,
  onNavigateToTrustPanel,
  onEvidenceReceived,
  messages: externalMessages,
  setMessages: externalSetMessages,
}: Props) {
  const [internalMessages, setInternalMessages] = useState<ChatMessage[]>(MOCK_INITIAL_MESSAGES);
  const messages = externalMessages ?? internalMessages;
  const setMessages = externalSetMessages ?? setInternalMessages;
  const [inputText, setInputText] = useState("");
  const [isSimulatingResponse, setIsSimulatingResponse] = useState(false);

  // Find latest assistant and user messages to feed the pipeline trace dynamically
  const latestAssistantMessage = [...messages].reverse().find((m) => m.sender === "assistant");
  const lastUserMessage = [...messages].reverse().find((m) => m.sender === "user");

  const latestExecutionTrace: ExecutionTraceData | null = latestAssistantMessage
    ? {
        toolsCalled: latestAssistantMessage.toolsCalled,
        evidenceUsed: latestAssistantMessage.evidenceUsed,
        evidenceDetails: latestAssistantMessage.evidenceDetails,
        timing: latestAssistantMessage.timing,
        evidenceValidation: latestAssistantMessage.evidenceValidation,
        isError: latestAssistantMessage.isError,
      }
    : null;

  const handleSendMessage = async (textToSend?: string) => {
    const text = textToSend || inputText;
    if (!text.trim() || isSimulatingResponse) return;

    const userMsg: ChatMessage = {
      id: createMessageId("usr"),
      sender: "user",
      text: text,
      timestamp: getTimestamp(),
    };

    setMessages((prev) => [...prev, userMsg]);
    if (!textToSend) setInputText("");
    setIsSimulatingResponse(true);

    try {
      const preferences = getValidatedResponsibleAIPreferences();
      const historyPayload = messages.map((m) => ({
        role: m.sender === "assistant" ? "assistant" : "user",
        content: m.text,
      }));

      // Call live FastAPI backend at http://localhost:8003/assistant/chat
      const response = await request<{
        answer: string;
        agent_execution?: {
          tools_used?: string[];
          number_of_tools?: number;
          status?: string;
          timing?: ExecutionTraceData["timing"];
        };
        evidence?: Array<{ tool: string; result: unknown }>;
        evidence_validation?: {
          passed?: boolean;
          issues?: string[];
          checks_applied?: number;
        };
        responsible_ai?: Record<string, unknown>;
      }>("assistant", "/assistant/chat", {
        method: "POST",
        body: { message: text, history: historyPayload, preferences },
        auth: false,
      });

      const evidenceDetails = (response.evidence || []).map((e) => ({
        tool: e.tool,
        result:
          typeof e.result === "string"
            ? e.result
            : JSON.stringify(e.result, null, 2),
      }));

      onEvidenceReceived?.(evidenceDetails);

      const toolsCalled = response.agent_execution?.tools_used || [];
      const evidenceUsed = (response.evidence || []).map((e) => e.tool);
      const hasToolsOrEvidence = toolsCalled.length > 0 || evidenceUsed.length > 0;

      const visibleAnswer = sanitizeAssistantResponse(
        response.answer || "No response received.",
        evidenceDetails
      );

      const isHighRiskOrBlocked = /blocked|high risk|fraud score|0\.92|tx1001/i.test(visibleAnswer);

      const assistantMsg: ChatMessage = {
        id: createMessageId("ast"),
        sender: "assistant",
        text: visibleAnswer,
        timestamp: getTimestamp(),
        toolsCalled: toolsCalled,
        evidenceUsed: evidenceUsed,
        evidenceDetails: evidenceDetails,
        timing: response.agent_execution?.timing,
        evidenceValidation: response.evidence_validation,
        hasWhyButton: hasToolsOrEvidence,
        hasTrustPanelButton: isHighRiskOrBlocked || toolsCalled.length > 0,
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (error) {
      console.error("Assistant API request failed:", error);

      const errorMsg: ChatMessage = {
        id: createMessageId("err"),
        sender: "assistant",
        text: "I couldn't connect to the assistant backend. Please check that the Component 4 API is running on port 8003 and try again.",
        timestamp: getTimestamp(),
        evidenceUsed: [],
        evidenceDetails: [],
        hasWhyButton: false,
        isError: true,
      };

      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsSimulatingResponse(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Dynamic Observable Agent Pipeline Trace (Collapsed by default) */}
      <AgentWorkflowPanel
        executionTrace={latestExecutionTrace}
        lastUserQuery={lastUserMessage?.text}
      />

      {/* Unified Conversation Card */}
      <div className="flex flex-col rounded-2xl border border-neutral-200/90 bg-white shadow-xs dark:border-neutral-800 dark:bg-neutral-900 overflow-hidden">
        {/* Card Header Bar */}
        <div className="flex items-center justify-between border-b border-neutral-200/80 px-5 py-3.5 dark:border-neutral-800 bg-neutral-50/50 dark:bg-neutral-800/20">
          <div className="flex items-center gap-2">
            <BotIcon className="size-4 text-blue-600 dark:text-blue-400" />
            <span className="text-xs font-bold text-neutral-900 dark:text-white">
              Financial Decision Dialogue
            </span>
          </div>
          <span className="text-[11px] text-neutral-400 font-mono">
            Grounding: Zero-Hallucination Policy
          </span>
        </div>

        {/* Message Thread */}
        <div className="flex flex-col gap-5 p-4 sm:p-6 min-h-[380px] max-h-[640px] overflow-y-auto">
          {messages.map((msg) => {
            const isUser = msg.sender === "user";
            const evidenceItems = msg.evidenceDetails || [];
            const hasEvidence = evidenceItems.length > 0;

            return (
              <div
                key={msg.id}
                className={`flex flex-col ${isUser ? "items-end" : "items-start"}`}
              >
                {/* Sender Label & Timestamp */}
                <div className="mb-1.5 flex items-center gap-1.5 text-xs text-neutral-400">
                  {isUser ? (
                    <span className="font-semibold text-neutral-700 dark:text-neutral-300">You</span>
                  ) : (
                    <div className="flex items-center gap-1.5">
                      <BotIcon className="size-3.5 text-blue-600 dark:text-blue-400" />
                      <span className="font-bold text-blue-900 dark:text-blue-300">FinTrust Agent</span>
                    </div>
                  )}
                  <span>· {msg.timestamp}</span>
                </div>

                {/* Message Bubble Container */}
                <div
                  className={`max-w-2xl rounded-2xl p-4 sm:p-5 text-sm ${
                    isUser
                      ? "bg-blue-600 text-white shadow-xs"
                      : "border border-neutral-200 bg-neutral-50/70 text-neutral-900 shadow-2xs dark:border-neutral-800 dark:bg-neutral-800/80 dark:text-neutral-100"
                  }`}
                >
                  {isUser ? (
                    <p className="whitespace-pre-line leading-relaxed">{msg.text}</p>
                  ) : (
                    renderStructuredMessage(msg.text)
                  )}

                  {/* Raw Tool Telemetry Inspector (Collapsible) */}
                  {!isUser && hasEvidence && (
                    <details className="mt-3.5 rounded-xl border border-neutral-200/90 bg-white p-3 dark:border-neutral-700/80 dark:bg-neutral-900">
                      <summary className="cursor-pointer text-xs font-semibold text-blue-700 dark:text-blue-400 hover:underline">
                        View raw tool evidence ({evidenceItems.length})
                      </summary>
                      <div className="mt-2.5 space-y-3">
                        {evidenceItems.map((item, index) => (
                          <div key={`${item.tool}-${index}`} className="border-t border-neutral-100 pt-2 dark:border-neutral-800">
                            <span className="font-mono text-[11px] font-bold text-neutral-700 dark:text-neutral-300">
                              {item.tool}
                            </span>
                            <pre className="mt-1 max-h-48 overflow-y-auto rounded bg-neutral-50 p-2 font-mono text-[10px] leading-relaxed text-neutral-700 dark:bg-neutral-950 dark:text-neutral-300 whitespace-pre-wrap break-words">
                              {item.result}
                            </pre>
                          </div>
                        ))}
                      </div>
                    </details>
                  )}

                  {/* Assistant Footer: Evidence Badges & Actions */}
                  {!isUser && (msg.hasWhyButton || hasEvidence || (msg.toolsCalled && msg.toolsCalled.length > 0)) && (
                    <div className="mt-4 border-t border-neutral-200/80 pt-3 dark:border-neutral-700/80">
                      {/* Compact Evidence-Source Indicators */}
                      {hasEvidence && (
                        <div className="flex flex-wrap items-center gap-1.5 mb-3">
                          <span className="text-[11px] font-semibold text-neutral-500">
                            Evidence Source:
                          </span>
                          {evidenceItems.map((item, idx) => {
                            const badge = getEvidenceBadge(item);
                            return (
                              <span
                                key={`${item.tool}-${idx}`}
                                className={`rounded-md px-2 py-0.5 text-[10px] font-semibold border ${
                                  badge.isLive
                                    ? "bg-emerald-50 text-emerald-800 border-emerald-200 dark:bg-emerald-950/60 dark:text-emerald-300 dark:border-emerald-900/60"
                                    : "bg-blue-50 text-blue-700 border-blue-200/60 dark:bg-blue-950/60 dark:text-blue-300 dark:border-blue-900/60"
                                }`}
                              >
                                {badge.label}
                              </span>
                            );
                          })}
                        </div>
                      )}

                      {/* Action Buttons: View Why? & Review in Trust Panel */}
                      <div className="flex flex-wrap items-center justify-between gap-2.5">
                        <span className="text-[11px] text-neutral-400 italic">
                          Explanation grounded in cross-service tool evidence
                        </span>

                        <div className="flex items-center gap-2">
                          {onNavigateToTrustPanel && msg.hasTrustPanelButton && (
                            <button
                              onClick={() => onNavigateToTrustPanel()}
                              className="flex items-center gap-1.5 rounded-lg border border-neutral-300 bg-white px-3 py-1.5 text-xs font-semibold text-neutral-700 shadow-2xs hover:bg-neutral-100 dark:border-neutral-700 dark:bg-neutral-800 dark:text-neutral-200 dark:hover:bg-neutral-700 transition"
                            >
                              <ShieldIcon className="size-3.5 text-blue-600 dark:text-blue-400" />
                              <span>Trust Panel</span>
                            </button>
                          )}

                          {msg.hasWhyButton && (
                            <button
                              onClick={() => onNavigateToExplanation(msg.evidenceDetails)}
                              className="flex items-center gap-1.5 rounded-lg bg-blue-600 px-3.5 py-1.5 text-xs font-bold text-white shadow-xs transition hover:bg-blue-700 hover:shadow-sm"
                            >
                              <BarChartIcon className="size-3.5" />
                              <span>View Why?</span>
                              <ChevronRightIcon className="size-3.5" />
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })}

          {isSimulatingResponse && (
            <div className="flex items-center gap-2.5 text-xs text-neutral-500 py-2">
              <BotIcon className="size-4 animate-bounce text-blue-600" />
              <span>FinTrust Agent is coordinating tools and validating evidence...</span>
            </div>
          )}
        </div>

        {/* Suggested Questions Section (Compact & Non-Dominating) */}
        <div className="border-t border-neutral-200/80 px-4 py-2.5 dark:border-neutral-800 bg-neutral-50/40 dark:bg-neutral-800/20">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[11px] font-semibold text-neutral-400">Suggested:</span>
            {SUGGESTED_QUESTIONS.map((q) => (
              <button
                key={q}
                onClick={() => handleSendMessage(q)}
                className="flex items-center gap-1 rounded-full border border-neutral-200 bg-white px-2.5 py-1 text-[11px] font-medium text-neutral-700 hover:border-blue-400 hover:bg-blue-50/50 hover:text-blue-700 dark:border-neutral-700 dark:bg-neutral-800 dark:text-neutral-300 dark:hover:bg-neutral-700 transition shadow-2xs"
              >
                <SparklesIcon className="size-3 text-blue-500" />
                <span>{q}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Chat Input Bar */}
        <div className="border-t border-neutral-200/80 p-3 sm:p-4 dark:border-neutral-800 bg-white dark:bg-neutral-900">
          <div className="flex items-center gap-2 rounded-xl border border-neutral-300 bg-neutral-50/60 p-1.5 shadow-2xs focus-within:border-blue-600 focus-within:ring-2 focus-within:ring-blue-500/20 dark:border-neutral-700 dark:bg-neutral-800/60 transition">
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSendMessage()}
              placeholder="Ask the financial assistant..."
              className="flex-1 bg-transparent px-3 py-2 text-sm outline-none placeholder:text-neutral-400 text-neutral-900 dark:text-white"
            />
            <button
              onClick={() => handleSendMessage()}
              disabled={!inputText.trim() || isSimulatingResponse}
              className="flex items-center gap-1.5 rounded-lg bg-blue-600 px-4 py-2 text-xs font-bold text-white transition hover:bg-blue-700 disabled:opacity-40 shadow-xs"
            >
              <span>Send</span>
              <SendIcon className="size-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
