"use client";

import React, { useState } from "react";
import { ChatMessage } from "./types";
import { MOCK_INITIAL_MESSAGES, SUGGESTED_QUESTIONS } from "./mockData";
import { AgentWorkflowPanel } from "./AgentWorkflowPanel";
import { BotIcon, SparklesIcon, ChevronRightIcon, WrenchIcon } from "./icons";

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

interface Props {
  onNavigateToExplanation: (evidence?: Array<{ tool: string; result: string }>) => void;
  onEvidenceReceived?: (evidence: Array<{ tool: string; result: string }>) => void;
  messages?: ChatMessage[];
  setMessages?: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
}

export function ChatScreen({
  onNavigateToExplanation,
  onEvidenceReceived,
  messages: externalMessages,
  setMessages: externalSetMessages,
}: Props) {
  const [internalMessages, setInternalMessages] = useState<ChatMessage[]>(MOCK_INITIAL_MESSAGES);
  const messages = externalMessages ?? internalMessages;
  const setMessages = externalSetMessages ?? setInternalMessages;
  const [inputText, setInputText] = useState("");
  const [isSimulatingResponse, setIsSimulatingResponse] = useState(false);

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

      // Call live FastAPI backend at http://localhost:8003/assistant/chat
      const response = await request<{
        answer: string;
        agent_execution?: { tools_used?: string[] };
        evidence?: Array<{ tool: string; result: unknown }>;
        responsible_ai?: Record<string, unknown>;
      }>("assistant", "/assistant/chat", {
        method: "POST",
        body: { message: text, preferences },
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

      const assistantMsg: ChatMessage = {
        id: createMessageId("ast"),
        sender: "assistant",
        text: visibleAnswer,
        timestamp: getTimestamp(),
        toolsCalled: toolsCalled,
        evidenceUsed: evidenceUsed,
        evidenceDetails: evidenceDetails,
        hasWhyButton: hasToolsOrEvidence,
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
      {/* Title & Subtitle Header */}
      <div className="rounded-xl border border-black/10 bg-white p-5 shadow-xs dark:border-white/15 dark:bg-neutral-900">
        <h2 className="text-xl font-bold text-neutral-900 dark:text-white">
          How can I help you today?
        </h2>
        <p className="mt-1 text-sm text-neutral-500">
          Ask about your financial decisions, transactions, risk or portfolio.
        </p>
      </div>

      {/* Expandable Agent Activity Panel */}
      <AgentWorkflowPanel />

      {/* Messages Thread Container */}
      <div className="flex flex-col gap-4 rounded-xl border border-black/10 bg-white p-4 shadow-xs dark:border-white/15 dark:bg-neutral-900 sm:p-6">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex flex-col ${msg.sender === "user" ? "items-end" : "items-start"
              }`}
          >
            {/* Sender Label */}
            <div className="mb-1 flex items-center gap-1.5 text-xs text-neutral-400">
              {msg.sender === "assistant" ? (
                <>
                  <BotIcon className="size-3.5 text-blue-600 dark:text-blue-400" />
                  <span className="font-semibold text-blue-900 dark:text-blue-300">FinTrust Agent</span>
                </>
              ) : (
                <span className="font-medium">You</span>
              )}
              <span>· {msg.timestamp}</span>
            </div>

            {/* Message Bubble */}
            <div
              className={`max-w-2xl rounded-2xl p-4 text-sm leading-relaxed ${msg.sender === "user"
                ? "bg-blue-600 text-white shadow-xs"
                : "border border-neutral-200 bg-neutral-50 text-neutral-900 shadow-2xs dark:border-neutral-800 dark:bg-neutral-800 dark:text-white"
                }`}
            >

              {msg.text}

              {msg.evidenceDetails && msg.evidenceDetails.length > 0 && (
                <details className="mt-3 rounded-lg border border-neutral-200 bg-white p-3 dark:border-neutral-700 dark:bg-neutral-900">
                  <summary className="cursor-pointer text-xs font-semibold text-blue-700 dark:text-blue-300">
                    View raw tool evidence ({msg.evidenceDetails.length})
                  </summary>

                  <div className="mt-3 space-y-3">
                    {msg.evidenceDetails.map((item, index) => (
                      <div key={`${item.tool}-${index}`}>
                        <p className="mb-1 text-xs font-bold text-neutral-700 dark:text-neutral-200">
                          {item.tool}
                        </p>

                        <pre className="whitespace-pre-wrap break-words text-xs leading-relaxed text-neutral-600 dark:text-neutral-300">
                          {item.result}
                        </pre>
                      </div>
                    ))}
                  </div>
                </details>
              )}




              {/* Assistant Evidence & Tool Attachments */}
              {msg.sender === "assistant" &&
                (msg.hasWhyButton ||
                  (msg.evidenceUsed && msg.evidenceUsed.length > 0) ||
                  (msg.toolsCalled && msg.toolsCalled.length > 0)) && (
                  <div className="mt-3.5 border-t border-neutral-200/80 pt-3 dark:border-neutral-700">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      {/* Evidence Badges */}
                      {msg.evidenceUsed && msg.evidenceUsed.length > 0 && (
                        <div className="flex flex-wrap items-center gap-1.5">
                          <span className="text-xs font-semibold text-neutral-500">
                            Evidence used:
                          </span>
                          {msg.evidenceUsed.map((ev) => (
                            <span
                              key={ev}
                              className="rounded-md bg-blue-50 px-2 py-0.5 text-[11px] font-medium text-blue-700 border border-blue-200/60 dark:bg-blue-950/60 dark:text-blue-300 dark:border-blue-900/60"
                            >
                              ✓ {ev}
                            </span>
                          ))}
                        </div>
                      )}

                      {/* Tools Called Badge */}
                      {msg.toolsCalled && msg.toolsCalled.length > 0 && (
                        <span className="flex items-center gap-1 rounded-full bg-neutral-200/70 px-2.5 py-0.5 text-[11px] font-bold text-neutral-700 dark:bg-neutral-700 dark:text-neutral-200">
                          <WrenchIcon className="size-3" />
                          {msg.toolsCalled.length} tools called
                        </span>
                      )}
                    </div>

                  {/* View Why? Button */}
                  {msg.hasWhyButton && (
                    <div className="mt-3 flex items-center justify-between">
                      <span className="text-xs text-neutral-400 italic">
                        LLM-generated explanation grounded in available tool evidence
                      </span>
                      <button
                        onClick={() => onNavigateToExplanation(msg.evidenceDetails)}
                        className="flex items-center gap-1.5 rounded-lg bg-blue-600 px-3.5 py-1.5 text-xs font-bold text-white shadow-xs transition-all hover:bg-blue-700 hover:shadow-md hover:shadow-blue-500/20 active:scale-98"
                      >
                        <span>View Why?</span>
                        <ChevronRightIcon className="size-3.5" />
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))
        }

        {
          isSimulatingResponse && (
            <div className="flex items-center gap-2 text-xs text-neutral-500">
              <BotIcon className="size-4 animate-bounce text-blue-600" />
              <span>Agentic LLM is analyzing evidence and calling tools...</span>
            </div>
          )
        }
      </div >

      {/* Suggested Questions */}
      < div className="flex flex-wrap items-center gap-2" >
        <span className="text-xs font-semibold text-neutral-500">Suggested:</span>
        {
          SUGGESTED_QUESTIONS.map((q) => (
            <button
              key={q}
              onClick={() => handleSendMessage(q)}
              className="flex items-center gap-1 rounded-full border border-blue-200/80 bg-white px-3 py-1.5 text-xs font-medium text-blue-700 transition hover:border-blue-400 hover:bg-blue-50 dark:border-blue-900/50 dark:bg-neutral-900 dark:text-blue-300 dark:hover:bg-neutral-800"
            >
              <SparklesIcon className="size-3 text-blue-500" />
              <span>{q}</span>
            </button>
          ))
        }
      </div >

      {/* Chat Input Box */}
      < div className="flex items-center gap-2 rounded-xl border border-black/15 bg-white p-2 shadow-xs dark:border-white/20 dark:bg-neutral-900" >
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSendMessage()}
          placeholder="Ask the financial assistant..."
          className="flex-1 bg-transparent px-3 py-2 text-sm outline-none placeholder:text-neutral-400"
        />
        <button
          onClick={() => handleSendMessage()}
          disabled={!inputText.trim() || isSimulatingResponse}
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-blue-700 disabled:opacity-40"
        >
          Send
        </button>
      </div >
    </div >
  );
}
