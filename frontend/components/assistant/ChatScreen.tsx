"use client";

import React, { useState } from "react";
import { ChatMessage, TabType } from "./types";
import { MOCK_INITIAL_MESSAGES, SUGGESTED_QUESTIONS } from "./mockData";
import { AgentWorkflowPanel } from "./AgentWorkflowPanel";
import { BotIcon, SparklesIcon, ChevronRightIcon, WrenchIcon, ShieldIcon } from "./icons";

import { request } from "@/lib/api/client";

interface Props {
  onNavigateToExplanation: () => void;
}

export function ChatScreen({ onNavigateToExplanation }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>(MOCK_INITIAL_MESSAGES);
  const [inputText, setInputText] = useState("");
  const [isSimulatingResponse, setIsSimulatingResponse] = useState(false);

  const handleSendMessage = async (textToSend?: string) => {
    const text = textToSend || inputText;
    if (!text.trim() || isSimulatingResponse) return;

    const userMsg: ChatMessage = {
      id: `usr_${Date.now()}`,
      sender: "user",
      text: text,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setMessages((prev) => [...prev, userMsg]);
    if (!textToSend) setInputText("");
    setIsSimulatingResponse(true);

    try {
      // Call live FastAPI backend at http://localhost:8003/assistant/chat
      const response = await request<{
        answer: string;
        agent_execution?: { tools_used?: string[] };
        evidence?: Array<{ tool: string; result: unknown }>;
      }>("assistant", "/assistant/chat", {
        method: "POST",
        body: { message: text },
        auth: false,
      });

      const assistantMsg: ChatMessage = {
        id: `ast_${Date.now()}`,
        sender: "assistant",
        text: response.answer || "No response received.",
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        toolsCalled: response.agent_execution?.tools_used || [],
        evidenceUsed: (response.evidence || []).map((e) => e.tool),
        hasWhyButton: true,
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch {
      // Fallback to local research mock data if backend is offline
      let assistantResponse: ChatMessage;

      if (text.toLowerCase().includes("risk") || text.toLowerCase().includes("portfolio")) {
        assistantResponse = {
          id: `ast_${Date.now()}`,
          sender: "assistant",
          text: "Your overall portfolio risk profile is moderate. However, recent transaction TX1001 flagged a localized spike due to non-standard transfer velocity and location telemetry.",
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          toolsCalled: ["get_portfolio_analysis", "get_fraud_analysis"],
          evidenceUsed: ["Portfolio Telemetry", "SHAP Attribution"],
          hasWhyButton: true,
        };
      } else if (text.toLowerCase().includes("evidence")) {
        assistantResponse = {
          id: `ast_${Date.now()}`,
          sender: "assistant",
          text: "The decision is backed by verified evidence: Fraud Analysis Model (Score 0.92), Blockchain Record AUDIT-2026-001 (Block 1847291), and SHAP feature attributions.",
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          toolsCalled: ["get_blockchain_audit", "get_fraud_analysis"],
          evidenceUsed: ["Blockchain Audit", "SHAP Explanation"],
          hasWhyButton: true,
        };
      } else {
        assistantResponse = {
          id: `ast_${Date.now()}`,
          sender: "assistant",
          text: "TX1001 was blocked because the transaction was classified as high risk. The strongest contributing factors were the unusually high transfer amount, unusual location, and use of a new device.",
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          toolsCalled: ["get_fraud_analysis", "get_blockchain_audit"],
          evidenceUsed: ["Fraud Analysis", "Blockchain Audit", "SHAP Explanation"],
          hasWhyButton: true,
        };
      }

      setMessages((prev) => [...prev, assistantResponse]);
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
            className={`flex flex-col ${
              msg.sender === "user" ? "items-end" : "items-start"
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
              className={`max-w-2xl rounded-2xl p-4 text-sm leading-relaxed ${
                msg.sender === "user"
                  ? "bg-blue-600 text-white shadow-xs"
                  : "border border-neutral-200 bg-neutral-50 text-neutral-900 shadow-2xs dark:border-neutral-800 dark:bg-neutral-800 dark:text-white"
              }`}
            >
              {msg.text}

              {/* Assistant Evidence & Tool Attachments */}
              {msg.sender === "assistant" && (
                <div className="mt-3.5 border-t border-neutral-200/80 pt-3 dark:border-neutral-700">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    {/* Evidence Badges */}
                    {msg.evidenceUsed && (
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
                    {msg.toolsCalled && (
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
                        SLM generated natural explanation grounded in SHAP attributions
                      </span>
                      <button
                        onClick={onNavigateToExplanation}
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
        ))}

        {isSimulatingResponse && (
          <div className="flex items-center gap-2 text-xs text-neutral-500">
            <BotIcon className="size-4 animate-bounce text-blue-600" />
            <span>Agentic LLM is analyzing evidence and calling tools...</span>
          </div>
        )}
      </div>

      {/* Suggested Questions */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-semibold text-neutral-500">Suggested:</span>
        {SUGGESTED_QUESTIONS.map((q) => (
          <button
            key={q}
            onClick={() => handleSendMessage(q)}
            className="flex items-center gap-1 rounded-full border border-blue-200/80 bg-white px-3 py-1.5 text-xs font-medium text-blue-700 transition hover:border-blue-400 hover:bg-blue-50 dark:border-blue-900/50 dark:bg-neutral-900 dark:text-blue-300 dark:hover:bg-neutral-800"
          >
            <SparklesIcon className="size-3 text-blue-500" />
            <span>{q}</span>
          </button>
        ))}
      </div>

      {/* Chat Input Box */}
      <div className="flex items-center gap-2 rounded-xl border border-black/15 bg-white p-2 shadow-xs dark:border-white/20 dark:bg-neutral-900">
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
      </div>
    </div>
  );
}
