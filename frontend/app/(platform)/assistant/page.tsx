"use client";

import React, { Suspense, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { TabType, ChatMessage } from "@/components/assistant/types";
import { INITIAL_ASSISTANT_MESSAGES, MOCK_INITIAL_MESSAGES } from "@/components/assistant/mockData";
import { AssistantHeader } from "@/components/assistant/AssistantHeader";
import { AssistantSidebar } from "@/components/assistant/AssistantSidebar";
import { ResearchBanner } from "@/components/assistant/ResearchBanner";
import { ChatScreen } from "@/components/assistant/ChatScreen";
import { ExplanationScreen } from "@/components/assistant/ExplanationScreen";
import { TrustPanelScreen } from "@/components/assistant/TrustPanelScreen";
import { ResponsibleAIScreen } from "@/components/assistant/ResponsibleAIScreen";
import { SettingsScreen } from "@/components/assistant/SettingsScreen";
import { Spinner } from "@/components/ui";

const VALID_TABS: TabType[] = ["assistant", "explanation", "trust-panel", "responsible-ai", "settings"];

function AssistantContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const tabQuery = searchParams.get("tab") as TabType | null;
  const activeTab: TabType = tabQuery && VALID_TABS.includes(tabQuery) ? tabQuery : "assistant";
  const selectedTx = searchParams.get("tx") || searchParams.get("transactionId") || undefined;
  const demoMode = searchParams.get("demo") === "true";

  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    if (demoMode) return MOCK_INITIAL_MESSAGES;
    if (selectedTx) {
      return [
        {
          id: "welcome_tx",
          sender: "assistant",
          text: `Hello! Transaction ${selectedTx} is selected. I can explain its fraud score, gateway decision, and contributing factors from Component 2. How can I assist you?`,
          timestamp: "Just now",
          hasWhyButton: false,
          hasTrustPanelButton: false,
        },
      ];
    }
    return INITIAL_ASSISTANT_MESSAGES;
  });

  const [latestEvidence, setLatestEvidence] = useState<Array<{ tool: string; result: string }>>([]);
  const [selectedEvidence, setSelectedEvidence] = useState<Array<{ tool: string; result: string }> | null>(null);
  const [isArchitectureOpen, setIsArchitectureOpen] = useState(false);

  const activeEvidence = selectedEvidence ?? latestEvidence;

  // If selectedTx changes on the URL before the user has started chatting, update the greeting
  React.useEffect(() => {
    if (demoMode) return;
    setMessages((prev) => {
      if (prev.length <= 1 && prev[0]?.sender === "assistant") {
        if (selectedTx) {
          return [
            {
              id: "welcome_tx",
              sender: "assistant",
              text: `Hello! Transaction ${selectedTx} is selected. I can explain its fraud score, gateway decision, and contributing factors from Component 2. How can I assist you?`,
              timestamp: "Just now",
              hasWhyButton: false,
              hasTrustPanelButton: false,
            },
          ];
        } else {
          return INITIAL_ASSISTANT_MESSAGES;
        }
      }
      return prev;
    });
  }, [selectedTx, demoMode]);

  const handleSelectTab = (tab: TabType) => {
    const params = new URLSearchParams();
    if (tab !== "assistant") {
      params.set("tab", tab);
    } else {
      setSelectedEvidence(null);
    }
    if (selectedTx) {
      params.set("tx", selectedTx);
    }
    if (demoMode) {
      params.set("demo", "true");
    }
    const q = params.toString();
    router.push(q ? `/assistant?${q}` : "/assistant");
  };

  const handleToggleArchitecture = () => {
    setIsArchitectureOpen((prev) => !prev);
  };

  return (
    <div className="flex flex-col gap-2">
      {/* Top Header */}
      <AssistantHeader
        isArchitectureOpen={isArchitectureOpen}
        onToggleArchitecture={handleToggleArchitecture}
      />

      {/* Unobtrusive System Architecture & Diagnostics Panel (Collapsed by default) */}
      <ResearchBanner
        isOpen={isArchitectureOpen}
        onClose={() => setIsArchitectureOpen(false)}
      />

      {/* Main Content Layout with Sidebar */}
      <div className="flex flex-col gap-6 md:flex-row">
        {/* Left Navigation Sidebar */}
        <AssistantSidebar
          activeTab={activeTab}
          onSelectTab={handleSelectTab}
          isArchitectureOpen={isArchitectureOpen}
          onToggleArchitecture={handleToggleArchitecture}
        />

        {/* Main Active Screen */}
        <main className="flex-1">
          {activeTab === "assistant" && (
            <ChatScreen
              messages={messages}
              setMessages={setMessages}
              selectedTransactionId={selectedTx}
              demoMode={demoMode}
              onNavigateToExplanation={(evidence) => {
                setSelectedEvidence(evidence ?? latestEvidence);
                handleSelectTab("explanation");
              }}
              onNavigateToTrustPanel={() => handleSelectTab("trust-panel")}
              onEvidenceReceived={(evidence) => {
                setLatestEvidence(evidence);
                setSelectedEvidence(null);
              }}
            />
          )}

          {activeTab === "explanation" && (
            <ExplanationScreen
              onNavigateToTrustPanel={() => handleSelectTab("trust-panel")}
              latestEvidence={activeEvidence}
            />
          )}

          {activeTab === "trust-panel" && (
            <TrustPanelScreen
              latestEvidence={activeEvidence}
              selectedTransactionId={selectedTx}
            />
          )}

          {activeTab === "responsible-ai" && (
            <ResponsibleAIScreen />
          )}

          {activeTab === "settings" && (
            <SettingsScreen />
          )}
        </main>
      </div>
    </div>
  );
}

export default function AssistantPage() {
  return (
    <Suspense
      fallback={
        <div className="flex h-64 items-center justify-center">
          <Spinner label="Loading assistant..." />
        </div>
      }
    >
      <AssistantContent />
    </Suspense>
  );
}
