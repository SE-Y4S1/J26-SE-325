"use client";

import React, { Suspense, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { TabType, ChatMessage } from "@/components/assistant/types";
import { MOCK_INITIAL_MESSAGES } from "@/components/assistant/mockData";
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

  const [messages, setMessages] = useState<ChatMessage[]>(MOCK_INITIAL_MESSAGES);
  const [latestEvidence, setLatestEvidence] = useState<Array<{ tool: string; result: string }>>([]);
  const [selectedEvidence, setSelectedEvidence] = useState<Array<{ tool: string; result: string }> | null>(null);
  const [isArchitectureOpen, setIsArchitectureOpen] = useState(false);

  const activeEvidence = selectedEvidence ?? latestEvidence;

  const handleSelectTab = (tab: TabType) => {
    if (tab === "assistant") {
      setSelectedEvidence(null);
      router.push("/assistant");
    } else {
      router.push(`/assistant?tab=${tab}`);
    }
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
            <TrustPanelScreen latestEvidence={activeEvidence} />
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
