"use client";

import React, { useState } from "react";
import { TabType, ChatMessage } from "@/components/assistant/types";
import { MOCK_INITIAL_MESSAGES } from "@/components/assistant/mockData";
import { AssistantHeader } from "@/components/assistant/AssistantHeader";
import { AssistantSidebar } from "@/components/assistant/AssistantSidebar";
import { ResearchBanner } from "@/components/assistant/ResearchBanner";
import { ChatScreen } from "@/components/assistant/ChatScreen";
import { ExplanationScreen } from "@/components/assistant/ExplanationScreen";
import { TrustPanelScreen } from "@/components/assistant/TrustPanelScreen";
import { ResponsibleAIScreen } from "@/components/assistant/ResponsibleAIScreen";

export default function AssistantPage() {
  const [activeTab, setActiveTab] = useState<TabType>("assistant");
  const [messages, setMessages] = useState<ChatMessage[]>(MOCK_INITIAL_MESSAGES);
  const [latestEvidence, setLatestEvidence] = useState<Array<{ tool: string; result: string }>>([]);

  return (
    <div className="flex flex-col gap-2">
      {/* Top Header */}
      <AssistantHeader />

      {/* Research Architecture Flow Banner */}
      <ResearchBanner />

      {/* Main Content Layout with Sidebar */}
      <div className="flex flex-col gap-6 md:flex-row">
        {/* Left Navigation Sidebar */}
        <AssistantSidebar
          activeTab={activeTab}
          onSelectTab={(tab) => setActiveTab(tab)}
        />

        {/* Main Active Screen */}
        <main className="flex-1">
          {activeTab === "assistant" && (
            <ChatScreen
              messages={messages}
              setMessages={setMessages}
              onNavigateToExplanation={() => setActiveTab("explanation")}
              onEvidenceReceived={(evidence) => setLatestEvidence(evidence)}
            />
          )}

          {activeTab === "explanation" && (
            <ExplanationScreen
              onNavigateToTrustPanel={() => setActiveTab("trust-panel")}
              latestEvidence={latestEvidence}
            />
          )}

          {activeTab === "trust-panel" && (
            <TrustPanelScreen latestEvidence={latestEvidence} />
          )}

          {(activeTab === "responsible-ai" || activeTab === "settings") && (
            <ResponsibleAIScreen />
          )}
        </main>
      </div>
    </div>
  );
}
