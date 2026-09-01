"use client";

import React, { useState } from "react";
import { TabType } from "@/components/assistant/types";
import { AssistantHeader } from "@/components/assistant/AssistantHeader";
import { AssistantSidebar } from "@/components/assistant/AssistantSidebar";
import { ResearchBanner } from "@/components/assistant/ResearchBanner";
import { ChatScreen } from "@/components/assistant/ChatScreen";
import { ExplanationScreen } from "@/components/assistant/ExplanationScreen";
import { TrustPanelScreen } from "@/components/assistant/TrustPanelScreen";
import { ResponsibleAIScreen } from "@/components/assistant/ResponsibleAIScreen";

export default function AssistantPage() {
  const [activeTab, setActiveTab] = useState<TabType>("assistant");

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
              onNavigateToExplanation={() => setActiveTab("explanation")}
            />
          )}

          {activeTab === "explanation" && (
            <ExplanationScreen
              onNavigateToTrustPanel={() => setActiveTab("trust-panel")}
            />
          )}

          {activeTab === "trust-panel" && <TrustPanelScreen />}

          {(activeTab === "responsible-ai" || activeTab === "settings") && (
            <ResponsibleAIScreen />
          )}
        </main>
      </div>
    </div>
  );
}
