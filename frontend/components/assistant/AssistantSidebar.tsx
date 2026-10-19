"use client";

import React from "react";
import { TabType } from "./types";
import { BotIcon, BarChartIcon, ShieldIcon, CpuIcon, LockIcon, LayersIcon } from "./icons";

interface Props {
  activeTab: TabType;
  onSelectTab: (tab: TabType) => void;
  onToggleArchitecture?: () => void;
  isArchitectureOpen?: boolean;
}

export function AssistantSidebar({
  activeTab,
  onSelectTab,
  onToggleArchitecture,
  isArchitectureOpen = false,
}: Props) {
  const navItems: { id: TabType; label: string; icon: React.ReactNode; badge?: string }[] = [
    { id: "assistant", label: "AI Assistant", icon: <BotIcon className="size-4" /> },
    { id: "explanation", label: "Explanations", icon: <BarChartIcon className="size-4" />, badge: "SHAP" },
    { id: "trust-panel", label: "Trust & Safety", icon: <ShieldIcon className="size-4" />, badge: "Action" },
    { id: "responsible-ai", label: "Responsible AI", icon: <CpuIcon className="size-4" /> },
    { id: "settings", label: "Settings", icon: <LockIcon className="size-4" /> },
  ];

  return (
    <aside className="w-full shrink-0 md:w-56 flex flex-col gap-3">
      <nav aria-label="Component Navigation" className="flex flex-row gap-1.5 overflow-x-auto rounded-2xl border border-neutral-200/90 bg-white p-2 shadow-xs dark:border-neutral-800 dark:bg-neutral-900 md:flex-col">
        <div className="hidden px-3 py-2 text-[10px] font-bold uppercase tracking-wider text-neutral-400 md:block">
          FinTrust Workspace
        </div>
        {navItems.map((item) => {
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onSelectTab(item.id)}
              className={`flex items-center justify-between gap-2.5 whitespace-nowrap rounded-xl px-3.5 py-2.5 text-xs font-semibold transition-all ${
                isActive
                  ? "bg-blue-600 text-white shadow-sm shadow-blue-500/25"
                  : "text-neutral-600 hover:bg-neutral-100 hover:text-neutral-900 dark:text-neutral-400 dark:hover:bg-neutral-800 dark:hover:text-white"
              }`}
            >
              <div className="flex items-center gap-2.5">
                {item.icon}
                <span>{item.label}</span>
              </div>
              {item.badge && (
                <span
                  className={`rounded-full px-1.5 py-0.5 text-[9px] font-bold ${
                    isActive
                      ? "bg-white/20 text-white"
                      : "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300"
                  }`}
                >
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Evaluator Architecture Quick Button */}
      {onToggleArchitecture && (
        <button
          onClick={onToggleArchitecture}
          className={`hidden md:flex items-center justify-between rounded-xl border p-3 text-xs font-semibold transition shadow-2xs ${
            isArchitectureOpen
              ? "border-blue-300 bg-blue-50 text-blue-900 dark:border-blue-900 dark:bg-blue-950/40 dark:text-blue-200"
              : "border-neutral-200 bg-neutral-50/70 text-neutral-600 hover:bg-neutral-100 hover:text-neutral-900 dark:border-neutral-800 dark:bg-neutral-900/60 dark:text-neutral-400 dark:hover:bg-neutral-800"
          }`}
        >
          <div className="flex items-center gap-2">
            <LayersIcon className="size-4 text-blue-600 dark:text-blue-400" />
            <span>Architecture & Lab</span>
          </div>
          <span className="text-[10px] font-bold opacity-75">
            {isArchitectureOpen ? "Open" : "View"}
          </span>
        </button>
      )}
    </aside>
  );
}
