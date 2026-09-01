"use client";

import React from "react";
import { TabType } from "./types";
import { BotIcon, BarChartIcon, ShieldIcon, CpuIcon, LockIcon } from "./icons";

interface Props {
  activeTab: TabType;
  onSelectTab: (tab: TabType) => void;
}

export function AssistantSidebar({ activeTab, onSelectTab }: Props) {
  const navItems: { id: TabType; label: string; icon: React.ReactNode; badge?: string }[] = [
    { id: "assistant", label: "AI Assistant", icon: <BotIcon className="size-4" /> },
    { id: "explanation", label: "Explanations", icon: <BarChartIcon className="size-4" />, badge: "SHAP" },
    { id: "trust-panel", label: "Trust & Safety", icon: <ShieldIcon className="size-4" />, badge: "Action" },
    { id: "responsible-ai", label: "Responsible AI", icon: <CpuIcon className="size-4" /> },
    { id: "settings", label: "Settings", icon: <LockIcon className="size-4" /> },
  ];

  return (
    <aside className="w-full shrink-0 md:w-56">
      <nav className="flex flex-row gap-1.5 overflow-x-auto rounded-xl border border-black/10 bg-white p-2 shadow-xs dark:border-white/15 dark:bg-neutral-900 md:flex-col">
        <div className="hidden px-3 py-2 text-[11px] font-bold uppercase tracking-wider text-neutral-400 md:block">
          Component Navigation
        </div>
        {navItems.map((item) => {
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onSelectTab(item.id)}
              className={`flex items-center justify-between gap-2.5 whitespace-nowrap rounded-lg px-3.5 py-2.5 text-xs font-medium transition-all ${
                isActive
                  ? "bg-blue-600 text-white shadow-sm shadow-blue-500/30"
                  : "text-neutral-600 hover:bg-neutral-100 hover:text-neutral-900 dark:text-neutral-400 dark:hover:bg-neutral-800 dark:hover:text-white"
              }`}
            >
              <div className="flex items-center gap-2.5">
                {item.icon}
                <span>{item.label}</span>
              </div>
              {item.badge && (
                <span
                  className={`rounded-full px-1.5 py-0.5 text-[10px] font-bold ${
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

      {/* Research Note Card */}
      <div className="mt-4 hidden rounded-xl border border-blue-100 bg-blue-50/50 p-3.5 text-xs text-blue-900 dark:border-blue-900/40 dark:bg-blue-950/20 dark:text-blue-300 md:block">
        <p className="font-semibold text-blue-950 dark:text-blue-200">Research Focus</p>
        <p className="mt-1 leading-relaxed text-[11px] opacity-90">
          Evaluates local SLM explainability and human oversight through the Trust Panel.
        </p>
      </div>
    </aside>
  );
}
