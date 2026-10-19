"use client";

import React from "react";
import { ShieldIcon, LayersIcon, CheckCircleIcon } from "./icons";
import { useAuth } from "@/lib/auth/context";

interface Props {
  isArchitectureOpen?: boolean;
  onToggleArchitecture?: () => void;
}

export function AssistantHeader({ isArchitectureOpen = false, onToggleArchitecture }: Props) {
  const { user } = useAuth();

  return (
    <header className="rounded-2xl border border-neutral-200/90 bg-white p-4 sm:p-5 shadow-xs mb-5 dark:border-neutral-800 dark:bg-neutral-900 transition-all">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        {/* Left branding */}
        <div className="flex items-center gap-3.5">
          <div className="flex size-11 items-center justify-center rounded-xl bg-blue-600 text-white shadow-sm shadow-blue-500/25">
            <ShieldIcon className="size-6" />
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h1 className="text-xl font-bold tracking-tight text-neutral-900 dark:text-white">
                FinTrust AI
              </h1>
              <span className="rounded-full bg-blue-100 px-2.5 py-0.5 text-[11px] font-semibold text-blue-800 dark:bg-blue-950 dark:text-blue-300 border border-blue-200/50 dark:border-blue-900/50">
                Enterprise Decision Support · Component 4
              </span>
            </div>
            <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-0.5">
              Explainable Agentic Financial Assistant & Human-in-the-Loop Trust Panel
            </p>
          </div>
        </div>

        {/* Right controls: Architecture Inspector & Status */}
        <div className="flex flex-wrap items-center gap-2.5 sm:gap-3">
          {/* Architecture Toggle Button */}
          {onToggleArchitecture && (
            <button
              onClick={onToggleArchitecture}
              className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold transition shadow-2xs border ${
                isArchitectureOpen
                  ? "bg-blue-600 text-white border-blue-600 shadow-sm shadow-blue-500/20"
                  : "bg-neutral-50 text-neutral-700 border-neutral-300 hover:bg-neutral-100 hover:text-neutral-900 dark:bg-neutral-800 dark:text-neutral-200 dark:border-neutral-700 dark:hover:bg-neutral-700"
              }`}
              title="Toggle System Architecture & Evaluator Diagnostics"
            >
              <LayersIcon className="size-3.5" />
              <span>{isArchitectureOpen ? "Diagnostics Open" : "Architecture"}</span>
            </button>
          )}

          {/* Actual Service Status Badge */}
          <div className="flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-800 dark:border-emerald-900/60 dark:bg-emerald-950/40 dark:text-emerald-300">
            <span className="relative flex size-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex size-2 rounded-full bg-emerald-500"></span>
            </span>
            <span>Online (Port 8003)</span>
          </div>

          {/* User profile indicator */}
          <div className="flex items-center gap-2 rounded-lg border border-neutral-200 bg-neutral-50/80 px-2.5 py-1 text-xs text-neutral-700 dark:border-neutral-800 dark:bg-neutral-800 dark:text-neutral-300">
            <div className="flex size-6 items-center justify-center rounded-full bg-neutral-900 text-[11px] font-bold text-white dark:bg-neutral-100 dark:text-neutral-900">
              {user?.display_name ? user.display_name.charAt(0).toUpperCase() : "E"}
            </div>
            <span className="hidden sm:inline font-medium">
              {user?.display_name || "Evaluator"}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
