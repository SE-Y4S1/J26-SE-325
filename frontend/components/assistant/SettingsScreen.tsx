"use client";

import React, { useState } from "react";
import { useAuth } from "@/lib/auth/context";
import { getToken, SERVICES } from "@/lib/api/client";
import { LockIcon, ShieldIcon, CheckCircleIcon, AlertTriangleIcon, WrenchIcon, CpuIcon } from "./icons";

export const ASSISTANT_SESSION_CACHE_KEYS = [
  "finagent_session_cache",
  "finagent_evidence_scratch",
  "finagent_temp_query",
  "finagent_workflow_scratch",
] as const;

export interface ClearCacheResult {
  success: boolean;
  clearedKeys: string[];
  error?: string;
}

/**
 * Safely clears ONLY explicitly defined temporary assistant UI session cache
 * and evidence scratch data without touching authentication tokens, Responsible AI
 * preferences, persistent governance records, or non-assistant session data.
 */
export function clearAssistantSessionCache(): ClearCacheResult {
  if (typeof window === "undefined" || !window.sessionStorage) {
    return {
      success: false,
      clearedKeys: [],
      error: "Session storage is not available in the current browser environment.",
    };
  }

  try {
    const clearedKeys: string[] = [];

    // 1. Remove explicitly defined static assistant keys
    for (const key of ASSISTANT_SESSION_CACHE_KEYS) {
      if (window.sessionStorage.getItem(key) !== null) {
        window.sessionStorage.removeItem(key);
        clearedKeys.push(key);
      }
    }

    // 2. Identify and remove dynamic assistant-scoped keys
    const dynamicKeysToRemove: string[] = [];
    for (let i = 0; i < window.sessionStorage.length; i++) {
      const key = window.sessionStorage.key(i);
      if (
        key &&
        (key.startsWith("finagent_session_") ||
         key.startsWith("finagent_scratch_") ||
         key.startsWith("finagent_cache_") ||
         key.startsWith("finagent_evidence_"))
      ) {
        dynamicKeysToRemove.push(key);
      }
    }

    for (const key of dynamicKeysToRemove) {
      if (!clearedKeys.includes(key)) {
        window.sessionStorage.removeItem(key);
        clearedKeys.push(key);
      }
    }

    return {
      success: true,
      clearedKeys,
    };
  } catch (err) {
    return {
      success: false,
      clearedKeys: [],
      error: err instanceof Error ? err.message : "Failed to clear assistant session cache.",
    };
  }
}

export function SettingsScreen() {
  const { user, logout } = useAuth();
  const token = getToken();
  const [cacheCleared, setCacheCleared] = useState(false);
  const [cacheError, setCacheError] = useState<string | null>(null);

  const handleClearCache = () => {
    try {
      const result = clearAssistantSessionCache();
      if (result.success) {
        setCacheError(null);
        setCacheCleared(true);
        setTimeout(() => setCacheCleared(false), 3000);
      } else {
        setCacheCleared(false);
        setCacheError(result.error || "Failed to clear session cache.");
        setTimeout(() => setCacheError(null), 5000);
      }
    } catch (err) {
      setCacheCleared(false);
      setCacheError(err instanceof Error ? err.message : "An unexpected error occurred while clearing cache.");
      setTimeout(() => setCacheError(null), 5000);
    }
  };

  const servicesList = [
    { name: "Platform Service", url: SERVICES.platform, port: "8100", role: "Identity, Auth & Portfolio Store", status: "Configured" },
    { name: "Portfolio Optimization", url: SERVICES.portfolio, port: "8000", role: "Forecasting & Multi-objective Optimization", status: "Configured" },
    { name: "Fraud Detection", url: SERVICES.fraud, port: "8001", role: "Real-time Dual-stream Anomaly Scoring", status: "Configured" },
    { name: "Blockchain Audit", url: SERVICES.audit, port: "8002", role: "SHA-256 State Anchoring & Ledger", status: "Configured" },
    { name: "Agentic Assistant", url: SERVICES.assistant, port: "8003", role: "Ollama LLM Agent & Explainability (C4)", status: "Active" },
  ];

  return (
    <div className="flex flex-col gap-6">
      {/* Title & Subtitle Header */}
      <div className="rounded-xl border border-black/10 bg-white p-5 shadow-xs dark:border-white/15 dark:bg-neutral-900">
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-blue-600 dark:text-blue-400">
          <LockIcon className="size-4" />
          <span>System & Account Configuration</span>
        </div>
        <h2 className="mt-1 text-2xl font-bold text-neutral-900 dark:text-white">
          Settings
        </h2>
        <p className="mt-1 text-sm text-neutral-500">
          Manage session credentials, connected service endpoints, and client application state.
        </p>
      </div>

      {/* Account & Session Card */}
      <div className="rounded-xl border border-black/10 bg-white p-5 shadow-xs dark:border-white/15 dark:bg-neutral-900">
        <div className="flex items-center justify-between border-b border-neutral-200/80 pb-3 dark:border-neutral-800">
          <div className="flex items-center gap-2">
            <ShieldIcon className="size-4 text-neutral-700 dark:text-neutral-300" />
            <h3 className="text-base font-semibold text-neutral-900 dark:text-white">
              Authenticated Session
            </h3>
          </div>
          <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-100 px-2.5 py-0.5 text-[11px] font-semibold text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300">
            <CheckCircleIcon className="size-3.5" />
            {token ? "Active Token" : "Demo Session"}
          </span>
        </div>

        <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-3">
          <div className="rounded-lg border border-neutral-200 bg-neutral-50 p-3.5 dark:border-neutral-800 dark:bg-neutral-800/50">
            <p className="text-[11px] font-medium text-neutral-500">Display Name</p>
            <p className="mt-1 text-sm font-bold text-neutral-900 dark:text-white">
              {user?.display_name || "Research Evaluator"}
            </p>
          </div>
          <div className="rounded-lg border border-neutral-200 bg-neutral-50 p-3.5 dark:border-neutral-800 dark:bg-neutral-800/50">
            <p className="text-[11px] font-medium text-neutral-500">Email Address</p>
            <p className="mt-1 text-sm font-bold text-neutral-900 dark:text-white truncate">
              {user?.email || "evaluator@university.edu"}
            </p>
          </div>
          <div className="rounded-lg border border-neutral-200 bg-neutral-50 p-3.5 dark:border-neutral-800 dark:bg-neutral-800/50">
            <p className="text-[11px] font-medium text-neutral-500">User ID</p>
            <p className="mt-1 text-sm font-bold text-neutral-900 dark:text-white">
              {user?.id ? `#${user.id}` : "#1 (Default)"}
            </p>
          </div>
        </div>

        <div className="mt-4 flex items-center justify-between pt-2">
          <p className="text-xs text-neutral-500">
            Authentication token is stored locally in browser storage and verified with Platform Service on port 8100.
          </p>
          <button
            onClick={logout}
            className="rounded-lg border border-neutral-300 bg-white px-3.5 py-1.5 text-xs font-semibold text-neutral-700 shadow-2xs hover:bg-neutral-100 dark:border-neutral-700 dark:bg-neutral-800 dark:text-neutral-200 dark:hover:bg-neutral-700 transition"
          >
            Sign Out
          </button>
        </div>
      </div>

      {/* Connected Microservices Architecture */}
      <div className="rounded-xl border border-black/10 bg-white p-5 shadow-xs dark:border-white/15 dark:bg-neutral-900">
        <div className="flex items-center gap-2 border-b border-neutral-200/80 pb-3 dark:border-neutral-800">
          <CpuIcon className="size-4 text-neutral-700 dark:text-neutral-300" />
          <h3 className="text-base font-semibold text-neutral-900 dark:text-white">
            Connected Microservices Endpoints
          </h3>
        </div>

        <p className="mt-2 text-xs text-neutral-500 leading-relaxed">
          The browser talks to each backend service directly via configured CORS headers. URLs reflect the local Docker Compose cluster or environment overrides.
        </p>

        <div className="mt-4 divide-y divide-neutral-200/80 rounded-lg border border-neutral-200 dark:divide-neutral-800 dark:border-neutral-800 overflow-hidden">
          {servicesList.map((svc) => (
            <div key={svc.name} className="flex flex-col sm:flex-row sm:items-center justify-between p-3.5 bg-neutral-50/50 dark:bg-neutral-800/30 gap-2">
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-neutral-900 dark:text-white">{svc.name}</span>
                  <span className="rounded bg-neutral-200 px-1.5 py-0.5 text-[10px] font-mono text-neutral-700 dark:bg-neutral-700 dark:text-neutral-300">
                    Port {svc.port}
                  </span>
                </div>
                <p className="text-[11px] text-neutral-500 mt-0.5">{svc.role}</p>
              </div>
              <div className="flex items-center gap-3">
                <code className="rounded bg-white px-2 py-0.5 text-xs font-mono text-blue-600 dark:bg-neutral-900 dark:text-blue-400 border border-neutral-200 dark:border-neutral-800">
                  {svc.url}
                </code>
                <span className="text-[11px] font-semibold text-emerald-600 dark:text-emerald-400">
                  {svc.status}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Client Cache & State Management */}
      <div className="rounded-xl border border-black/10 bg-white p-5 shadow-xs dark:border-white/15 dark:bg-neutral-900">
        <div className="flex items-center gap-2 border-b border-neutral-200/80 pb-3 dark:border-neutral-800">
          <WrenchIcon className="size-4 text-neutral-700 dark:text-neutral-300" />
          <h3 className="text-base font-semibold text-neutral-900 dark:text-white">
            Client Cache & Diagnostics
          </h3>
        </div>

        <div className="mt-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <p className="text-xs font-semibold text-neutral-900 dark:text-white">
              Assistant Session Cache
            </p>
            <p className="text-[11px] text-neutral-500">
              Clear temporary UI session storage and evidence scratch data without affecting your login token or Responsible AI preference settings.
            </p>
          </div>
          <button
            onClick={handleClearCache}
            className="shrink-0 rounded-lg bg-neutral-900 px-3.5 py-2 text-xs font-semibold text-white hover:bg-neutral-800 dark:bg-white dark:text-neutral-900 dark:hover:bg-neutral-100 transition shadow-2xs"
          >
            {cacheCleared ? "Cache Cleared!" : "Clear Session Cache"}
          </button>
        </div>

        {cacheCleared && (
          <div
            role="status"
            className="mt-3 flex items-center gap-2 rounded-lg bg-emerald-100 p-2.5 text-xs font-semibold text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200"
          >
            <CheckCircleIcon className="size-4 shrink-0" />
            <span>Assistant session cache cleared successfully.</span>
          </div>
        )}

        {cacheError && (
          <div
            role="alert"
            className="mt-3 flex items-center gap-2 rounded-lg bg-rose-100 p-2.5 text-xs font-semibold text-rose-800 dark:bg-rose-950 dark:text-rose-200"
          >
            <AlertTriangleIcon className="size-4 shrink-0 text-rose-700 dark:text-rose-300" />
            <span>{cacheError}</span>
          </div>
        )}
      </div>

      {/* Governance & Privacy Notice */}
      <div className="flex items-start gap-3 rounded-xl border border-neutral-200 bg-neutral-50/70 p-4 shadow-2xs dark:border-neutral-800 dark:bg-neutral-900/40">
        <AlertTriangleIcon className="size-5 shrink-0 text-neutral-500 dark:text-neutral-400" />
        <div>
          <h4 className="text-xs font-bold uppercase tracking-wider text-neutral-700 dark:text-neutral-300">
            Responsible AI & Privacy Preferences
          </h4>
          <p className="mt-1 text-xs leading-relaxed text-neutral-600 dark:text-neutral-400">
            Looking for AI explanation toggles, personalized financial context gating, or empirical research consent controls? Those governance controls are managed on the dedicated <strong>Responsible AI</strong> screen.
          </p>
        </div>
      </div>
    </div>
  );
}
