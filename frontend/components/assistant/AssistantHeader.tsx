"use client";

import React from "react";
import { BotIcon, ShieldIcon } from "./icons";

export function AssistantHeader() {
  return (
    <header className="rounded-xl border border-blue-100 bg-gradient-to-r from-blue-50/70 via-white to-neutral-50 p-4 sm:p-5 shadow-xs mb-6 dark:border-blue-900/40 dark:from-neutral-900 dark:via-neutral-900 dark:to-neutral-900">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        {/* Left branding */}
        <div className="flex items-center gap-3.5">
          <div className="flex size-11 items-center justify-center rounded-xl bg-blue-600 text-white shadow-md shadow-blue-500/20">
            <ShieldIcon className="size-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-tight text-neutral-900 dark:text-white">
                FinTrust AI
              </h1>
              <span className="rounded-full bg-blue-100 px-2.5 py-0.5 text-xs font-semibold text-blue-700 dark:bg-blue-950 dark:text-blue-300">
                PROTOTYPE — Component 4
              </span>
            </div>
            <p className="text-xs font-medium text-neutral-500 dark:text-neutral-400">
              Agentic Financial Assistant & Trust Panel
            </p>
          </div>
        </div>

        {/* Right status & Profile */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50/80 px-3 py-1 text-xs font-medium text-emerald-700 dark:border-emerald-900/50 dark:bg-emerald-950/40 dark:text-emerald-300">
            <span className="relative flex size-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex size-2 rounded-full bg-emerald-500"></span>
            </span>
            <span>AI Online</span>
          </div>

          <div className="flex items-center gap-2.5 rounded-lg border border-black/10 bg-white px-3 py-1.5 shadow-2xs dark:border-white/15 dark:bg-neutral-800">
            <div className="flex size-7 items-center justify-center rounded-full bg-blue-600 text-xs font-bold text-white">
              U
            </div>
            <div className="hidden text-left sm:block">
              <p className="text-xs font-semibold text-neutral-800 dark:text-neutral-200">
                Research User
              </p>
              <p className="text-[10px] text-neutral-500">Univ. 4th Year Eval</p>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
