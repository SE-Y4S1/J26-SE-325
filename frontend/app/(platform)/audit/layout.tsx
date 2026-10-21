"use client";

/**
 * Component 3 — AI Decision Assurance Engine with a tamper-evident evidence ledger
 * (Abeysekara W C S M).
 *
 * One heading and a row of sub-pages, styled like the main navigation, so each task has its
 * own screen instead of everything stacked on one page.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

import { isNavActive } from "@/lib/nav";

const TABS = [
  { href: "/audit", label: "Overview", exact: true },
  { href: "/audit/decisions", label: "Decisions" },
  { href: "/audit/review", label: "Review queue" },
  { href: "/audit/verify", label: "Verify" },
  { href: "/audit/policies", label: "Policies" },
  { href: "/audit/evaluation", label: "Evaluation" },
];

export default function AuditLayout({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Audit trail</h1>
        <p className="mt-1 text-sm text-neutral-500">
          Each AI decision is routed by the assurance engine — automatic, verification, or human
          review — and anchored with its assurance result and policy. Component 3 · Abeysekara W C S M.
        </p>
      </div>

      <nav aria-label="Audit sections" className="flex gap-1 overflow-x-auto border-b border-black/10 pb-2 text-sm dark:border-white/15">
        {TABS.map((tab) => {
          const active = tab.exact ? pathname === tab.href : isNavActive(pathname, tab.href);
          return (
            <Link
              key={tab.href}
              href={tab.href}
              aria-current={active ? "page" : undefined}
              className={`shrink-0 rounded-lg px-3 py-1.5 transition ${
                active
                  ? "bg-neutral-900 text-white dark:bg-white dark:text-neutral-900"
                  : "text-neutral-600 hover:bg-black/5 dark:text-neutral-400 dark:hover:bg-white/10"
              }`}
            >
              {tab.label}
            </Link>
          );
        })}
      </nav>

      {children}
    </div>
  );
}
