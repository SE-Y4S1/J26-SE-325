"use client";

/**
 * Small shared primitives.
 *
 * Deliberately hand-rolled rather than pulling in a component library: the project has no
 * design system, and four components' worth of UI is easier to keep consistent against a
 * handful of readable primitives than against someone else's API.
 *
 * Tailwind v4 CSS-first — theme tokens live in app/globals.css under @theme inline; there
 * is no tailwind.config.ts.
 */

import type { ReactNode } from "react";

export function Card({
  title,
  subtitle,
  children,
  className = "",
}: {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-xl border border-black/10 bg-white p-5 shadow-sm dark:border-white/15 dark:bg-neutral-900 ${className}`}
    >
      {title && (
        <header className="mb-4">
          <h2 className="text-base font-semibold">{title}</h2>
          {subtitle && <p className="mt-1 text-sm text-neutral-500">{subtitle}</p>}
        </header>
      )}
      {children}
    </section>
  );
}

export function Button({
  children,
  variant = "primary",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "danger";
}) {
  const styles = {
    primary: "bg-neutral-900 text-white hover:bg-neutral-700 dark:bg-white dark:text-neutral-900",
    secondary:
      "border border-black/15 hover:bg-black/5 dark:border-white/20 dark:hover:bg-white/10",
    danger: "bg-red-600 text-white hover:bg-red-500",
  }[variant];

  return (
    <button
      {...props}
      className={`rounded-lg px-4 py-2 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50 ${styles} ${props.className ?? ""}`}
    >
      {children}
    </button>
  );
}

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-sm font-medium">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-xs text-neutral-500">{hint}</span>}
    </label>
  );
}

export function Input(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={`w-full rounded-lg border border-black/15 bg-transparent px-3 py-2 text-sm outline-none focus:border-neutral-500 dark:border-white/20 ${props.className ?? ""}`}
    />
  );
}

/** Non-blocking message. `tone` carries meaning: an infeasible withdrawal is a real answer,
 *  not an error, so it uses "warn" rather than "error". */
export function Notice({
  tone = "info",
  title,
  children,
}: {
  tone?: "info" | "warn" | "error" | "success";
  title?: string;
  children?: ReactNode;
}) {
  const styles = {
    info: "border-blue-500/30 bg-blue-500/5 text-blue-900 dark:text-blue-200",
    warn: "border-amber-500/40 bg-amber-500/10 text-amber-900 dark:text-amber-200",
    error: "border-red-500/40 bg-red-500/10 text-red-900 dark:text-red-200",
    success: "border-emerald-500/40 bg-emerald-500/10 text-emerald-900 dark:text-emerald-200",
  }[tone];

  return (
    <div className={`rounded-lg border p-3 text-sm ${styles}`}>
      {title && <p className="font-semibold">{title}</p>}
      {children && <div className={title ? "mt-1" : ""}>{children}</div>}
    </div>
  );
}

export function Stat({
  label,
  value,
  hint,
}: {
  label: string;
  value: ReactNode;
  hint?: string;
}) {
  return (
    <div>
      <p className="text-xs uppercase tracking-wide text-neutral-500">{label}</p>
      <p className="mt-1 text-xl font-semibold tabular-nums">{value}</p>
      {hint && <p className="mt-0.5 text-xs text-neutral-500">{hint}</p>}
    </div>
  );
}

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <p className="flex items-center gap-2 text-sm text-neutral-500">
      <span className="inline-block size-3 animate-spin rounded-full border-2 border-current border-t-transparent" />
      {label}
    </p>
  );
}

// --- formatting ---------------------------------------------------------------------
// Shared so every screen renders money and percentages identically. Percentages come from
// the API as fractions (0.0021), never pre-multiplied.

export const money = (value: number, currency = "USD") =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    maximumFractionDigits: 2,
  }).format(value);

export const percent = (fraction: number, digits = 3) =>
  `${(fraction * 100).toFixed(digits)}%`;

export const compact = (value: number) =>
  new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value);
