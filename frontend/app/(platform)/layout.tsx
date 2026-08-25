"use client";

/**
 * Shared shell for every component's screens.
 *
 * The navigation is the contract with the rest of the team: each of the four components owns
 * a top-level section, and adding one means adding a NAV entry plus a folder. Nothing else
 * in this file changes.
 */

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, type ReactNode } from "react";

import { Spinner } from "@/components/ui";
import { AuthProvider, useAuth } from "@/lib/auth/context";

interface NavItem {
  href: string;
  label: string;
  component: string;
  ready: boolean;
}

const NAV: NavItem[] = [
  // Component 1 — this work.
  { href: "/portfolio", label: "Portfolio", component: "1", ready: true },
  { href: "/withdraw", label: "Withdraw", component: "1", ready: true },
  { href: "/optimize", label: "Optimize", component: "1", ready: true },
  { href: "/forecast", label: "Forecast", component: "1", ready: true },
  // Teammates' components. `ready: false` renders them visibly pending rather than hiding
  // them, so the integration surface is obvious to anyone looking at the app.
  { href: "/fraud", label: "Fraud", component: "2", ready: false },
  { href: "/audit", label: "Audit", component: "3", ready: false },
  { href: "/assistant", label: "Assistant", component: "4", ready: false },
];

function Shell({ children }: { children: ReactNode }) {
  const { user, loading, logout } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    // Wait for `loading` before redirecting, or a signed-in user is bounced to /login on
    // every hard refresh while the token is still being verified.
    if (!loading && !user) router.replace("/login");
  }, [loading, user, router]);

  if (loading) {
    return (
      <main className="flex min-h-dvh items-center justify-center">
        <Spinner label="Restoring session" />
      </main>
    );
  }

  if (!user) return null; // redirect in flight

  return (
    <div className="min-h-dvh">
      <header className="border-b border-black/10 dark:border-white/15">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 p-4">
          <Link href="/portfolio" className="font-semibold">
            Smart Finance Platform
          </Link>
          <div className="flex items-center gap-3 text-sm">
            <span className="text-neutral-500">{user.display_name}</span>
            <button onClick={logout} className="underline">
              Sign out
            </button>
          </div>
        </div>

        <nav className="mx-auto flex max-w-6xl gap-1 overflow-x-auto px-4 pb-2 text-sm">
          {NAV.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                title={`Component ${item.component}${item.ready ? "" : " — not yet implemented"}`}
                className={`shrink-0 rounded-lg px-3 py-1.5 transition ${
                  active
                    ? "bg-neutral-900 text-white dark:bg-white dark:text-neutral-900"
                    : "hover:bg-black/5 dark:hover:bg-white/10"
                } ${item.ready ? "" : "opacity-50"}`}
              >
                {item.label}
                {!item.ready && <span className="ml-1 text-xs">·</span>}
              </Link>
            );
          })}
        </nav>
      </header>

      <main className="mx-auto max-w-6xl p-4 sm:p-6">{children}</main>
    </div>
  );
}

export default function PlatformLayout({ children }: { children: ReactNode }) {
  return (
    <AuthProvider>
      <Shell>{children}</Shell>
    </AuthProvider>
  );
}
