"use client";

/**
 * Auth state for the whole app.
 *
 * Client-side because the browser calls each backend directly and holds the token itself —
 * there is no server session to read. `AuthProvider` wraps the platform route group.
 */

import { useRouter } from "next/navigation";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { getToken } from "@/lib/api/client";
import * as platform from "@/lib/api/platform";
import type { User } from "@/lib/api/types";

/**
 * Opt-in demo mode: stand in a fake signed-in user so a screen can be shown without the
 * platform service running.
 *
 * This exists because Component 4's assistant UI needs to be demonstrable on its own. It is
 * off unless explicitly switched on, because when it is on the platform has no access
 * control at all -- every guard below sees a signed-in user and lets the visitor through.
 * NEXT_PUBLIC_* is inlined at build time, so a production build without it cannot enable
 * this at runtime.
 */
const DEMO_AUTH = process.env.NEXT_PUBLIC_DEMO_AUTH === "1";

const DEMO_USER: User = {
  id: 1,
  email: "evaluator@university.edu",
  display_name: "Research Evaluator",
  created_at: "2026-08-30T00:00:00Z",
};

/** The demo user in demo mode, otherwise signed out. */
const fallbackUser = (): User | null => (DEMO_AUTH ? DEMO_USER : null);

interface AuthState {
  user: User | null;
  /** True until the initial token check finishes. Guards must wait for this, or they
   *  bounce a signed-in user to /login on every hard refresh. */
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, displayName: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    // A token in storage is not proof of a valid session — it may be expired or signed with
    // a rotated secret. Verify against /auth/me before trusting it.
    let cancelled = false;

    async function restore() {
      if (!getToken()) {
        if (!cancelled) {
          setUser(fallbackUser());
          setLoading(false);
        }
        return;
      }

      try {
        const current = await platform.me();
        if (!cancelled) {
          setUser(current ?? fallbackUser());
          setLoading(false);
        }
      } catch {
        // A rejected token is a signed-out visitor, not a signed-in one -- unless we are
        // deliberately demoing without a backend.
        if (!cancelled) {
          setUser(fallbackUser());
          setLoading(false);
        }
      }
    }

    void restore();
    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const result = await platform.login(email, password);
    setUser(result.user);
  }, []);

  const register = useCallback(
    async (email: string, displayName: string, password: string) => {
      const result = await platform.register(email, displayName, password);
      setUser(result.user);
    },
    [],
  );

  const logout = useCallback(() => {
    platform.logout();
    setUser(null);
    router.push("/login");
  }, [router]);

  const value = useMemo(
    () => ({ user, loading, login, register, logout }),
    [user, loading, login, register, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used inside an AuthProvider");
  }
  return context;
}
