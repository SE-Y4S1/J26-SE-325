/** Typed client for the platform service: identity and portfolio storage. */

import { request, setToken } from "./client";
import type {
  Portfolio,
  PortfolioCreate,
  PortfolioSummary,
  PortfolioUpdate,
  TokenResponse,
  User,
} from "./types";

// --- auth -------------------------------------------------------------------------------

export async function register(
  email: string,
  displayName: string,
  password: string,
): Promise<TokenResponse> {
  const result = await request<TokenResponse>("platform", "/auth/register", {
    method: "POST",
    // No token exists yet, and sending a stale one from a previous session would be
    // rejected before registration is even attempted.
    auth: false,
    body: { email, display_name: displayName, password },
  });
  setToken(result.access_token);
  return result;
}

export async function login(email: string, password: string): Promise<TokenResponse> {
  const result = await request<TokenResponse>("platform", "/auth/login", {
    method: "POST",
    auth: false,
    body: { email, password },
  });
  setToken(result.access_token);
  return result;
}

export function logout(): void {
  setToken(null);
}

/** The current user, or null when the token is missing or no longer valid. */
export async function me(): Promise<User | null> {
  try {
    return await request<User>("platform", "/auth/me");
  } catch {
    // Any failure here means "not signed in" as far as the UI is concerned. Distinguishing
    // expired from malformed would not change what the app does about it.
    return null;
  }
}

// --- portfolios -------------------------------------------------------------------------

export function listPortfolios(): Promise<PortfolioSummary[]> {
  return request<PortfolioSummary[]>("platform", "/portfolios");
}

export function getPortfolio(id: number): Promise<Portfolio> {
  return request<Portfolio>("platform", `/portfolios/${id}`);
}

export function createPortfolio(data: PortfolioCreate): Promise<Portfolio> {
  return request<Portfolio>("platform", "/portfolios", { method: "POST", body: data });
}

export function updatePortfolio(id: number, data: PortfolioUpdate): Promise<Portfolio> {
  return request<Portfolio>("platform", `/portfolios/${id}`, { method: "PUT", body: data });
}

export function deletePortfolio(id: number): Promise<void> {
  return request<void>("platform", `/portfolios/${id}`, { method: "DELETE" });
}
