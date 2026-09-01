/**
 * Shared fetch wrapper for every backend service.
 *
 * The browser calls each service DIRECTLY (CORS is enabled on each) rather than proxying
 * through Next.js route handlers. Everything here therefore runs client-side.
 *
 * Components 2-4 get their own base URL and a `createClient` call; nothing else changes.
 */

/** One service the frontend talks to. Ports match docker-compose. */
export const SERVICES = {
  platform: process.env.NEXT_PUBLIC_PLATFORM_URL ?? "http://localhost:8100",
  portfolio: process.env.NEXT_PUBLIC_PORTFOLIO_URL ?? "http://localhost:8000",
  fraud: process.env.NEXT_PUBLIC_FRAUD_URL ?? "http://localhost:8001",
  audit: process.env.NEXT_PUBLIC_AUDIT_URL ?? "http://localhost:8002",
  assistant: process.env.NEXT_PUBLIC_ASSISTANT_URL ?? "http://localhost:8003",
} as const;

export type ServiceName = keyof typeof SERVICES;

const TOKEN_KEY = "j26_access_token";

/**
 * An HTTP failure carrying the server's message.
 *
 * FastAPI puts the useful text in `detail`, and validation errors put a whole array there.
 * Surfacing that beats a generic "Request failed" — "target_amount exceeds total portfolio
 * value" tells the user what to change.
 */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly detail?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }

  /** 401/403 — the caller should send the user back to login. */
  get isAuthError(): boolean {
    return this.status === 401 || this.status === 403;
  }

  /** 503 — the service is up but a dependency is not. /forecast does this until a model
   *  is trained, which is an expected state rather than a bug. */
  get isUnavailable(): boolean {
    return this.status === 503;
  }
}

// --- token storage ---------------------------------------------------------------------
// NOTE: localStorage is readable by any script on the page, so an XSS would expose the
// token. The alternative (an httpOnly cookie) requires proxying through Next.js, which this
// architecture deliberately does not do. Documented as a known trade-off in the README.

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(TOKEN_KEY);
  } catch {
    // Private mode / storage disabled. Treat as logged out rather than crashing render.
    return null;
  }
}

export function setToken(token: string | null): void {
  if (typeof window === "undefined") return;
  try {
    if (token) window.localStorage.setItem(TOKEN_KEY, token);
    else window.localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* ignore: the app still works for the current session */
  }
}

function messageFrom(status: number, body: unknown): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    // FastAPI validation errors: [{loc, msg, type}, ...]
    if (Array.isArray(detail)) {
      const parts = detail
        .map((item) => {
          if (item && typeof item === "object" && "msg" in item) {
            const loc = "loc" in item && Array.isArray(item.loc) ? item.loc.slice(1).join(".") : "";
            return loc ? `${loc}: ${(item as { msg: string }).msg}` : (item as { msg: string }).msg;
          }
          return String(item);
        })
        .filter(Boolean);
      if (parts.length) return parts.join("; ");
    }
  }
  return `Request failed (${status})`;
}

export interface RequestOptions {
  method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  body?: unknown;
  /** Send the bearer token. Default true; login/register set it false. */
  auth?: boolean;
  signal?: AbortSignal;
}

/** Issue a request against one service and parse the result. */
export async function request<T>(
  service: ServiceName,
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const { method = "GET", body, auth = true, signal } = options;

  const headers: Record<string, string> = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";

  if (auth) {
    const token = getToken();
    if (token) headers.Authorization = `Bearer ${token}`;
  }

  let response: Response;
  try {
    response = await fetch(`${SERVICES[service]}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
      // Never cache: these are live decisions, and a stale withdrawal plan is worse than
      // no plan. Next 16 does not cache fetch by default, but it still prerenders routes at
      // build time, which would otherwise bake a response captured during `next build`.
      cache: "no-store",
    });
  } catch (cause) {
    // Distinguish "backend is down" from a real HTTP error — the fix is completely
    // different, and a bare TypeError here is what a CORS rejection also looks like.
    if (signal?.aborted) throw cause;
    throw new ApiError(
      0,
      `Cannot reach the ${service} service at ${SERVICES[service]}. Is it running?`,
      cause,
    );
  }

  if (response.status === 204) return undefined as T;

  const text = await response.text();
  const payload: unknown = text ? safeParse(text) : null;

  if (!response.ok) {
    throw new ApiError(response.status, messageFrom(response.status, payload), payload);
  }
  return payload as T;
}

function safeParse(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}
