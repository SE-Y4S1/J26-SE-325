/**
 * The fetch wrapper every screen goes through.
 *
 * These are the behaviours the UI depends on being right: an unreachable backend must be
 * distinguishable from an HTTP error, a 503 from /forecast must be recognisable as an
 * expected state rather than a bug, and FastAPI's validation-error array must reach the user
 * as readable text instead of "[object Object]".
 */

import { ApiError, getToken, request, setToken } from "../client";

function jsonResponse(status: number, body: unknown) {
  return {
    status,
    ok: status >= 200 && status < 300,
    text: async () => (body === undefined ? "" : JSON.stringify(body)),
  } as unknown as Response;
}

// jsdom provides no global fetch, so there is nothing for jest.spyOn to attach to --
// the mock is installed directly.
const fetchMock = jest.fn();
global.fetch = fetchMock as unknown as typeof fetch;

beforeEach(() => {
  window.localStorage.clear();
  jest.restoreAllMocks();
  fetchMock.mockReset();
});

describe("ApiError", () => {
  it("flags 401 and 403 as auth errors so the caller can bounce to login", () => {
    expect(new ApiError(401, "x").isAuthError).toBe(true);
    expect(new ApiError(403, "x").isAuthError).toBe(true);
    expect(new ApiError(500, "x").isAuthError).toBe(false);
  });

  it("flags only 503 as unavailable", () => {
    // /forecast returns 503 until a model is trained. The Forecast screen renders an
    // explanation for that rather than an error, so the distinction has to hold.
    expect(new ApiError(503, "x").isUnavailable).toBe(true);
    expect(new ApiError(500, "x").isUnavailable).toBe(false);
  });
});

describe("error messages", () => {
  it("surfaces FastAPI's string detail verbatim", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(400, { detail: "target_amount exceeds total portfolio value" }),
    );

    await expect(request("portfolio", "/portfolio/withdraw")).rejects.toThrow(
      "target_amount exceeds total portfolio value",
    );
  });

  it("renders a pydantic validation array as readable text", async () => {
    // Without this the user sees "[object Object]" and has no idea which field is wrong.
    fetchMock.mockResolvedValue(
      jsonResponse(422, {
        detail: [
          { loc: ["body", "target_amount"], msg: "must be greater than 0", type: "x" },
          { loc: ["body", "deadline_days"], msg: "must be an integer", type: "y" },
        ],
      }),
    );

    await expect(request("portfolio", "/portfolio/withdraw")).rejects.toThrow(
      "target_amount: must be greater than 0; deadline_days: must be an integer",
    );
  });

  it("falls back to the status when there is no detail", async () => {
    fetchMock.mockResolvedValue(jsonResponse(500, { oops: true }));
    await expect(request("portfolio", "/x")).rejects.toThrow("Request failed (500)");
  });

  it("does not crash on a non-JSON body", async () => {
    // A proxy or gateway returning an HTML error page must not produce a JSON parse crash
    // on top of the original failure.
    fetchMock.mockResolvedValue({
      status: 502,
      ok: false,
      text: async () => "<html>Bad Gateway</html>",
    } as unknown as Response);

    await expect(request("portfolio", "/x")).rejects.toThrow(ApiError);
  });

  it("reports an unreachable service with the URL and a question to act on", async () => {
    // A bare TypeError here is also what a CORS rejection looks like, and the fix for each
    // is completely different -- so the message has to name the service and its address.
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));

    await expect(request("platform", "/auth/me")).rejects.toThrow(
      /Cannot reach the platform service at http:\/\/localhost:8100\. Is it running\?/,
    );
    await expect(request("platform", "/auth/me")).rejects.toMatchObject({ status: 0 });
  });
});

describe("requests", () => {
  it("attaches the bearer token when one is stored", async () => {
    setToken("tok-123");
    fetchMock.mockResolvedValue(jsonResponse(200, { ok: true }));

    await request("platform", "/auth/me");

    const [, init] = fetchMock.mock.calls[0];
    expect((init?.headers as Record<string, string>).Authorization).toBe("Bearer tok-123");
  });

  it("omits the token when auth is false", async () => {
    // Login and register must not send a stale token from a previous session.
    setToken("tok-123");
    fetchMock.mockResolvedValue(jsonResponse(200, { access_token: "new" }));

    await request("platform", "/auth/login", { method: "POST", body: {}, auth: false });

    const [, init] = fetchMock.mock.calls[0];
    expect((init?.headers as Record<string, string>).Authorization).toBeUndefined();
  });

  it("never caches", async () => {
    // Next 16 does not cache fetch by default, but it still prerenders routes at build time,
    // which would otherwise bake a withdrawal plan captured during `next build`.
    fetchMock.mockResolvedValue(jsonResponse(200, { ok: true }));

    await request("portfolio", "/health");

    expect(fetchMock.mock.calls[0][1]).toMatchObject({ cache: "no-store" });
  });

  it("returns undefined for 204 rather than trying to parse a body", async () => {
    fetchMock.mockResolvedValue({
      status: 204,
      ok: true,
      text: async () => "",
    } as unknown as Response);

    await expect(request("platform", "/portfolios/1")).resolves.toBeUndefined();
  });
});

describe("token storage", () => {
  it("round-trips and clears", () => {
    expect(getToken()).toBeNull();
    setToken("abc");
    expect(getToken()).toBe("abc");
    setToken(null);
    expect(getToken()).toBeNull();
  });

  it("treats unavailable storage as logged out rather than crashing", () => {
    // Private browsing throws on localStorage access. A crash here would blank the whole
    // app on render.
    jest.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("storage disabled");
    });
    expect(getToken()).toBeNull();
  });
});
