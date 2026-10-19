/**
 * Market data and active-set calls on Component 1.
 *
 * Search and quote are lookups only: they never register a symbol for analysis, because
 * they run while the user is still typing. `trackSymbols` is what makes a symbol "held" —
 * call it when a portfolio is saved, then poll `analysisStatus` until each symbol is ready.
 */

import { request } from "./client";
import type {
  AnalysisStatusResponse,
  ChannelStatus,
  KeywordProfile,
  KeywordProfileUpdate,
  MarketSearchResponse,
  QuoteResponse,
  AnalysisForecastResponse,
  FundamentalsResponse,
  SentimentResponse,
  TechnicalResponse,
  TrackResponse,
} from "./types";

export type ExchangeName = "cse" | "binance" | "yahoo";

/** One search box across the Colombo Stock Exchange, Binance and Yahoo Finance. */
export function searchInstruments(
  query: string,
  { limit = 20, signal }: { limit?: number; signal?: AbortSignal } = {},
): Promise<MarketSearchResponse> {
  const params = new URLSearchParams({ q: query, limit: String(limit) });
  return request<MarketSearchResponse>("portfolio", `/market/search?${params}`, { signal });
}

/**
 * Live price, liquidity and volatility for one instrument, in its native currency and in
 * the platform base currency (LKR). 503 means the exchange is unreachable, not a bug.
 */
export function getQuote(
  symbol: string,
  { exchange, quantity }: { exchange?: ExchangeName; quantity?: number } = {},
): Promise<QuoteResponse> {
  return request<QuoteResponse>("portfolio", "/market/quote", {
    method: "POST",
    body: { symbol, exchange: exchange ?? null, quantity: quantity ?? null },
  });
}

/** Register a portfolio's symbols for analysis; preparation runs in the background. */
export function trackSymbols(symbols: string[]): Promise<TrackResponse> {
  return request<TrackResponse>("portfolio", "/analysis/track", {
    method: "POST",
    body: { symbols },
  });
}

/** Preparation state per symbol: preparing, ready, partial or failed. */
export function analysisStatus(symbols?: string[]): Promise<AnalysisStatusResponse> {
  const query = symbols?.length ? `?${new URLSearchParams({ symbols: symbols.join(",") })}` : "";
  return request<AnalysisStatusResponse>("portfolio", `/analysis/status${query}`);
}

/** Re-run preparation (news goes stale within hours). Omit symbols for every held symbol. */
export function refreshAnalysis(symbols?: string[]): Promise<AnalysisStatusResponse> {
  return request<AnalysisStatusResponse>("portfolio", "/analysis/refresh", {
    method: "POST",
    body: { symbols: symbols ?? null },
  });
}

/** Every news/social channel, whether it can run, and why not. */
export async function listChannels(): Promise<ChannelStatus[]> {
  const body = await request<{ channels: ChannelStatus[] }>("portfolio", "/analysis/channels");
  return body.channels;
}

/** A held symbol's keyword profile (company / sector / macro tiers). */
export function getKeywords(symbol: string): Promise<KeywordProfile> {
  return request<KeywordProfile>("portfolio", `/analysis/keywords/${encodeURIComponent(symbol)}`);
}

/** Correct a profile. Lists replace the stored list; the profile becomes "reviewed". */
export function updateKeywords(symbol: string, changes: KeywordProfileUpdate): Promise<KeywordProfile> {
  return request<KeywordProfile>("portfolio", `/analysis/keywords/${encodeURIComponent(symbol)}`, {
    method: "PUT",
    body: changes,
  });
}

/** Per-tier sentiment, attention spike and top sources for a held symbol. */
export function getSentiment(symbol: string): Promise<SentimentResponse> {
  return request<SentimentResponse>("portfolio", `/analysis/sentiment/${encodeURIComponent(symbol)}`);
}

/** Statements, ratios, F/Z scores and red flags (CSE PDF / yfinance), or tokenomics (crypto). */
export function getFundamentals(symbol: string): Promise<FundamentalsResponse> {
  return request<FundamentalsResponse>("portfolio", `/analysis/fundamentals/${encodeURIComponent(symbol)}`);
}

/** Trend, momentum, volume, range and relative strength against the symbol's market. */
export function getTechnical(symbol: string): Promise<TechnicalResponse> {
  return request<TechnicalResponse>("portfolio", `/analysis/technical/${encodeURIComponent(symbol)}`);
}

/**
 * Forecast bands per horizon for a held symbol. `method` says what produced each band:
 * "hybrid" (TimesFM + LSTM head) or "empirical" (the symbol's own past returns, used when no
 * trained head can run here). Never present an empirical band as a model forecast.
 */
export function getAnalysisForecast(symbol: string): Promise<AnalysisForecastResponse> {
  return request<AnalysisForecastResponse>("portfolio", `/analysis/forecast/${encodeURIComponent(symbol)}`);
}
