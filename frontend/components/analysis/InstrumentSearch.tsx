"use client";

/**
 * One search box across the Colombo Stock Exchange, Binance and Yahoo Finance.
 *
 * A lookup only: picking a result fills the row, it does not register the symbol for
 * analysis -- that happens when the portfolio is saved, so half-typed symbols never enter
 * the active set.
 */

import { useEffect, useRef, useState } from "react";

import { Input } from "@/components/ui";
import { searchInstruments } from "@/lib/api/market";
import type { Instrument } from "@/lib/api/types";

const DEBOUNCE_MS = 300;

export const EXCHANGE_LABEL: Record<string, string> = { cse: "CSE", binance: "Binance", yahoo: "Yahoo" };

export function InstrumentSearch({
  value,
  onChange,
  onPick,
}: {
  value: string;
  onChange: (symbol: string) => void;
  onPick: (instrument: Instrument) => void;
}) {
  const [results, setResults] = useState<Instrument[]>([]);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const controller = useRef<AbortController | null>(null);

  useEffect(() => {
    if (query.trim().length < 2) return;
    const timer = setTimeout(async () => {
      controller.current?.abort();
      controller.current = new AbortController();
      try {
        const response = await searchInstruments(query.trim(), { limit: 8, signal: controller.current.signal });
        setResults(response.results);
        setOpen(true);
      } catch {
        // Aborted by a newer keystroke, or search unavailable: the user can still type a symbol.
      }
    }, DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [query]);

  return (
    <div className="relative">
      <Input
        value={value}
        aria-label="Symbol"
        placeholder="Search: keells, AAPL, sol…"
        onChange={(e) => {
          onChange(e.target.value.toUpperCase());
          setQuery(e.target.value);
        }}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        className="w-40"
      />
      {open && results.length > 0 && (
        <ul
          role="listbox"
          className="absolute z-10 mt-1 max-h-64 w-80 overflow-auto rounded-lg border border-black/10 bg-white text-sm shadow-lg dark:border-white/15 dark:bg-neutral-900"
        >
          {results.map((instrument) => (
            <li key={`${instrument.exchange}:${instrument.symbol}`}>
              <button
                type="button"
                role="option"
                aria-selected={false}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => {
                  onPick(instrument);
                  setOpen(false);
                }}
                className="flex w-full items-baseline justify-between gap-2 px-3 py-2 text-left hover:bg-black/5 dark:hover:bg-white/10"
              >
                <span>
                  <span className="font-medium">{instrument.symbol}</span>{" "}
                  <span className="text-neutral-500">{instrument.name}</span>
                </span>
                <span className="shrink-0 text-xs text-neutral-500">
                  {EXCHANGE_LABEL[instrument.exchange] ?? instrument.exchange} · {instrument.currency}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
