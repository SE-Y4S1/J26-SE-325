"""Where fundamentals come from, per exchange.

  CSE      the company's report archive (POST /api/financials): interim and annual reports
           as text PDFs on cdn.cse.lk, each with an UPLOAD timestamp. The upload time is the
           point-in-time key: a backtest may only use a report after it was uploaded.
           (The www.cse.lk path for the same file returns an HTML page, not the PDF.)
  Yahoo    yfinance quarterly statements, mapped onto the same canonical fields as the CSE
           extractor. yfinance gives no filing date, so availability is estimated as period
           end + 45 days (the SEC 10-Q deadline for large filers) and flagged as estimated.
  Binance  crypto has no financial statements. "Fundamentals" are token economics from
           CoinGecko's free markets endpoint (no key needed, verified 2026-10-10): market
           cap, fully diluted value, supply, volume/market-cap, drawdown from all-time high.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from data.cache import CACHE_ROOT, _safe_symbol
from data.http import UpstreamError, request_bytes, request_json
from fundamentals.extract import BANK, GENERAL, run_checks

logger = logging.getLogger(__name__)

CSE_FINANCIALS_URL = "https://www.cse.lk/api/financials"
CSE_CDN = "https://cdn.cse.lk/"
COINGECKO_MARKETS = "https://api.coingecko.com/api/v3/coins/markets"
REPORT_DIR = CACHE_ROOT / "reports"
YAHOO_FILING_LAG = timedelta(days=45)

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"], start=1)}


@dataclass(frozen=True)
class ReportRef:
    symbol: str
    report_id: str
    kind: str                 # quarterly | annual
    title: str
    url: str
    uploaded_at: datetime
    period_end: datetime | None


def _ms(value: Any) -> datetime | None:
    try:
        return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def period_end_from_title(title: str) -> datetime | None:
    """'Interim Financial Statements for the Quarter ended 30th June 2026' -> 2026-06-30."""
    match = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+),?\s+(\d{4})", title)
    if match and match.group(2).lower() in _MONTHS:
        try:
            return datetime(int(match.group(3)), _MONTHS[match.group(2).lower()], int(match.group(1)),
                            tzinfo=timezone.utc)
        except ValueError:
            return None
    match = re.search(r"(\d{2})[./-](\d{2})[./-](\d{4})", title)
    if match:
        try:
            return datetime(int(match.group(3)), int(match.group(2)), int(match.group(1)), tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def parse_cse_index(symbol: str, payload: dict[str, Any]) -> list[ReportRef]:
    """Normalise the financials payload. Separate from the fetch for recorded-fixture tests."""
    refs: list[ReportRef] = []
    for key, kind in (("infoQuarterlyData", "quarterly"), ("infoAnnualData", "annual")):
        for row in payload.get(key) or []:
            path, uploaded = row.get("path"), _ms(row.get("uploadedDate"))
            if not path or not str(path).lower().endswith(".pdf") or uploaded is None:
                continue
            title = (row.get("fileText") or "").strip()
            refs.append(ReportRef(
                symbol=symbol, report_id=str(row.get("id")), kind=kind, title=title,
                url=CSE_CDN + str(path).lstrip("/"), uploaded_at=uploaded,
                period_end=period_end_from_title(title) or _ms(row.get("manualDate")),
            ))
    refs.sort(key=lambda r: r.uploaded_at, reverse=True)
    return refs


def cse_report_index(symbol: str) -> list[ReportRef]:
    payload = request_json("cse", "POST", CSE_FINANCIALS_URL, data={"symbol": symbol})
    if not isinstance(payload, dict):
        raise UpstreamError("cse", f"financials for {symbol} returned no index")
    return parse_cse_index(symbol, payload)


def download_report(ref: ReportRef) -> Path:
    """Cached PDF for a report (reports never change once uploaded)."""
    directory = REPORT_DIR / "cse" / _safe_symbol(ref.symbol)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{ref.report_id}.pdf"
    if path.exists() and path.stat().st_size > 0:
        return path
    body = request_bytes("cse", ref.url)
    if not body.startswith(b"%PDF"):
        raise UpstreamError("cse", f"{ref.url} did not return a PDF")
    tmp = path.with_suffix(".pdf.tmp")
    tmp.write_bytes(body)
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------------------
# Yahoo
# --------------------------------------------------------------------------------------

_YAHOO_INCOME = {
    "revenue": "Total Revenue", "cost_of_sales": "Cost Of Revenue", "gross_profit": "Gross Profit",
    "operating_profit": "Operating Income", "finance_cost": "Interest Expense",
    "profit_before_tax": "Pretax Income", "tax": "Tax Provision", "net_profit": "Net Income",
    "net_profit_parent": "Net Income Common Stockholders", "eps_basic": "Basic EPS",
    "net_interest_income": "Net Interest Income", "interest_income": "Interest Income",
    "interest_expense": "Interest Expense",
}
_YAHOO_BALANCE = {
    "total_assets": "Total Assets", "current_assets": "Current Assets",
    "non_current_assets": "Total Non Current Assets", "current_liabilities": "Current Liabilities",
    "total_liabilities": "Total Liabilities Net Minority Interest",
    "total_equity": "Total Equity Gross Minority Interest", "equity_parent": "Stockholders Equity",
    "retained_earnings": "Retained Earnings", "stated_capital": "Capital Stock",
    "inventories": "Inventory", "receivables": "Receivables", "cash": "Cash And Cash Equivalents",
    "borrowings": "Total Debt",
}
_YAHOO_CASHFLOW = {"cfo": "Operating Cash Flow", "cfi": "Investing Cash Flow",
                   "cff": "Financing Cash Flow", "capex": "Capital Expenditure"}


def _column(frame, column_index: int, mapping: dict[str, str]) -> dict[str, float]:  # noqa: ANN001
    import math

    out: dict[str, float] = {}
    if frame is None or frame.empty or column_index >= len(frame.columns):
        return out
    column = frame.columns[column_index]
    for name, row in mapping.items():
        if row in frame.index:
            value = frame.at[row, column]
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                out[name] = value
    return out


def statement_from_yahoo_frames(income, balance, cashflow, *, info: dict[str, Any] | None = None  # noqa: ANN001
                                ) -> dict[str, Any] | None:
    """Canonical statement from yfinance quarterly frames (newest column first).

    The comparative column is the same quarter a year earlier (column 4), matching what a
    CSE interim report prints, so the ratio code treats both sources identically.
    """
    if income is None or income.empty:
        return None
    current = {**_column(income, 0, _YAHOO_INCOME), **_column(balance, 0, _YAHOO_BALANCE),
               **_column(cashflow, 0, _YAHOO_CASHFLOW)}
    comparative = {**_column(income, 4, _YAHOO_INCOME), **_column(balance, 4, _YAHOO_BALANCE),
                   **_column(cashflow, 4, _YAHOO_CASHFLOW)}
    shares = None
    if balance is not None and not balance.empty and "Ordinary Shares Number" in balance.index:
        shares = balance.at["Ordinary Shares Number", balance.columns[0]]
    if shares and current.get("equity_parent"):
        current["naps"] = current["equity_parent"] / float(shares)
    info = info or {}
    if info.get("dividendRate"):
        current["dps"] = float(info["dividendRate"]) / 4  # quarterly, to match the period
    for values in (current, comparative):
        if "total_liabilities" in values and "total_equity" in values:
            values["total_equity_and_liabilities"] = values["total_liabilities"] + values["total_equity"]

    period_end = income.columns[0].to_pydatetime().replace(tzinfo=timezone.utc)
    template = BANK if "net_interest_income" in current and "revenue" not in current else GENERAL
    checks = run_checks(current)
    key_present = sum(1 for k in ("net_profit", "total_assets", "total_equity") if k in current) / 3
    return {
        "template": template, "period_months": 3, "unit": 1.0, "current": current,
        "comparative": comparative, "pages": {}, "checks": checks,
        "confidence": round((sum(checks.values()) / len(checks) if checks else 0.6) * key_present, 3),
        "usable": key_present >= 2 / 3, "notes": ["source: yfinance; filing date estimated"],
        "period_end": period_end.isoformat(),
        "available_from": (period_end + YAHOO_FILING_LAG).isoformat(),
        "available_from_estimated": True,
    }


def yahoo_statement(symbol: str) -> dict[str, Any] | None:
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    try:
        info = ticker.info or {}
    except Exception:  # noqa: BLE001 - info is optional decoration
        info = {}
    return statement_from_yahoo_frames(ticker.quarterly_income_stmt, ticker.quarterly_balance_sheet,
                                       ticker.quarterly_cashflow, info=info)


# --------------------------------------------------------------------------------------
# Crypto
# --------------------------------------------------------------------------------------

def tokenomics_from_market(row: dict[str, Any]) -> dict[str, Any]:
    def f(key: str) -> float | None:
        value = row.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    mcap, fdv, volume = f("market_cap"), f("fully_diluted_valuation"), f("total_volume")
    circ, total, max_supply = f("circulating_supply"), f("total_supply"), f("max_supply")
    return {
        "coingecko_id": row.get("id"), "name": row.get("name"),
        "market_cap_usd": mcap, "market_cap_rank": row.get("market_cap_rank"),
        "fdv_usd": fdv, "volume_24h_usd": volume,
        "volume_to_market_cap": volume / mcap if volume and mcap else None,
        "circulating_supply": circ, "max_supply": max_supply,
        # Share of eventual supply not yet circulating: future dilution for holders.
        "supply_overhang": 1 - circ / (max_supply or total) if circ and (max_supply or total) else None,
        "mcap_to_fdv": mcap / fdv if mcap and fdv else None,
        "drawdown_from_ath": (f("ath_change_percentage") or 0.0) / 100 if f("ath_change_percentage") is not None else None,
        "as_of": row.get("last_updated"),
    }


def crypto_tokenomics(base_asset: str) -> dict[str, Any] | None:
    """Tokenomics for a base asset ('BTC'). When several coins share a ticker, the largest
    by market cap is taken -- the one a Binance USDT pair almost always refers to."""
    rows = request_json("coingecko", "GET", COINGECKO_MARKETS,
                        params={"vs_currency": "usd", "symbols": base_asset.lower(), "order": "market_cap_desc"})
    if not isinstance(rows, list) or not rows:
        return None
    best = max(rows, key=lambda r: r.get("market_cap") or 0)
    return tokenomics_from_market(best)
