"""Colab Part D: historical fundamentals backfill (CSE report archive + SEC EDGAR).

Run on Colab (fast link); the development machine cannot download ~282 x 70 report PDFs.
Resumable: every finished report is appended to a JSONL checkpoint, so a disconnect loses
at most the report in flight. Re-running skips everything already done.

    python -m experiments.fundamentals_backfill --out artifacts/backfill [--cse-limit 20] [--no-us]

Output (small enough for the slow link home):
    <out>/fundamentals_backfill.parquet   symbol, report_id, available_from, record, ratios
Import locally with:
    python -m fundamentals.pipeline --import fundamentals_backfill.parquet

POINT IN TIME
Every record carries `available_from` -- the CSE upload timestamp, or the SEC `filed` date.
Ratios that need a price use the close on or before that date (CSE history beyond ~1 year
is not public, so older CSE records have price-free ratios only).

US SOURCE
SEC EDGAR companyfacts (XBRL), not yfinance: yfinance returns ~4-5 quarters, EDGAR the
full filing history WITH filing dates. SEC requires a descriptive User-Agent with a contact
address (SEC_USER_AGENT env var).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# Canonical field -> us-gaap tags in preference order. Flow tags are durations, balance tags
# instants; EDGAR does not say which, so the split is explicit here.
EDGAR_FLOWS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "cost_of_sales": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "gross_profit": ["GrossProfit"],
    "operating_profit": ["OperatingIncomeLoss"],
    "finance_cost": ["InterestExpense"],
    "profit_before_tax": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                          "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "tax": ["IncomeTaxExpenseBenefit"],
    "net_profit": ["ProfitLoss", "NetIncomeLoss"],
    "net_profit_parent": ["NetIncomeLoss"],
    "eps_basic": ["EarningsPerShareBasic"],
    "dps": ["CommonStockDividendsPerShareDeclared"],
    "net_interest_income": ["InterestIncomeExpenseNet"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
}
EDGAR_INSTANTS = {
    "total_assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "total_liabilities": ["Liabilities"],
    "total_equity_and_liabilities": ["LiabilitiesAndStockholdersEquity"],
    "total_equity": ["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "StockholdersEquity"],
    "equity_parent": ["StockholdersEquity"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "borrowings": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "inventories": ["InventoryNet"],
    "stated_capital": ["CommonStockValue"],
}
_PER_SHARE_UNITS = ("USD/shares",)


# --------------------------------------------------------------------------------------
# Checkpoint
# --------------------------------------------------------------------------------------

class Checkpoint:
    def __init__(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        self.path = out / "backfill_checkpoint.jsonl"
        self.done: set[str] = set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    self.done.add(json.loads(line)["key"])
                except (json.JSONDecodeError, KeyError):
                    continue  # a torn final line from a disconnect

    def add(self, key: str, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, **row}, default=str) + "\n")
        self.done.add(key)

    def rows(self) -> list[dict[str, Any]]:
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("record"):
                out.append(row)
        return out


# --------------------------------------------------------------------------------------
# CSE
# --------------------------------------------------------------------------------------

def _price_on(bars, when: datetime) -> float | None:  # noqa: ANN001 - DataFrame
    if bars is None or bars.empty:
        return None
    stamp = when.replace(tzinfo=None)
    before = bars[bars["timestamp"] <= stamp]
    return float(before["close"].iloc[-1]) if not before.empty else None


def backfill_cse(checkpoint: Checkpoint, *, limit: int | None, include_annual: bool) -> None:
    from data.exchanges import cse
    from fundamentals.pipeline import process_cse_report
    from fundamentals.ratios import compute_ratios
    from fundamentals.reports import cse_report_index

    securities = [s for s in cse.list_securities() if ".N" in s["symbol"] or ".X" in s["symbol"]]
    securities = securities[:limit] if limit else securities
    logger.info("CSE: %d securities", len(securities))
    for n, security in enumerate(securities, 1):
        symbol = security["symbol"]
        try:
            refs = cse_report_index(symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: index failed: %s", symbol, exc)
            continue
        refs = [r for r in refs if include_annual or r.kind == "quarterly"]
        try:
            bars = cse.daily_bars(symbol, security["provider_id"]) if security.get("provider_id") else None
        except Exception:  # noqa: BLE001 - prices are optional for the backfill
            bars = None
        previous = None
        for ref in sorted(refs, key=lambda r: r.uploaded_at):
            key = f"cse:{symbol}:{ref.report_id}"
            if key in checkpoint.done:
                continue
            try:
                record = process_cse_report(ref, company_name=security["name"])
            except Exception as exc:  # noqa: BLE001
                checkpoint.add(key, {"error": f"{type(exc).__name__}: {exc}"[:300]})
                continue
            ratios = None
            if record.get("usable"):
                ratios = compute_ratios(record, price=_price_on(bars, ref.uploaded_at), previous=previous)
                record["ratios"] = ratios
                previous = record
            checkpoint.add(key, {"symbol": symbol, "report_id": ref.report_id,
                                 "available_from": record["available_from"], "record": record, "ratios": ratios})
        logger.info("CSE %d/%d %s done", n, len(securities), symbol)


# --------------------------------------------------------------------------------------
# SEC EDGAR
# --------------------------------------------------------------------------------------

def _sec_get(url: str) -> Any:
    import requests

    agent = os.environ.get("SEC_USER_AGENT")
    if not agent:
        raise SystemExit("Set SEC_USER_AGENT='<project> <contact email>' (SEC requires a contact).")
    for attempt in range(3):
        response = requests.get(url, headers={"User-Agent": agent}, timeout=60)
        if response.status_code == 200:
            return response.json()
        time.sleep(2 * (attempt + 1))  # SEC allows 10 req/s; back off on 429/5xx
    raise RuntimeError(f"SEC {url}: HTTP {response.status_code}")


def _days(start: str | None, end: str) -> int | None:
    if not start:
        return None
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def edgar_statements(facts: dict[str, Any]) -> list[dict[str, Any]]:
    """One canonical statement per 10-Q / 10-K filing, keyed by accession.

    current = values for the filing's own period (3-month flows for a 10-Q, 12-month for a
    10-K; balance at period end); comparative = the same-length period ending ~1 year earlier,
    and the earliest balance instant in the same filing (prior year-end).
    """
    gaap = (facts.get("facts") or {}).get("us-gaap") or {}
    filings: dict[str, dict[str, Any]] = {}

    def entries(tag: str):  # noqa: ANN202
        units = (gaap.get(tag) or {}).get("units") or {}
        for unit, rows in units.items():
            if unit == "USD" or unit in _PER_SHARE_UNITS:
                yield from rows

    for field, tags in {**EDGAR_FLOWS, **EDGAR_INSTANTS}.items():
        is_flow = field in EDGAR_FLOWS
        for tag in tags:
            found_any = False
            for row in entries(tag):
                form = row.get("form", "")
                if form not in ("10-Q", "10-K"):
                    continue
                accn = row["accn"]
                filing = filings.setdefault(accn, {"form": form, "filed": row["filed"], "rows": {}})
                filing["rows"].setdefault(field, []).append((row.get("start"), row["end"], float(row["val"]), is_flow))
                found_any = True
            if found_any:
                break  # first tag with data wins for this field

    statements = []
    for accn, filing in filings.items():
        months = 3 if filing["form"] == "10-Q" else 12
        target = (80, 100) if months == 3 else (350, 380)
        current: dict[str, float] = {}
        comparative: dict[str, float] = {}
        period_end = None
        for field, rows in filing["rows"].items():
            if rows[0][3]:  # flow
                matching = [r for r in rows if (d := _days(r[0], r[1])) and target[0] <= d <= target[1]]
            else:
                matching = rows
            if not matching:
                continue
            latest = max(matching, key=lambda r: r[1])
            earliest = min(matching, key=lambda r: r[1])
            current[field] = latest[2]
            if earliest[1] != latest[1]:
                comparative[field] = earliest[2]
            period_end = max(period_end or latest[1], latest[1])
        if "total_assets" not in current or period_end is None:
            continue
        from fundamentals.extract import run_checks

        for values in (current, comparative):
            # Many filers (KO among them) tag only LiabilitiesAndStockholdersEquity.
            if "total_liabilities" not in values and {"total_equity_and_liabilities", "total_equity"} <= values.keys():
                values["total_liabilities"] = values["total_equity_and_liabilities"] - values["total_equity"]
            if "total_equity_and_liabilities" not in values and {"total_liabilities", "total_equity"} <= values.keys():
                values["total_equity_and_liabilities"] = values["total_liabilities"] + values["total_equity"]
        checks = run_checks(current)
        statements.append({
            "report_id": f"sec-{accn}", "kind": "quarterly" if months == 3 else "annual",
            "title": f"{filing['form']} for period ended {period_end}", "url": None, "source": "sec_edgar",
            "template": "general", "period_months": months, "unit": 1.0,
            "current": current, "comparative": comparative, "checks": checks,
            "confidence": round(sum(checks.values()) / len(checks), 3) if checks else 0.6,
            "usable": "net_profit" in current and "total_assets" in current,
            "period_end": f"{period_end}T00:00:00+00:00",
            "available_from": f"{filing['filed']}T00:00:00+00:00",
        })
    return sorted(statements, key=lambda s: s["available_from"])


def backfill_us(checkpoint: Checkpoint, symbols: list[str]) -> None:
    import yfinance as yf

    from fundamentals.ratios import compute_ratios

    tickers = {row["ticker"].upper(): int(row["cik_str"]) for row in _sec_get(SEC_TICKERS).values()}
    for symbol in symbols:
        cik = tickers.get(symbol.upper())
        if cik is None:
            logger.info("%s: not an SEC filer (ETF/FX); skipped", symbol)
            continue
        try:
            statements = edgar_statements(_sec_get(SEC_FACTS.format(cik=cik)))
            closes = yf.download(symbol, period="max", progress=False, auto_adjust=False)["Close"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: EDGAR/yfinance failed: %s", symbol, exc)
            continue
        closes = closes.squeeze() if hasattr(closes, "squeeze") else closes
        previous = None
        for statement in statements:
            key = f"us:{symbol}:{statement['report_id']}"
            if key in checkpoint.done:
                continue
            when = datetime.fromisoformat(statement["available_from"]).replace(tzinfo=None)
            before = closes[closes.index <= when]
            price = float(before.iloc[-1]) if len(before) else None
            ratios = compute_ratios(statement, price=price, previous=previous) if statement["usable"] else None
            statement["ratios"] = ratios
            previous = statement if statement["usable"] else previous
            checkpoint.add(key, {"symbol": symbol, "report_id": statement["report_id"],
                                 "available_from": statement["available_from"], "record": statement, "ratios": ratios})
        logger.info("US %s: %d filings", symbol, len(statements))


def export(checkpoint: Checkpoint, out: Path) -> Path:
    import pandas as pd

    rows = checkpoint.rows()
    frame = pd.DataFrame([{
        "symbol": r["symbol"], "report_id": r["report_id"], "available_from": r["available_from"],
        "record": json.dumps(r["record"], default=str),
        "ratios": json.dumps(r["ratios"], default=str) if r.get("ratios") else None,
    } for r in rows])
    path = out / "fundamentals_backfill.parquet"
    frame.to_parquet(path, index=False)
    usable = sum(1 for r in rows if (r["record"] or {}).get("usable"))
    logger.info("exported %d records (%d usable) to %s", len(frame), usable, path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("artifacts/backfill"))
    parser.add_argument("--cse-limit", type=int, default=None, help="first N securities (smoke runs)")
    parser.add_argument("--include-annual", action="store_true", help="also annual reports (large PDFs)")
    parser.add_argument("--no-cse", action="store_true")
    parser.add_argument("--no-us", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    checkpoint = Checkpoint(args.out)
    if not args.no_cse:
        backfill_cse(checkpoint, limit=args.cse_limit, include_annual=args.include_annual)
    if not args.no_us:
        import yaml

        universe = yaml.safe_load(Path("configs/universe.yaml").read_text(encoding="utf-8"))
        backfill_us(checkpoint, [e["symbol"] for e in universe.get("equities", [])])
    export(checkpoint, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
