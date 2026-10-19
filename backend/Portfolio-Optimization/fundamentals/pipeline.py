"""Fundamentals for one held symbol, the active-set step, and the point-in-time store.

STORE
artifacts/fundamentals/<symbol>.json holds every processed report as a record keyed by
report id, each with `available_from` (CSE upload time; Yahoo period end + 45 days). The
store only grows. `as_of(symbol, when)` returns the newest usable record available at
`when`, which is what training and backtests must use -- never "the latest report", which
would leak a future filing into a past decision.

LIVE COST
Locally only the newest MAX_NEW_REPORTS reports are fetched per run (PDFs are ~0.2-1 MB and
the development link is slow). The historical archive is backfilled on Colab
(experiments/colab_fundamentals_backfill.ipynb) and imported with
`python -m fundamentals.pipeline --import <parquet>`.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.instruments import Exchange, Instrument

logger = logging.getLogger(__name__)

STORE_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "fundamentals"
PEERS_PATH = STORE_DIR / "peers.parquet"
MAX_NEW_REPORTS = 2

# Fields surfaced in the API summary (the full extraction stays in the store).
SUMMARY_FIELDS = ["revenue", "net_interest_income", "gross_profit", "operating_profit", "profit_before_tax",
                  "net_profit", "net_profit_parent", "total_assets", "total_liabilities", "total_equity",
                  "borrowings", "cfo", "eps_basic", "dps", "naps", "loans", "deposits"]


def _store_path(symbol: str) -> Path:
    from data.cache import _safe_symbol

    STORE_DIR.mkdir(parents=True, exist_ok=True)
    return STORE_DIR / f"{_safe_symbol(symbol)}.json"


def load_records(symbol: str) -> dict[str, dict[str, Any]]:
    path = _store_path(symbol)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("unreadable fundamentals store %s: %s", path, exc)
        return {}


def save_records(symbol: str, records: dict[str, dict[str, Any]]) -> None:
    path = _store_path(symbol)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(records, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


def as_of(symbol: str, when: datetime) -> dict[str, Any] | None:
    """Newest usable record whose report was available at `when` (point-in-time)."""
    stamp = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
    usable = [r for r in load_records(symbol).values()
              if r.get("usable") and datetime.fromisoformat(r["available_from"]) <= stamp]
    return max(usable, key=lambda r: r["available_from"], default=None)


def _ordered(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records.values(), key=lambda r: r["available_from"])


# --------------------------------------------------------------------------------------
# Per-exchange processing
# --------------------------------------------------------------------------------------

def process_cse_report(ref, *, company_name: str) -> dict[str, Any]:  # noqa: ANN001 - ReportRef
    """Download + extract + flags + tone for one CSE report. Never raises on a bad PDF:
    the record carries the reason and `usable: false`."""
    from fundamentals.extract import extract_statements, is_scanned, pdf_pages
    from fundamentals.redflags import commentary, flags, outlook_tone
    from fundamentals.reports import download_report

    base = {
        "report_id": ref.report_id, "kind": ref.kind, "title": ref.title, "url": ref.url,
        "available_from": ref.uploaded_at.isoformat(),
        "period_end": ref.period_end.isoformat() if ref.period_end else None,
        "source": "cse_pdf", "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    pages = pdf_pages(download_report(ref))
    if is_scanned(pages):
        return {**base, "usable": False, "confidence": 0.0, "reason": "scanned PDF (no text layer); OCR not supported"}

    statement = extract_statements(pages, company_name=company_name)
    if not statement.usable:
        statement = _llm_rescue(pages, statement)
    return {
        **base, **statement.to_dict(),
        "flags": flags(pages, statement.to_dict()),
        "outlook": outlook_tone(commentary(pages)),
    }


def _llm_rescue(pages: dict[int, str], statement):  # noqa: ANN001, ANN202 - Statement
    """Parser fallback: ask the local model for the key fields from the statement pages,
    then re-run the identity checks. Kept only if it scores better than the parser."""
    from dataclasses import replace

    from fundamentals.extract import MIN_CONFIDENCE, _KEY_FIELDS, run_checks
    from news.llm import chat_json

    page_ids = sorted(set(statement.pages.values()))
    if not page_ids:
        return statement
    text = "\n".join(pages[i] for i in page_ids)[:6000]
    fields = sorted({*_KEY_FIELDS[statement.template], "total_liabilities", "total_equity_and_liabilities",
                     "profit_before_tax", "tax", "current_assets", "current_liabilities"})
    try:
        payload = chat_json(
            "Extract figures from these financial statement pages for the CURRENT period, group "
            "column. Return JSON with numeric values exactly as printed (brackets as negative), "
            f"null when absent, for keys: {', '.join(fields)}.", text, timeout=300)
    except Exception as exc:  # noqa: BLE001
        logger.info("llm extraction unavailable: %s", exc)
        return statement
    current = dict(statement.current)
    for name in fields:
        value = payload.get(name)
        if isinstance(value, (int, float)) and name not in current:
            current[name] = float(value) * statement.unit
    checks = run_checks(current)
    if not checks:
        return statement
    confidence = sum(checks.values()) / len(checks) * 0.9  # model-read: never full confidence
    if confidence <= statement.confidence:
        return statement
    notes = [*statement.notes, "key fields read by local LLM after parser failure"]
    rescued = replace(statement, current=current, checks=checks, confidence=confidence, notes=notes)
    return rescued if rescued.confidence >= MIN_CONFIDENCE else statement


def refresh_cse(instrument: Instrument) -> dict[str, dict[str, Any]]:
    from fundamentals.reports import cse_report_index

    records = load_records(instrument.symbol)
    # Interims only: CSE annual reports are full glossy PDFs (often 10 MB+), and the Q4
    # interim carries the same year-end figures. Annuals come in through the Colab backfill.
    new = [ref for ref in cse_report_index(instrument.symbol)
           if ref.kind == "quarterly" and ref.report_id not in records]
    for ref in new[:MAX_NEW_REPORTS]:
        try:
            records[ref.report_id] = process_cse_report(ref, company_name=instrument.name)
        except Exception as exc:  # noqa: BLE001 - one bad report must not block the next
            logger.warning("report %s for %s failed: %s", ref.report_id, instrument.symbol, exc)
    save_records(instrument.symbol, records)
    return records


def refresh_yahoo(instrument: Instrument) -> dict[str, dict[str, Any]]:
    from fundamentals.reports import yahoo_statement

    records = load_records(instrument.symbol)
    statement = yahoo_statement(instrument.symbol)
    if statement is not None:
        key = f"yahoo-{statement['period_end'][:10]}"
        records[key] = {**statement, "report_id": key, "kind": "quarterly", "source": "yfinance",
                        "title": f"Quarter ended {statement['period_end'][:10]}", "url": None,
                        "processed_at": datetime.now(timezone.utc).isoformat()}
        save_records(instrument.symbol, records)
    return records


# --------------------------------------------------------------------------------------
# Peers
# --------------------------------------------------------------------------------------

def peer_ratios(template: str, *, exclude: str) -> list[dict[str, Any]]:
    """Latest ratios of every other analysed symbol with the same template, plus the Colab
    backfill's peer table when it has been imported."""
    from data.cache import _safe_symbol

    peers: list[dict[str, Any]] = []
    for path in STORE_DIR.glob("*.json"):
        if path.stem == _safe_symbol(exclude):
            continue
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        usable = [r for r in records.values() if r.get("usable") and r.get("ratios")]
        if usable:
            latest = max(usable, key=lambda r: r["available_from"])
            if latest["ratios"].get("template") == template:
                peers.append(latest["ratios"])
    if PEERS_PATH.exists():
        try:
            import pandas as pd

            table = pd.read_parquet(PEERS_PATH)
            table = table[(table["template"] == template) & (table["symbol"] != exclude)]
            peers.extend(table.drop(columns=["symbol"]).to_dict("records"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("peer table unreadable: %s", exc)
    return peers


# --------------------------------------------------------------------------------------
# Active-set step
# --------------------------------------------------------------------------------------

def _price(instrument: Instrument) -> float | None:
    from data import active_set

    metrics = active_set.metrics_for(instrument.symbol)
    return float(metrics["last_close"]) if metrics and metrics.get("last_close") else None


def fundamentals_step(instrument: Instrument) -> tuple[bool, dict[str, Any] | str]:
    if instrument.exchange is Exchange.BINANCE:
        from fundamentals.reports import crypto_tokenomics

        base = instrument.provider_id or instrument.symbol.removesuffix("USDT")
        tokenomics = crypto_tokenomics(base)
        if tokenomics is None:
            return False, f"no CoinGecko market data for {base}"
        return True, {"source": "coingecko", "kind": "tokenomics", "tokenomics": tokenomics, "usable": True}

    if instrument.exchange is Exchange.CSE:
        records = refresh_cse(instrument)
    else:
        records = refresh_yahoo(instrument)
    if not records:
        return False, "no financial reports found"

    ordered = _ordered(records)
    usable = [r for r in ordered if r.get("usable")]
    if not usable:
        latest = ordered[-1]
        return False, f"latest report not usable: {latest.get('reason') or 'low extraction confidence'}"

    from fundamentals.ratios import compute_ratios, sector_relative

    latest = usable[-1]
    previous = usable[-2] if len(usable) > 1 else None
    latest["ratios"] = compute_ratios(latest, price=_price(instrument), previous=previous)
    save_records(instrument.symbol, records)

    available = datetime.fromisoformat(latest["available_from"])
    block = {
        "source": latest.get("source"),
        "kind": "statements",
        "usable": True,
        "report": {
            "title": latest.get("title"), "url": latest.get("url"), "kind": latest.get("kind"),
            "period_end": latest.get("period_end"), "available_from": latest["available_from"],
            "available_from_estimated": bool(latest.get("available_from_estimated")),
            "age_days": (datetime.now(timezone.utc) - available).days,
        },
        "template": latest.get("template"),
        "confidence": latest.get("confidence"),
        "checks": latest.get("checks"),
        "summary": {k: latest["current"][k] for k in SUMMARY_FIELDS if k in (latest.get("current") or {})},
        "ratios": latest["ratios"],
        "sector_relative": sector_relative(latest["ratios"], peer_ratios(latest["ratios"]["template"],
                                                                         exclude=instrument.symbol)),
        "flags": latest.get("flags"),
        "outlook": latest.get("outlook"),
        "reports_stored": len(records),
    }
    return True, block


# --------------------------------------------------------------------------------------
# CLI: import the Colab backfill
# --------------------------------------------------------------------------------------

def import_backfill(path: Path) -> int:
    """Merge the Colab Part D export (one row per symbol x report, `record` JSON column) into
    the store, and write the peer table. Returns records imported."""
    import pandas as pd

    table = pd.read_parquet(path)
    count = 0
    for symbol, group in table.groupby("symbol"):
        records = load_records(symbol)
        for row in group.itertuples():
            record = json.loads(row.record)
            records.setdefault(str(record["report_id"]), record)
            count += 1
        save_records(symbol, records)
    if "ratios" in table.columns:
        peers = []
        for row in table.itertuples():
            ratios = json.loads(row.ratios) if isinstance(row.ratios, str) else None
            if ratios:
                peers.append({"symbol": row.symbol, "available_from": row.available_from, **ratios})
        if peers:
            frame = pd.DataFrame(peers).sort_values("available_from").drop_duplicates("symbol", keep="last")
            frame = frame.drop(columns=["available_from", "f_signals"], errors="ignore")
            STORE_DIR.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(PEERS_PATH, index=False)
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fundamentals store maintenance.")
    parser.add_argument("--import", dest="import_path", type=Path, required=True,
                        help="parquet exported by experiments/colab_fundamentals_backfill.ipynb")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    print(f"imported {import_backfill(args.import_path)} report records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
