"""Phase 2b tests: statement extraction, ratios and scores, red flags, report sources, the
point-in-time store, and the fundamentals step/endpoint.

Fixtures (tests/fixtures/fundamentals/), recorded 2026-10-10:
  jkh_jun2026_pages.json    statement + commentary pages of JKH's June 2026 interim (general)
  comb_jun2026_pages.json   the same for Commercial Bank's June 2026 interim (bank, YTD first)
  cse_financials_jkh.json   JKH's CSE report index
  coingecko_markets_sol.json  CoinGecko markets row(s) for SOL
Expected values were read off the printed statements by hand.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from fundamentals import extract, pipeline, ratios, redflags, reports

FIXTURES = Path(__file__).parent / "fixtures" / "fundamentals"


def pages(name: str) -> dict[int, str]:
    raw = json.loads((FIXTURES / f"{name}_jun2026_pages.json").read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


@pytest.fixture(scope="module")
def jkh():
    return extract.extract_statements(pages("jkh"), company_name="John Keells Holdings PLC")


@pytest.fixture(scope="module")
def comb():
    return extract.extract_statements(pages("comb"), company_name="Commercial Bank of Ceylon PLC")


# --------------------------------------------------------------------------------------
# Line parsing
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("line, label, values", [
    ("Finance cost 7  (8,124,365)  (5,706,359) 42", "Finance cost", [-8124365, -5706359, 42]),
    ("Revenue from insurance contracts  6,204,105  5,441,044 14", "Revenue from insurance contracts",
     [6204105, 5441044, 14]),
    ("Proceeds from sale of a subsidiary -  401,420", "Proceeds from sale of a subsidiary", [0, 401420]),
    ("Basic  0.004  (0.046)", "Basic", [0.004, -0.046]),
    ("Dividend per share 11  0.10  0.05", "Dividend per share", [0.10, 0.05]),
    ("CONSOLIDATED INCOME STATEMENT", "CONSOLIDATED INCOME STATEMENT", []),
])
def test_split_line(line, label, values) -> None:
    got_label, got_values, _ = extract.split_line(line)
    assert got_label == label
    assert got_values == pytest.approx(values)


def test_unit_and_period_detection() -> None:
    assert extract.detect_unit("Note : All values are in Rs '000s") == 1_000
    assert extract.detect_unit("Rs.'000 Rs.'000 %") == 1_000
    assert extract.detect_unit("Rs. Mn") == 1_000_000
    assert extract.detect_period_months("For the quarter ended 30 June") == 3
    assert extract.detect_period_months("For the six months ended For the quarter ended") == 6
    assert extract.detect_period_months("no header") is None


# --------------------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------------------

def test_jkh_general_template(jkh) -> None:
    assert jkh.template == extract.GENERAL
    assert jkh.period_months == 3
    assert jkh.pages == {"income": 8, "balance": 10, "cashflow": 11}
    c = jkh.current
    # Rs '000 scaled to rupees; "Total revenue", not the first revenue sub-line.
    assert c["revenue"] == pytest.approx(141_647_460_000)
    assert c["net_profit_parent"] == pytest.approx(62_215_000)
    assert jkh.comparative["net_profit_parent"] == pytest.approx(-803_695_000)
    assert c["total_assets"] == pytest.approx(934_898_114_000)
    # Unlabelled subtotal under "Current assets".
    assert c["current_assets"] == pytest.approx(236_986_945_000)
    # JKH prints no total-liabilities line; derived from total equity and liabilities.
    assert c["total_liabilities"] == pytest.approx(934_898_114_000 - 440_544_707_000)
    # Debt summed over loans, short-term borrowings and overdrafts (current + non-current).
    assert c["borrowings"] == pytest.approx((106_646_628 + 55_678_985 + 26_871_166 + 54_102_027) * 1000)
    assert c["cfo"] == pytest.approx(12_334_943_000)
    # Per-share fields are not scaled by the thousands unit.
    assert c["eps_basic"] == pytest.approx(0.004)
    assert c["naps"] == pytest.approx(23.46)
    assert all(jkh.checks.values()) and len(jkh.checks) == 4
    assert jkh.confidence == pytest.approx(1.0)
    assert jkh.usable


def test_comb_bank_template(comb) -> None:
    assert comb.template == extract.BANK
    # Banks lead with the six-month YTD column.
    assert comb.period_months == 6
    c = comb.current
    assert c["net_interest_income"] == pytest.approx(79_693_465_000)
    assert c["total_equity"] == pytest.approx(363_888_905_000)       # not the parent-only line
    assert c["equity_parent"] == pytest.approx(357_399_155_000)
    assert c["loans"] == pytest.approx(2_212_677_022_000)
    assert c["deposits"] == pytest.approx(3_015_155_478_000)
    assert c["eps_basic"] == pytest.approx(21.14)
    assert comb.checks["net_interest_income"] and comb.checks["assets_equal_equity_plus_liabilities"]
    assert comb.usable


def test_broken_identity_lowers_confidence() -> None:
    tampered = pages("jkh")
    tampered[10] = tampered[10].replace("Total assets  934,898,114", "Total assets  999,898,114")
    statement = extract.extract_statements(tampered)
    assert statement.checks["assets_equal_equity_plus_liabilities"] is False
    assert statement.confidence < 1.0


def test_no_statements_is_unusable() -> None:
    statement = extract.extract_statements({0: "Chairperson's message. Good quarter."})
    assert not statement.usable
    assert "no income statement page found" in statement.notes


def test_conglomerate_with_insurer_stays_general() -> None:
    noisy = pages("jkh")
    noisy[3] = noisy[3] + "\nSegment note: Gross written premiums 9,000,000"
    assert extract.extract_statements(noisy).template == extract.GENERAL


def test_finance_company_template() -> None:
    assert extract.detect_template("Net interest income 1,000", "LOLC Finance PLC") == extract.FINANCE
    assert extract.detect_template("Net interest income 1,000", "Hatton National Bank PLC") == extract.BANK
    assert extract.detect_template("Gross written premiums 10", "") == extract.INSURANCE


# --------------------------------------------------------------------------------------
# Ratios and scores
# --------------------------------------------------------------------------------------

def test_jkh_ratios(jkh) -> None:
    r = ratios.compute_ratios(jkh.to_dict(), price=18.40)
    assert r["annualised"] is True
    assert r["gross_margin"] == pytest.approx(24_584_583 / 141_647_460, abs=1e-4)
    assert r["revenue_growth_yoy"] == pytest.approx(141_647_460 / 114_153_194 - 1, abs=1e-4)
    # Loss to profit is improvement (growth against |base|), not a sign-flipped number.
    assert r["earnings_growth_yoy"] > 1
    assert r["pb"] == pytest.approx(18.40 / 23.46, abs=1e-3)
    assert r["pe"] == pytest.approx(18.40 / (0.004 * 4), abs=1e-1)
    assert r["f_score_max"] == 9 and 0 <= r["f_score"] <= 9
    assert r["z_score"] is not None and r["z_zone"] in ("distress", "grey", "safe")


def test_bank_ratios_skip_z_score(comb) -> None:
    r = ratios.compute_ratios(comb.to_dict(), price=150.0)
    assert r["z_score"] is None and r["z_zone"] is None
    assert r["cost_to_income"] == pytest.approx(30_609_471 / 109_074_161, abs=1e-4)
    assert r["loans_to_deposits"] == pytest.approx(2_212_677_022 / 3_015_155_478, abs=1e-4)
    # 6-month profit annualised x2.
    assert r["eps_annualised"] == pytest.approx(42.28)


def test_missing_inputs_give_none_not_zero() -> None:
    r = ratios.compute_ratios({"template": "general", "period_months": 12, "current": {"revenue": 100.0}})
    assert r["net_margin"] is None and r["roe"] is None and r["pe"] is None
    assert r["f_score"] is None          # fewer than 5 signals computable


def test_altman_zones() -> None:
    assert ratios._z_zone(3.0) == "distress"
    assert ratios._z_zone(5.0) == "grey"
    assert ratios._z_zone(7.0) == "safe"


def test_sector_relative_percentiles() -> None:
    peers = [{"roe": 0.05, "pe": 5.0}, {"roe": 0.10, "pe": 10.0}, {"roe": 0.20, "pe": 20.0}]
    rel = ratios.sector_relative({"roe": 0.15, "pe": 8.0}, peers)
    assert rel["roe"] == pytest.approx(2 / 3, abs=1e-3)
    # Lower P/E is better for the holder, so the percentile is inverted.
    assert rel["pe"] == pytest.approx(1 - 1 / 3, abs=1e-3)
    assert ratios.sector_relative({"roe": 0.15}, peers[:2])["roe"] is None


def test_qoq_only_between_quarters(jkh) -> None:
    previous = {"period_months": 3, "current": {"revenue": 120_000_000_000.0, "net_profit_parent": 50_000_000.0}}
    r = ratios.compute_ratios(jkh.to_dict(), previous=previous)
    assert r["revenue_growth_qoq"] == pytest.approx(141_647_460_000 / 120_000_000_000 - 1, abs=1e-4)
    annual = {"period_months": 12, "current": {"revenue": 500.0}}
    assert "revenue_growth_qoq" not in ratios.compute_ratios(jkh.to_dict(), previous=annual)


# --------------------------------------------------------------------------------------
# Red flags and tone
# --------------------------------------------------------------------------------------

def test_going_concern_flags() -> None:
    doubt = {0: "There is a material uncertainty that may cast significant doubt on the Group's ability "
                "to continue as a going concern."}
    policy = {0: "The financial statements have been prepared on a going concern basis."}
    assert redflags.flags(doubt)["going_concern_doubt"] is True
    assert redflags.flags(policy)["going_concern_doubt"] is False


def test_flags_on_real_reports(jkh, comb) -> None:
    jkh_flags = redflags.flags(pages("jkh"), jkh.to_dict())
    assert jkh_flags["unaudited"] is True and jkh_flags["going_concern_doubt"] is False
    comb_flags = redflags.flags(pages("comb"), comb.to_dict())
    # A bank's contingents (guarantees, credit lines) are its business: not scored.
    assert comb_flags["contingent_to_equity"] is None
    as_general = {**comb.to_dict(), "template": "general"}
    # COMB prints contingent liabilities and commitments of 1,099,834,661 ('000).
    assert redflags.flags(pages("comb"), as_general)["contingent_to_equity"] == pytest.approx(
        1_099_834_661 / 363_888_905, abs=1e-2)


def test_commentary_excludes_statement_pages() -> None:
    text = redflags.commentary(pages("jkh"))
    assert "CHAIRPERSON" in text.upper()
    assert "STATEMENT OF FINANCIAL POSITION" not in text


def test_outlook_falls_back_to_lexicon(monkeypatch) -> None:
    monkeypatch.setattr("news.scoring.laya_router", lambda: None)

    def no_ollama(*args, **kwargs):
        raise RuntimeError("Ollama is not running")

    monkeypatch.setattr("news.llm.chat_json", no_ollama)
    result = redflags.outlook_tone("Strong growth and record momentum; revenue increased and margins improved.")
    assert result == {"tone": "positive", "confidence": pytest.approx(1.0), "method": "lexicon"}
    assert redflags.outlook_tone("") is None


# --------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------

def test_cse_report_index() -> None:
    payload = json.loads((FIXTURES / "cse_financials_jkh.json").read_text(encoding="utf-8"))
    refs = reports.parse_cse_index("JKH.N0000", payload)
    assert len(refs) == 59 + 15
    newest = refs[0]
    assert newest.kind == "quarterly"
    assert newest.url == "https://cdn.cse.lk/cmt/upload_report_file/508_1785229278745.pdf"
    assert newest.period_end == datetime(2026, 6, 30, tzinfo=timezone.utc)
    assert all(a.uploaded_at >= b.uploaded_at for a, b in zip(refs, refs[1:]))


@pytest.mark.parametrize("title, expected", [
    ("Interim Financial Statements for the Quarter ended 30th June 2026", datetime(2026, 6, 30)),
    ("Annual Report as at 31st March 2026", datetime(2026, 3, 31)),
    ("Press Release 31/03/2018", datetime(2018, 3, 31)),
    ("Annual Report 2024/25", None),
])
def test_period_end_from_title(title, expected) -> None:
    got = reports.period_end_from_title(title)
    assert (got.replace(tzinfo=None) if got else None) == expected


def test_yahoo_frames_map_to_canonical_fields() -> None:
    dates = pd.to_datetime(["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30", "2025-06-30"])
    income = pd.DataFrame({d: [100.0 + i, 40.0, 60.0, 30.0, 25.0, 5.0, 20.0, 20.0, 1.0]
                           for i, d in enumerate(dates)},
                          index=["Total Revenue", "Cost Of Revenue", "Gross Profit", "Operating Income",
                                 "Pretax Income", "Tax Provision", "Net Income",
                                 "Net Income Common Stockholders", "Basic EPS"])
    balance = pd.DataFrame({d: [1000.0, 400.0, 200.0, 600.0, 400.0, 400.0, 100.0, 50.0] for d in dates},
                           index=["Total Assets", "Current Assets", "Current Liabilities",
                                  "Total Liabilities Net Minority Interest", "Total Equity Gross Minority Interest",
                                  "Stockholders Equity", "Retained Earnings", "Ordinary Shares Number"])
    statement = reports.statement_from_yahoo_frames(income, balance, None, info={"dividendRate": 0.4})
    assert statement["current"]["revenue"] == 100.0
    assert statement["comparative"]["revenue"] == 104.0       # same quarter, a year earlier
    assert statement["current"]["naps"] == pytest.approx(8.0)
    assert statement["current"]["dps"] == pytest.approx(0.1)
    assert statement["checks"]["gross_profit"] and statement["checks"]["profit_after_tax"]
    assert statement["available_from_estimated"] is True
    assert statement["available_from"].startswith("2026-08-14")


def test_tokenomics_takes_the_largest_coin(monkeypatch) -> None:
    rows = json.loads((FIXTURES / "coingecko_markets_sol.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(reports, "request_json", lambda *a, **k: rows)
    t = reports.crypto_tokenomics("SOL")
    assert t["coingecko_id"] == "solana"
    assert 0 < t["volume_to_market_cap"] < 1
    assert 0 < t["mcap_to_fdv"] <= 1


# --------------------------------------------------------------------------------------
# Store and step
# --------------------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "STORE_DIR", tmp_path / "fundamentals")
    monkeypatch.setattr(pipeline, "PEERS_PATH", tmp_path / "fundamentals" / "peers.parquet")
    return tmp_path


def test_as_of_is_point_in_time(store) -> None:
    pipeline.save_records("X", {
        "a": {"usable": True, "available_from": "2026-02-15T00:00:00+00:00", "report_id": "a"},
        "b": {"usable": True, "available_from": "2026-05-15T00:00:00+00:00", "report_id": "b"},
        "c": {"usable": False, "available_from": "2026-06-01T00:00:00+00:00", "report_id": "c"},
    })
    assert pipeline.as_of("X", datetime(2026, 4, 1, tzinfo=timezone.utc))["report_id"] == "a"
    assert pipeline.as_of("X", datetime(2026, 7, 1, tzinfo=timezone.utc))["report_id"] == "b"
    assert pipeline.as_of("X", datetime(2026, 1, 1, tzinfo=timezone.utc)) is None


def test_cse_step_end_to_end(store, monkeypatch) -> None:
    from data.instruments import Exchange, Instrument
    from data.schema import AssetClass

    payload = json.loads((FIXTURES / "cse_financials_jkh.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(reports, "cse_report_index", lambda symbol: reports.parse_cse_index(symbol, payload))
    monkeypatch.setattr(reports, "download_report", lambda ref: Path("unused.pdf"))
    monkeypatch.setattr(extract, "pdf_pages", lambda path: pages("jkh"))
    monkeypatch.setattr(redflags, "outlook_tone", lambda text: {"tone": "positive", "confidence": 0.7, "method": "test"})
    monkeypatch.setattr(pipeline, "_price", lambda instrument: 18.40)

    jkh = Instrument(symbol="JKH.N0000", name="John Keells Holdings PLC", exchange=Exchange.CSE,
                     asset_class=AssetClass.CSE_EQUITY, currency="LKR", provider_id="508")
    ok, block = pipeline.fundamentals_step(jkh)
    assert ok, block
    assert block["report"]["url"].startswith("https://cdn.cse.lk/")
    assert block["template"] == "general"
    assert block["ratios"]["pb"] == pytest.approx(18.40 / 23.46, abs=1e-3)
    assert block["outlook"]["tone"] == "positive"
    # Only the newest MAX_NEW_REPORTS are processed per run; the rest wait for the backfill.
    assert block["reports_stored"] == pipeline.MAX_NEW_REPORTS
    # A second run finds nothing new and does not re-download.
    monkeypatch.setattr(reports, "download_report", lambda ref: pytest.fail("re-downloaded"))
    monkeypatch.setattr(pipeline, "MAX_NEW_REPORTS", 0)
    ok, _ = pipeline.fundamentals_step(jkh)
    assert ok


def test_scanned_report_is_recorded_unusable(store, monkeypatch) -> None:
    ref = reports.ReportRef("X.N0000", "1", "quarterly", "Quarter ended 31st March 2010",
                            "https://cdn.cse.lk/x.pdf", datetime(2010, 5, 1, tzinfo=timezone.utc), None)
    monkeypatch.setattr(reports, "download_report", lambda ref: Path("unused.pdf"))
    monkeypatch.setattr(extract, "pdf_pages", lambda path: {0: "", 1: ""})
    record = pipeline.process_cse_report(ref, company_name="X")
    assert record["usable"] is False and "scanned" in record["reason"]


def test_import_backfill_writes_store_and_peers(store) -> None:
    record = {"report_id": "9", "usable": True, "available_from": "2025-01-01T00:00:00+00:00"}
    table = pd.DataFrame([{
        "symbol": "HNB.N0000", "available_from": record["available_from"], "record": json.dumps(record),
        "ratios": json.dumps({"template": "bank", "roe": 0.15}),
    }])
    path = store / "backfill.parquet"
    table.to_parquet(path)
    assert pipeline.import_backfill(path) == 1
    assert pipeline.load_records("HNB.N0000")["9"]["usable"] is True
    assert pipeline.peer_ratios("bank", exclude="COMB.N0000") == [{"template": "bank", "roe": 0.15}]


def test_fundamentals_endpoint(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from data import active_set
    from service.api import app

    rows = {
        "JKH.N0000": {"symbol": "JKH.N0000", "status": "ready", "error": None, "metrics": {"fundamentals": {
            "source": "cse_pdf", "kind": "statements", "usable": True, "template": "general",
            "report": {"available_from": "2026-07-28T00:00:00+00:00", "age_days": 74},
            "ratios": {"roe": 0.01, "f_signals": {"cfo_positive": True}},
            "outlook": {"tone": "neutral", "confidence": 0.5, "method": "lexicon"}}}},
        "BTCUSDT": {"symbol": "BTCUSDT", "status": "partial", "metrics": {},
                    "error": "sentiment: x; fundamentals: no CoinGecko market data for BTC"},
    }
    monkeypatch.setattr(active_set, "status", lambda symbols=None: [rows[s] for s in symbols or [] if s in rows])
    client = TestClient(app)
    body = client.get("/analysis/fundamentals/jkh.n0000").json()
    assert body["fundamentals"]["template"] == "general"
    assert body["fundamentals"]["outlook"]["tone"] == "neutral"
    body = client.get("/analysis/fundamentals/BTCUSDT").json()
    assert body["fundamentals"] is None
    assert body["detail"] == "fundamentals: no CoinGecko market data for BTC"
    assert client.get("/analysis/fundamentals/NOPE").status_code == 404


# --------------------------------------------------------------------------------------
# Colab Part D backfill (pure parts)
# --------------------------------------------------------------------------------------

def _fact(start, end, val, accn, form, filed):
    return {"start": start, "end": end, "val": val, "accn": accn, "form": form, "filed": filed}


def test_edgar_statements_pick_quarter_and_comparative() -> None:
    from experiments.fundamentals_backfill import edgar_statements

    q = ("0001", "10-Q", "2026-08-01")
    facts = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            _fact("2026-04-01", "2026-06-30", 120.0, *q),
            _fact("2026-01-01", "2026-06-30", 230.0, *q),     # six-month YTD: ignored for a 10-Q
            _fact("2025-04-01", "2025-06-30", 100.0, *q),
        ]}},
        "NetIncomeLoss": {"units": {"USD": [_fact("2026-04-01", "2026-06-30", 12.0, *q),
                                            _fact("2025-04-01", "2025-06-30", 10.0, *q)]}},
        "Assets": {"units": {"USD": [_fact(None, "2026-06-30", 1000.0, *q), _fact(None, "2025-12-31", 900.0, *q)]}},
        "Liabilities": {"units": {"USD": [_fact(None, "2026-06-30", 600.0, *q)]}},
        "StockholdersEquity": {"units": {"USD": [_fact(None, "2026-06-30", 400.0, *q)]}},
        "EarningsPerShareBasic": {"units": {"USD/shares": [_fact("2026-04-01", "2026-06-30", 1.5, *q)]}},
    }}}
    [statement] = edgar_statements(facts)
    assert statement["available_from"] == "2026-08-01T00:00:00+00:00"
    assert statement["period_months"] == 3
    assert statement["current"]["revenue"] == 120.0 and statement["comparative"]["revenue"] == 100.0
    assert statement["current"]["total_assets"] == 1000.0 and statement["comparative"]["total_assets"] == 900.0
    assert statement["current"]["eps_basic"] == 1.5
    assert statement["checks"]["assets_equal_equity_plus_liabilities"] is True
    assert statement["usable"]


def test_checkpoint_resumes_and_skips_torn_lines(tmp_path) -> None:
    from experiments.fundamentals_backfill import Checkpoint

    checkpoint = Checkpoint(tmp_path)
    checkpoint.add("cse:A:1", {"symbol": "A", "report_id": "1", "available_from": "x", "record": {"usable": True}})
    checkpoint.add("cse:A:2", {"error": "boom"})
    with checkpoint.path.open("a", encoding="utf-8") as handle:
        handle.write('{"key": "cse:A:3", "rec')                # disconnect mid-write
    resumed = Checkpoint(tmp_path)
    assert resumed.done == {"cse:A:1", "cse:A:2"}
    assert [r["report_id"] for r in resumed.rows()] == ["1"]
