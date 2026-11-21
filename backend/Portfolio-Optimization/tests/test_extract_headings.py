"""Why two thirds of CSE report records came back unusable, and the fixes that follow from it.

Diagnosis, from the 7,602 CSE records in the local backfill: balance-sheet identities pass 4,659
times and fail 27, so the extractor is accurate when it finds numbers. It is not finding enough:
"no income statement page found" on ~3,500 records, "reporting period not detected" on ~4,700.
The income heading pattern excluded "STATEMENT OF PROFIT OR LOSS AND OTHER COMPREHENSIVE INCOME"
(a combined statement) and did not know "STATEMENT OF COMPREHENSIVE INCOME" at all; the period
detector wanted the exact words "three months ended" and missed "3 months ended" and "three
months period ended".

These tests use synthetic pages. Whether they lift the real usable rate is only measurable by
re-running Part D on Colab against the PDFs.
"""

from __future__ import annotations

import pytest

from fundamentals import extract


def income_page(heading: str, period: str = "For the three months ended 30 September 2025") -> str:
    return f"""{heading}
Sample Holdings PLC
{period}                    Rs.'000
                             2025          2024
Revenue  1,000,000  900,000
Cost of sales  (600,000)  (540,000)
Gross profit  400,000  360,000
Profit before tax  150,000  120,000
Income tax expense  (30,000)  (24,000)
Profit for the period  120,000  96,000
Basic  1.20  0.96
"""


BALANCE = """STATEMENT OF FINANCIAL POSITION
As at 30 September 2025                    Rs.'000
Non-current assets
Total non-current assets  700,000  650,000
Current assets
Total current assets  300,000  250,000
Total assets  1,000,000  900,000
Total equity  600,000  540,000
Total liabilities  400,000  360,000
Total equity and liabilities  1,000,000  900,000
"""

OCI_ONLY = """STATEMENT OF COMPREHENSIVE INCOME
Sample Holdings PLC
For the three months ended 30 September 2025                    Rs.'000
Profit for the period  120,000  96,000
Other comprehensive income
Total comprehensive income for the period  125,000  99,000
"""


@pytest.mark.parametrize("heading", [
    "STATEMENT OF PROFIT OR LOSS AND OTHER COMPREHENSIVE INCOME",     # one combined statement
    "CONSOLIDATED STATEMENT OF PROFIT OR LOSS AND OTHER COMPREHENSIVE INCOME",
    "STATEMENT OF COMPREHENSIVE INCOME",
    "CONSOLIDATED STATEMENT OF COMPREHENSIVE INCOME",
    "STATEMENT OF PROFIT AND LOSS",
    "PROFIT AND LOSS ACCOUNT",
    "STATEMENT OF INCOME",
    "INCOME STATEMENT",                                               # already worked
    "STATEMENT OF PROFIT OR LOSS",                                    # already worked
])
def test_every_common_income_statement_heading_is_found(heading: str) -> None:
    statement = extract.extract_statements({0: income_page(heading), 1: BALANCE}, company_name="Sample Holdings PLC")

    assert "income" in statement.pages, f"{heading!r}: {statement.notes}"
    assert statement.current["revenue"] == pytest.approx(1_000_000 * 1000)
    assert statement.current["net_profit"] == pytest.approx(120_000 * 1000)
    assert statement.confidence >= extract.MIN_CONFIDENCE and statement.usable


def test_a_comprehensive_income_only_page_does_not_displace_the_real_income_statement() -> None:
    """Widening the heading admits OCI pages. The page with the most parsed fields must still win."""
    pages = {0: income_page("STATEMENT OF PROFIT OR LOSS"), 1: OCI_ONLY, 2: BALANCE}
    statement = extract.extract_statements(pages, company_name="Sample Holdings PLC")

    assert statement.pages["income"] == 0
    assert statement.current["revenue"] == pytest.approx(1_000_000 * 1000)


def test_an_oci_page_alone_yields_profit_but_never_an_invented_revenue() -> None:
    """The page does hold "Profit for the period", so net profit is real; revenue is simply not
    there. Ratios that need revenue come back None, and confidence stays below a full statement's."""
    full = extract.extract_statements({0: income_page("STATEMENT OF PROFIT OR LOSS"), 1: BALANCE},
                                      company_name="Sample Holdings PLC")
    partial = extract.extract_statements({0: OCI_ONLY, 1: BALANCE}, company_name="Sample Holdings PLC")

    assert "revenue" not in partial.current
    assert partial.current["net_profit"] == pytest.approx(120_000 * 1000)
    assert partial.confidence < full.confidence


@pytest.mark.parametrize("text, months", [
    ("For the three months ended 30 September 2025", 3),
    ("For the three months period ended 30 September 2025", 3),
    ("For the 3 months ended 30 Sep 2025", 3),
    ("THREE MONTHS ENDED 30.09.2025", 3),
    ("For the quarter ended 30 June 2026", 3),
    ("For the quarter ending 30 June 2026", 3),
    ("For the six months ended 30 September 2025", 6),
    ("For the 6 months period ended 30 September 2025", 6),
    ("For the half year ended 30 September 2025", 6),
    ("For the half-year ended 30 September 2025", 6),
    ("For the nine months ended 31 December 2025", 9),
    ("For the 9 months ended 31 December 2025", 9),
    ("For the twelve months ended 31 March 2026", 12),
    ("For the 12 months ended 31 March 2026", 12),
    ("For the year ended 31 March 2026", 12),
    ("For the financial year ended 31 March 2026", 12),
])
def test_period_detection_covers_the_ways_cse_reports_phrase_it(text: str, months: int) -> None:
    assert extract.detect_period_months(text) == months


def test_period_detection_still_takes_the_first_period_stated() -> None:
    # A quarterly report prints the current period first and a year-end comparative later.
    assert extract.detect_period_months("For the three months ended 30 June   For the year ended 31 March") == 3
    assert extract.detect_period_months("For the six months ended For the quarter ended") == 6


@pytest.mark.parametrize("text", ["no header at all", "For the period ended 30 September 2025", "Year-end review"])
def test_period_detection_declines_rather_than_guessing(text: str) -> None:
    assert extract.detect_period_months(text) is None


@pytest.mark.parametrize("text, unit", [
    ("Rs.'000", 1_000), ("Rs. '000", 1_000), ("LKR '000", 1_000), ("LKR'000", 1_000),
    ("(Amounts in Rs. thousands)", 1_000), ("All values are in LKR thousands", 1_000), ("In thousands of rupees", 1_000),
    ("Rs. Mn", 1_000_000), ("LKR Mn", 1_000_000), ("(in Rs. millions)", 1_000_000), ("LKR Million", 1_000_000),
    ("Rs. Bn", 1_000_000_000),
    ("Rupees", 1.0), ("", 1.0),
])
def test_unit_detection_covers_lkr_and_spelled_out_units(text: str, unit: float) -> None:
    """A missed unit is silent: the identities are scale-invariant, so numbers 1,000x too small
    pass every check and corrupt every ratio that mixes them with a market price."""
    assert extract.detect_unit(text) == unit
