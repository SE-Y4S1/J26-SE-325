"""CSE interim/annual report text -> canonical statement fields.

CSE reports are text PDFs whose statement pages extract cleanly line by line (verified on
JKH and COMB, June 2026). Each statement line is

    <label> [note ref] <current> <comparative> [change % | more columns...]

with brackets for negatives and '-' for nil. The FIRST value column is the reporting entity's
current period: the group (consolidated) column where a report has group and company columns,
and the cumulative year-to-date column where a bank reports both YTD and the quarter.

TEMPLATES
Banks and finance companies report net interest income, impairment and deposits instead of
revenue and cost of sales; insurers report premiums. The template is detected from the text
(a statement with "net interest income" is a bank/finance company), so a mis-tagged sector
cannot route a bank through the general template.

CONFIDENCE
Every extraction is checked against accounting identities (assets = liabilities + equity,
gross profit = revenue - cost of sales, ...). confidence = checks passed / checks applicable,
discounted when key fields are missing. Below MIN_CONFIDENCE the statement is stored for
inspection but excluded from decisions -- a wrong ratio is worse than a missing one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

MIN_CONFIDENCE = 0.5

GENERAL, BANK, FINANCE, INSURANCE = "general", "bank", "finance", "insurance"

# Statement headings. Searched over the whole page: COMB prints its headings at the foot.
_HEADINGS = {
    "income": re.compile(r"INCOME STATEMENT|STATEMENT OF PROFIT OR LOSS(?! AND OTHER COMPREHENSIVE)", re.I),
    "balance": re.compile(r"STATEMENT OF FINANCIAL POSITION|BALANCE SHEET", re.I),
    "cashflow": re.compile(r"STATEMENT OF CASH FLOWS?|CASH FLOW STATEMENT", re.I),
}
_GROUP = re.compile(r"CONSOLIDATED|\bGROUP\b", re.I)

# Canonical field -> label patterns, matched against the start of the cleaned label. The
# first matching line wins, so put specific patterns before general ones.
_INCOME = {
    "revenue": [r"total revenue", r"revenue", r"turnover", r"sales\b"],
    "cost_of_sales": [r"cost of sales", r"cost of revenue"],
    "gross_profit": [r"gross profit"],
    "operating_profit": [r"results? from operating activities", r"operating profit(?! before taxes on)",
                         r"profit from operations"],
    "finance_cost": [r"finance costs?", r"interest expense on borrowings"],
    "finance_income": [r"finance income"],
    "profit_before_tax": [r"profit before (income )?tax", r"profit/\(loss\) before tax"],
    "tax": [r"(less\s*:\s*)?income tax expense", r"tax expense", r"income tax"],
    "net_profit": [r"profit for the (period|year)", r"profit/\(loss\) for the (period|year)"],
    "net_profit_parent": [r"equity holders of the (parent|company|bank)", r"owners of the (parent|company)"],
    "eps_basic": [r"basic( earnings per (ordinary )?share)?\b"],
    "dps": [r"dividends? per share"],
    # bank / finance
    "gross_income": [r"gross income"],
    "interest_income": [r"interest income"],
    "interest_expense": [r"(less\s*:\s*)?interest expenses?"],
    "net_interest_income": [r"net interest income"],
    "fee_income": [r"net fee and commission income"],
    "total_operating_income": [r"total operating income"],
    "impairment": [r"(less\s*:\s*)?impairment (charges?|losses?)"],
    "operating_expenses": [r"(less\s*:\s*)?(total )?operating expenses"],
    # insurance
    "gross_written_premium": [r"gross written premiums?"],
    "net_earned_premium": [r"net earned premiums?"],
    "claims": [r"net (claims|benefits)( and benefits)? (paid|incurred)", r"net insurance benefits"],
}
_BALANCE = {
    "total_assets": [r"total assets"],
    "inventories": [r"inventories"],
    "receivables": [r"trade and other receivables"],
    "cash": [r"cash in hand and at bank", r"cash and cash equivalents", r"cash and bank balances"],
    "total_equity": [r"total equity$", r"total shareholders'? (equity|funds)$"],
    "equity_parent": [r"total equity attributable to equity holders", r"equity attributable to (equity holders|owners)"],
    "stated_capital": [r"stated capital", r"share capital"],
    "retained_earnings": [r"retained earnings", r"revenue reserves?", r"accumulated profits?"],
    "total_liabilities": [r"total liabilities$"],
    "total_equity_and_liabilities": [r"total (equity and liabilities|liabilities and equity)"],
    "naps": [r"net assets? (value )?per (ordinary )?share"],
    # bank / finance
    "loans": [r"financial assets at amortised cost\W+loans and (advances|receivables)",
              r"loans and (advances|receivables)( to (other )?customers)?"],
    "deposits": [r"financial liabilities at amortised cost\W+due to depositors", r"due to depositors",
                 r"deposits from customers"],
}
# Summed rather than first-match: debt is spread over several lines.
_BORROWINGS = re.compile(
    r"^(interest[- ]bearing (loans and )?borrowings|short[- ]term borrowings|bank overdrafts?|"
    r"borrowings|debentures|subordinated liabilities|other borrowings)", re.I)
_CASHFLOW = {
    "cfo": [r"net cash (flows? )?(from|generated from|\(used in\)|from/\(used in\)|from / \(used in\)) operating",
            r"net cash flow from operating activities"],
    "cfi": [r"net cash (flows? )?(from|used in|\(used in\)|from/\(used in\)|from / \(used in\))\s*(/ \(used in\) )?investing"],
    "cff": [r"net cash (flows? )?(from|used in|\(used in\)|from/\(used in\)|from / \(used in\))\s*(/ \(used in\) )?financing"],
    "capex": [r"purchase (and construction )?of property,? plant and equipment"],
}

# Section headers whose unlabeled subtotal line (JKH prints current-assets total with no
# label) belongs to a canonical field.
_SECTIONS = [
    (re.compile(r"^non[- ]?current assets", re.I), "non_current_assets"),
    (re.compile(r"^current assets", re.I), "current_assets"),
    (re.compile(r"^non[- ]?current liabilities", re.I), "non_current_liabilities"),
    (re.compile(r"^current liabilities", re.I), "current_liabilities"),
    (re.compile(r"^equity attributable to equity holders", re.I), "equity_parent"),
]
_TOTAL_SECTION = re.compile(r"^total (non[- ]?current |current )(assets|liabilities)", re.I)

PER_SHARE_FIELDS = {"eps_basic", "dps", "naps"}

_NUMBER = re.compile(r"^\(?-?[\d,]*\d(\.\d+)?\)?%?$")
_DASH = {"-", "–", "—"}


@dataclass
class Statement:
    """Extracted fields for one report: current and comparative columns, in absolute units
    (per-share fields in currency per share)."""

    template: str
    period_months: int | None
    unit: float
    current: dict[str, float] = field(default_factory=dict)
    comparative: dict[str, float] = field(default_factory=dict)
    pages: dict[str, int] = field(default_factory=dict)
    checks: dict[str, bool] = field(default_factory=dict)
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.confidence >= MIN_CONFIDENCE

    def to_dict(self) -> dict[str, Any]:
        return {
            "template": self.template, "period_months": self.period_months, "unit": self.unit,
            "current": self.current, "comparative": self.comparative, "pages": self.pages,
            "checks": self.checks, "confidence": round(self.confidence, 3), "usable": self.usable,
            "notes": self.notes,
        }


def _to_number(token: str) -> float | None:
    if token in _DASH:
        return 0.0
    if not _NUMBER.match(token):
        return None
    negative = token.startswith("(") or token.startswith("-")
    digits = token.strip("()%").lstrip("-").replace(",", "")
    try:
        value = float(digits)
    except ValueError:
        return None
    return -value if negative else value


def split_line(line: str) -> tuple[str, list[float], list[str]]:
    """(label, values, raw value tokens). Values are the trailing run of numeric tokens."""
    tokens = line.split()
    values: list[float] = []
    raw: list[str] = []
    while tokens:
        number = _to_number(tokens[-1])
        if number is None:
            break
        values.insert(0, number)
        raw.insert(0, tokens.pop())
    label = " ".join(tokens)
    # A leading 1-2 digit integer before at least two values is a note reference ("Finance
    # cost 7 (8,124,365) (5,706,359) 42"), not a value.
    if len(values) >= 3 and re.fullmatch(r"\d{1,2}", raw[0]):
        values, raw = values[1:], raw[1:]
    return label, values, raw


def _clean_label(label: str) -> str:
    label = re.sub(r"[^\w\s/()%'&,.:-]", " ", label)  # PDF bullets / replacement chars
    return re.sub(r"\s+", " ", label).strip().lower()


def detect_unit(text: str) -> float:
    lowered = text.lower()
    if re.search(r"rs\.?\s*'?\s*000|'000s?|rs\.?\s*thousands?", lowered):
        return 1_000.0
    if re.search(r"rs\.?\s*(mn|million)", lowered):
        return 1_000_000.0
    if re.search(r"rs\.?\s*(bn|billion)", lowered):
        return 1_000_000_000.0
    return 1.0


def detect_period_months(text: str) -> int | None:
    match = re.search(r"(quarter|three months|six months|nine months|twelve months|year) ended", text, re.I)
    if not match:
        return None
    return {"quarter": 3, "three months": 3, "six months": 6, "nine months": 9,
            "twelve months": 12, "year": 12}[match.group(1).lower()]


def detect_template(text: str, company_name: str = "") -> str:
    lowered = text.lower()
    if ("gross written premium" in lowered or "net earned premium" in lowered) and "cost of sales" not in lowered:
        return INSURANCE
    if "net interest income" in lowered:
        if re.search(r"finance|leasing|credit|investments?\b", company_name, re.I) and "bank" not in company_name.lower():
            return FINANCE
        return BANK
    return GENERAL


def _parse_page(text: str, patterns: dict[str, list[str]], *, sum_borrowings: bool = False
                ) -> tuple[dict[str, float], dict[str, float]]:
    current: dict[str, float] = {}
    comparative: dict[str, float] = {}
    compiled = {name: [re.compile(rf"^{p}", re.I) for p in pats] for name, pats in patterns.items()}
    section: str | None = None
    ranks: dict[str, int] = {}
    borrowings = [0.0, 0.0]
    found_borrowing = False
    for line in text.splitlines():
        label, values, _ = split_line(line)
        clean = _clean_label(label)
        if not values:
            for header, name in _SECTIONS:
                if header.match(clean):
                    section = name
            continue
        if not clean and section and len(values) >= 2:
            current.setdefault(section, values[0])
            comparative.setdefault(section, values[1])
            section = None
            continue
        total_section = _TOTAL_SECTION.match(clean)
        if total_section:
            name = ("non_current_" if "non" in total_section.group(1).lower() else "current_") + \
                total_section.group(2).lower()
            current.setdefault(name, values[0])
            if len(values) > 1:
                comparative.setdefault(name, values[1])
            continue
        if sum_borrowings and _BORROWINGS.match(clean) and len(values) >= 2:
            borrowings[0] += values[0]
            borrowings[1] += values[1]
            found_borrowing = True
        for name, regexes in compiled.items():
            rank = next((i for i, r in enumerate(regexes) if r.match(clean)), None)
            # A better-ranked pattern on a later line wins ("Total revenue" over "Revenue
            # from contracts..."); among equals the first line wins.
            if rank is None or rank >= ranks.get(name, len(regexes)):
                continue
            ranks[name] = rank
            current[name] = values[0]
            if len(values) > 1:
                comparative[name] = values[1]
            else:
                comparative.pop(name, None)
            break
    if found_borrowing:
        current["borrowings"], comparative["borrowings"] = borrowings
    return current, comparative


def _candidates(pages: dict[int, str], kind: str) -> list[int]:
    hits = [i for i, text in sorted(pages.items()) if _HEADINGS[kind].search(text)]
    # Group/consolidated pages first; within each, document order.
    return sorted(hits, key=lambda i: (not _GROUP.search(pages[i]), i))


def _best(pages: dict[int, str], kind: str, patterns: dict[str, list[str]], *, sum_borrowings: bool = False):
    best: tuple[int, dict, dict] | None = None
    for index in _candidates(pages, kind):
        current, comparative = _parse_page(pages[index], patterns, sum_borrowings=sum_borrowings)
        if best is None or len(current) > len(best[1]):
            best = (index, current, comparative)
    if best is None:
        return None
    index, current, comparative = best
    # Statements spill onto the next page (COMB's equity section): fill only fields still
    # missing, and only from the immediately following page.
    following = pages.get(index + 1)
    if following is not None:
        more_current, more_comparative = _parse_page(following, patterns, sum_borrowings=sum_borrowings)
        for name, value in more_current.items():
            if name not in current:
                current[name] = value
                if name in more_comparative:
                    comparative[name] = more_comparative[name]
    return index, current, comparative


def _close(a: float, b: float, tolerance: float = 0.01) -> bool:
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / scale <= tolerance


def run_checks(values: dict[str, float]) -> dict[str, bool]:
    """Accounting identities that apply given the fields present."""
    checks: dict[str, bool] = {}
    v = values
    if {"total_assets", "total_equity_and_liabilities"} <= v.keys():
        checks["assets_equal_equity_plus_liabilities"] = _close(v["total_assets"], v["total_equity_and_liabilities"])
    elif {"total_assets", "total_equity", "total_liabilities"} <= v.keys():
        checks["assets_equal_equity_plus_liabilities"] = _close(
            v["total_assets"], v["total_equity"] + v["total_liabilities"])
    if {"current_assets", "non_current_assets", "total_assets"} <= v.keys():
        checks["asset_subtotals"] = _close(v["current_assets"] + v["non_current_assets"], v["total_assets"])
    if {"revenue", "cost_of_sales", "gross_profit"} <= v.keys():
        # cost of sales is printed negative (bracketed) in most reports, positive in some.
        checks["gross_profit"] = _close(v["revenue"] - abs(v["cost_of_sales"]), v["gross_profit"])
    if {"profit_before_tax", "tax", "net_profit"} <= v.keys():
        checks["profit_after_tax"] = _close(v["profit_before_tax"] - abs(v["tax"]), v["net_profit"])
    if {"interest_income", "interest_expense", "net_interest_income"} <= v.keys():
        checks["net_interest_income"] = _close(
            v["interest_income"] - abs(v["interest_expense"]), v["net_interest_income"])
    return checks


_KEY_FIELDS = {
    GENERAL: ["revenue", "net_profit", "total_assets", "total_equity"],
    BANK: ["net_interest_income", "net_profit", "total_assets", "total_equity"],
    FINANCE: ["net_interest_income", "net_profit", "total_assets", "total_equity"],
    INSURANCE: ["net_profit", "total_assets", "total_equity"],
}


def extract_statements(pages: dict[int, str], *, company_name: str = "") -> Statement:
    """Extract one report. `pages` maps page index -> extracted text."""
    # Template from the income statement pages only: a conglomerate's segment notes mention
    # premiums or interest income (JKH owns an insurer) without it being an insurer.
    income_pages = _candidates(pages, "income")
    template = detect_template("\n".join(pages[i] for i in income_pages[:2]), company_name)

    current: dict[str, float] = {}
    comparative: dict[str, float] = {}
    found_pages: dict[str, int] = {}
    unit = 1.0
    period_months: int | None = None
    notes: list[str] = []

    for kind, patterns, sums in (("income", _INCOME, False), ("balance", _BALANCE, True), ("cashflow", _CASHFLOW, False)):
        result = _best(pages, kind, patterns, sum_borrowings=sums)
        if result is None:
            notes.append(f"no {kind} statement page found")
            continue
        index, page_current, page_comparative = result
        found_pages[kind] = index
        page_unit = detect_unit(pages[index])
        if kind == "income":
            unit = page_unit
            period_months = detect_period_months(pages[index])
        for name, value in page_current.items():
            scale = 1.0 if name in PER_SHARE_FIELDS else page_unit
            current.setdefault(name, value * scale)
            if name in page_comparative:
                comparative.setdefault(name, page_comparative[name] * scale)

    # Derive what reports leave implicit.
    for values in (current, comparative):
        if "total_liabilities" not in values and {"total_equity_and_liabilities", "total_equity"} <= values.keys():
            values["total_liabilities"] = values["total_equity_and_liabilities"] - values["total_equity"]
        if "total_equity_and_liabilities" not in values and {"total_liabilities", "total_equity"} <= values.keys():
            values["total_equity_and_liabilities"] = values["total_liabilities"] + values["total_equity"]
        if "equity_parent" not in values and "total_equity" in values:
            values["equity_parent"] = values["total_equity"]
        if "net_profit_parent" not in values and "net_profit" in values:
            values["net_profit_parent"] = values["net_profit"]

    checks = run_checks(current)
    # total_equity_and_liabilities is derived when missing; the identity is only evidence
    # when both sides were printed.
    passed = sum(checks.values())
    check_score = passed / len(checks) if checks else 0.0
    key = _KEY_FIELDS[template]
    coverage = sum(1 for name in key if name in current) / len(key)
    confidence = check_score * coverage if checks else 0.4 * coverage
    if period_months is None:
        notes.append("reporting period not detected; flows are not annualised")
        confidence *= 0.8

    return Statement(
        template=template, period_months=period_months, unit=unit, current=current,
        comparative=comparative, pages=found_pages, checks=checks, confidence=confidence, notes=notes,
    )


def pdf_pages(path) -> dict[int, str]:  # noqa: ANN001 - str | Path
    """Text of every page. Pages without a text layer (scans) come back empty."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    out: dict[int, str] = {}
    for index, page in enumerate(reader.pages):
        try:
            out[index] = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - one malformed page must not lose the report
            out[index] = ""
    return out


def is_scanned(pages: dict[int, str]) -> bool:
    """A report whose pages carry almost no text is an image scan (OCR not supported)."""
    return sum(len(t) for t in pages.values()) < 200 * max(1, len(pages)) * 0.1
