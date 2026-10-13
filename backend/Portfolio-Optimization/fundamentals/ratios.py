"""Ratios, Piotroski F-score and Altman Z''-score from one extracted statement.

Inputs are the canonical fields from extract.py (or the yfinance mapping in reports.py):
`current` and `comparative` columns. In a CSE interim report the comparative income column is
the same period a year earlier (so growth is year-on-year) and the comparative balance column
is the previous financial year-end.

ANNUALISATION
Flow items (profit, cash flow) cover `period_months`; return ratios scale them to a year so a
quarter's ROE is comparable with an annual report's. Valuation multiples use the annualised
EPS. Annualising a single quarter ignores seasonality -- the value is labelled as such.

Z-SCORE CHOICE
Altman's Z''-score for emerging markets (Altman 2005) rather than the original 1968 Z: it
needs no market value of equity (thin CSE prices make that noisy) and was fitted on
non-US, non-manufacturing firms. It does not apply to banks, finance companies or insurers,
whose leverage is their business model; those get None.

Every ratio is None when an input is missing or a denominator is ~0. A missing ratio is
reported as missing, never as 0.
"""

from __future__ import annotations

import math
from typing import Any

from fundamentals.extract import BANK, FINANCE, GENERAL, INSURANCE

FINANCIAL_TEMPLATES = {BANK, FINANCE, INSURANCE}


def _div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b) or abs(b) < 1e-9:
        return None
    return a / b


def _avg(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return (a + b) / 2


def _growth(now: float | None, before: float | None) -> float | None:
    """Growth over a base that may be negative: measured against |base|, so a loss narrowing
    from -800 to +62 reads as improvement rather than a sign-flipped number."""
    if now is None or before is None or abs(before) < 1e-9:
        return None
    return (now - before) / abs(before)


def _round(values: dict[str, Any]) -> dict[str, Any]:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in values.items()}


def compute_ratios(statement: dict[str, Any], *, price: float | None = None,
                   previous: dict[str, Any] | None = None) -> dict[str, Any]:
    """All ratios for one statement dict (Statement.to_dict() shape).

    `previous` is the prior stored report for the same symbol; used for quarter-on-quarter
    growth when both are single quarters.
    """
    cur: dict[str, float] = statement.get("current") or {}
    cmp_: dict[str, float] = statement.get("comparative") or {}
    template = statement.get("template") or GENERAL
    months = statement.get("period_months")
    annualise = 12 / months if months else 1.0
    g = cur.get

    net_parent = g("net_profit_parent", g("net_profit"))
    equity_parent = g("equity_parent", g("total_equity"))
    out: dict[str, Any] = {
        "template": template,
        "period_months": months,
        "annualised": bool(months and months != 12),
        # Profitability
        "net_margin": _div(g("net_profit"), g("revenue")),
        "gross_margin": _div(g("gross_profit"), g("revenue")),
        "operating_margin": _div(g("operating_profit"), g("revenue")),
        "roe": _div(_mul(net_parent, annualise), _avg(equity_parent, cmp_.get("equity_parent"))),
        "roa": _div(_mul(g("net_profit"), annualise), _avg(g("total_assets"), cmp_.get("total_assets"))),
        # Growth (year-on-year from the comparative column)
        "revenue_growth_yoy": _growth(g("revenue", g("net_interest_income")),
                                      cmp_.get("revenue", cmp_.get("net_interest_income"))),
        "earnings_growth_yoy": _growth(net_parent, cmp_.get("net_profit_parent", cmp_.get("net_profit"))),
        # Leverage and liquidity
        "debt_to_equity": _div(g("borrowings"), g("total_equity")),
        "liabilities_to_assets": _div(g("total_liabilities"), g("total_assets")),
        "interest_cover": _div(g("operating_profit"), abs(g("finance_cost")) if g("finance_cost") else None),
        "current_ratio": _div(g("current_assets"), g("current_liabilities")),
        # Earnings quality
        "cfo_to_net_profit": _div(g("cfo"), g("net_profit")) if (g("net_profit") or 0) > 0 else None,
        "dividend_cover": _div(g("eps_basic"), g("dps")),
    }

    if template in (BANK, FINANCE):
        out.update({
            # Proxy: NII over average total assets (interest-earning assets are not split out
            # in the summary statements).
            "nim_on_assets": _div(_mul(g("net_interest_income"), annualise),
                                  _avg(g("total_assets"), cmp_.get("total_assets"))),
            "cost_to_income": _div(abs(g("operating_expenses")) if g("operating_expenses") else None,
                                   g("total_operating_income")),
            "impairment_to_loans": _div(_mul(abs(g("impairment")) if g("impairment") else None, annualise),
                                        g("loans")),
            "loans_to_deposits": _div(g("loans"), g("deposits")),
            "equity_to_assets": _div(g("total_equity"), g("total_assets")),
        })

    # Valuation (needs a price in the statement's currency)
    eps_annual = _mul(g("eps_basic"), annualise)
    out.update({
        "eps_annualised": eps_annual,
        "pe": _div(price, eps_annual) if price and eps_annual and eps_annual > 0 else None,
        "pb": _div(price, g("naps")) if price else None,
        "dividend_yield": _div(_mul(g("dps"), annualise), price) if price else None,
        "earnings_yield": _div(eps_annual, price) if price else None,
    })

    # Quarter-on-quarter, only between two single quarters.
    if previous and months == 3 and previous.get("period_months") == 3:
        prev = previous.get("current") or {}
        out["revenue_growth_qoq"] = _growth(g("revenue", g("net_interest_income")),
                                            prev.get("revenue", prev.get("net_interest_income")))
        out["earnings_growth_qoq"] = _growth(net_parent, prev.get("net_profit_parent", prev.get("net_profit")))

    f_score, f_max, f_signals = piotroski(cur, cmp_, annualise)
    out.update({"f_score": f_score, "f_score_max": f_max, "f_signals": f_signals})
    out["z_score"] = None if template in FINANCIAL_TEMPLATES else altman_z_em(cur, annualise)
    out["z_zone"] = _z_zone(out["z_score"])
    return _round(out)


def _mul(a: float | None, b: float) -> float | None:
    return None if a is None else a * b


def piotroski(cur: dict[str, float], cmp_: dict[str, float], annualise: float
              ) -> tuple[int | None, int, dict[str, bool]]:
    """Piotroski (2000) F-score over the signals computable from one report's two columns.

    Returns (score, signals available, per-signal result). With fewer than 5 signals
    available the score is None: a "3/3" would read as strong when it means "mostly unknown".
    The comparative balance column is the prior year-end, not the prior-year same date, so
    the delta signals compare against the last audited position.
    """
    signals: dict[str, bool] = {}

    def has(*names: str, src: dict[str, float] = cur) -> bool:
        return all(src.get(n) is not None for n in names)

    if has("net_profit", "total_assets"):
        signals["roa_positive"] = cur["net_profit"] > 0
    if has("cfo"):
        signals["cfo_positive"] = cur["cfo"] > 0
    if has("net_profit", "total_assets") and has("net_profit", "total_assets", src=cmp_):
        signals["roa_improving"] = cur["net_profit"] / cur["total_assets"] > cmp_["net_profit"] / cmp_["total_assets"]
    if has("cfo", "net_profit"):
        signals["accruals"] = cur["cfo"] > cur["net_profit"]
    if has("borrowings", "total_assets") and has("borrowings", "total_assets", src=cmp_):
        signals["leverage_falling"] = cur["borrowings"] / cur["total_assets"] <= cmp_["borrowings"] / cmp_["total_assets"]
    if has("current_assets", "current_liabilities") and has("current_assets", "current_liabilities", src=cmp_):
        signals["liquidity_improving"] = (cur["current_assets"] / cur["current_liabilities"]
                                          > cmp_["current_assets"] / cmp_["current_liabilities"])
    if has("stated_capital") and has("stated_capital", src=cmp_):
        # Small option-plan issues (<1%) are not dilution in Piotroski's sense.
        signals["no_dilution"] = cur["stated_capital"] <= cmp_["stated_capital"] * 1.01
    if has("gross_profit", "revenue") and has("gross_profit", "revenue", src=cmp_):
        signals["margin_improving"] = cur["gross_profit"] / cur["revenue"] > cmp_["gross_profit"] / cmp_["revenue"]
    if has("revenue", "total_assets") and has("revenue", "total_assets", src=cmp_):
        signals["turnover_improving"] = cur["revenue"] / cur["total_assets"] > cmp_["revenue"] / cmp_["total_assets"]

    available = len(signals)
    score = sum(signals.values()) if available >= 5 else None
    return score, available, signals


def altman_z_em(cur: dict[str, float], annualise: float) -> float | None:
    """Z'' = 3.25 + 6.56 X1 + 3.26 X2 + 6.72 X3 + 1.05 X4 (emerging-market constant included).

    X1 working capital / TA, X2 retained earnings / TA, X3 EBIT / TA (annualised),
    X4 book equity / total liabilities.
    """
    ta = cur.get("total_assets")
    needed = ("current_assets", "current_liabilities", "retained_earnings", "total_equity", "total_liabilities")
    if not ta or any(cur.get(n) is None for n in needed):
        return None
    ebit = cur.get("operating_profit")
    if ebit is None and cur.get("profit_before_tax") is not None:
        ebit = cur["profit_before_tax"] + abs(cur.get("finance_cost") or 0.0)
    if ebit is None or not cur["total_liabilities"]:
        return None
    x1 = (cur["current_assets"] - cur["current_liabilities"]) / ta
    x2 = cur["retained_earnings"] / ta
    x3 = ebit * annualise / ta
    x4 = cur["total_equity"] / cur["total_liabilities"]
    return 3.25 + 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4


def _z_zone(z: float | None) -> str | None:
    # Altman's Z''-EM cut-offs: < 1.10 distress, 1.10-2.60 grey, > 2.60 safe (scaled +3.25
    # in the EM form, so 4.35 / 5.85).
    if z is None:
        return None
    if z < 4.35:
        return "distress"
    if z <= 5.85:
        return "grey"
    return "safe"


# Ratios worth comparing against sector peers, and whether higher is better.
PEER_RATIOS: dict[str, bool] = {
    "roe": True, "roa": True, "net_margin": True, "revenue_growth_yoy": True,
    "earnings_growth_yoy": True, "debt_to_equity": False, "current_ratio": True,
    "interest_cover": True, "pe": False, "pb": False, "dividend_yield": True,
    "nim_on_assets": True, "cost_to_income": False, "impairment_to_loans": False,
}

MIN_PEERS = 3


def sector_relative(ratios: dict[str, Any], peers: list[dict[str, Any]]) -> dict[str, float | None]:
    """Percentile (0-1, higher = better for the holder) of each ratio among peers of the same
    template. None when fewer than MIN_PEERS peers report the ratio."""
    out: dict[str, float | None] = {}
    for name, higher_is_better in PEER_RATIOS.items():
        value = ratios.get(name)
        values = [p[name] for p in peers if p.get(name) is not None]
        if value is None or len(values) < MIN_PEERS:
            out[name] = None
            continue
        below = sum(1 for v in values if v < value) + 0.5 * sum(1 for v in values if v == value)
        percentile = below / len(values)
        out[name] = round(percentile if higher_is_better else 1 - percentile, 3)
    return out
