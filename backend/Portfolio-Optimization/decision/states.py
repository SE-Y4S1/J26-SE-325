"""Compact Laya states, one per question group.

The SAME builders serve live inference (from the active-set snapshot) and training (from
point-in-time history in labels.py), so the model is trained on exactly the inputs it will
see. Every state is a small dict of rounded numbers and short labels -- well inside Laya's
512-token budget -- and never contains raw text from social posts.
"""

from __future__ import annotations

from typing import Any


def _r(value: Any, digits: int = 3) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    return value


def _pick(source: dict[str, Any] | None, keys: tuple[str, ...], digits: int = 3) -> dict[str, Any]:
    source = source or {}
    return {k: _r(source.get(k), digits) for k in keys if source.get(k) is not None}


def technical_state(technical: dict[str, Any] | None, *, asset_class: str) -> dict[str, Any]:
    return {
        "asset_class": asset_class,
        **_pick(technical, ("trend", "price_vs_sma_50", "price_vs_sma_200", "ma_cross_recent", "rsi",
                            "macd_hist", "mfi", "atr_pct", "bollinger_pct_b", "adx", "obv_trend",
                            "return_20d", "return_63d", "drawdown_52w", "dist_to_support",
                            "dist_to_resistance", "relative_strength_63d")),
    }


def fundamental_state(fundamentals: dict[str, Any] | None, *, asset_class: str) -> dict[str, Any] | None:
    """From a fundamentals block (pipeline step) or a stored record carrying `ratios`.
    None when there is nothing usable -- the group is then not asked."""
    if not fundamentals:
        return None
    if fundamentals.get("kind") == "tokenomics":
        return {"asset_class": asset_class, "kind": "tokenomics",
                **_pick(fundamentals.get("tokenomics"), ("market_cap_rank", "volume_to_market_cap",
                                                         "supply_overhang", "mcap_to_fdv", "drawdown_from_ath"))}
    ratios = fundamentals.get("ratios")
    if not fundamentals.get("usable", True) or not ratios:
        return None
    flags = fundamentals.get("flags") or {}
    state = {
        "asset_class": asset_class,
        "template": ratios.get("template"),
        **_pick(ratios, ("roe", "roa", "net_margin", "revenue_growth_yoy", "earnings_growth_yoy", "debt_to_equity",
                         "interest_cover", "current_ratio", "cfo_to_net_profit", "pe", "pb", "dividend_yield",
                         "nim_on_assets", "cost_to_income", "impairment_to_loans", "z_score", "z_zone")),
        "f_score": f"{ratios['f_score']}/{ratios['f_score_max']}" if ratios.get("f_score") is not None else None,
        "red_flags": [k for k, v in flags.items() if v is True and k != "unaudited"],
    }
    # Sector percentiles are deliberately left out: the historical (training) records have
    # none, and a field present only at inference is an input the model never learned.
    outlook = fundamentals.get("outlook") or {}
    if outlook.get("tone"):
        state["management_outlook"] = outlook["tone"]
    return {k: v for k, v in state.items() if v not in (None, [], {})}


def news_state(sentiment: dict[str, Any] | None) -> dict[str, Any] | None:
    if not sentiment or not sentiment.get("items_relevant"):
        return None
    tiers = sentiment.get("tiers") or {}
    return {
        "overall_sentiment": _r(sentiment.get("overall_sentiment")),
        **{f"{tier}_sentiment": _r((tiers.get(tier) or {}).get("mean_sentiment")) for tier in ("company", "sector", "macro")},
        "company_momentum": _r((tiers.get("company") or {}).get("momentum")),
        "attention_spike": _r((tiers.get("company") or {}).get("attention_spike"), 2),
        "items": sentiment.get("items_relevant"),
        "official_items": sum(1 for s in sentiment.get("top_sources") or [] if s.get("kind") == "official"),
    }


def summarise_stage1(answers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Stage 1 answers -> the compact form Stage 2 reads (label + probability)."""
    out: dict[str, Any] = {}
    for qid, answer in answers.items():
        if "choice" in answer:
            out[qid] = f"{answer['choice']} ({answer.get('confidence', 0):.2f})"
        elif "noul" in answer:
            out[qid] = round(float(answer["noul"]), 2)
        elif "score" in answer:
            out[qid] = round(float(answer["score"]), 2)
    return out


def decision_state(*, stage1: dict[str, Any], forecast: dict[str, Any] | None, position: dict[str, Any],
                   request: dict[str, Any]) -> dict[str, Any]:
    """Stage 2 state for one holding. `position`: share_of_portfolio, days_to_exit (value / (cap x
    ADV)), unrealized_pnl_pct, needs_fx; `request`: deadline_days, urgency, target_share."""
    state = {"analysis": stage1, "position": {k: _r(v) for k, v in position.items()},
             "request": {k: _r(v) for k, v in request.items()}}
    if forecast and forecast.get("p50") is not None:
        state["forecast"] = {"horizon_days": forecast.get("horizon"), "method": forecast.get("method"),
                             **{k: _r(forecast.get(k), 4) for k in ("p10", "p50", "p90")}}
    return state


def portfolio_state(*, holdings: list[dict[str, Any]], request: dict[str, Any]) -> dict[str, Any]:
    """Portfolio-level state for the urgency / risk-tolerance questions."""
    return {"request": {k: _r(v) for k, v in request.items()},
            "holdings": [{k: _r(v) for k, v in h.items()} for h in holdings[:8]]}
