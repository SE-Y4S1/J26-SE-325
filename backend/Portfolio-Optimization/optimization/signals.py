"""Per-holding market signals for the liquidation planner, from the active-set snapshot.

The withdrawal request never waits on a model or an exchange: everything here is read from
what the background preparation already stored (data/active_set.py). A holding that is not
tracked, or still `preparing`, simply gets no signals, and the planner treats it exactly as
before (historical volatility, fuzzy liquidity/volatility rules only). The response says so.

FORECAST -> GA / FUZZY
The band whose horizon best matches the deadline is used (h5 up to 10 days, else h21):
    drift_daily           = log(1 + p50) / h
    forecast_sigma_daily  = (log(1 + p90) - log(1 + p10)) / (2 x 1.2816 x sqrt(h))
                            (p10-p90 of a normal spans +-1.2816 sigma)
    forecast_outlook      = 0.5 + 0.5 x tanh(p50 / half-band)   in [0, 1], 1 = bullish
Drift and outlook are taken only from the HYBRID model. An empirical band (the symbol's own
past returns, used when no head can run) contributes its spread to timing risk but no drift:
"it went up last year" is not a forecast, and steering a sale on it would be momentum
trading presented as a model.
"""

from __future__ import annotations

import math
from typing import Any

Z_P90 = 1.2816


def _band(forecast: dict[str, Any] | None, deadline_days: int) -> dict[str, Any] | None:
    if not forecast:
        return None
    preferred = "h5" if deadline_days <= 10 else "h21"
    for key in (preferred, "h21" if preferred == "h5" else "h5"):
        band = forecast.get(key)
        if band and band.get("method") in ("hybrid", "empirical") and band.get("p50") is not None:
            return band
    return None


def holding_signals(snapshot: dict[str, Any] | None, deadline_days: int) -> dict[str, float]:
    """Numeric fields merged into the GA/fuzzy holding dict. Empty when nothing is prepared."""
    if not snapshot:
        return {}
    metrics = snapshot.get("metrics") or {}
    out: dict[str, float] = {}

    band = _band(metrics.get("forecast"), deadline_days)
    if band:
        h = int(band["horizon"])
        p10, p50, p90 = (float(band[k]) for k in ("p10", "p50", "p90"))
        if p10 > -1 and p90 > -1:
            out["forecast_sigma_daily"] = max(0.0, (math.log1p(p90) - math.log1p(p10)) / (2 * Z_P90 * math.sqrt(h)))
        if band["method"] == "hybrid" and p50 > -1:
            out["drift_daily"] = math.log1p(p50) / h
            half_band = max((p90 - p10) / 2, 1e-6)
            out["forecast_outlook"] = 0.5 + 0.5 * math.tanh(p50 / half_band)

    sentiment = metrics.get("sentiment") or {}
    if sentiment.get("overall_sentiment") is not None:
        out["news_sentiment"] = (float(sentiment["overall_sentiment"]) + 1) / 2
    company = (sentiment.get("tiers") or {}).get("company") or {}
    if company.get("attention_spike") is not None:
        # The aggregate's spike is a z-like score; 3 = an extreme burst.
        out["attention_spike"] = min(1.0, max(0.0, float(company["attention_spike"]) / 3))
    return out


def market_context(snapshot: dict[str, Any] | None, signals: dict[str, float], deadline_days: int) -> dict[str, Any]:
    """Compact, display-ready context for one holding (WithdrawalResponse.market_context)."""
    if not snapshot:
        return {"status": "untracked", "signals_used": {}}
    metrics = snapshot.get("metrics") or {}
    context: dict[str, Any] = {
        "status": snapshot.get("status"),
        "prepared_at": snapshot.get("prepared_at"),
        "data_age_hours": round(snapshot["age_hours"], 1) if snapshot.get("age_hours") is not None else None,
        "signals_used": {k: round(v, 6) for k, v in signals.items()},
    }
    band = _band(metrics.get("forecast"), deadline_days)
    if band:
        context["forecast"] = {k: band.get(k) for k in ("horizon", "method", "p10", "p50", "p90", "model_version", "reason")}

    sentiment = metrics.get("sentiment")
    if sentiment:
        tiers = sentiment.get("tiers") or {}
        context["sentiment"] = {
            "overall": sentiment.get("overall_sentiment"),
            **{tier: (tiers.get(tier) or {}).get("mean_sentiment") for tier in ("company", "sector", "macro")},
            "attention_spike": (tiers.get("company") or {}).get("attention_spike"),
            "items": sentiment.get("items_relevant"),
            "top_sources": [
                {k: s.get(k) for k in ("channel", "title", "url", "polarity", "published_at")}
                for s in (sentiment.get("top_sources") or [])[:3]
            ],
        }

    technical = metrics.get("technical")
    if technical:
        context["technical"] = {k: technical.get(k) for k in (
            "trend", "rsi", "adx", "return_20d", "relative_strength_63d", "drawdown_52w",
            "dist_to_support", "ma_cross_recent")}

    fundamentals = metrics.get("fundamentals")
    if fundamentals:
        if fundamentals.get("kind") == "tokenomics":
            t = fundamentals.get("tokenomics") or {}
            context["fundamentals"] = {"kind": "tokenomics", **{k: t.get(k) for k in (
                "market_cap_rank", "volume_to_market_cap", "supply_overhang", "mcap_to_fdv", "drawdown_from_ath")}}
        else:
            r = fundamentals.get("ratios") or {}
            flags = fundamentals.get("flags") or {}
            context["fundamentals"] = {
                "kind": "statements", "template": fundamentals.get("template"),
                **{k: r.get(k) for k in ("roe", "pe", "pb", "debt_to_equity", "earnings_growth_yoy",
                                         "f_score", "f_score_max", "z_score", "z_zone")},
                "red_flags": [k for k, v in flags.items() if v is True and k != "unaudited"],
                "outlook_tone": (fundamentals.get("outlook") or {}).get("tone"),
                "report_age_days": (fundamentals.get("report") or {}).get("age_days"),
                "report_url": (fundamentals.get("report") or {}).get("url"),
            }
    return context
