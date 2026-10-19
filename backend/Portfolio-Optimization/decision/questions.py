"""Typed questions for the two-stage Laya decision model.

Stage 1 analyses one holding from three angles, each its own compact state (so each fits
Laya's 512-token budget) and each answer separately auditable. Stage 2 takes the Stage 1
answers plus the request and the holding's position and decides what to do with it.

Question schema is Laya's own (verified against laya 0.4.1, agent._check_question):
  choice  {"type": "choice", "instructions": str, "criteria": {label: description}}
  noul    {"type": "noul",   "instructions": str}            -> P(true)
  score   {"type": "score",  "instructions": str, "criteria": [level0, level1, ...]}
Label sets are fixed: training targets (labels.py) and the engine index into them by name.
"""

from __future__ import annotations

TECHNICAL = {
    "trend": {
        "type": "choice",
        "instructions": "Given this stock's price trend and momentum indicators, over the next month the trend is most likely:",
        "criteria": {"bullish": "rising, outperforming its market", "neutral": "sideways or in line with the market",
                     "bearish": "falling, underperforming its market"},
    },
    "momentum_exhausted": {
        "type": "noul",
        "instructions": "Is the recent move stretched (overbought or oversold) and likely to reverse within two weeks?",
    },
    "near_support": {
        "type": "noul",
        "instructions": "Is the price close to a support level that is likely to hold over the next two weeks?",
    },
}

FUNDAMENTAL = {
    "financial_health": {
        "type": "score",
        "instructions": "Rate the company's financial health from its latest reported statements and ratios:",
        "criteria": ["severe distress", "weak", "adequate", "sound", "very strong"],
    },
    "valuation": {
        "type": "choice",
        "instructions": "Relative to its fundamentals and sector peers, the shares look:",
        "criteria": {"cheap": "undervalued, likely to outperform", "fair": "fairly valued",
                     "expensive": "overvalued, likely to underperform"},
    },
    "earnings_deteriorating": {
        "type": "noul",
        "instructions": "Will the next reported earnings be lower than a year earlier?",
    },
    "distress_risk": {
        "type": "noul",
        "instructions": "Is there a material risk of financial distress (a collapse in value or solvency) within a year?",
    },
}

NEWS = {
    "sentiment_shock": {
        "type": "noul",
        "instructions": "Does the recent news flow amount to a sharp negative shock for this stock?",
    },
    "news_material": {
        "type": "noul",
        "instructions": "Is the recent news material enough to move this stock's price?",
    },
}

ACTIONS = ("sell_now", "spread", "defer", "exclude")

DECISION = {
    "action": {
        "type": "choice",
        "instructions": ("An investor must raise cash from this portfolio by the deadline. For THIS holding, "
                         "considering the analysis, its liquidity and its forecast, the best action is:"),
        "criteria": {"sell_now": "sell early, at the start of the window",
                     "spread": "sell gradually across the window",
                     "defer": "sell late, near the deadline",
                     "exclude": "do not sell it; raise the cash from other holdings"},
    },
    "downside_risk": {
        "type": "noul",
        "instructions": "Will this holding fall by more than its normal range before the deadline?",
    },
}

PORTFOLIO = {
    "urgency": {
        "type": "score",
        "instructions": "How urgently should this withdrawal be executed, given the deadline and market conditions?",
        "criteria": ["no rush", "low", "moderate", "high", "immediate"],
    },
    "risk_tolerance": {
        "type": "score",
        "instructions": "How much price risk can this withdrawal tolerate while waiting for better fills?",
        "criteria": ["none", "low", "moderate", "high", "very high"],
    },
}

STAGE1 = {"technical": TECHNICAL, "fundamental": FUNDAMENTAL, "news": NEWS}
ALL_QUESTIONS = {**TECHNICAL, **FUNDAMENTAL, **NEWS, **DECISION, **PORTFOLIO}
