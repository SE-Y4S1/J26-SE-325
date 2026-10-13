"""Training data for the Laya decision model: point-in-time states, outcome-based labels.

Every row is built from information available on its date only (bars up to t, the latest
fundamentals record uploaded on or before t), and labelled from what happened AFTER t.

STAGE 1 LABELS
  trend                  21-session return in excess of the benchmark: bullish > +3%,
                         bearish < -3% (soft edges, so a +2.9% month is not pure "neutral")
  momentum_exhausted     a >8% 20-session move followed by a >2% reversal within 10 sessions
  near_support           within 3% of the 60-session low AND that low held (no close >2% below
                         it) over the next 10 sessions
  valuation              63-session excess return: cheap > +5%, expensive < -5%
  earnings_deteriorating the NEXT report's year-on-year earnings growth is negative
  distress_risk          a >50% drawdown within 252 sessions, or a later report in the
                         distress zone / with a going-concern doubt
  financial_health       SECONDARY (analyst-rule) label from F-score and Z''-zone, trained
                         with a deliberately soft target so it weighs less than outcomes
  News questions have no historical labels (sentiment history starts when a symbol is first
  tracked); the base model answers them zero-shot.

STAGE 2 LABELS (per holding, per sampled withdrawal scenario)
  Each action's realised cost per unit sold, with the same square-root impact model the GA
  uses and the prices that actually followed:
      sell_now   impact(all on day 0)
      spread     impact(1/D per day) + mean over the window of (1 - P_d / P_0)
      defer      impact(all on day D) + (1 - P_D / P_0)
      exclude    the other holdings' mean best cost  -  (P_D / P_0 - 1)    (keeping it earns its
                 own move; someone else covers) -- impossible when the rest cannot cover
  target = softmax(-cost / tau). downside_risk = the realised low breached 1.28 sigma sqrt(D).
  Portfolio urgency / risk tolerance: from whether waiting paid (defer vs sell-now cost).

The labels approximate "the GA under each strategy" per holding with that same cost model,
rather than re-running the full GA for every action combination; this keeps a dataset of
~10^5 rows buildable on Colab. Rows are split by DATE (train before the cutoff, test after).
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decision import questions as Q
from decision.states import decision_state, fundamental_state, portfolio_state, technical_state

logger = logging.getLogger(__name__)

TREND_EDGE, VALUATION_EDGE = 0.03, 0.05
EDGE_SOFTNESS = 0.01
SAMPLE_EVERY = 10
WARMUP = 60
HORIZONS = {"trend": 21, "momentum": 10, "support": 10, "valuation": 63, "distress": 252}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


def three_way(x: float, edge: float, labels: tuple[str, str, str]) -> dict[str, float]:
    """Soft (up, flat, down) distribution for a signed outcome with a +/- edge."""
    up = _sigmoid((x - edge) / EDGE_SOFTNESS)
    down = _sigmoid((-x - edge) / EDGE_SOFTNESS)
    flat = max(0.0, 1.0 - up - down)
    total = up + flat + down
    return {labels[0]: up / total, labels[1]: flat / total, labels[2]: down / total}


def noul(value: bool, certainty: float = 0.9) -> dict[str, float]:
    return {"true": certainty if value else 1 - certainty, "false": 1 - certainty if value else certainty}


def soft_level(level: int, n: int, spread: float = 0.15) -> dict[str, float]:
    probs = [spread if abs(i - level) == 1 else 0.0 for i in range(n)]
    probs[level] = 1.0
    total = sum(probs)
    return {str(i): p / total for i, p in enumerate(probs)}


def softmax_costs(costs: dict[str, float]) -> dict[str, float]:
    finite = {k: v for k, v in costs.items() if math.isfinite(v)}
    if not finite:
        return {k: 1 / len(costs) for k in costs}
    tau = max(0.002, float(np.std(list(finite.values()))))
    best = min(finite.values())
    weights = {k: (math.exp(-(v - best) / tau) if math.isfinite(v) else 0.0) for k, v in costs.items()}
    total = sum(weights.values())
    return {k: w / total for k, w in weights.items()}


# --------------------------------------------------------------------------------------
# Stage 1
# --------------------------------------------------------------------------------------

def _fwd(close: np.ndarray, t: int, h: int) -> float | None:
    return float(close[t + h] / close[t] - 1) if t + h < len(close) else None


def _bench_fwd(benchmark: pd.Series | None, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if benchmark is None or benchmark.empty:
        return 0.0
    a, b = benchmark[benchmark.index <= start], benchmark[benchmark.index <= end]
    return float(b.iloc[-1] / a.iloc[-1] - 1) if len(a) and len(b) else 0.0


def _record_at(records: list[dict[str, Any]], when: pd.Timestamp, *, after: bool = False) -> dict[str, Any] | None:
    stamp = when.tz_localize("UTC") if when.tzinfo is None else when
    usable = sorted((r for r in records if r.get("usable") and r.get("ratios")), key=lambda r: r["available_from"])
    if after:
        return next((r for r in usable if pd.Timestamp(r["available_from"]) > stamp), None)
    before = [r for r in usable if pd.Timestamp(r["available_from"]) <= stamp]
    return before[-1] if before else None


def health_level(ratios: dict[str, Any]) -> int | None:
    f, f_max = ratios.get("f_score"), ratios.get("f_score_max")
    zone = ratios.get("z_zone")
    parts = []
    if f is not None and f_max:
        parts.append((0.6, f / f_max))
    if zone:
        parts.append((0.4, {"distress": 0.0, "grey": 0.5, "safe": 1.0}[zone]))
    if not parts:
        return None
    score = sum(w * v for w, v in parts) / sum(w for w, _ in parts)
    return int(round(score * 4))


def stage1_rows(symbol: str, bars: pd.DataFrame, *, asset_class: str, benchmark: pd.Series | None = None,
                records: list[dict[str, Any]] | None = None, sample_every: int = SAMPLE_EVERY) -> list[dict[str, Any]]:
    """Labelled Stage 1 rows for one symbol: technical rows always, fundamental rows when a
    point-in-time report exists."""
    from features.technical import technical_snapshot

    frame = bars.sort_values("timestamp").reset_index(drop=True)
    close = frame["close"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    stamps = pd.to_datetime(frame["timestamp"])
    records = records or []
    rows: list[dict[str, Any]] = []

    for t in range(WARMUP, len(frame) - HORIZONS["support"], sample_every):
        when = stamps.iloc[t]
        snap = technical_snapshot(frame.iloc[: t + 1], benchmark[benchmark.index <= when] if benchmark is not None else None)
        meta = {"symbol": symbol, "date": str(when.date()), "asset_class": asset_class}

        gold: dict[str, dict[str, float]] = {}
        f21 = _fwd(close, t, HORIZONS["trend"])
        if f21 is not None:
            excess = f21 - _bench_fwd(benchmark, when, stamps.iloc[t + HORIZONS["trend"]])
            gold["trend"] = three_way(excess, TREND_EDGE, ("bullish", "neutral", "bearish"))
        past20, f10 = snap.get("return_20d"), _fwd(close, t, HORIZONS["momentum"])
        if past20 is not None and f10 is not None:
            exhausted = abs(past20) > 0.08 and f10 * past20 < 0 and abs(f10) > 0.02
            gold["momentum_exhausted"] = noul(exhausted)
        if snap.get("support"):
            near = (snap.get("dist_to_support") or 1.0) <= 0.03
            held = low[t + 1 : t + 1 + HORIZONS["support"]].min() >= snap["support"] * 0.98
            gold["near_support"] = noul(near and held)
        if gold:
            rows.append({"state": technical_state(snap, asset_class=asset_class),
                         "questions": {q: Q.TECHNICAL[q] for q in gold}, "gold": {q: {"probabilities": p} for q, p in gold.items()},
                         "meta": {**meta, "group": "technical"}})

        record = _record_at(records, when)
        state = fundamental_state(record, asset_class=asset_class) if record else None
        if not state:
            continue
        gold = {}
        f63 = _fwd(close, t, HORIZONS["valuation"])
        if f63 is not None:
            excess = f63 - _bench_fwd(benchmark, when, stamps.iloc[t + HORIZONS["valuation"]])
            gold["valuation"] = three_way(excess, VALUATION_EDGE, ("cheap", "fair", "expensive"))
        following = _record_at(records, when, after=True)
        growth = (following or {}).get("ratios", {}).get("earnings_growth_yoy")
        if growth is not None:
            gold["earnings_deteriorating"] = noul(growth < 0)
        if t + HORIZONS["distress"] < len(close):
            window = close[t : t + HORIZONS["distress"] + 1]
            later = [r for r in records if r.get("usable") and pd.Timestamp(r["available_from"]) > (
                when.tz_localize("UTC") if when.tzinfo is None else when)][:4]
            distressed = (window.min() / close[t] - 1 < -0.5) or any(
                (r.get("ratios") or {}).get("z_zone") == "distress" or (r.get("flags") or {}).get("going_concern_doubt")
                for r in later)
            gold["distress_risk"] = noul(distressed)
        level = health_level(record["ratios"])
        if level is not None:
            gold["financial_health"] = soft_level(level, 5, spread=0.35)   # secondary: softer target
        if gold:
            rows.append({"state": state, "questions": {q: Q.FUNDAMENTAL[q] for q in gold},
                         "gold": {q: {"probabilities": p} for q, p in gold.items()},
                         "meta": {**meta, "group": "fundamental"}})
    return rows


# --------------------------------------------------------------------------------------
# Stage 2
# --------------------------------------------------------------------------------------

def action_costs(path: np.ndarray, *, value: float, adv: float, sigma: float, deadline: int,
                 others_best: float | None, participation_cap: float = 0.10) -> dict[str, float]:
    """Realised cost per unit notional of each action. `path` = closes from t to t+deadline."""
    from optimization.objectives import IMPACT_COEFFICIENT, IMPACT_EXPONENT

    def impact(notional_per_day: float) -> float:
        participation = min(notional_per_day / max(adv, 1e-9), 1.0)
        return IMPACT_COEFFICIENT * sigma * participation ** IMPACT_EXPONENT

    p0 = path[0]
    end = path[min(deadline, len(path) - 1)]
    window = path[: max(1, deadline)]
    costs = {
        "sell_now": impact(value),
        "spread": impact(value / max(deadline, 1)) + float(np.mean(1 - window / p0)),
        "defer": impact(value) + float(1 - end / p0),
        "exclude": math.inf if others_best is None else others_best - float(end / p0 - 1),
    }
    # A position that cannot be exited inside the deadline at the cap is infeasible to defer.
    if value > participation_cap * adv * max(deadline, 1):
        costs["defer"] = math.inf
    return costs


def scenario_rows(
    panel: dict[str, pd.DataFrame], *, n_scenarios: int, rng: random.Random,
    stage1_summary: Callable[[str, pd.Timestamp], dict[str, Any]],
    asset_classes: dict[str, str],
) -> list[dict[str, Any]]:
    """Sampled withdrawal scenarios -> decision rows (per holding) + portfolio rows."""
    rows: list[dict[str, Any]] = []
    symbols = [s for s, f in panel.items() if len(f) > WARMUP + 25]
    if len(symbols) < 2:
        return rows
    for _ in range(n_scenarios):
        chosen = rng.sample(symbols, k=min(len(symbols), rng.randint(2, 5)))
        deadline = rng.choice([1, 3, 5, 10, 21])
        dates = sorted(set.intersection(*[set(pd.to_datetime(panel[s]["timestamp"])[WARMUP:-deadline - 1])
                                          for s in chosen]))
        if not dates:
            continue
        when = rng.choice(dates)
        holdings = {}
        for symbol in chosen:
            frame = panel[symbol].sort_values("timestamp").reset_index(drop=True)
            t = int(np.searchsorted(pd.to_datetime(frame["timestamp"]).to_numpy(), np.datetime64(when)))
            close = frame["close"].to_numpy(dtype=float)
            returns = np.diff(np.log(close[max(0, t - 60) : t + 1]))
            adv = float((frame["close"] * frame["volume"]).iloc[max(0, t - 20) : t + 1].mean() or 0.0)
            holdings[symbol] = {"price": close[t], "path": close[t : t + deadline + 1],
                                "sigma": float(returns.std()) if len(returns) > 5 else 0.02,
                                "adv": adv if adv > 0 else close[t] * 1e5,
                                "value": rng.uniform(0.05, 2.0) * (adv if adv > 0 else close[t] * 1e5),
                                "basis": close[t] * rng.uniform(0.7, 1.3)}
        total = sum(h["value"] for h in holdings.values())
        target_share = rng.uniform(0.1, 0.6)
        request = {"deadline_days": deadline, "urgency": round(rng.uniform(0, 1), 2), "target_share": target_share}

        provisional = {s: action_costs(h["path"], value=h["value"] * target_share, adv=h["adv"], sigma=h["sigma"],
                                       deadline=deadline, others_best=None) for s, h in holdings.items()}
        best = {s: min(c.values()) for s, c in provisional.items()}
        waited = []
        for symbol, h in holdings.items():
            others = [best[o] for o in holdings if o != symbol]
            can_cover = total - h["value"] >= target_share * total * 1.05
            costs = action_costs(h["path"], value=h["value"] * target_share, adv=h["adv"], sigma=h["sigma"],
                                 deadline=deadline, others_best=float(np.mean(others)) if others and can_cover else None)
            breached = float(h["path"].min() / h["price"] - 1) < -1.2816 * h["sigma"] * math.sqrt(max(deadline, 1))
            position = {"share_of_portfolio": h["value"] / total, "days_to_exit": h["value"] / (0.1 * h["adv"]),
                        "unrealized_pnl_pct": h["price"] / h["basis"] - 1, "needs_fx": False}
            if math.isfinite(costs["defer"]):
                waited.append(costs["defer"] - costs["sell_now"])
            rows.append({
                "state": decision_state(stage1=stage1_summary(symbol, when), forecast=None,
                                        position=position, request=request),
                "questions": Q.DECISION,
                "gold": {"action": {"probabilities": softmax_costs(costs)},
                         "downside_risk": {"probabilities": noul(breached)}},
                "meta": {"symbol": symbol, "date": str(pd.Timestamp(when).date()), "group": "decision",
                         "asset_class": asset_classes.get(symbol, "equity")},
            })
        if waited:
            # Waiting cost more than selling now (positive) -> urgent, low tolerance.
            level = int(np.clip(round(2 + float(np.mean(waited)) / 0.01), 0, 4))
            summary = [{"symbol": s, "share_of_portfolio": h["value"] / total,
                        "days_to_exit": h["value"] / (0.1 * h["adv"])} for s, h in holdings.items()]
            rows.append({
                "state": portfolio_state(holdings=summary, request=request),
                "questions": Q.PORTFOLIO,
                "gold": {"urgency": {"probabilities": soft_level(level, 5)},
                         "risk_tolerance": {"probabilities": soft_level(4 - level, 5)}},
                "meta": {"date": str(pd.Timestamp(when).date()), "group": "portfolio"},
            })
    return rows


# --------------------------------------------------------------------------------------
# Dataset files
# --------------------------------------------------------------------------------------

def split_by_date(rows: list[dict[str, Any]], test_fraction: float = 0.2) -> tuple[list, list, str]:
    dates = sorted({r["meta"]["date"] for r in rows})
    if not dates:
        return [], [], ""
    cutoff = dates[int(len(dates) * (1 - test_fraction))]
    return ([r for r in rows if r["meta"]["date"] < cutoff], [r for r in rows if r["meta"]["date"] >= cutoff], cutoff)


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """laya.train reads {state, questions, gold}; `meta` rides along for evaluation slicing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def evaluate(answer_rows: list[tuple[dict[str, Any], dict[str, dict]]]) -> pd.DataFrame:
    """Per question: accuracy of the argmax against the gold argmax, Brier score and mean
    confidence. `answer_rows` = [(row, answers-from-the-model), ...]."""
    stats: dict[str, dict[str, list[float]]] = {}
    for row, answers in answer_rows:
        for qid, gold in row["gold"].items():
            answer = answers.get(qid)
            if not answer:
                continue
            target = gold["probabilities"]
            if "probabilities" in answer and Q.ALL_QUESTIONS[qid]["type"] == "choice":
                predicted = answer["probabilities"]
            elif "noul" in answer:
                predicted = {"true": float(answer["noul"]), "false": 1 - float(answer["noul"])}
            else:
                predicted = {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
            keys = list(target)
            brier = sum((predicted.get(k, 0.0) - target[k]) ** 2 for k in keys)
            hit = max(keys, key=lambda k: predicted.get(k, 0.0)) == max(keys, key=lambda k: target[k])
            s = stats.setdefault(qid, {"accuracy": [], "brier": [], "confidence": []})
            s["accuracy"].append(float(hit))
            s["brier"].append(brier)
            s["confidence"].append(float(answer.get("confidence", 0.0)))
    return pd.DataFrame([{"question": q, "n": len(v["accuracy"]), **{k: float(np.mean(x)) for k, x in v.items()}}
                         for q, v in sorted(stats.items())])
