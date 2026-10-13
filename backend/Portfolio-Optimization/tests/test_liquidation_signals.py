"""Phase 4 tests: forecast/sentiment signals, the GA's new cost terms, the fuzzy signal
layer, currency handling and the enriched /portfolio/withdraw response."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from optimization import fuzzy_withdrawal, ga_withdrawal
from optimization.ga_withdrawal import GAConfig, optimize_withdrawal, risk_aversion_for, simulate_schedule
from optimization.signals import holding_signals, market_context

FAST = GAConfig(population_size=40, n_generations=40, stagnation_patience=15)


def snapshot(method: str = "hybrid", p=(-0.06, -0.02, 0.02), sentiment=-0.6, spike=2.4) -> dict:
    return {
        "status": "ready", "prepared_at": datetime.now(timezone.utc).isoformat(), "age_hours": 2.0,
        "metrics": {
            "forecast": {"h5": {"horizon": 5, "method": method, "p10": p[0], "p50": p[1], "p90": p[2],
                                "model_version": "hybrid_h5-v0.1" if method == "hybrid" else None}},
            "sentiment": {"overall_sentiment": sentiment, "items_relevant": 12,
                          "tiers": {"company": {"mean_sentiment": sentiment, "attention_spike": spike}},
                          "top_sources": []},
            "technical": {"trend": "bearish", "rsi": 28.0},
            "fundamentals": {"kind": "statements", "template": "general",
                             "ratios": {"f_score": 3, "f_score_max": 9, "z_zone": "distress"},
                             "flags": {"going_concern_doubt": True, "unaudited": True},
                             "report": {"age_days": 40, "url": "https://cdn.cse.lk/x.pdf"}},
        },
    }


# --------------------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------------------

def test_hybrid_band_gives_drift_spread_and_outlook() -> None:
    s = holding_signals(snapshot(), deadline_days=3)
    assert s["drift_daily"] == pytest.approx(math.log1p(-0.02) / 5)
    assert s["forecast_sigma_daily"] == pytest.approx((math.log1p(0.02) - math.log1p(-0.06)) / (2 * 1.2816 * math.sqrt(5)))
    assert s["forecast_outlook"] < 0.5                       # bearish
    assert s["news_sentiment"] == pytest.approx(0.2)
    assert s["attention_spike"] == pytest.approx(0.8)


def test_empirical_band_contributes_spread_but_no_drift() -> None:
    s = holding_signals(snapshot(method="empirical"), deadline_days=3)
    assert "forecast_sigma_daily" in s
    assert "drift_daily" not in s and "forecast_outlook" not in s


def test_no_snapshot_no_signals() -> None:
    assert holding_signals(None, 3) == {}
    assert market_context(None, {}, 3) == {"status": "untracked", "signals_used": {}}


def test_context_lists_red_flags_but_not_unaudited() -> None:
    context = market_context(snapshot(), holding_signals(snapshot(), 3), 3)
    assert context["fundamentals"]["red_flags"] == ["going_concern_doubt"]
    assert context["forecast"]["method"] == "hybrid"
    assert context["technical"]["trend"] == "bearish"


# --------------------------------------------------------------------------------------
# GA cost terms
# --------------------------------------------------------------------------------------

def twin(**overrides) -> dict:
    base = {"value": 100_000.0, "price": 100.0, "adv_usd": 400_000.0, "daily_volatility": 0.01,
            "volatility_pct": 0.5}
    return {**base, **overrides}


def test_falling_forecast_is_sold_first_rising_one_deferred() -> None:
    holdings = {"FALL": twin(drift_daily=-0.01), "RISE": twin(drift_daily=0.01)}
    plan = optimize_withdrawal(holdings, 60_000, deadline_days=5, config=FAST)
    first_day = {step.symbol for step in plan.steps if step.execution_day == 0}
    sold = {s: sum(st.quantity for st in plan.steps if st.symbol == s) for s in holdings}
    assert "FALL" in first_day
    assert sold["FALL"] > sold["RISE"]


def test_fx_cost_charged_only_on_foreign_proceeds() -> None:
    order = (np.array([0, 1]), np.array([1.0, 1.0]))
    holdings = {"LKR": twin(), "USD": twin(fx_cost_pct=0.0075)}
    plan = simulate_schedule(order, holdings, target_amount=150_000, deadline_days=5, participation_cap=0.5)
    usd_sold = sum(st.quantity * 100.0 for st in plan.steps if st.symbol == "USD")
    assert plan.fx_cost == pytest.approx(0.0075 * usd_sold)


def test_ga_prefers_selling_winners_over_crystallising_losses() -> None:
    holdings = {"LOSER": twin(cost_basis=130.0), "WINNER": twin(cost_basis=60.0)}
    plan = optimize_withdrawal(holdings, 50_000, deadline_days=3, config=FAST)
    sold = {s: sum(st.quantity for st in plan.steps if st.symbol == s) for s in holdings}
    assert sold["WINNER"] > sold["LOSER"]
    assert plan.realized_pnl > 0


def test_cost_breakdown_sums_to_expected_loss() -> None:
    holdings = {"A": twin(drift_daily=-0.002, fx_cost_pct=0.01, cost_basis=90.0), "B": twin()}
    plan = optimize_withdrawal(holdings, 120_000, deadline_days=4, config=FAST)
    total = plan.slippage_cost + plan.timing_cost + plan.drift_cost + plan.fx_cost
    assert plan.expected_realized_loss == pytest.approx(total)


def test_risk_tolerance_scales_timing_aversion() -> None:
    assert risk_aversion_for(0.5) == pytest.approx(ga_withdrawal.TIMING_RISK_AVERSION)
    assert risk_aversion_for(0.0) > risk_aversion_for(1.0)
    holdings = {"A": twin(adv_usd=50_000.0)}
    cautious = optimize_withdrawal(holdings, 60_000, deadline_days=10, risk_tolerance=0.0, config=FAST)
    relaxed = optimize_withdrawal(holdings, 60_000, deadline_days=10, risk_tolerance=1.0, config=FAST)
    assert cautious.timing_cost / max(cautious.raised_amount, 1) >= 0
    assert cautious.days_required <= relaxed.days_required


# --------------------------------------------------------------------------------------
# Fuzzy signal layer
# --------------------------------------------------------------------------------------

def test_signal_adjustment_direction_and_trace() -> None:
    sell_earlier, rules = fuzzy_withdrawal.compute_signal_adjustment(0.1, 0.1, 0.9)
    hold, _ = fuzzy_withdrawal.compute_signal_adjustment(0.9, 0.9, 0.0)
    neutral, _ = fuzzy_withdrawal.compute_signal_adjustment(None, None, None)
    assert sell_earlier > 15 and hold < -10 and neutral == pytest.approx(0.0)
    assert rules[0].rule_id.startswith("S")


@pytest.mark.parametrize("outlook", [0.2, 0.5, 0.8])
def test_signal_adjustment_monotone_in_sentiment(outlook: float) -> None:
    values = [fuzzy_withdrawal.compute_signal_adjustment(s, outlook, 0.3)[0] for s in np.linspace(0, 1, 11)]
    assert all(a >= b - 1e-9 for a, b in zip(values, values[1:]))


def test_priorities_unchanged_without_signals() -> None:
    plain = fuzzy_withdrawal.compute_portfolio_priorities({"A": twin()}, 0.5)["A"]
    signalled = fuzzy_withdrawal.compute_portfolio_priorities(
        {"A": twin(news_sentiment=0.1, forecast_outlook=0.1, attention_spike=0.9)}, 0.5)["A"]
    assert plain.signal_adjustment == 0.0 and plain.signal_rules == ()
    assert signalled.sell_priority > plain.sell_priority
    trace = fuzzy_withdrawal.rule_trace_to_dict({"A": signalled})[0]
    assert trace["signal_adjustment"] > 0 and trace["signal_rules"]


# --------------------------------------------------------------------------------------
# Endpoint
# --------------------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from data import active_set, fx
    from service import events as events_mod
    from service.api import app
    from service.events import NullProducer

    monkeypatch.setattr(events_mod, "_producer_cache", {"producer": NullProducer()})
    monkeypatch.setattr(active_set, "touch", lambda symbols: None)
    monkeypatch.setattr(active_set, "metrics_for", lambda symbol: None)
    snapshots = {"JKH.N0000": snapshot()}
    monkeypatch.setattr(active_set, "snapshot_for", lambda symbol: snapshots.get(symbol))
    rates = {("USD", "LKR"): 300.0}

    def fake_rate(from_ccy, to_ccy="LKR"):
        value = 1.0 if from_ccy == to_ccy else rates[(fx.normalise_currency(from_ccy), to_ccy)]
        return fx.FxRate(from_ccy, to_ccy, value, datetime.now(timezone.utc), "test")

    monkeypatch.setattr(fx, "get_rate", fake_rate)
    with TestClient(app) as test_client:
        yield test_client


MIXED = [
    {"symbol": "JKH.N0000", "quantity": 20_000, "current_price": 18.4, "avg_daily_volume": 9.5e7,
     "cost_basis": 21.0, "currency": "LKR"},
    {"symbol": "AAPL", "quantity": 100, "current_price": 336.6, "avg_daily_volume": 1.5e10,
     "cost_basis": 150.0, "currency": "USD"},
]


def test_withdraw_converts_currencies_and_reports_context(client) -> None:
    body = client.post("/portfolio/withdraw", json={
        "holdings": MIXED, "target_amount": 5_000_000, "urgency": 0.5, "deadline_days": 3}).json()
    assert body["currency"]["base"] == "LKR"
    assert body["currency"]["rates"]["USD"]["rate"] == 300.0
    # AAPL is worth 100 x 336.6 x 300 = LKR 10.1M, so the LKR 5M target is reachable.
    assert body["feasible"] is True
    assert body["market_context"]["JKH.N0000"]["forecast"]["method"] == "hybrid"
    assert body["market_context"]["AAPL"]["status"] == "untracked"
    breakdown = body["cost_breakdown"]
    assert breakdown["slippage"] + breakdown["timing_risk"] + breakdown["forecast_drift"] + \
        breakdown["fx_conversion"] == pytest.approx(body["expected_realized_loss"], abs=0.05)
    methods = [row["method"] for row in body["baseline_comparison"]]
    assert methods == ["fuzzy_ga", "pro_rata", "largest_first", "most_liquid_first"]
    jkh_trace = next(t for t in body["fuzzy_rule_trace"] if t["symbol"] == "JKH.N0000")
    assert jkh_trace["signal_adjustment"] > 0                 # bearish + negative news + spike


def test_withdraw_target_beyond_base_value_is_422(client) -> None:
    response = client.post("/portfolio/withdraw", json={"holdings": MIXED, "target_amount": 9e9})
    assert response.status_code == 422 and "LKR" in response.json()["detail"]


def test_agent_inputs_are_the_ones_the_plan_used(client) -> None:
    body = client.post("/portfolio/withdraw", json={
        "holdings": MIXED, "target_amount": 2_000_000, "urgency": 0.3, "deadline_days": 1,
        "use_agent": True}).json()
    grounded = [s for s in body["agent_reasoning_trace"] if s["tool"] == "run_fuzzy_ga_withdrawal"]
    assert grounded, body["agent_reasoning_trace"]
    args = grounded[0]["tool_arguments"]
    # Same-day deadline: the scripted agent raises urgency; the plan must use ITS value.
    assert body["effective_inputs"]["urgency"] == pytest.approx(args["urgency"])
    assert args["urgency"] > 0.3


def test_optimize_applies_max_weight(client) -> None:
    holdings = [{**h, "currency": "LKR"} for h in MIXED] + [
        {"symbol": f"X{i}", "quantity": 1000, "current_price": 100.0, "avg_daily_volume": 1e8} for i in range(4)]
    body = client.post("/portfolio/optimize", json={"holdings": holdings, "max_weight": 0.3}).json()
    assert max(body["recommended_weights"].values()) <= 0.3 + 1e-6


def test_currency_is_derived_when_a_base_is_named(client, monkeypatch) -> None:
    from data import active_set

    monkeypatch.setattr(active_set, "get_instrument", lambda symbol: None)
    holdings = [{**h} for h in MIXED]
    for h in holdings:
        h.pop("currency")
    body = client.post("/portfolio/withdraw", json={
        "holdings": holdings, "target_amount": 5_000_000, "base_currency": "LKR", "deadline_days": 3}).json()
    assert body["market_context"]["JKH.N0000"]["currency"] == "LKR"      # CSE shape
    assert body["market_context"]["AAPL"]["currency"] == "USD"           # bare US ticker
    assert set(body["currency"]["rates"]) == {"USD"}


def test_legacy_request_converts_nothing(client) -> None:
    holdings = [{k: v for k, v in h.items() if k != "currency"} for h in MIXED]
    body = client.post("/portfolio/withdraw", json={"holdings": holdings, "target_amount": 30_000}).json()
    assert body["currency"]["rates"] == {}
    assert all(c["currency"] == body["currency"]["base"] for c in body["market_context"].values())


def test_plan_is_currency_scale_invariant() -> None:
    """The same book in USD and in LKR (x300) must get the same schedule. With absolute
    penalty weights the LKR version under-raised, because execution cost (which scales with
    money) outgrew a shortfall penalty that did not."""
    usd = {"A": twin(adv_usd=150_000.0), "B": twin(adv_usd=2_000_000.0), "C": twin(adv_usd=60_000.0)}
    lkr = {s: {**h, "value": h["value"] * 300, "price": h["price"] * 300, "adv_usd": h["adv_usd"] * 300}
           for s, h in usd.items()}
    a = optimize_withdrawal(usd, 150_000, deadline_days=3, config=FAST)
    b = optimize_withdrawal(lkr, 150_000 * 300, deadline_days=3, config=FAST)
    assert a.feasible and b.feasible
    assert [(s.symbol, s.execution_day) for s in a.steps] == [(s.symbol, s.execution_day) for s in b.steps]
    assert b.expected_realized_loss == pytest.approx(a.expected_realized_loss * 300, rel=1e-6)
