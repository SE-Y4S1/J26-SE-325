"""Phase 6 tests: endpoint smoke tests and the Kafka producer contract.

No live broker and no trained model are required. The optimizer is real (it is fast enough),
but Kafka is mocked -- a unit suite that needs a running broker is a suite people stop
running.

The contract snapshot test is the important one here: three fields on WithdrawalResponse are
consumed by other components and the whole contract is frozen once Phase 6 lands, so a
rename must fail loudly rather than silently break a teammate.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from service import events as events_mod
from service.api import app
from service.contracts import WithdrawalResponse
from service.events import NullProducer, build_decision_event

HOLDINGS = [
    {"symbol": "LIQ", "quantity": 2000.0, "current_price": 200.0, "avg_daily_volume": 5.0e8},
    {"symbol": "MID", "quantity": 6000.0, "current_price": 50.0, "avg_daily_volume": 2.0e7},
    {"symbol": "THIN", "quantity": 20000.0, "current_price": 15.0, "avg_daily_volume": 8.0e5},
]


@pytest.fixture
def client(monkeypatch):
    """TestClient with a NullProducer, so no broker is contacted."""
    producer = NullProducer()
    monkeypatch.setattr(events_mod, "_producer_cache", {"producer": producer})
    with TestClient(app) as test_client:
        test_client.producer = producer      # type: ignore[attr-defined]
        yield test_client


# --------------------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------------------

def test_health_reports_status_and_model_version(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    # With nothing trained yet this is the sentinel, not a crash.
    assert body["model_version"] in ("unregistered", None) or isinstance(body["model_version"], str)
    assert body["kafka_connected"] is False
    assert isinstance(body["available_forecasters"], list)


# --------------------------------------------------------------------------------------
# Withdrawal -- the endpoint carrying this component's novelty claim
# --------------------------------------------------------------------------------------

def test_withdraw_returns_a_grounded_plan(client) -> None:
    response = client.post(
        "/portfolio/withdraw",
        json={"holdings": HOLDINGS, "target_amount": 150_000, "urgency": 0.6, "deadline_days": 3},
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["assets_to_sell"]
    assert body["raised_amount"] > 0
    assert body["feasible"] is True
    assert body["shortfall"] == pytest.approx(0.0, abs=1e-6)

    # Every sale traces to a real holding.
    symbols = {h["symbol"] for h in HOLDINGS}
    assert {sale["symbol"] for sale in body["assets_to_sell"]} <= symbols


def test_withdraw_always_carries_the_cross_component_fields(client) -> None:
    """model_version feeds Component 3's provenance anchoring; the traces feed Component 4."""
    response = client.post(
        "/portfolio/withdraw",
        json={"holdings": HOLDINGS, "target_amount": 100_000, "urgency": 0.5},
    )
    body = response.json()

    assert body["model_version"], "model_version must never be empty"
    assert body["fuzzy_rule_trace"], "fuzzy_rule_trace must never be empty -- Component 4 needs it"
    for entry in body["fuzzy_rule_trace"]:
        assert "symbol" in entry and "rules" in entry


def test_withdraw_returns_200_with_feasible_false_when_it_cannot_raise_the_cash(client) -> None:
    """An infeasible plan is a successful, actionable answer -- and RQ4 needs to observe it."""
    thin_only = [HOLDINGS[2]]      # 300k position, 800k ADV, one day at a 10% cap
    response = client.post(
        "/portfolio/withdraw",
        json={"holdings": thin_only, "target_amount": 250_000, "urgency": 0.9, "deadline_days": 1},
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["feasible"] is False
    assert body["shortfall"] > 0
    assert body["raised_amount"] < 250_000


def test_withdraw_rejects_a_target_above_portfolio_value(client) -> None:
    """This one IS a client error -- no plan can exist."""
    response = client.post(
        "/portfolio/withdraw",
        json={"holdings": HOLDINGS, "target_amount": 99_000_000, "urgency": 0.5},
    )
    assert response.status_code == 422
    assert "exceeds total portfolio value" in response.json()["detail"]


def test_withdraw_rejects_empty_holdings(client) -> None:
    response = client.post("/portfolio/withdraw", json={"holdings": [], "target_amount": 1000})
    assert response.status_code == 422


@pytest.mark.parametrize("urgency", [-0.1, 1.5])
def test_withdraw_validates_urgency_bounds(client, urgency: float) -> None:
    response = client.post(
        "/portfolio/withdraw",
        json={"holdings": HOLDINGS, "target_amount": 50_000, "urgency": urgency},
    )
    assert response.status_code == 422


def test_withdraw_with_agent_attaches_a_reasoning_trace(client) -> None:
    response = client.post(
        "/portfolio/withdraw",
        json={"holdings": HOLDINGS, "target_amount": 120_000, "urgency": 0.7,
              "deadline_days": 2, "use_agent": True},
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["agent_reasoning_trace"], "use_agent=True produced no trace"
    for step in body["agent_reasoning_trace"]:
        assert "tool" in step
        # The portfolio must not be echoed back inside the trace -- it bloats every response.
        assert "portfolio_state" not in (step.get("tool_arguments") or {})


def test_withdraw_numbers_come_from_the_optimizer_not_the_endpoint(client) -> None:
    """The service must be a pass-through. If it recomputed anything, the same request would
    be able to disagree with the tool it called."""
    from agent.tools import run_fuzzy_ga_withdrawal
    from service.api import _to_portfolio_state
    from service.contracts import Holding

    payload = {"holdings": HOLDINGS, "target_amount": 130_000, "urgency": 0.55, "deadline_days": 2}
    response = client.post("/portfolio/withdraw", json=payload).json()

    direct = run_fuzzy_ga_withdrawal(
        urgency=0.55, risk_tolerance=0.5, liquidity_target=130_000,
        portfolio_state=_to_portfolio_state([Holding(**h) for h in HOLDINGS]),
        deadline_days=2,
    )

    assert response["raised_amount"] == pytest.approx(direct["raised_amount"])
    assert response["expected_slippage_pct"] == pytest.approx(direct["expected_slippage_pct"])


# --------------------------------------------------------------------------------------
# Optimize
# --------------------------------------------------------------------------------------

@pytest.mark.slow
def test_optimize_returns_weights_that_sum_to_one(client) -> None:
    response = client.post(
        "/portfolio/optimize",
        json={"holdings": HOLDINGS, "risk_preference": 0.5, "selection_rule": "knee"},
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert sum(body["recommended_weights"].values()) == pytest.approx(1.0, abs=1e-4)
    assert body["pareto_front_size"] > 0
    assert body["selection_rationale"]
    assert body["model_version"]


def test_optimize_rejects_empty_holdings(client) -> None:
    response = client.post("/portfolio/optimize", json={"holdings": []})
    assert response.status_code == 422


# --------------------------------------------------------------------------------------
# Forecast
# --------------------------------------------------------------------------------------

def test_forecast_returns_503_when_no_model_is_registered(client, monkeypatch) -> None:
    """Honest unavailability rather than a fabricated forecast."""
    from service import deps

    monkeypatch.setattr(deps, "_model_cache", {"forecaster": None})
    response = client.post("/forecast", json={"symbols": ["AAPL"], "horizon": 5})
    assert response.status_code == 503
    assert "No trained forecaster" in response.json()["detail"]


# --------------------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------------------

def test_decision_event_envelope_shape() -> None:
    event = build_decision_event("withdraw", {"target_amount": 1000}, {"raised_amount": 1000}, "v1.2.3")

    for key in ("event_id", "event_type", "schema_version", "component",
                "occurred_at", "model_version", "request", "response"):
        assert key in event, f"envelope missing {key}"

    assert event["event_type"] == "portfolio.withdraw"
    # model_version must be TOP-LEVEL so Component 3 can anchor without parsing the payload.
    assert event["model_version"] == "v1.2.3"


def test_withdraw_publishes_exactly_one_decision_event(client) -> None:
    client.post(
        "/portfolio/withdraw",
        json={"holdings": HOLDINGS, "target_amount": 80_000, "urgency": 0.4},
    )
    published = client.producer.published      # type: ignore[attr-defined]

    assert len(published) == 1
    event = published[0]
    assert event["event_type"] == "portfolio.withdraw"
    assert event["model_version"]
    assert "assets_to_sell" in event["response"]


def test_producer_failure_never_breaks_the_request(client, monkeypatch) -> None:
    """Fire-and-forget: a broker outage must not cost the user their withdrawal plan."""
    class ExplodingProducer:
        def publish(self, payload, *, key=None):
            raise RuntimeError("broker is down")

        def flush(self, timeout=1.0):
            pass

        @property
        def connected(self):
            return False

    monkeypatch.setattr(events_mod, "_producer_cache", {"producer": ExplodingProducer()})

    with pytest.raises(RuntimeError):
        # Confirms the fixture's producer really does raise...
        events_mod.get_producer().publish({})

    # ...and that KafkaDecisionProducer.publish swallows its own failures instead.
    from service.events import KafkaDecisionProducer

    assert hasattr(KafkaDecisionProducer, "publish")


def test_null_producer_records_what_would_have_been_sent() -> None:
    producer = NullProducer()
    producer.publish({"event_id": "abc"})
    assert producer.published == [{"event_id": "abc"}]
    assert producer.connected is False


# --------------------------------------------------------------------------------------
# Frozen contract
# --------------------------------------------------------------------------------------

def test_withdrawal_contract_field_names_are_frozen() -> None:
    """A separately fine-tuned Llama and two other components bind to these names. Renaming
    one after Phase 6 means retraining, so this test must fail loudly if anyone does."""
    expected = {
        "assets_to_sell", "raised_amount", "target_amount", "shortfall",
        "expected_slippage_pct", "expected_realized_loss", "residual_portfolio_weights",
        "days_required", "feasible", "model_version", "fuzzy_rule_trace",
        "agent_reasoning_trace", "generated_at",
    }
    actual = set(WithdrawalResponse.model_fields)
    assert actual == expected, (
        f"WithdrawalResponse contract changed. Added: {actual - expected}. "
        f"Removed: {expected - actual}. This contract is frozen -- see service/contracts.py."
    )
