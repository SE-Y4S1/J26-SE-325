"""Execution pipeline tests: Component 1 plan -> Component 2 -> Component 3 -> holdings.

Components 2 and 3 are stubbed at the HTTP boundary (service.pipeline's call functions), so
these tests pin the platform's own behaviour: which orders a plan becomes, what is sent
downstream, and above all that holdings change only when screening and anchoring both clear.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from service import pipeline

HOLDINGS = [
    {"symbol": "SPY", "quantity": 100, "current_price": 500.0, "avg_daily_volume": 4.0e10},
    {"symbol": "THIN", "quantity": 1000, "current_price": 10.0, "avg_daily_volume": 8.0e5},
]
PASSWORD = "a-good-long-passphrase"

WITHDRAW_PLAN = {
    "assets_to_sell": [
        {"symbol": "SPY", "sell_fraction": 0.2, "quantity": 20, "expected_price": 499.0},
        {"symbol": "THIN", "sell_fraction": 0.0, "quantity": 0, "expected_price": 10.0},
    ],
    "raised_amount": 9980.0,
    "target_amount": 10000.0,
    "model_version": "c1-test-v1",
    "fuzzy_rule_trace": [{"rule": "R3", "strength": 0.8}],
    "currency": {"base": "USD"},
}


def _engine(decision: str, score: float = 0.2, benchmark: bool = False) -> dict:
    return {
        "transaction_id": "set-by-stub",
        "fraud_score": score,
        "decision": decision,
        "model_version": "heuristic-untrained",
        "benchmark_model": benchmark,
        "explanation": {"rules": [], "streams": {"heuristic": score}, "top_factors": [], "notes": []},
    }


def _audit(status: str, final: str) -> dict:
    return {
        "success": True,
        "status": status,
        "policyResult": {"action": final},
        "record": {"finalDecision": final, "recordHash": "0x" + "cd" * 32},
        "blockchain": {"success": True, "isMock": True},
    }


class Stubs:
    """Records every downstream request and answers with what the test configured."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.fraud_requests: list[dict] = []
        self.audit_requests: list[dict] = []
        self.reviews: list[tuple[str, bool]] = []
        self.fraud = pipeline.Downstream(True, 200, _engine("ALLOW"))
        self.audit = pipeline.Downstream(True, 200, _audit("COMPLETED", "APPROVE"))
        self.review = pipeline.Downstream(True, 200, {"success": True, "record": {"finalDecision": "HUMAN_APPROVED"}})
        self.retry = pipeline.Downstream(True, 200, {"success": True, "record": {"finalDecision": "APPROVE"}})

        def score(payload: dict) -> pipeline.Downstream:
            self.fraud_requests.append(payload)
            if self.fraud.body:
                self.fraud.body["transaction_id"] = payload["transaction_id"]
            return self.fraud

        def anchor(payload: dict) -> pipeline.Downstream:
            self.audit_requests.append(payload)
            return self.audit

        def review(transaction_id: str, approve: bool) -> pipeline.Downstream:
            self.reviews.append((transaction_id, approve))
            return self.review

        monkeypatch.setattr(pipeline, "score_transaction", score)
        monkeypatch.setattr(pipeline, "anchor_decision", anchor)
        monkeypatch.setattr(pipeline, "submit_review", review)
        monkeypatch.setattr(pipeline, "retry_anchor", lambda _id: self.retry)


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> Stubs:
    return Stubs(monkeypatch)


@pytest.fixture
def owner(client: TestClient, registered):
    headers, _ = registered("owner@example.com", PASSWORD)
    portfolio = client.post("/portfolios", headers=headers,
                            json={"name": "Main", "holdings": HOLDINGS}).json()
    return headers, portfolio["id"]


def _execute(client: TestClient, headers: dict, portfolio_id: int, kind="withdrawal", plan=None):
    return client.post(f"/portfolios/{portfolio_id}/executions", headers=headers,
                       json={"kind": kind, "plan": WITHDRAW_PLAN if plan is None else plan})


def _quantity(client: TestClient, headers: dict, portfolio_id: int, symbol: str) -> float | None:
    holdings = client.get(f"/portfolios/{portfolio_id}", headers=headers).json()["holdings"]
    return next((h["quantity"] for h in holdings if h["symbol"] == symbol), None)


# --------------------------------------------------------------------------------------
# The happy path and what crosses each boundary
# --------------------------------------------------------------------------------------

def test_allow_and_anchored_approve_executes_the_plan(client, owner, stubs) -> None:
    headers, pid = owner
    response = _execute(client, headers, pid)

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "EXECUTED"
    assert body["orders"] == [{"symbol": "SPY", "side": "SELL", "quantity": 20.0, "price": 499.0}]
    assert body["amount"] == 9980.0
    assert body["plan"]["fuzzy_rule_trace"] == WITHDRAW_PLAN["fuzzy_rule_trace"]
    assert _quantity(client, headers, pid, "SPY") == 80


def test_component2_receives_a_pseudonymous_transaction(client, owner, stubs) -> None:
    headers, pid = owner
    execution_id = _execute(client, headers, pid).json()["id"]

    sent = stubs.fraud_requests[0]
    assert sent["transaction_id"] == execution_id
    assert sent["amount"] == 9980.0 and sent["currency"] == "USD" and sent["product_code"] == "W"
    assert sent["account_key"].startswith("acct-")
    assert "owner" not in str(sent) and "@" not in str(sent)


def test_component3_anchors_component2s_decision_and_the_plan_hash(client, owner, stubs) -> None:
    headers, pid = owner
    body = _execute(client, headers, pid).json()

    sent = stubs.audit_requests[0]
    assert sent["transactionId"] == body["id"]
    assert sent["upstreamDecision"] == {"source": "component2", "decision": "ALLOW",
                                        "modelVersion": "heuristic-untrained"}
    assert sent["planHash"] == body["plan_hash"]
    assert sent["modelVersion"] == "c1:c1-test-v1|c2:heuristic-untrained"
    assert sent["riskScore"] == 20.0
    assert sent["confidence"] == 50          # heuristic, not a benchmark model


def test_plan_hash_is_deterministic() -> None:
    orders = [{"symbol": "SPY", "side": "SELL", "quantity": 1.0, "price": 2.0}]
    first = pipeline.plan_hash("withdrawal", 1, {"b": 1, "a": 2}, orders)
    assert first == pipeline.plan_hash("withdrawal", 1, {"a": 2, "b": 1}, orders)
    assert first != pipeline.plan_hash("withdrawal", 2, {"a": 2, "b": 1}, orders)
    assert first.startswith("0x") and len(first) == 66


# --------------------------------------------------------------------------------------
# Enforcement: nothing moves unless both stages clear
# --------------------------------------------------------------------------------------

def test_block_is_enforced_and_holdings_do_not_move(client, owner, stubs) -> None:
    headers, pid = owner
    stubs.fraud = pipeline.Downstream(True, 200, _engine("BLOCK", 0.95))
    stubs.audit = pipeline.Downstream(True, 200, _audit("COMPLETED", "REJECT"))

    body = _execute(client, headers, pid).json()

    assert body["status"] == "BLOCKED"
    assert _quantity(client, headers, pid, "SPY") == 100


def test_block_is_enforced_even_when_component3_is_down(client, owner, stubs) -> None:
    headers, pid = owner
    stubs.fraud = pipeline.Downstream(True, 200, _engine("BLOCK", 0.95))
    stubs.audit = pipeline.Downstream(False, None, None, "connection refused")

    body = _execute(client, headers, pid).json()

    assert body["status"] == "BLOCKED"
    assert "could not yet be anchored" in body["status_reason"]


def test_component2_down_holds_and_skips_component3(client, owner, stubs) -> None:
    headers, pid = owner
    stubs.fraud = pipeline.Downstream(False, None, None, "connection refused")

    body = _execute(client, headers, pid).json()

    assert body["status"] == "HELD"
    assert stubs.audit_requests == []
    assert _quantity(client, headers, pid, "SPY") == 100


def test_allow_without_an_anchor_is_held(client, owner, stubs) -> None:
    headers, pid = owner
    stubs.audit = pipeline.Downstream(True, 503, {"status": "BLOCKCHAIN_UNAVAILABLE"}, "HTTP 503")

    body = _execute(client, headers, pid).json()

    assert body["status"] == "HELD"
    assert _quantity(client, headers, pid, "SPY") == 100


def test_component3_policy_rejection_blocks(client, owner, stubs) -> None:
    headers, pid = owner
    stubs.audit = pipeline.Downstream(True, 200, _audit("COMPLETED", "REJECT"))

    assert _execute(client, headers, pid).json()["status"] == "BLOCKED"


# --------------------------------------------------------------------------------------
# Step-up and retry
# --------------------------------------------------------------------------------------

def _pending(client, owner, stubs):
    headers, pid = owner
    stubs.fraud = pipeline.Downstream(True, 200, _engine("STEP_UP", 0.6))
    stubs.audit = pipeline.Downstream(True, 200, _audit("HUMAN_REVIEW_REQUIRED", "PENDING_HUMAN_REVIEW"))
    body = _execute(client, headers, pid).json()
    assert body["status"] == "PENDING_REVIEW"
    assert "step-up" in body["status_reason"]
    return headers, pid, body["id"]


def test_step_up_waits_then_executes_after_reauthentication(client, owner, stubs) -> None:
    headers, pid, execution_id = _pending(client, owner, stubs)
    assert _quantity(client, headers, pid, "SPY") == 100

    confirmed = client.post(f"/executions/{execution_id}/confirm", headers=headers,
                            json={"password": PASSWORD, "approve": True})

    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["status"] == "EXECUTED"
    assert stubs.reviews == [(execution_id, True)]
    assert _quantity(client, headers, pid, "SPY") == 80


def test_step_up_with_wrong_password_changes_nothing(client, owner, stubs) -> None:
    headers, pid, execution_id = _pending(client, owner, stubs)

    response = client.post(f"/executions/{execution_id}/confirm", headers=headers,
                           json={"password": "not-the-password", "approve": True})

    assert response.status_code == 400
    assert stubs.reviews == []
    assert client.get(f"/executions/{execution_id}", headers=headers).json()["status"] == "PENDING_REVIEW"


def test_step_up_rejection_is_recorded(client, owner, stubs) -> None:
    headers, pid, execution_id = _pending(client, owner, stubs)

    body = client.post(f"/executions/{execution_id}/confirm", headers=headers,
                       json={"password": PASSWORD, "approve": False}).json()

    assert body["status"] == "REJECTED"
    assert _quantity(client, headers, pid, "SPY") == 100


def test_review_not_anchored_leaves_execution_pending(client, owner, stubs) -> None:
    headers, _pid, execution_id = _pending(client, owner, stubs)
    stubs.review = pipeline.Downstream(False, None, None, "down")

    response = client.post(f"/executions/{execution_id}/confirm", headers=headers,
                           json={"password": PASSWORD, "approve": True})

    assert response.status_code == 503
    assert client.get(f"/executions/{execution_id}", headers=headers).json()["status"] == "PENDING_REVIEW"


def test_blocked_execution_cannot_be_confirmed(client, owner, stubs) -> None:
    headers, pid = owner
    stubs.fraud = pipeline.Downstream(True, 200, _engine("BLOCK", 0.95))
    execution_id = _execute(client, headers, pid).json()["id"]

    response = client.post(f"/executions/{execution_id}/confirm", headers=headers,
                           json={"password": PASSWORD, "approve": True})

    assert response.status_code == 409


def test_held_execution_retries_without_rescoring(client, owner, stubs) -> None:
    headers, pid = owner
    stubs.audit = pipeline.Downstream(False, None, None, "down")
    execution_id = _execute(client, headers, pid).json()["id"]

    stubs.audit = pipeline.Downstream(True, 200, _audit("COMPLETED", "APPROVE"))
    retried = client.post(f"/executions/{execution_id}/retry", headers=headers).json()

    assert retried["status"] == "EXECUTED"
    assert len(stubs.fraud_requests) == 1
    assert _quantity(client, headers, pid, "SPY") == 80


def test_retry_publishes_a_record_component3_already_holds(client, owner, stubs) -> None:
    headers, pid = owner
    stubs.audit = pipeline.Downstream(True, 503, {"status": "BLOCKCHAIN_UNAVAILABLE"}, "HTTP 503")
    execution_id = _execute(client, headers, pid).json()["id"]

    stubs.audit = pipeline.Downstream(True, 409, {"success": False}, "HTTP 409")
    retried = client.post(f"/executions/{execution_id}/retry", headers=headers).json()

    assert retried["status"] == "EXECUTED"


def test_holdings_changed_while_pending_rejects_instead_of_overselling(client, owner, stubs) -> None:
    headers, pid, execution_id = _pending(client, owner, stubs)
    client.put(f"/portfolios/{pid}", headers=headers, json={"holdings": [
        {**HOLDINGS[0], "quantity": 5}, HOLDINGS[1]]})

    body = client.post(f"/executions/{execution_id}/confirm", headers=headers,
                       json={"password": PASSWORD, "approve": True}).json()

    assert body["status"] == "REJECTED"
    assert _quantity(client, headers, pid, "SPY") == 5


# --------------------------------------------------------------------------------------
# Plan validation and isolation
# --------------------------------------------------------------------------------------

def test_selling_more_than_held_is_rejected_before_any_downstream_call(client, owner, stubs) -> None:
    headers, pid = owner
    plan = {**WITHDRAW_PLAN, "assets_to_sell": [
        {"symbol": "SPY", "sell_fraction": 1.0, "quantity": 101, "expected_price": 500.0}]}

    response = _execute(client, headers, pid, plan=plan)

    assert response.status_code == 422
    assert stubs.fraud_requests == []


def test_rebalance_executes_component1s_orders(client, owner, stubs) -> None:
    headers, pid = owner
    plan = {
        "recommended_weights": {"SPY": 0.5, "THIN": 0.5},
        "rebalance_orders": [
            {"symbol": "SPY", "side": "SELL", "quantity": 40, "price": 500.0},
            {"symbol": "THIN", "side": "BUY", "quantity": 2000, "price": 10.0},
        ],
        "currency": {"base": "USD", "rates": {}},
        "model_version": "c1-test-v1",
    }

    body = _execute(client, headers, pid, kind="rebalance", plan=plan).json()

    assert body["status"] == "EXECUTED"
    assert body["amount"] == pytest.approx(40000)
    assert stubs.fraud_requests[0]["product_code"] == "R"
    assert _quantity(client, headers, pid, "SPY") == pytest.approx(60)
    assert _quantity(client, headers, pid, "THIN") == pytest.approx(3000)


def test_rebalance_without_component1_orders_is_rejected(client, owner, stubs) -> None:
    """Weights alone are never turned into trades here: they are fractions of a converted
    total this service cannot reproduce."""
    headers, pid = owner
    plan = {"recommended_weights": {"SPY": 0.5, "THIN": 0.5}}

    assert _execute(client, headers, pid, kind="rebalance", plan=plan).status_code == 422
    assert stubs.fraud_requests == []


def test_rebalance_into_an_unheld_symbol_is_rejected(client, owner, stubs) -> None:
    headers, pid = owner
    plan = {"rebalance_orders": [{"symbol": "NEW", "side": "BUY", "quantity": 1, "price": 1.0}]}

    assert _execute(client, headers, pid, kind="rebalance", plan=plan).status_code == 422


def test_selling_a_whole_position_removes_it(client, owner, stubs) -> None:
    headers, pid = owner
    plan = {**WITHDRAW_PLAN, "assets_to_sell": [
        {"symbol": "THIN", "sell_fraction": 1.0, "quantity": 1000, "expected_price": 10.0}],
        "raised_amount": 10000.0}

    assert _execute(client, headers, pid, plan=plan).json()["status"] == "EXECUTED"
    assert _quantity(client, headers, pid, "THIN") is None


def test_one_user_cannot_see_or_act_on_anothers_execution(client, registered, owner, stubs) -> None:
    headers, pid = owner
    stubs.fraud = pipeline.Downstream(True, 200, _engine("STEP_UP", 0.6))
    stubs.audit = pipeline.Downstream(True, 200, _audit("HUMAN_REVIEW_REQUIRED", "PENDING_HUMAN_REVIEW"))
    execution_id = _execute(client, headers, pid).json()["id"]
    other, _ = registered("other@example.com", PASSWORD)

    assert client.get(f"/executions/{execution_id}", headers=other).status_code == 404
    assert client.post(f"/executions/{execution_id}/confirm", headers=other,
                       json={"password": PASSWORD, "approve": True}).status_code == 404
    assert client.get("/executions", headers=other).json() == []
    assert _execute(client, other, pid).status_code == 404


def test_executions_require_authentication(client) -> None:
    assert client.get("/executions").status_code == 401
    assert client.post("/portfolios/1/executions", json={"kind": "withdrawal", "plan": {}}).status_code == 401
