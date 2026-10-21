"""Component 3's assurance engine, seen from the platform.

What the platform sends (signals taken only from Component 2's response, plus a keyed
subject reference) and how it acts on the route Component 3 returns. Components 2 and 3 are
stubbed at the HTTP boundary, as in test_executions.py.
"""

from __future__ import annotations

import json

import jwt
import pytest

from service import pipeline
from tests.test_executions import PASSWORD, _engine, _execute, _quantity, Stubs  # noqa: F401

# The cross-language vector: Component 3's tests/auth.test.js asserts the same values.
VECTOR_SECRET = "vector-secret-not-for-production"
VECTOR_17 = "54a2a6c26d59c6e0d39d5e5e39a43a517fd0b79e2fb44ae435a4a4997dab9e00"
VECTOR_18 = "38d28f38aa46b80df1cb544b0371ad1d33fd53d25abdf0746070bc3680183475"


def _trained(score: float = 0.2, lstm: float = 0.18, gnn: float = 0.22, rules=None) -> dict:
    body = _engine("ALLOW", score=score, benchmark=True)
    body["model_version"] = "fraud-bundle-2026-10"
    body["explanation"] = {"rules": rules or ["score_below_step_up_threshold"],
                           "streams": {"lstm": lstm, "gnn": gnn, "fused": score},
                           "top_factors": [], "notes": []}
    return body


def _assured(status: str, final: str, route: str, quality: str = "FULL") -> dict:
    return {
        "success": True, "status": status,
        "policyResult": {"action": final, "policyId": "P002"},
        "assurance": {"route": route, "assessedRoute": route, "assuranceScore": 0.5,
                      "signalQuality": quality, "policyId": "P002", "reasons": ["…"]},
        "record": {"finalDecision": final, "recordHash": "0x" + "cd" * 32},
        "blockchain": {"success": True, "isMock": True},
    }


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> Stubs:
    return Stubs(monkeypatch)


@pytest.fixture
def owner(client, registered):
    headers, _ = registered("owner@example.com", PASSWORD)
    portfolio = client.post("/portfolios", headers=headers, json={"name": "Main", "holdings": [
        {"symbol": "SPY", "quantity": 100, "current_price": 500.0, "avg_daily_volume": 4.0e10},
    ]}).json()
    me = client.get("/auth/me", headers=headers).json()
    return headers, portfolio["id"], me["id"]


# --------------------------------------------------------------------------------------
# What is sent
# --------------------------------------------------------------------------------------

def test_trained_component2_streams_become_independent_model_scores(client, owner, stubs) -> None:
    headers, pid, user_id = owner
    stubs.fraud = pipeline.Downstream(True, 200, _trained())
    _execute(client, headers, pid)

    signals = stubs.audit_requests[0]["signals"]
    assert signals["modelScores"] == {"lstm": 0.18, "gnn": 0.22}       # "fused" is derived: excluded
    assert signals["rules"] == ["score_below_step_up_threshold"]
    assert signals["benchmarkModel"] is True
    assert signals["sourceModelVersion"] == "fraud-bundle-2026-10"
    assert signals["subjectRef"] == pipeline.subject_ref(user_id)


def test_heuristic_mode_sends_no_second_opinion(client, owner, stubs) -> None:
    headers, pid, _ = owner
    _execute(client, headers, pid)                     # default stub: untrained heuristic

    sent = stubs.audit_requests[0]
    assert sent["signals"]["modelScores"] == {}         # one heuristic score is not two models
    assert sent["signals"]["benchmarkModel"] is False
    assert sent["confidence"] == 50                     # legacy field kept for older Component 3s


def test_signals_never_carry_identity_or_invented_values(client, owner, stubs) -> None:
    headers, pid, user_id = owner
    stubs.fraud = pipeline.Downstream(True, 200, _trained())
    _execute(client, headers, pid)
    text = json.dumps(stubs.audit_requests[0])
    assert "owner@example.com" not in text
    assert f'"{user_id}"' not in text and pipeline.account_key(user_id) not in text


def test_malformed_component2_values_are_dropped_not_forwarded() -> None:
    fraud = _trained()
    fraud["explanation"]["streams"].update({"bad name": 0.1, "over": 1.5, "flag": True, "text": "0.3"})
    fraud["explanation"]["rules"] = ["ok_rule", "", "x" * 65, 7]
    signals = pipeline.assurance_signals(fraud)
    assert signals["modelScores"] == {"lstm": 0.18, "gnn": 0.22}
    assert signals["rules"] == ["ok_rule"]
    assert "subjectRef" not in signals                  # no user given


def test_subject_ref_matches_component3_and_needs_the_key(monkeypatch) -> None:
    monkeypatch.setenv("JWT_SECRET", VECTOR_SECRET)
    monkeypatch.delenv("AUDIT_PSEUDONYM_KEY", raising=False)
    assert pipeline.subject_ref(17) == VECTOR_17
    assert pipeline.subject_ref(18) == VECTOR_18
    monkeypatch.setenv("JWT_SECRET", "a-different-secret")
    assert pipeline.subject_ref(17) != VECTOR_17
    monkeypatch.setenv("AUDIT_PSEUDONYM_KEY", "override")
    assert pipeline.subject_ref(17) not in (VECTOR_17, None)


def test_component3_calls_carry_a_short_lived_service_token(monkeypatch) -> None:
    captured = {}

    class _Response:
        status = 200
        def read(self): return b"{}"
        def __enter__(self): return self
        def __exit__(self, *exc): return False

    def fake_urlopen(request, timeout):
        captured["auth"] = request.get_header("Authorization")
        return _Response()

    monkeypatch.setattr(pipeline.urllib.request, "urlopen", fake_urlopen)
    pipeline.anchor_decision({"transactionId": "x"})
    token = captured["auth"].removeprefix("Bearer ")
    claims = jwt.decode(token, options={"verify_signature": False})
    assert claims["sub"] == "service:platform"
    assert claims["iss"] == "j26-se-325-platform"
    assert claims["exp"] - claims["iat"] <= 5 * 60


def test_the_service_token_is_not_a_user(client) -> None:
    response = client.get("/auth/me", headers={"Authorization": f"Bearer {pipeline.service_token()}"})
    assert response.status_code == 401


# --------------------------------------------------------------------------------------
# How the route is acted on
# --------------------------------------------------------------------------------------

def test_auto_route_executes(client, owner, stubs) -> None:
    headers, pid, _ = owner
    stubs.fraud = pipeline.Downstream(True, 200, _trained())
    stubs.audit = pipeline.Downstream(True, 200, _assured("COMPLETED", "APPROVE", "AUTO"))
    body = _execute(client, headers, pid).json()
    assert body["status"] == "EXECUTED"
    assert body["audit"]["assurance"]["route"] == "AUTO"      # kept in the trace for the UI
    assert _quantity(client, headers, pid, "SPY") == 80


def test_verify_route_asks_the_account_holder_to_step_up(client, owner, stubs) -> None:
    headers, pid, _ = owner
    stubs.audit = pipeline.Downstream(True, 200, _assured("HUMAN_REVIEW_REQUIRED", "PENDING_HUMAN_REVIEW", "VERIFY", "DEGRADED"))
    body = _execute(client, headers, pid).json()
    assert body["status"] == "PENDING_REVIEW"
    assert "step-up" in body["status_reason"]
    assert _quantity(client, headers, pid, "SPY") == 100


def test_human_review_route_is_held_for_an_analyst_not_the_account_holder(client, owner, stubs) -> None:
    headers, pid, _ = owner
    stubs.audit = pipeline.Downstream(True, 200, _assured("HUMAN_REVIEW_REQUIRED", "PENDING_HUMAN_REVIEW", "HUMAN_REVIEW"))
    body = _execute(client, headers, pid).json()
    assert body["status"] == "HELD"
    assert "analyst" in body["status_reason"]
    assert _quantity(client, headers, pid, "SPY") == 100
    # The account holder cannot approve it themselves.
    confirm = client.post(f"/executions/{body['id']}/confirm", headers=headers, json={"password": PASSWORD, "approve": True})
    assert confirm.status_code == 409
    assert stubs.reviews == []


def _held_for_analyst(client, owner, stubs) -> str:
    headers, pid, _ = owner
    stubs.audit = pipeline.Downstream(True, 200, _assured("HUMAN_REVIEW_REQUIRED", "PENDING_HUMAN_REVIEW", "HUMAN_REVIEW"))
    execution_id = _execute(client, headers, pid).json()["id"]
    stubs.audit = pipeline.Downstream(True, 409, {"success": False, "message": "already exists"})
    return execution_id


def test_retry_while_the_analyst_has_not_decided_stays_held(client, owner, stubs) -> None:
    headers, pid, _ = owner
    execution_id = _held_for_analyst(client, owner, stubs)
    stubs.retry = pipeline.Downstream(True, 200, {
        "success": True, "status": "HUMAN_REVIEW_REQUIRED",
        "assurance": {"route": "HUMAN_REVIEW", "signalQuality": "FULL"},
        "record": {"finalDecision": "PENDING_HUMAN_REVIEW"},
    })
    body = client.post(f"/executions/{execution_id}/retry", headers=headers).json()
    assert body["status"] == "HELD"
    assert _quantity(client, headers, pid, "SPY") == 100


def test_retry_after_an_analyst_approval_executes(client, owner, stubs) -> None:
    headers, pid, _ = owner
    execution_id = _held_for_analyst(client, owner, stubs)
    stubs.retry = pipeline.Downstream(True, 200, {
        "success": True, "status": "COMPLETED", "assurance": {"route": "HUMAN_REVIEW", "signalQuality": "FULL"},
        "record": {"finalDecision": "HUMAN_APPROVED"},
    })
    body = client.post(f"/executions/{execution_id}/retry", headers=headers).json()
    assert body["status"] == "EXECUTED"
    assert _quantity(client, headers, pid, "SPY") == 80
    assert len(stubs.fraud_requests) == 1                    # not rescored


def test_retry_after_an_analyst_rejection_rejects(client, owner, stubs) -> None:
    headers, pid, _ = owner
    execution_id = _held_for_analyst(client, owner, stubs)
    stubs.retry = pipeline.Downstream(True, 200, {"success": True, "status": "COMPLETED", "record": {"finalDecision": "HUMAN_REJECTED"}})
    body = client.post(f"/executions/{execution_id}/retry", headers=headers).json()
    assert body["status"] == "REJECTED"
    assert _quantity(client, headers, pid, "SPY") == 100


# --------------------------------------------------------------------------------------
# Backward compatibility
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("assurance", [
    None,                                                           # Component 3 before the engine
    {"route": "HUMAN_REVIEW", "signalQuality": "LEGACY"},           # its P001 baseline
])
def test_without_an_engine_route_review_maps_to_step_up_as_before(client, owner, stubs, assurance) -> None:
    headers, pid, _ = owner
    body = {"success": True, "status": "HUMAN_REVIEW_REQUIRED", "policyResult": {"action": "HUMAN_REVIEW"},
            "record": {"finalDecision": "PENDING_HUMAN_REVIEW"}}
    if assurance:
        body["assurance"] = assurance
    stubs.audit = pipeline.Downstream(True, 200, body)
    assert _execute(client, headers, pid).json()["status"] == "PENDING_REVIEW"


def test_a_retry_from_an_older_component3_without_status_still_reads_as_completed(client, owner, stubs) -> None:
    headers, pid, _ = owner
    stubs.audit = pipeline.Downstream(True, 409, {"success": False})
    stubs.retry = pipeline.Downstream(True, 200, {"success": True, "record": {"finalDecision": "APPROVE"}})
    assert _execute(client, headers, pid).json()["status"] == "EXECUTED"


def test_block_still_wins_over_any_route(client, owner, stubs) -> None:
    headers, pid, _ = owner
    stubs.fraud = pipeline.Downstream(True, 200, _engine("BLOCK", score=0.95))
    stubs.audit = pipeline.Downstream(True, 200, _assured("COMPLETED", "REJECT", "AUTO"))
    assert _execute(client, headers, pid).json()["status"] == "BLOCKED"


def test_component3_unreachable_still_holds(client, owner, stubs) -> None:
    headers, pid, _ = owner
    stubs.audit = pipeline.Downstream(False, None, None, "connection refused")
    assert _execute(client, headers, pid).json()["status"] == "HELD"
