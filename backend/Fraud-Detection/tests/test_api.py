"""API validation and decisions."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from fraud_engine.api import create_app
from fraud_engine.config import load_settings


def _client(tmp_path: Path, **gateway_overrides: float) -> TestClient:
    settings = load_settings("synthetic")
    settings["log_dir"] = str(tmp_path)
    settings["gateway"].update(gateway_overrides)
    return TestClient(create_app(settings, bundle_path=None))


def test_health_and_allow(tmp_path: Path) -> None:
    client = _client(tmp_path)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    response = client.post(
        "/v1/transactions/score",
        json={"transaction_id": "tx-1", "amount": 25, "account_key": "acct-a"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] in {"ALLOW", "STEP_UP", "BLOCK"}
    assert body["model_version"] == "heuristic-untrained"
    assert body["benchmark_model"] is False
    assert 0 <= body["fraud_score"] <= 1
    log_text = (tmp_path / "decisions.jsonl").read_text(encoding="utf-8")
    assert "acct-a" not in log_text
    assert "amount" not in log_text


def test_missing_field_and_sensitive_field_are_rejected(tmp_path: Path) -> None:
    client = _client(tmp_path)
    missing = client.post("/v1/transactions/score", json={"transaction_id": "tx-2"})
    assert missing.status_code == 422
    sensitive = client.post(
        "/v1/transactions/score",
        json={"transaction_id": "tx-3", "amount": 10, "email": "person@example.com"},
    )
    assert sensitive.status_code == 422
    nested = client.post(
        "/v1/transactions/score",
        json={"transaction_id": "tx-4", "amount": 10, "features": {"card_number": 4111111111111111}},
    )
    assert nested.status_code == 422


def test_hard_block_reputation_and_dashboard(tmp_path: Path) -> None:
    client = _client(tmp_path)
    blocked = client.post(
        "/v1/transactions/score",
        json={"transaction_id": "tx-block", "amount": 10, "ip_reputation": 0.01},
    )
    assert blocked.status_code == 200
    assert blocked.json()["decision"] == "BLOCK"
    stepped = client.post(
        "/v1/transactions/score",
        json={"transaction_id": "tx-step", "amount": 10, "ip_reputation": 0.25},
    )
    assert stepped.json()["decision"] == "STEP_UP"
    page = client.get("/v1/dashboard")
    assert page.status_code == 200
    assert "tx-block" in page.text
    assert "BLOCK" in page.text


def test_configured_score_bands_and_hard_amount_block(tmp_path: Path) -> None:
    client = _client(tmp_path)
    allow = client.post("/v1/transactions/score", json={"transaction_id": "band-allow", "amount": 100_000})
    step = client.post("/v1/transactions/score", json={"transaction_id": "band-step", "amount": 600_000})
    block = client.post("/v1/transactions/score", json={"transaction_id": "band-block", "amount": 900_000})
    hard = client.post("/v1/transactions/score", json={"transaction_id": "band-hard", "amount": 1_000_000})
    assert allow.json()["decision"] == "ALLOW"
    assert allow.json()["fraud_score"] == 0.1
    assert "score_below_step_up_threshold" in allow.json()["explanation"]["rules"]
    assert step.json()["decision"] == "STEP_UP"
    assert step.json()["fraud_score"] == 0.6
    assert block.json()["decision"] == "BLOCK"
    assert block.json()["fraud_score"] == 0.9
    assert "score_above_block_threshold" in block.json()["explanation"]["rules"]
    assert hard.json()["decision"] == "BLOCK"
    assert "amount_hard_block" in hard.json()["explanation"]["rules"]
    assert 0 <= hard.json()["fraud_score"] <= 1


def test_rate_limit_raises_the_decision(tmp_path: Path) -> None:
    client = _client(tmp_path, rate_limit=2, rate_window_seconds=60, rate_limit_block_multiplier=2)
    decisions = []
    for index in range(5):
        response = client.post(
            "/v1/transactions/score",
            json={"transaction_id": f"rate-{index}", "amount": 10, "account_key": "same-account"},
        )
        body = response.json()
        decisions.append(body["decision"])
        assert 0 <= body["fraud_score"] <= 1
    assert decisions == ["ALLOW", "ALLOW", "STEP_UP", "STEP_UP", "BLOCK"]
    assert "rate_limit_block" in client.post(
        "/v1/transactions/score",
        json={"transaction_id": "rate-5", "amount": 10, "account_key": "same-account"},
    ).json()["explanation"]["rules"]


def test_drift_can_raise_a_low_score_to_step_up(tmp_path: Path) -> None:
    client = _client(tmp_path, drift_delta=0.1, drift_window=5)
    for index in range(5):
        response = client.post(
            "/v1/transactions/score",
            json={"transaction_id": f"drift-high-{index}", "amount": 900_000, "account_key": f"acct-{index}"},
        )
        assert response.json()["decision"] == "BLOCK"
    quiet = client.post(
        "/v1/transactions/score",
        json={"transaction_id": "drift-quiet", "amount": 10, "account_key": "acct-quiet"},
    )
    body = quiet.json()
    assert body["fraud_score"] < 0.55
    assert body["decision"] == "STEP_UP"
    assert "concept_drift_step_up" in body["explanation"]["rules"]


def test_simulated_signal_must_be_labelled(tmp_path: Path) -> None:
    client = _client(tmp_path)
    rejected = client.post(
        "/v1/transactions/score",
        json={
            "transaction_id": "sim-bad",
            "amount": 10,
            "simulated_signals": {"origin": "customer", "typing_interval_ms": 80},
        },
    )
    assert rejected.status_code == 422
    accepted = client.post(
        "/v1/transactions/score",
        json={
            "transaction_id": "sim-ok",
            "amount": 10,
            "simulated_signals": {"origin": "simulated", "typing_interval_ms": 80},
        },
    )
    assert accepted.status_code == 200
    assert any("Simulated signals" in note for note in accepted.json()["explanation"]["notes"])


def test_recent_decisions_are_returned_as_json(tmp_path: Path) -> None:
    client = _client(tmp_path)
    scored = client.post(
        "/v1/transactions/score",
        json={"transaction_id": "history-1", "amount": 25, "account_key": "acct-hidden"},
    )
    assert scored.status_code == 200
    listed = client.get("/v1/decisions")
    assert listed.status_code == 200
    body = listed.json()
    assert body["scope"] == "this API process only"
    assert body["decisions"][0]["transaction_id"] == "history-1"
    assert body["decisions"][0]["decision"] == scored.json()["decision"]
    assert "scored_at" in body["decisions"][0]
    assert "acct-hidden" not in listed.text


def test_service_failure_is_a_503(tmp_path: Path) -> None:
    client = _client(tmp_path)

    def _boom(_payload: dict, now: float | None = None) -> dict:
        raise RuntimeError("model crashed")

    client.app.state.service.score = _boom
    response = client.post("/v1/transactions/score", json={"transaction_id": "tx-9", "amount": 10})
    assert response.status_code == 503
    assert "crashed" not in response.text


def test_scored_transaction_can_be_read_back(tmp_path: Path) -> None:
    client = _client(tmp_path)
    scored = client.post(
        "/v1/transactions/score",
        json={"transaction_id": "tx-lookup", "amount": 40, "account_key": "acct-l"},
    ).json()
    found = client.get("/v1/transactions/tx-lookup")
    assert found.status_code == 200
    assert found.json() == scored
    assert "acct-l" not in found.text
    assert client.get("/v1/transactions/never-scored").status_code == 404
