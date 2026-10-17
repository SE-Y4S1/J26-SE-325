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


def test_service_failure_is_a_503(tmp_path: Path) -> None:
    client = _client(tmp_path)

    def _boom(_payload: dict, now: float | None = None) -> dict:
        raise RuntimeError("model crashed")

    client.app.state.service.score = _boom
    response = client.post("/v1/transactions/score", json={"transaction_id": "tx-9", "amount": 10})
    assert response.status_code == 503
    assert "crashed" not in response.text
