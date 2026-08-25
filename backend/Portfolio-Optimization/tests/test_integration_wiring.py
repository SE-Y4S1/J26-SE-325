"""Tests for the frontend-wiring additions: CORS and optional JWT verification.

These cover the two things that stand between a working browser call and a broken one, and
that are invisible until a browser is actually pointed at the service.
"""

from __future__ import annotations

import time

import jwt
import pytest
from fastapi.testclient import TestClient

from service.api import app
from service.auth import CallerIdentity, auth_required, require_user
from service.cors import DEFAULT_ORIGINS, allowed_origins

SECRET = "wiring-test-secret-long-enough-to-be-accepted-0123456789"


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def _token(user_id: int = 1, *, secret: str = SECRET, expires_in: int = 3600,
           issuer: str = "j26-se-325-platform") -> str:
    now = int(time.time())
    return jwt.encode(
        {"sub": str(user_id), "iat": now, "exp": now + expires_in,
         "iss": issuer, "email": "user@example.com"},
        secret, algorithm="HS256",
    )


# --------------------------------------------------------------------------------------
# CORS
# --------------------------------------------------------------------------------------

def test_default_origins_cover_the_next_dev_server() -> None:
    """Both spellings, because a browser treats localhost and 127.0.0.1 as distinct origins
    and Next prints whichever the user happened to open."""
    assert "http://localhost:3000" in DEFAULT_ORIGINS
    assert "http://127.0.0.1:3000" in DEFAULT_ORIGINS


def test_allowed_origins_reads_the_env_var(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://a.example.com, https://b.example.com/")
    # Trailing slashes are stripped: the browser's Origin header never has one, so a
    # configured "https://b.example.com/" would silently never match.
    assert allowed_origins() == ["https://a.example.com", "https://b.example.com"]


def test_blank_env_falls_back_to_defaults(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_ORIGINS", "   ")
    assert allowed_origins() == list(DEFAULT_ORIGINS)


def test_preflight_is_answered_for_the_withdraw_endpoint(client: TestClient) -> None:
    """THE check that matters for the browser. Without a successful preflight the actual
    POST is never sent, and the failure surfaces as an opaque network error."""
    response = client.options(
        "/portfolio/withdraw",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,authorization",
        },
    )
    assert response.status_code in (200, 204)
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    allow_headers = response.headers.get("access-control-allow-headers", "").lower()
    assert "authorization" in allow_headers and "content-type" in allow_headers


def test_actual_response_carries_the_allow_origin_header(client: TestClient) -> None:
    response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_unlisted_origin_is_not_granted_access(client: TestClient) -> None:
    """A site the user happens to be visiting must not be able to read the response."""
    response = client.get("/health", headers={"Origin": "https://evil.example.com"})
    assert response.headers.get("access-control-allow-origin") != "https://evil.example.com"


# --------------------------------------------------------------------------------------
# Optional auth
# --------------------------------------------------------------------------------------

def test_auth_is_off_by_default(monkeypatch) -> None:
    """The existing 338 tests call these endpoints unauthenticated and must keep passing."""
    monkeypatch.delenv("AUTH_REQUIRED", raising=False)
    assert auth_required() is False
    assert require_user(credentials=None) is None


@pytest.mark.parametrize("value,expected", [
    ("true", True), ("1", True), ("yes", True), ("on", True), ("TRUE", True),
    ("false", False), ("0", False), ("", False), ("maybe", False),
])
def test_auth_required_parsing(monkeypatch, value: str, expected: bool) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", value)
    assert auth_required() is expected


def test_enforced_auth_rejects_a_missing_token(monkeypatch) -> None:
    from fastapi import HTTPException

    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("JWT_SECRET", SECRET)

    with pytest.raises(HTTPException) as excinfo:
        require_user(credentials=None)
    assert excinfo.value.status_code == 401


def test_enforced_auth_accepts_a_platform_token(monkeypatch) -> None:
    from fastapi.security import HTTPAuthorizationCredentials

    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("JWT_SECRET", SECRET)

    identity = require_user(
        credentials=HTTPAuthorizationCredentials(scheme="Bearer", credentials=_token(42))
    )
    assert isinstance(identity, CallerIdentity)
    assert identity.user_id == 42
    assert identity.email == "user@example.com"


@pytest.mark.parametrize("bad_token,reason", [
    (_token(secret="a-totally-different-secret-value-here"), "signed with another secret"),
    (_token(expires_in=-10), "expired"),
    (_token(issuer="somebody-else"), "wrong issuer"),
    ("not-a-jwt", "malformed"),
])
def test_enforced_auth_rejects_bad_tokens(monkeypatch, bad_token: str, reason: str) -> None:
    from fastapi import HTTPException
    from fastapi.security import HTTPAuthorizationCredentials

    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("JWT_SECRET", SECRET)

    with pytest.raises(HTTPException) as excinfo:
        require_user(
            credentials=HTTPAuthorizationCredentials(scheme="Bearer", credentials=bad_token)
        )
    assert excinfo.value.status_code == 401, reason


def test_enforced_auth_without_a_secret_fails_closed(monkeypatch) -> None:
    """Enforcement switched on but no secret configured. Falling back to "allow" would
    silently disable the control that was just requested, so it must refuse -- and with 500,
    because the fault is the deployment's rather than the caller's."""
    from fastapi import HTTPException
    from fastapi.security import HTTPAuthorizationCredentials

    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.delenv("JWT_SECRET", raising=False)

    with pytest.raises(HTTPException) as excinfo:
        require_user(
            credentials=HTTPAuthorizationCredentials(scheme="Bearer", credentials=_token())
        )
    assert excinfo.value.status_code == 500


def test_withdraw_still_works_unauthenticated_when_auth_is_off(client: TestClient) -> None:
    """End-to-end proof the additions are non-breaking."""
    response = client.post(
        "/portfolio/withdraw",
        json={
            "holdings": [
                {"symbol": "SPY", "quantity": 500, "current_price": 580.0,
                 "avg_daily_volume": 4.0e10}
            ],
            "target_amount": 50_000, "urgency": 0.5,
        },
    )
    assert response.status_code == 200
    assert response.json()["feasible"] is True
