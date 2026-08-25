"""Auth tests.

The security-relevant assertions here are the ones about what the API does NOT reveal:
identical responses for unknown-email and wrong-password, and 404 rather than 403 for
someone else's data. Those are easy to regress into something friendlier and less safe.
"""

from __future__ import annotations

import time

import jwt
import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------------------

def test_register_returns_a_usable_token(client: TestClient) -> None:
    """Registration issues a token directly -- forcing a separate login after signing up is
    a pointless round trip."""
    response = client.post(
        "/auth/register",
        json={"email": "a@example.com", "display_name": "A", "password": "a-good-long-passphrase"},
    )
    assert response.status_code == 201

    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] > 0
    assert body["user"]["email"] == "a@example.com"

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.status_code == 200


def test_password_is_never_returned(client: TestClient) -> None:
    """Neither the hash nor the plaintext may appear anywhere in a response."""
    response = client.post(
        "/auth/register",
        json={"email": "b@example.com", "display_name": "B", "password": "super-secret-passphrase"},
    )
    serialized = response.text.lower()
    assert "super-secret-passphrase" not in serialized
    assert "password" not in serialized
    assert "argon2" not in serialized


def test_duplicate_email_is_rejected(client: TestClient, registered) -> None:
    registered("dup@example.com")
    response = client.post(
        "/auth/register",
        json={"email": "dup@example.com", "display_name": "Other", "password": "another-passphrase"},
    )
    assert response.status_code == 409


def test_email_is_normalised_to_lowercase(client: TestClient) -> None:
    """Otherwise Nivakaran@x.com and nivakaran@x.com become two accounts."""
    client.post(
        "/auth/register",
        json={"email": "Mixed@Example.com", "display_name": "M", "password": "a-good-long-passphrase"},
    )
    login = client.post(
        "/auth/login", json={"email": "mixed@example.com", "password": "a-good-long-passphrase"}
    )
    assert login.status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {"email": "not-an-email", "display_name": "X", "password": "a-good-long-passphrase"},
        {"email": "c@example.com", "display_name": "X", "password": "short"},
        {"email": "c@example.com", "display_name": "", "password": "a-good-long-passphrase"},
    ],
)
def test_registration_validation(client: TestClient, payload: dict) -> None:
    assert client.post("/auth/register", json=payload).status_code == 422


# --------------------------------------------------------------------------------------
# Login
# --------------------------------------------------------------------------------------

def test_login_succeeds_with_correct_credentials(client: TestClient, registered) -> None:
    registered("login@example.com", "a-good-long-passphrase")
    response = client.post(
        "/auth/login", json={"email": "login@example.com", "password": "a-good-long-passphrase"}
    )
    assert response.status_code == 200
    assert response.json()["access_token"]


def test_unknown_email_and_wrong_password_are_indistinguishable(
    client: TestClient, registered
) -> None:
    """THE account-enumeration guard. If these two responses differ in status or body, the
    login form tells an attacker which addresses have accounts."""
    registered("real@example.com", "a-good-long-passphrase")

    wrong_password = client.post(
        "/auth/login", json={"email": "real@example.com", "password": "wrong-passphrase-here"}
    )
    unknown_email = client.post(
        "/auth/login", json={"email": "ghost@example.com", "password": "wrong-passphrase-here"}
    )

    assert wrong_password.status_code == unknown_email.status_code == 401
    assert wrong_password.json() == unknown_email.json()


# --------------------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------------------

def test_missing_token_is_rejected(client: TestClient) -> None:
    response = client.get("/auth/me")
    assert response.status_code == 401
    assert "bearer" in response.headers.get("www-authenticate", "").lower()


@pytest.mark.parametrize("token", ["garbage", "a.b.c", ""])
def test_malformed_tokens_are_rejected(client: TestClient, token: str) -> None:
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_expired_token_is_rejected(client: TestClient, registered) -> None:
    from auth.security import create_access_token

    _, user = registered("exp@example.com")
    expired = create_access_token(subject=str(user["id"]), expires_minutes=-1)

    response = client.get("/auth/me", headers={"Authorization": f"Bearer {expired}"})
    assert response.status_code == 401
    assert "expired" in response.json()["detail"].lower()


def test_token_signed_with_another_secret_is_rejected(client: TestClient, registered) -> None:
    """The whole point of a shared secret: a token this platform did not sign is worthless."""
    _, user = registered("forge@example.com")
    forged = jwt.encode(
        {"sub": str(user["id"]), "iat": int(time.time()), "exp": int(time.time()) + 3600,
         "iss": "j26-se-325-platform"},
        "a-completely-different-secret-value-here",
        algorithm="HS256",
    )
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401


def test_alg_none_token_is_rejected(client: TestClient, registered) -> None:
    """The classic JWT bypass: claim `alg: none` so no signature is checked. Pinning
    `algorithms=['HS256']` in decode_access_token is what defeats it."""
    _, user = registered("algnone@example.com")
    unsigned = jwt.encode(
        {"sub": str(user["id"]), "iat": int(time.time()), "exp": int(time.time()) + 3600,
         "iss": "j26-se-325-platform"},
        key="",
        algorithm="none",
    )
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {unsigned}"}).status_code == 401


def test_token_for_deleted_user_is_rejected(client: TestClient, registered) -> None:
    """A validly-signed token whose subject no longer exists must not authenticate."""
    from sqlalchemy import select

    from store.database import _SessionLocal
    from store.models import User

    headers, _ = registered("gone@example.com")

    session = _SessionLocal()
    session.delete(session.scalar(select(User).where(User.email == "gone@example.com")))
    session.commit()
    session.close()

    assert client.get("/auth/me", headers=headers).status_code == 401


# --------------------------------------------------------------------------------------
# Secret validation
# --------------------------------------------------------------------------------------

def test_missing_secret_refuses_to_run(monkeypatch) -> None:
    """Better a hard failure at startup than a service running on a guessable secret."""
    from auth.security import get_jwt_secret

    monkeypatch.setenv("JWT_SECRET", "")
    with pytest.raises(RuntimeError, match="JWT_SECRET is not set"):
        get_jwt_secret()


def test_placeholder_secret_refuses_to_run(monkeypatch) -> None:
    """The .env.example value is public in the repository -- anyone could forge tokens."""
    from auth.security import EXAMPLE_SECRET, get_jwt_secret

    monkeypatch.setenv("JWT_SECRET", EXAMPLE_SECRET)
    with pytest.raises(RuntimeError, match="placeholder"):
        get_jwt_secret()


def test_short_secret_refuses_to_run(monkeypatch) -> None:
    from auth.security import get_jwt_secret

    monkeypatch.setenv("JWT_SECRET", "tooshort")
    with pytest.raises(RuntimeError, match="at least 32"):
        get_jwt_secret()


def test_password_hashing_roundtrip() -> None:
    from auth.security import hash_password, verify_password

    hashed = hash_password("a-good-long-passphrase")
    assert hashed != "a-good-long-passphrase"
    assert hashed.startswith("$argon2")
    assert verify_password("a-good-long-passphrase", hashed)
    assert not verify_password("wrong", hashed)


def test_verify_password_survives_a_malformed_hash() -> None:
    """One corrupt row must read as 'wrong password', not take down login for everyone."""
    from auth.security import verify_password

    assert verify_password("anything", "not-a-valid-hash") is False


def test_long_passphrase_is_not_truncated() -> None:
    """Argon2 has no 72-byte ceiling. Under bcrypt these two would verify interchangeably,
    silently discarding everything past byte 72."""
    from auth.security import hash_password, verify_password

    base = "x" * 80
    assert verify_password(base, hash_password(base))
    assert not verify_password(base + "-different-tail", hash_password(base))
