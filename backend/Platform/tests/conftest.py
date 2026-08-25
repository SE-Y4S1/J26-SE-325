"""Shared fixtures.

Every test runs against a fresh in-memory database and a fixed JWT secret, so tests cannot
leak state into one another and none of them depend on the developer's real .env.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

# Set BEFORE importing anything that reads it: auth.security.get_jwt_secret() raises when
# JWT_SECRET is missing, and service.api validates it during app startup.
os.environ.setdefault(
    "JWT_SECRET", "unit-test-secret-long-enough-to-pass-validation-0123456789"
)
os.environ.setdefault("DATABASE_URL", "sqlite://")      # in-memory

from fastapi.testclient import TestClient  # noqa: E402

from service.api import app  # noqa: E402
from store import database  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_database() -> Iterator[None]:
    """Give every test an empty in-memory database.

    `create_db_engine` uses a StaticPool for in-memory URLs so one connection is shared
    app-wide; without that, `create_all` and the request handlers would each get their own
    blank database. Because the app's lifespan calls `init_db()` on startup anyway, this
    fixture just needs to reset the schema rather than build a competing engine.
    """
    database.init_db("sqlite://")
    database.reset_db()
    yield


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def registered(client: TestClient):
    """Register a user and return (auth_headers, user_dict)."""

    def _register(email: str = "user@example.com", password: str = "a-good-long-passphrase"):
        response = client.post(
            "/auth/register",
            json={"email": email, "display_name": email.split("@")[0], "password": password},
        )
        assert response.status_code == 201, response.text
        body = response.json()
        return {"Authorization": f"Bearer {body['access_token']}"}, body["user"]

    return _register
