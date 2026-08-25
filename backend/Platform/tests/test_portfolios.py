"""Portfolio CRUD tests.

The isolation tests are the important ones: this service holds every user's holdings, and
one user reading another's would be the single worst bug it could have.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

HOLDINGS = [
    {"symbol": "SPY", "quantity": 500, "current_price": 580.0, "avg_daily_volume": 4.0e10},
    {"symbol": "THIN", "quantity": 6000, "current_price": 15.0, "avg_daily_volume": 8.0e5},
]


def _create(client: TestClient, headers: dict, name: str = "Demo", holdings=None):
    return client.post(
        "/portfolios", headers=headers,
        json={"name": name, "holdings": HOLDINGS if holdings is None else holdings},
    )


# --------------------------------------------------------------------------------------
# Access control
# --------------------------------------------------------------------------------------

def test_every_portfolio_route_requires_authentication(client: TestClient) -> None:
    assert client.get("/portfolios").status_code == 401
    assert client.post("/portfolios", json={"name": "x"}).status_code == 401
    assert client.get("/portfolios/1").status_code == 401
    assert client.put("/portfolios/1", json={"name": "x"}).status_code == 401
    assert client.delete("/portfolios/1").status_code == 401


def test_one_user_cannot_read_anothers_portfolio(client: TestClient, registered) -> None:
    """THE isolation guard. 404 rather than 403 on purpose -- a 403 would confirm the id is
    real, letting anyone enumerate how many portfolios the platform holds."""
    alice_headers, _ = registered("alice@example.com")
    bob_headers, _ = registered("bob@example.com")

    portfolio_id = _create(client, alice_headers, "Alice Portfolio").json()["id"]

    assert client.get(f"/portfolios/{portfolio_id}", headers=bob_headers).status_code == 404
    assert client.put(f"/portfolios/{portfolio_id}", headers=bob_headers,
                      json={"name": "Hijacked"}).status_code == 404
    assert client.delete(f"/portfolios/{portfolio_id}", headers=bob_headers).status_code == 404

    # And Alice's portfolio is untouched by all that.
    assert client.get(f"/portfolios/{portfolio_id}", headers=alice_headers).json()["name"] == "Alice Portfolio"


def test_listing_shows_only_your_own(client: TestClient, registered) -> None:
    alice_headers, _ = registered("alice2@example.com")
    bob_headers, _ = registered("bob2@example.com")

    _create(client, alice_headers, "A1")
    _create(client, alice_headers, "A2")
    _create(client, bob_headers, "B1")

    assert {p["name"] for p in client.get("/portfolios", headers=alice_headers).json()} == {"A1", "A2"}
    assert {p["name"] for p in client.get("/portfolios", headers=bob_headers).json()} == {"B1"}


# --------------------------------------------------------------------------------------
# Create / read
# --------------------------------------------------------------------------------------

def test_create_returns_the_stored_portfolio(client: TestClient, registered) -> None:
    headers, _ = registered()
    response = _create(client, headers)

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Demo"
    assert len(body["holdings"]) == 2
    # 500*580 + 6000*15
    assert body["total_value"] == 290_000 + 90_000


def test_symbols_are_normalised_to_uppercase(client: TestClient, registered) -> None:
    """Component 1 looks symbols up by exact string; 'spy' and 'SPY' must not diverge."""
    headers, _ = registered()
    response = _create(client, headers, holdings=[
        {"symbol": "spy", "quantity": 1, "current_price": 1.0, "avg_daily_volume": 1.0}
    ])
    assert response.json()["holdings"][0]["symbol"] == "SPY"


def test_holdings_match_component_one_contract(client: TestClient, registered) -> None:
    """The stored shape must be postable straight to /portfolio/withdraw. Any renaming here
    would force a translation layer, which is where field drift hides."""
    headers, _ = registered()
    holding = _create(client, headers).json()["holdings"][0]

    assert set(holding) == {"symbol", "quantity", "current_price", "avg_daily_volume", "cost_basis"}


def test_duplicate_name_for_same_user_is_rejected(client: TestClient, registered) -> None:
    headers, _ = registered()
    _create(client, headers, "Same")
    assert _create(client, headers, "Same").status_code == 409


def test_different_users_may_reuse_a_name(client: TestClient, registered) -> None:
    """The uniqueness constraint is per user, not global."""
    alice_headers, _ = registered("alice3@example.com")
    bob_headers, _ = registered("bob3@example.com")

    assert _create(client, alice_headers, "Main").status_code == 201
    assert _create(client, bob_headers, "Main").status_code == 201


def test_missing_portfolio_is_404(client: TestClient, registered) -> None:
    headers, _ = registered()
    assert client.get("/portfolios/9999", headers=headers).status_code == 404


# --------------------------------------------------------------------------------------
# Update
# --------------------------------------------------------------------------------------

def test_rename_without_resending_holdings(client: TestClient, registered) -> None:
    """holdings=None means 'leave them alone'. Without that, renaming would require the
    client to round-trip every holding."""
    headers, _ = registered()
    portfolio_id = _create(client, headers).json()["id"]

    response = client.put(f"/portfolios/{portfolio_id}", headers=headers, json={"name": "Renamed"})
    body = response.json()

    assert response.status_code == 200
    assert body["name"] == "Renamed"
    assert len(body["holdings"]) == 2


def test_empty_holdings_list_clears_them(client: TestClient, registered) -> None:
    """holdings=[] is distinct from holdings=None -- it means 'remove them all'."""
    headers, _ = registered()
    portfolio_id = _create(client, headers).json()["id"]

    body = client.put(f"/portfolios/{portfolio_id}", headers=headers, json={"holdings": []}).json()
    assert body["holdings"] == []
    assert body["total_value"] == 0


def test_replacing_holdings_does_not_leave_orphans(client: TestClient, registered) -> None:
    """delete-orphan cascade must actually remove the detached rows, not just unlink them."""
    from sqlalchemy import select

    from store.database import _SessionLocal
    from store.models import Holding

    headers, _ = registered()
    portfolio_id = _create(client, headers).json()["id"]

    client.put(f"/portfolios/{portfolio_id}", headers=headers, json={"holdings": [
        {"symbol": "QQQ", "quantity": 10, "current_price": 490.0, "avg_daily_volume": 2.0e10}
    ]})

    session = _SessionLocal()
    remaining = session.scalars(select(Holding)).all()
    session.close()

    assert len(remaining) == 1, f"orphaned holdings left behind: {[h.symbol for h in remaining]}"
    assert remaining[0].symbol == "QQQ"


def test_updated_at_advances_on_change(client: TestClient, registered) -> None:
    headers, _ = registered()
    created = _create(client, headers).json()
    updated = client.put(f"/portfolios/{created['id']}", headers=headers,
                         json={"name": "Touched"}).json()
    assert updated["updated_at"] >= created["updated_at"]


# --------------------------------------------------------------------------------------
# Delete
# --------------------------------------------------------------------------------------

def test_delete_removes_the_portfolio_and_its_holdings(client: TestClient, registered) -> None:
    from sqlalchemy import select

    from store.database import _SessionLocal
    from store.models import Holding

    headers, _ = registered()
    portfolio_id = _create(client, headers).json()["id"]

    assert client.delete(f"/portfolios/{portfolio_id}", headers=headers).status_code == 204
    assert client.get(f"/portfolios/{portfolio_id}", headers=headers).status_code == 404

    session = _SessionLocal()
    orphans = session.scalars(select(Holding)).all()
    session.close()
    assert orphans == [], "holdings outlived their portfolio"


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------

def test_zero_price_is_rejected(client: TestClient, registered) -> None:
    """Component 1 divides by price when converting notional to quantity."""
    headers, _ = registered()
    response = _create(client, headers, holdings=[
        {"symbol": "X", "quantity": 1, "current_price": 0.0, "avg_daily_volume": 1.0}
    ])
    assert response.status_code == 422


def test_negative_quantity_is_rejected(client: TestClient, registered) -> None:
    headers, _ = registered()
    response = _create(client, headers, holdings=[
        {"symbol": "X", "quantity": -5, "current_price": 10.0, "avg_daily_volume": 1.0}
    ])
    assert response.status_code == 422


def test_avg_daily_volume_is_required(client: TestClient, registered) -> None:
    """Component 1's liquidity cost model divides by ADV; omitting it is not a valid holding."""
    headers, _ = registered()
    response = _create(client, headers, holdings=[
        {"symbol": "X", "quantity": 1, "current_price": 10.0}
    ])
    assert response.status_code == 422
