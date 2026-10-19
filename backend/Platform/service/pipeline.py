"""Execution pipeline: Component 1 plan -> Component 2 fraud gateway -> Component 3 audit.

This is the one place a service calls other components on the request path, and it does
so on purpose. Executing a plan changes holdings, holdings live here, and no holding may
change before the transaction has been screened and the decision anchored. Every other
route in this service still calls nothing.

Failure policy, in one place:

* Component 2 unreachable  -> HELD. Fail closed: an unscreened transaction never executes.
* Component 2 BLOCK        -> BLOCKED, whether or not Component 3 is up. A block needs no
                              anchor to be enforced; it is still anchored when possible.
* Component 3 unreachable  -> HELD (unless blocked). Fail closed: an ALLOW is not acted on
                              until the decision is provably recorded.
* STEP_UP or HUMAN_REVIEW  -> PENDING_REVIEW until the account holder re-authenticates.
* ALLOW + anchored APPROVE -> EXECUTED; holdings change.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Component 1 rounds quantities; a sell may exceed the held quantity by float error only.
QUANTITY_TOLERANCE = 1e-6

TRANSACTION_TYPES = {"withdrawal": "PORTFOLIO_WITHDRAWAL", "rebalance": "PORTFOLIO_REBALANCE"}
PRODUCT_CODES = {"withdrawal": "W", "rebalance": "R"}


class PlanError(ValueError):
    """The submitted plan cannot be executed against the portfolio as it stands."""


def fraud_service_url() -> str:
    return os.getenv("FRAUD_SERVICE_URL", "http://localhost:8001").rstrip("/")


def audit_service_url() -> str:
    return os.getenv("AUDIT_SERVICE_URL", "http://localhost:8002").rstrip("/")


def downstream_timeout() -> float:
    return float(os.getenv("PIPELINE_TIMEOUT_SECONDS", "10"))


def new_execution_id() -> str:
    return f"EXE-{uuid.uuid4().hex}"


# --------------------------------------------------------------------------------------
# Orders
# --------------------------------------------------------------------------------------

def derive_orders(kind: str, plan: dict, holdings: dict[str, dict]) -> list[dict]:
    """Turn a Component 1 response into concrete orders, checked against current holdings.

    `holdings` maps symbol -> {"quantity", "current_price"}.
    """
    if kind == "withdrawal":
        orders = _withdrawal_orders(plan, holdings)
    elif kind == "rebalance":
        orders = _rebalance_orders(plan, holdings)
    else:
        raise PlanError(f"Unknown execution kind {kind!r}")
    if not orders:
        raise PlanError("The plan contains no orders to execute")
    return orders


def _withdrawal_orders(plan: dict, holdings: dict[str, dict]) -> list[dict]:
    sales = plan.get("assets_to_sell")
    if not isinstance(sales, list):
        raise PlanError("A withdrawal plan must carry assets_to_sell from /portfolio/withdraw")
    orders = []
    for sale in sales:
        symbol = str(sale.get("symbol", "")).upper().strip()
        quantity = float(sale.get("quantity") or 0)
        price = float(sale.get("expected_price") or 0)
        if quantity <= 0:
            continue
        held = holdings.get(symbol)
        if held is None:
            raise PlanError(f"The plan sells {symbol}, which this portfolio does not hold")
        if quantity > held["quantity"] + QUANTITY_TOLERANCE:
            raise PlanError(
                f"The plan sells {quantity:g} {symbol} but the portfolio holds {held['quantity']:g}"
            )
        if price <= 0:
            raise PlanError(f"The plan has no expected price for {symbol}")
        orders.append({"symbol": symbol, "side": "SELL",
                       "quantity": min(quantity, held["quantity"]), "price": price})
    return orders


def _rebalance_orders(plan: dict, holdings: dict[str, dict]) -> list[dict]:
    """Component 1's own rebalance_orders, checked against what is actually held.

    Never re-derived from recommended_weights: those are fractions of an FX-converted total,
    and this service stores no currency per holding, so sizing trades here would mis-size
    every leg of a mixed-currency portfolio. Component 1 sizes them with the rates it used.
    """
    legs = plan.get("rebalance_orders")
    if not isinstance(legs, list):
        raise PlanError("A rebalance plan must carry rebalance_orders from /portfolio/optimize")
    orders = []
    for leg in legs:
        symbol = str(leg.get("symbol", "")).upper().strip()
        side = str(leg.get("side", "")).upper()
        quantity = float(leg.get("quantity") or 0)
        price = float(leg.get("price") or 0)
        if side not in ("BUY", "SELL") or quantity <= 0 or price <= 0:
            raise PlanError(f"Malformed rebalance order for {symbol or 'an unnamed symbol'}")
        held = holdings.get(symbol)
        if held is None:
            # Without a held position there is nothing to add a buy to; the platform does not
            # open positions it was never told the price or liquidity of.
            raise PlanError(f"The plan trades {symbol}, which this portfolio does not hold")
        if side == "SELL" and quantity > held["quantity"] + QUANTITY_TOLERANCE:
            raise PlanError(
                f"The plan sells {quantity:g} {symbol} but the portfolio holds {held['quantity']:g}"
            )
        orders.append({"symbol": symbol, "side": side,
                       "quantity": min(quantity, held["quantity"]) if side == "SELL" else quantity,
                       "price": price})
    return orders


def transaction_amount(kind: str, plan: dict, orders: list[dict]) -> float:
    """The money the transaction moves, in the plan's base currency: cash raised for a
    withdrawal, turnover for a rebalance. Order prices are in that currency too."""
    if kind == "withdrawal":
        raised = plan.get("raised_amount")
        if isinstance(raised, (int, float)) and raised > 0:
            return float(raised)
    return float(sum(o["quantity"] * o["price"] for o in orders))


def plan_hash(kind: str, portfolio_id: int, plan: dict, orders: list[dict]) -> str:
    """SHA-256 over the plan and the orders derived from it, anchored by Component 3.

    Sorted keys and fixed separators, so the same plan always hashes the same.
    """
    canonical = json.dumps(
        {"kind": kind, "portfolio_id": portfolio_id, "plan": plan, "orders": orders},
        sort_keys=True, separators=(",", ":"), default=str,
    )
    return "0x" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def account_key(user_id: int) -> str:
    """A stable pseudonym for Component 2's per-account history. Never the email or id."""
    return "acct-" + hashlib.sha256(f"platform-user:{user_id}".encode()).hexdigest()[:24]


def apply_orders(holdings: dict[str, Any], orders: list[dict]) -> list[str]:
    """Apply orders to ORM holdings in place. Returns symbols whose position closed out.

    Re-validates sells: holdings may have changed while an execution waited for review.
    """
    for order in orders:
        holding = holdings.get(order["symbol"])
        if holding is None:
            raise PlanError(f"{order['symbol']} is no longer held")
        if order["side"] == "SELL" and order["quantity"] > holding.quantity + QUANTITY_TOLERANCE:
            raise PlanError(
                f"{order['symbol']}: {holding.quantity:g} held, {order['quantity']:g} to sell"
            )
    closed = []
    for order in orders:
        holding = holdings[order["symbol"]]
        signed = order["quantity"] if order["side"] == "BUY" else -order["quantity"]
        holding.quantity = max(0.0, holding.quantity + signed)
        if holding.quantity <= QUANTITY_TOLERANCE:
            closed.append(order["symbol"])
    return closed


# --------------------------------------------------------------------------------------
# Downstream calls
# --------------------------------------------------------------------------------------

@dataclass
class Downstream:
    """One HTTP call's outcome. `reachable=False` means no usable answer came back."""

    reachable: bool
    status_code: int | None
    body: dict | None
    error: str | None = None


def _post_json(url: str, payload: dict, timeout: float) -> Downstream:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return Downstream(True, response.status, json.loads(response.read().decode("utf-8")))
    except urllib.error.HTTPError as err:
        try:
            body = json.loads(err.read().decode("utf-8"))
        except (ValueError, OSError):
            body = None
        return Downstream(True, err.code, body if isinstance(body, dict) else None, f"HTTP {err.code}")
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError) as err:
        logger.warning("downstream call to %s failed: %s", url, err)
        return Downstream(False, None, None, str(err))


def score_transaction(payload: dict) -> Downstream:
    return _post_json(f"{fraud_service_url()}/v1/transactions/score", payload, downstream_timeout())


def anchor_decision(payload: dict) -> Downstream:
    return _post_json(f"{audit_service_url()}/api/audit/evaluate", payload, downstream_timeout())


def submit_review(transaction_id: str, approve: bool) -> Downstream:
    return _post_json(
        f"{audit_service_url()}/api/audit/review",
        {"transactionId": transaction_id, "reviewerDecision": "APPROVE" if approve else "REJECT"},
        downstream_timeout(),
    )


def retry_anchor(transaction_id: str) -> Downstream:
    quoted = urllib.parse.quote(transaction_id, safe="")
    return _post_json(f"{audit_service_url()}/api/audit/{quoted}/retry", {}, downstream_timeout())


# --------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------

def fraud_payload(execution_id: str, kind: str, amount: float, currency: str,
                  user_id: int, device_key: str | None) -> dict:
    payload = {
        "transaction_id": execution_id,
        "amount": round(amount, 2),
        "currency": currency,
        "product_code": PRODUCT_CODES[kind],
        "account_key": account_key(user_id),
    }
    if device_key:
        payload["device_key"] = device_key
    return payload


def audit_payload(execution_id: str, kind: str, amount: float, source_model_version: str,
                  fraud: dict, plan_digest: str) -> dict:
    """Component 3's evaluate request for a decision Component 2 already enforced.

    `confidence` is not something Component 2 reports. It is set from whether the score came
    from a benchmark-validated model (90) or from the untrained heuristic or an unvalidated
    bundle (50), so P001 routes a high heuristic score to human review instead of treating
    it as a confident automated rejection.
    """
    return {
        "transactionId": execution_id,
        "riskScore": round(float(fraud["fraud_score"]) * 100, 2),
        "confidence": 90 if fraud.get("benchmark_model") else 50,
        "amount": round(amount, 2),
        "transactionType": TRANSACTION_TYPES[kind],
        "modelVersion": f"c1:{source_model_version}|c2:{fraud.get('model_version', 'unknown')}",
        "upstreamDecision": {
            "source": "component2",
            "decision": fraud["decision"],
            "modelVersion": str(fraud.get("model_version", "unknown")),
        },
        "planHash": plan_digest,
    }


def audit_record(result: Downstream) -> dict:
    """What the trace keeps from Component 3: its outcome, the record and the chain receipt."""
    body = result.body or {}
    return {
        "reachable": result.reachable,
        "status_code": result.status_code,
        "status": body.get("status") or ("UNREACHABLE" if not result.reachable else "ERROR"),
        "message": body.get("message") or result.error,
        "policy": body.get("policyResult"),
        "record": body.get("record"),
        "blockchain": body.get("blockchain"),
    }


def decide(fraud: Downstream, audit: Downstream | None) -> tuple[str, str]:
    """Final execution status from the two stage outcomes. Pure, so every branch is testable."""
    if not fraud.reachable or fraud.status_code != 200 or not fraud.body:
        return "HELD", "The fraud gateway (Component 2) did not return a decision; nothing was executed."

    decision = fraud.body.get("decision")
    if decision == "BLOCK":
        anchored = audit is not None and audit.reachable and audit.status_code == 200
        suffix = "" if anchored else " The block could not yet be anchored by Component 3."
        return "BLOCKED", f"Component 2 blocked this transaction.{suffix}"

    if audit is None or not audit.reachable:
        return "HELD", "The audit bridge (Component 3) is unreachable; the decision is not yet anchored."
    if audit.status_code != 200 or not audit.body:
        return "HELD", f"The audit bridge (Component 3) could not anchor the decision ({audit.error or audit.status_code})."

    outcome = audit.body.get("status")
    final = (audit.body.get("record") or {}).get("finalDecision")
    if outcome == "HUMAN_REVIEW_REQUIRED":
        why = "Component 2 asked for step-up verification" if decision == "STEP_UP" \
            else "Component 3's policy requires human review"
        return "PENDING_REVIEW", f"{why}. Re-authenticate to approve or reject."
    if outcome == "COMPLETED" and final == "REJECT":
        return "BLOCKED", "Component 3's policy rejected this transaction."
    if outcome == "COMPLETED" and final == "APPROVE" and decision == "ALLOW":
        return "EXECUTED", "Screened by Component 2 and anchored by Component 3."
    return "HELD", f"Unexpected stage outcome (fraud={decision}, audit={outcome}/{final})."
