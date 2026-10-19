import json
import os
import urllib.error
import urllib.parse
import urllib.request

from langchain_core.tools import tool
from preferences_context import get_current_preferences
from request_auth import get_caller_authorization
from tools.audit_tools import query_live_audit_service
from tools.fraud_tools import query_live_fraud_service


PLATFORM_SERVICE_URL = os.getenv("PLATFORM_SERVICE_URL", "http://localhost:8100").rstrip("/")
PLATFORM_TIMEOUT_SECONDS = float(os.getenv("PLATFORM_SERVICE_TIMEOUT", "1.5"))


def query_execution_trace(
    execution_id: str,
    base_url: str = PLATFORM_SERVICE_URL,
    timeout: float = PLATFORM_TIMEOUT_SECONDS,
) -> dict:
    """
    Read an execution trace from the platform service AS THE CALLER (GET /executions/{id}).
    Without the caller's own bearer token nothing is requested: the trace holds their plan
    and orders, and this component must not mint a token for some other user to read it.
    """
    authorization = get_caller_authorization()
    if not authorization:
        return {
            "available": False,
            "trace": None,
            "reason": "No signed-in caller; the plan behind this execution needs sign-in to read",
        }

    quoted = urllib.parse.quote(str(execution_id).strip(), safe="")
    req = urllib.request.Request(
        f"{base_url}/executions/{quoted}",
        headers={"Accept": "application/json", "Authorization": authorization},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return {
                "available": True,
                "trace": json.loads(response.read().decode("utf-8")),
                "reason": "Execution trace read from the platform service",
            }
    except urllib.error.HTTPError as err:
        reason = (
            "No execution with this ID belongs to the signed-in user"
            if err.code == 404
            else f"Platform service returned HTTP {err.code}"
        )
        return {"available": True, "trace": None, "reason": reason}
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError) as err:
        return {"available": False, "trace": None, "reason": f"Cannot reach the platform service ({err})"}


def _plan_section(trace: dict, explanation_enabled: bool) -> str:
    plan = trace.get("plan") or {}
    orders = "\n".join(
        f"- {o['side']} {o['quantity']:,.4f} {o['symbol']} at {o['price']:,.4f}"
        for o in trace.get("orders") or []
    ) or "- none"
    lines = [
        f"Kind: {trace.get('kind')}",
        f"Amount: {trace.get('currency')} {float(trace.get('amount', 0.0)):,.2f}",
        f"Component 1 Model Version: {trace.get('source_model_version')}",
        f"Plan Hash (anchored by Component 3): {trace.get('plan_hash')}",
        "Orders:",
        orders,
    ]
    if explanation_enabled:
        if plan.get("selection_rationale"):
            lines += ["Allocation Rationale (Component 1, MOEA/D):", str(plan["selection_rationale"])]
        rules = plan.get("fuzzy_rule_trace") or []
        if rules:
            lines.append("Fuzzy Rules That Fired (Component 1):")
            lines += [f"- {json.dumps(rule, sort_keys=True, default=str)}" for rule in rules[:8]]
        steps = plan.get("agent_reasoning_trace") or []
        if steps:
            lines.append("Component 1 Agent Steps:")
            lines += [f"- {json.dumps(step, sort_keys=True, default=str)}" for step in steps[:5]]
    else:
        lines.append("Component 1 rule-level explanation suppressed by user preference (AI explanation disabled).")
    return "\n".join(lines)


def _fraud_section(entry: dict | None, reason: str) -> str:
    if entry is None:
        return f"No live Component 2 decision available ({reason})."
    factors = "\n".join(
        f"- {f.get('name')}: contribution {float(f.get('contribution', 0.0)):+.4f}"
        for f in entry.get("top_factors") or []
    ) or "- none returned"
    return "\n".join([
        f"Decision: {entry.get('decision')}",
        f"Fraud Score: {float(entry.get('risk_score', 0.0)):.4f}",
        f"Model Version: {entry.get('model_version')}"
        + ("" if entry.get("benchmark_model") else " (NOT a benchmark model)"),
        f"Gateway Rules: {entry.get('reason')}",
        "Feature Attributions (Component 2's own):",
        factors,
    ])


def _audit_section(result: dict) -> str:
    if not result.get("available"):
        return f"Component 3 unreachable ({result.get('error')})."
    data = result.get("data") or {}
    if result.get("status_code") == 404 or data.get("status") == "NOT_FOUND":
        return "Component 3 holds no audit record for this execution."
    details = data.get("details") or {}
    payload = details.get("canonicalPayload") or {}
    return "\n".join([
        f"Verification Status: {data.get('status')}",
        f"Integrity Verified: {data.get('verified')}",
        f"Final Decision Anchored: {payload.get('finalDecision', 'N/A')}",
        f"Upstream Decision Anchored: {payload.get('upstreamDecision', 'N/A')}",
        f"Plan Hash Anchored: {payload.get('planHash', 'N/A')}",
        f"On-Chain Record Hash: {details.get('blockchainOnChainHash', 'N/A')}",
        f"Ledger: {'simulated adapter' if (details.get('onChainData') or {}).get('isMock') else 'configured chain'}",
    ])


@tool
def get_execution_explanation(execution_id: str) -> str:
    """
    Explain an executed (or held, blocked, pending) portfolio plan end to end, using its
    execution ID (it starts with EXE-). Combines why Component 1 proposed the plan, why
    Component 2's fraud gateway decided as it did, and whether Component 3's audit record
    for it verifies. Use this whenever the user asks about an execution ID or why their
    withdrawal or rebalance was executed, blocked, held, or sent for verification.
    """
    target_id = str(execution_id or "").strip().strip("'\"")
    if not target_id.startswith("EXE-"):
        return (
            "An execution ID starts with 'EXE-' and is shown on the withdrawal or rebalance "
            f"result. '{target_id}' is not one; ask the user for the execution ID."
        )

    explanation_enabled = get_current_preferences().get("explanation_enabled", True)
    trace_res = query_execution_trace(target_id)
    fraud_res = query_live_fraud_service(target_id)
    audit_res = query_live_audit_service(target_id)
    trace = trace_res.get("trace")

    status_block = (
        f"Execution Status: {trace.get('status')}\nStatus Reason: {trace.get('status_reason')}"
        if trace
        else f"Execution Status: unknown ({trace_res.get('reason')})"
    )
    plan_block = _plan_section(trace, explanation_enabled) if trace else (
        f"Component 1 plan not available: {trace_res.get('reason')}."
    )
    from governance_store import default_governance_store
    gov_decision = default_governance_store.get_latest_decision(target_id)
    gov_block = ""
    if gov_decision:
        gov_status = gov_decision.get("decision_type", "").upper()
        gov_block = f"""
4. HUMAN OVERSIGHT GOVERNANCE STATUS:
- Recorded State: {gov_status} by user (Decision ID: {gov_decision.get('decision_id')})
- Timestamp: {gov_decision.get('timestamp')}
- Enforcement Rule: {"EXECUTION BLOCKED — Human oversight rejected this plan; executing real transactions is strictly prohibited." if gov_status == "REJECTED" else "Human oversight confirmed this decision."}
"""

    return f"""
EXECUTION EXPLANATION — LIVE CROSS-COMPONENT TRACE

Execution ID:
{target_id}

{status_block}

1. WHY THIS PLAN (Component 1, portfolio optimization):
{plan_block}

2. WHY THIS DECISION (Component 2, fraud gateway; deterministic, no language model):
{_fraud_section(fraud_res.get("entry"), fraud_res.get("reason", ""))}

3. PROOF IT WAS NOT ALTERED (Component 3, audit anchor):
{_audit_section(audit_res)}
{gov_block}
PROVENANCE NOTICE:
- Each section is read live from the component that produced it; none is generated here.
- Component 4's local synthetic Random Forest SHAP prototype is NOT used for this execution.
- Probability calibration not established.
"""
