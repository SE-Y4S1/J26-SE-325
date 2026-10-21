
from preferences_context import get_request_transaction_id
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from langchain_core.tools import tool
from preferences_context import get_current_preferences
from xai.shap_explainer import explain_transaction


DATA_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "mock_financial_data.json"
)

FRAUD_SERVICE_URL = os.getenv("FRAUD_SERVICE_URL", "http://localhost:8001").rstrip("/")
FRAUD_TIMEOUT_SECONDS = float(os.getenv("FRAUD_SERVICE_TIMEOUT", "1.5"))


def load_data():
    if not DATA_FILE.exists():
        return {}
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def _normalise_engine_response(body: dict) -> dict:
    """Map Component 2's /v1/transactions/{id} response onto the entry fields used below."""
    explanation = body.get("explanation") or {}
    streams = explanation.get("streams") or {}
    rules = [str(r) for r in explanation.get("rules") or []]
    notes = [str(n) for n in explanation.get("notes") or []]
    decision = str(body.get("decision", "UNKNOWN"))
    return {
        "transaction_id": body.get("transaction_id"),
        # The engine never returns a user identifier; it is keyed by a pseudonymous account.
        "user_id": "not disclosed by Component 2",
        "risk_score": body.get("fraud_score", 0.0),
        # The gateway spells it STEP_UP; the risk mapping below uses the older STEP-UP.
        "decision": "STEP-UP" if decision == "STEP_UP" else decision,
        "reason": "; ".join(rules) if rules else "; ".join(notes) or "No gateway rule fired",
        "behavioral_score": streams.get("lstm", streams.get("heuristic", 0.0)),
        "graph_score": streams.get("gnn", 0.0),
        "timestamp": "N/A",
        "model_version": body.get("model_version", "unknown"),
        "benchmark_model": bool(body.get("benchmark_model", False)),
        "top_factors": explanation.get("top_factors") or [],
        "attack_type": None,
    }


def query_live_fraud_service(
    transaction_id: str,
    base_url: str = FRAUD_SERVICE_URL,
    timeout: float = FRAUD_TIMEOUT_SECONDS,
) -> dict:
    """
    Query Component 2's live service using a safe read-only request.
    Does NOT invoke POST /v1/transactions/score (no state mutation, no decision triggering).

    Queries GET {base_url}/v1/transactions/{transaction_id}, which returns the decision and
    explanation the gateway already produced for that transaction.
    Returns:
        {
            "available": bool,
            "status_code": int | None,
            "entry": dict | None,
            "error": str | None,
            "reason": str,
        }
    """
    target_id = str(transaction_id).strip()
    url = f"{base_url}/v1/transactions/{urllib.parse.quote(target_id, safe='')}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status_code = getattr(response, "status", 200)
            text = response.read().decode("utf-8")
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                return {
                    "available": True,
                    "status_code": status_code,
                    "entry": None,
                    "error": "Malformed JSON returned by Component 2",
                    "reason": "Component 2 returned non-JSON payload",
                }
            if not isinstance(data, dict) or "decision" not in data:
                return {
                    "available": True,
                    "status_code": status_code,
                    "entry": None,
                    "error": "Unexpected payload returned by Component 2",
                    "reason": "Component 2 returned a payload without a decision",
                }
            return {
                "available": True,
                "status_code": status_code,
                "entry": _normalise_engine_response(data),
                "error": None,
                "reason": "Record found in Component 2 live decision store",
            }

    except urllib.error.HTTPError as err:
        if err.code == 404:
            return {
                "available": True,
                "status_code": 404,
                "entry": None,
                "error": None,
                "reason": f"Transaction '{target_id}' not found in Component 2 live decision store",
            }
        try:
            detail = json.loads(err.read().decode("utf-8")).get("detail", str(err))
        except Exception:
            detail = str(err)
        return {
            "available": True,
            "status_code": err.code,
            "entry": None,
            "error": f"HTTP {err.code}: {detail}",
            "reason": f"Component 2 returned HTTP {err.code} ({detail})",
        }
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as err:
        return {
            "available": False,
            "status_code": None,
            "entry": None,
            "error": f"Service unavailable ({err})",
            "reason": f"Cannot connect to Component 2 at {base_url} ({err})",
        }
    except Exception as err:
        return {
            "available": False,
            "status_code": None,
            "entry": None,
            "error": f"Unexpected error querying Component 2 ({err})",
            "reason": f"Query failure: {err}",
        }


@tool
def get_fraud_analysis(transaction_id: str = "", demo_mode: bool = False) -> str:
    """
    Retrieve fraud analysis and risk assessment for a specific transaction ID.
    Performs a safe read-only query against Component 2's live decision store (GET /v1/transactions/{id}).
    
    Parameters:
    - transaction_id: The specific transaction identifier (e.g., 'PP1-DEMO-001', 'EXE-1', 'TX1001').
      Must be provided. Never invent or assume a default transaction ID when none is specified.
    - demo_mode: Set to True ONLY when explicitly requested for synthetic demonstration purposes.
    """
    prefs = get_current_preferences()
    explanation_enabled = prefs.get("explanation_enabled", True)

    raw_id = (transaction_id or "").strip()
    if not raw_id:
        req_id = get_request_transaction_id()
        if req_id:
            raw_id = req_id.strip()

    from preferences_context import get_request_demo_mode
    effective_demo_mode = bool(demo_mode or prefs.get("demo_mode", False) or get_request_demo_mode())

    if not raw_id:
        return (
            "TRANSACTION ID REQUIRED:\n"
            "No specific transaction ID was provided. Please specify the transaction ID "
            "(for example, a TX... or execution ID) to inspect fraud analysis."
        )

    target_id = raw_id

    # Authorization Check: if querying an execution ID (starts with EXE- / exe-)
    from request_auth import get_caller_authorization
    authorization = get_caller_authorization()
    if target_id.upper().startswith("EXE-"):
        from tools.execution_tools import query_execution_trace
        trace_res = query_execution_trace(target_id)
        if trace_res.get("available") and not trace_res.get("trace"):
            reason = trace_res.get("reason", "")
            return (
                f"AUTHORIZATION DENIED:\n"
                f"Execution '{target_id}' was not found or does not belong to the authenticated user "
                f"({reason}). Unauthorized users cannot inspect another user's execution or fraud decision."
            )

    from governance_store import default_governance_store
    gov_decision = default_governance_store.get_latest_decision(target_id)
    gov_section = ""
    if gov_decision:
        gov_status = gov_decision.get("decision_type", "").upper()
        gov_section = f"""
HUMAN OVERSIGHT GOVERNANCE STATUS:
- Recorded Human Decision: {gov_status} (Decision ID: {gov_decision.get('decision_id')})
- Decision Timestamp: {gov_decision.get('timestamp')}
- Policy Enforcement: {"ACTION REJECTED BY USER — Financial execution is strictly blocked; do not execute real financial transactions." if gov_status == "REJECTED" else "Action confirmed by human oversight."}
"""

    # Query Component 2's live decision store (safe read-only GET /v1/transactions/{id})
    live_res = query_live_fraud_service(target_id)

    # 1. LIVE SERVICE RECORD FOUND
    if live_res.get("available") and live_res.get("entry") is not None:
        entry = live_res["entry"]
        tx_id = entry.get("transaction_id", target_id)
        user_id = entry.get("user_id", "not disclosed by Component 2")
        risk_score = float(entry.get("risk_score", 0.0))
        decision = str(entry.get("decision", "UNKNOWN"))
        reason = str(entry.get("reason", "No recorded reason"))
        behavioral_score = float(entry.get("behavioral_score", 0.0))
        graph_score = float(entry.get("graph_score", 0.0))
        timestamp = str(entry.get("timestamp", "N/A"))
        attack_type = entry.get("attack_type")
        model_version = str(entry.get("model_version", "unknown"))
        benchmark_model = bool(entry.get("benchmark_model", False))
        top_factors = entry.get("top_factors") or []

        # Deterministic risk level mapping from gateway decision
        if decision == "BLOCK":
            risk_level = "HIGH"
        elif decision == "STEP-UP":
            risk_level = "MODERATE"
        elif decision == "ALLOW":
            risk_level = "LOW"
        else:
            risk_level = "UNKNOWN"

        attack_line = f"- Tagged Scenario: {attack_type}\n" if attack_type else ""
        factor_lines = "\n".join(
            f"- {f.get('name')}: contribution {float(f.get('contribution', 0.0)):+.4f} ({'increases' if float(f.get('contribution', 0.0)) > 0 else 'decreases'} fraud risk)"
            if isinstance(f, dict)
            else f"- {f}"
            for f in top_factors
        ) or "- None returned (the untrained heuristic scores on amount and reputation only)"
        model_line = (
            "trained bundle validated on a public benchmark"
            if benchmark_model
            else "NOT a benchmark model (untrained heuristic or unvalidated bundle)"
        )

        reason_lower = reason.lower()
        has_device_rule = "device" in reason_lower
        has_location_rule = "location" in reason_lower or "ip" in reason_lower
        has_amount_factor = any(
            isinstance(f, dict) and f.get("name") in ("TransactionAmt", "amount")
            for f in top_factors
        )

        device_signal_desc = (
            f"Supported by recorded rule ({reason})"
            if has_device_rule
            else "Unavailable (device fingerprint and new-device telemetry not collected or retained in gateway audit trail)"
        )
        location_signal_desc = (
            f"Supported by recorded rule ({reason})"
            if has_location_rule
            else "Unavailable (geographic location and IP coordinates not collected or retained in gateway audit trail)"
        )
        amt_factor_parts = [
            f"{f.get('name')}: {float(f.get('contribution', 0.0)):+.4f}"
            for f in top_factors
            if isinstance(f, dict) and f.get("name") in ("TransactionAmt", "amount")
        ]
        if has_amount_factor and amt_factor_parts:
            amount_signal_desc = f"Evaluated by gateway risk model / feature attribution ({', '.join(amt_factor_parts)})"
        else:
            amount_signal_desc = "Evaluated during gateway ingestion; threshold details not retained in audit trail"

        return f"""
FRAUD ANALYSIS — LIVE COMPONENT 2 AUDIT RECORD

Data Source:
Live Component 2 Gateway Decision Store ({FRAUD_SERVICE_URL}/v1/transactions).
This record is live governance evidence from Component 2's real-time fraud engine.

Transaction ID:
{tx_id}

User ID:
{user_id}

Recorded Fraud Score:
{risk_score:.4f}

Recorded Risk Level:
{risk_level}

Recorded Status:
{decision}

RECORDED CONTRIBUTING FACTORS:
- {reason}

COMPONENT 2 DUAL-STREAM MODEL SIGNALS:
- Behavioral Stream Anomaly Score (LSTM Autoencoder): {behavioral_score:.4f}
- Relational Stream Risk Score (GNN): {graph_score:.4f}
- Fused Final Risk Score: {risk_score:.4f}
- Model Version: {model_version} ({model_line})
- Audit Timestamp: {timestamp}
{attack_line}
COMPONENT 2 FEATURE ATTRIBUTIONS (the engine's own linear attributions, largest first):
{factor_lines}

GENUINE SIGNALS & TELEMETRY STATUS:
- Device Signals: {device_signal_desc}
- Location Signals: {location_signal_desc}
- Amount Risk Assessment: {amount_signal_desc}
- Keystroke Dynamics / Biometrics: Unavailable (not collected or retained)

MODEL PROVENANCE & EXPLANATION NOTICE:
- The fraud score ({risk_score:.4f}) and decision ({decision}) were produced by Component 2's dual-stream engine and deterministic security gateway.
- Component 4's local synthetic Random Forest prototype is NOT used for this live transaction.
- Local SHAP attributions are not computed for live records because Component 4's prototype does not model Component 2's LSTM/GNN architecture and the live audit trail does not retain raw tabular inputs.
- Probability calibration not established.
{gov_section}"""

    # 2. RECORD NOT FOUND OR LIVE SERVICE UNAVAILABLE
    fallback_reason = live_res.get("reason") or "Live Component 2 service returned no record"

    if not effective_demo_mode:
        status_label = "RECORD_NOT_FOUND" if live_res.get("status_code") == 404 else "SERVICE_UNAVAILABLE"
        return f"""
FRAUD ANALYSIS — RECORD NOT FOUND / SERVICE UNAVAILABLE

Data Source:
Component 2 Live Gateway Decision Store ({FRAUD_SERVICE_URL}/v1/transactions).

Transaction ID:
{target_id}

Lookup Status:
{status_label}

Live Service Detail:
{fallback_reason}

Recorded Fraud Score:
Unavailable (no verified live decision)

Recorded Risk Level:
Unavailable

Recorded Status:
Unavailable

SIGNALS & TELEMETRY STATUS:
- Device Signals: Unavailable
- Location Signals: Unavailable
- Amount Risk Assessment: Unavailable
- Keystroke Dynamics / Biometrics: Unavailable

EXPLANATION NOTICE:
No verified live fraud record exists for transaction '{target_id}' in Component 2's decision store ({fallback_reason}).
FinTrust Assistant does not substitute mock fallback data or fabricate fraud scores for unknown transactions.
If you wish to view synthetic baseline demonstration data, please request demo mode explicitly.
{gov_section}"""

    # 3. EXPLICIT DEMO MODE ONLY (VISIBLY LABELLED SYNTHETIC DEMO SPECIFICATION)
    data = load_data()
    transaction = data.get("transactions", {}).get(target_id)

    if transaction is None:
        return (
            f"No demo transaction was found for ID '{target_id}' in local synthetic demonstration data. "
            f"(Live service note: {fallback_reason})"
        )

    reasons = transaction.get("reasons", [])
    reason_text = " ".join(reasons).lower()

    new_device = int(
        "new device" in reason_text
        or "unfamiliar device" in reason_text
    )

    unusual_location = int(
        "unusual location" in reason_text
        or "unfamiliar location" in reason_text
    )

    amount_lkr = float(transaction.get("amount_lkr", 0.0))

    reasons_text = "\n".join(
        f"- {reason}" for reason in reasons
    ) or "- No contributing factors provided"

    if explanation_enabled:
        # Explain using the local synthetic prototype model
        shap_result = explain_transaction(
            new_device=new_device,
            unusual_location=unusual_location,
            amount_lkr=amount_lkr,
        )

        shap_lines = "\n".join(
            (
                f"- {item['feature']}: "
                f"SHAP value = {item['shap_value']:.6f}; "
                f"{item['impact']}"
            )
            for item in shap_result["shap_explanation"]
        )

        shap_section = f"""LOCAL SHAP PROTOTYPE — NOT THE PRODUCTION MODEL

Prototype prediction:
{shap_result["prediction"]}

Prototype fraud probability:
{shap_result["fraud_probability"]:.4f}

Prototype input features:
- New device: {new_device}
- Unusual location: {unusual_location}
- Amount (LKR): {amount_lkr:,.2f}

SHAP CONTRIBUTIONS:
{shap_lines}

LIMITATION:
The SHAP values explain a locally trained synthetic Random Forest prototype,
not the production model or Component 2 engine.
Device and location inputs were inferred from mock reason labels.
These results are prototype evidence, not verified production output.
Probability calibration not established."""
    else:
        shap_section = """AI EXPLANATION NOTICE:
Optional SHAP feature-attribution details were suppressed based on user preference (AI explanation disabled).
The recorded score and contributing factors reflect the recorded mock baseline without prototype SHAP attribution.

LIMITATION:
Device and location inputs were inferred from mock reason labels.
These results are baseline mock evidence, not verified production output.
Probability calibration not established."""

    return f"""
FRAUD ANALYSIS — DEMO / SYNTHETIC SPECIFICATION (EXPLICIT DEMO MODE)

Data Source:
Local mock baseline (mock_financial_data.json) — DEMO MODE EXPLICITLY ENABLED.
Live Component 2 status ({FRAUD_SERVICE_URL}): {fallback_reason}.

DEMO NOTICE:
This is SYNTHETIC DEMO EVIDENCE for demonstration purposes only.
It is NOT live transaction evidence from Component 2 and does NOT represent an authenticated user transaction.

Transaction ID:
{target_id} [DEMO]

Transaction Amount:
LKR {amount_lkr:,.2f} (Demo Baseline)

Recorded Fraud Score:
{transaction["fraud_score"]} (Demo Baseline)

Recorded Risk Level:
{transaction["risk_level"]} (Demo Baseline)

Recorded Status:
{transaction["status"]} (Demo Baseline)

RECORDED CONTRIBUTING FACTORS:
{reasons_text}

SIGNALS & TELEMETRY NOTE:
Device and location inputs were inferred from mock demo labels for prototype demonstration.
They are not verified real-world telemetry observations.

{shap_section}
{gov_section}"""


