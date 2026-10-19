
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
def get_fraud_analysis(transaction_id: str = "TX1001") -> str:
    """
    Retrieve fraud analysis and risk assessment for a transaction.
    Attempts a safe read-only query against Component 2's live audit log,
    falling back to mock data if the transaction is not found or the service is unavailable.

    Use this tool when explaining a transaction's fraud risk.
    If no ID is supplied, use TX1001.
    """
    prefs = get_current_preferences()
    explanation_enabled = prefs.get("explanation_enabled", True)
    target_id = str(transaction_id).strip() if transaction_id else "TX1001"

    # Query Component 2's live decision store (safe read-only GET /v1/transactions/{id})
    live_res = query_live_fraud_service(target_id)

    # 1. LIVE SERVICE RECORD FOUND
    if live_res.get("available") and live_res.get("entry") is not None:
        entry = live_res["entry"]
        tx_id = entry.get("transaction_id", target_id)
        user_id = entry.get("user_id", "N/A")
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
            f"- {f.get('name')}: contribution {float(f.get('contribution', 0.0)):+.4f} "
            f"({'increases' if float(f.get('contribution', 0.0)) > 0 else 'decreases'} fraud risk)"
            for f in top_factors
        ) or "- None returned (the untrained heuristic scores on amount and reputation only)"
        model_line = (
            "trained bundle validated on a public benchmark"
            if benchmark_model
            else "NOT a benchmark model (untrained heuristic or unvalidated bundle)"
        )

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

TRANSACTION TELEMETRY NOTE:
Raw transaction telemetry (amount, device fingerprint, geographic location, keystroke dynamics) is evaluated during ingestion and is not retained in the gateway audit trail.

MODEL PROVENANCE & EXPLANATION NOTICE:
- The fraud score ({risk_score:.4f}) and decision ({decision}) were produced by Component 2's dual-stream engine and deterministic security gateway.
- Component 4's local synthetic Random Forest prototype is NOT used for this live transaction.
- Local SHAP attributions are not computed for live records because Component 4's prototype does not model Component 2's LSTM/GNN architecture and the live audit trail does not retain raw tabular inputs.
- Probability calibration not established.
"""

    # 2. FALLBACK TO MOCK DATA (SERVICE OFFLINE, NOT FOUND, TIMED OUT, OR MALFORMED)
    fallback_reason = live_res.get("reason") or "Live Component 2 service returned no record"

    data = load_data()
    transaction = data.get("transactions", {}).get(target_id)

    if transaction is None:
        return (
            f"No transaction was found for ID '{target_id}' in either live Component 2 "
            f"audit records or fallback mock data. (Live service note: {fallback_reason})"
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

        shap_section = f"""LOCAL SHAP PROTOTYPE — NOT THE TEAM MODEL

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
FRAUD ANALYSIS — FALLBACK MOCK DATA

Data Source:
Local mock baseline (mock_financial_data.json).
Live service status ({FRAUD_SERVICE_URL}): {fallback_reason}.

FALLBACK NOTICE:
This is FALLBACK MOCK DATA because the live Component 2 service did not provide a record for {target_id}.
Reason: {fallback_reason}.
This record is NOT a live lookup result from Component 2.

Transaction ID:
{target_id}

Transaction Amount:
LKR {amount_lkr:,.2f}

Recorded Fraud Score:
{transaction["fraud_score"]}

Recorded Risk Level:
{transaction["risk_level"]}

Recorded Status:
{transaction["status"]}

RECORDED CONTRIBUTING FACTORS:
{reasons_text}

{shap_section}
"""

