
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from langchain_core.tools import tool
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


def query_live_fraud_service(
    transaction_id: str,
    base_url: str = FRAUD_SERVICE_URL,
    timeout: float = FRAUD_TIMEOUT_SECONDS,
) -> dict:
    """
    Query Component 2's live service using a safe read-only request.
    Does NOT invoke POST /score (no state mutation, no decision triggering).

    Queries GET {base_url}/audit?limit=200 to inspect the gateway's recorded decisions.
    Returns:
        {
            "available": bool,
            "status_code": int | None,
            "entry": dict | None,
            "error": str | None,
            "reason": str,
        }
    """
    url = f"{base_url}/audit?limit=200"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status_code = getattr(response, "status", 200)
            raw_bytes = response.read()
            text = raw_bytes.decode("utf-8")
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                return {
                    "available": True,
                    "status_code": status_code,
                    "entry": None,
                    "error": "Malformed JSON returned by Component 2 audit service",
                    "reason": "Component 2 returned non-JSON payload",
                }

            entries = data.get("entries", []) if isinstance(data, dict) else []
            target_id = str(transaction_id).strip()

            for entry in entries:
                if str(entry.get("transaction_id", "")).strip() == target_id:
                    return {
                        "available": True,
                        "status_code": status_code,
                        "entry": entry,
                        "error": None,
                        "reason": "Record found in Component 2 live audit log",
                    }

            return {
                "available": True,
                "status_code": status_code,
                "entry": None,
                "error": None,
                "reason": f"Transaction '{target_id}' not found in Component 2 live audit log ({len(entries)} recent entries checked)",
            }

    except urllib.error.HTTPError as err:
        try:
            err_body = err.read().decode("utf-8")
            err_data = json.loads(err_body)
            detail = err_data.get("detail", str(err))
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
    target_id = str(transaction_id).strip() if transaction_id else "TX1001"

    # Query Component 2's live audit service (safe read-only GET /audit)
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

        return f"""
FRAUD ANALYSIS — LIVE COMPONENT 2 AUDIT RECORD

Data Source:
Live Component 2 Gateway Audit Log ({FRAUD_SERVICE_URL}/audit).
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
- Behavioral Stream Anomaly Score (Simulated LSTM Autoencoder): {behavioral_score:.4f}
- Relational Stream Risk Score (Simulated GNN): {graph_score:.4f}
- Fused Final Risk Score: {risk_score:.4f}
- Audit Timestamp: {timestamp}
{attack_line}
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

    # Explain using the local synthetic prototype model
    shap_result = explain_transaction(
        new_device=new_device,
        unusual_location=unusual_location,
        amount_lkr=amount_lkr,
    )

    reasons_text = "\n".join(
        f"- {reason}" for reason in reasons
    ) or "- No contributing factors provided"

    shap_lines = "\n".join(
        (
            f"- {item['feature']}: "
            f"SHAP value = {item['shap_value']:.6f}; "
            f"{item['impact']}"
        )
        for item in shap_result["shap_explanation"]
    )

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

LOCAL SHAP PROTOTYPE — NOT THE TEAM MODEL

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
Probability calibration not established.
"""

