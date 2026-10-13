
import json
from pathlib import Path

from langchain_core.tools import tool
from xai.shap_explainer import explain_transaction


DATA_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "mock_financial_data.json"
)


def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


@tool
def get_fraud_analysis(transaction_id: str = "TX1001") -> str:
    """
    Retrieve mock transaction information and a SHAP explanation
    from the local prototype model.

    Use this tool when explaining a transaction's fraud risk.
    If no ID is supplied, use TX1001.
    """

    data = load_data()
    transaction = data["transactions"].get(transaction_id)

    if transaction is None:
        return f"No transaction was found for ID {transaction_id}."

    reasons = transaction.get("reasons", [])

    # These feature values are inferred from mock reason labels.
    # They are NOT verified transaction telemetry.
    reason_text = " ".join(reasons).lower()

    new_device = int(
        "new device" in reason_text
        or "unfamiliar device" in reason_text
    )

    unusual_location = int(
        "unusual location" in reason_text
        or "unfamiliar location" in reason_text
    )

    amount_lkr = float(transaction["amount_lkr"])

    # Explain using the local synthetic prototype model.
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
FRAUD ANALYSIS — MOCK TRANSACTION

Transaction ID:
{transaction_id}

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
The SHAP values explain a locally trained synthetic Random Forest,
not the model that produced the recorded fraud score above.
Device and location inputs were inferred from mock reason labels.
These results are prototype evidence, not verified production output.
"""
