import json
from pathlib import Path

from langchain_core.tools import tool


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
    Retrieve fraud detection information for a transaction.

    IMPORTANT:
    If the user asks why a transaction was blocked,
    suspicious, flagged, or detected as fraudulent,
    ALWAYS call this tool.

    If the user does not provide a transaction ID,
    use the default mock transaction ID TX1001.

    Do not simply tell the user that the tool exists.
    Actually call the tool and use its returned information.
    """

    data = load_data()

    transaction = data["transactions"].get(transaction_id)

    if transaction is None:
        return f"No transaction was found for ID {transaction_id}."

    reasons = "\n".join(
        f"- {reason}" for reason in transaction["reasons"]
    )

    return f"""
FRAUD ANALYSIS

Transaction ID:
{transaction_id}

Transaction Amount:
LKR {transaction["amount_lkr"]:,}

Fraud Score:
{transaction["fraud_score"]}

Risk Level:
{transaction["risk_level"]}

Status:
{transaction["status"]}

CONTRIBUTING FACTORS:

{reasons}

MODEL INTERPRETATION:

The combination of the above factors increased
the transaction's fraud risk score.
"""