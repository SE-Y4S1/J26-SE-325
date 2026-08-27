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
def get_blockchain_audit(transaction_id: str = "TX1001") -> str:
    """
    Retrieve audit information for a financial transaction.

    Use this tool when the user asks about:
    blockchain records, audit history, transaction verification,
    or recorded financial decisions.
    """

    data = load_data()

    audit = data["audit"].get(transaction_id)

    if audit is None:
        return f"No audit record was found for {transaction_id}."

    return f"""
BLOCKCHAIN AUDIT RECORD

Transaction ID:
{transaction_id}

Audit ID:
{audit["audit_id"]}

Decision:
{audit["decision"]}

Recorded At:
{audit["timestamp"]}

Block Number:
{audit["block_number"]}

Record Status:
{audit["status"]}

The audit record indicates that the decision
was recorded and can be independently verified.
"""