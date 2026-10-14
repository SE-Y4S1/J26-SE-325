import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from langchain_core.tools import tool


DATA_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "mock_financial_data.json"
)

AUDIT_SERVICE_URL = os.getenv("AUDIT_SERVICE_URL", "http://localhost:8002").rstrip("/")
AUDIT_TIMEOUT_SECONDS = float(os.getenv("AUDIT_SERVICE_TIMEOUT", "1.5"))


def load_data():
    if not DATA_FILE.exists():
        return {}
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def query_live_audit_service(transaction_id: str, base_url: str = AUDIT_SERVICE_URL, timeout: float = AUDIT_TIMEOUT_SECONDS) -> dict:
    """
    Query the live blockchain audit service for transaction verification.
    Returns a dict with query status and payload/error details.
    """
    quoted_id = urllib.parse.quote(str(transaction_id), safe="")
    url = f"{base_url}/api/audit/{quoted_id}/verify"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status_code = getattr(response, "status", 200)
            raw_bytes = response.read()
            text = raw_bytes.decode("utf-8")
            try:
                data = json.loads(text)
                return {
                    "available": True,
                    "status_code": status_code,
                    "data": data,
                    "error": None,
                }
            except json.JSONDecodeError:
                return {
                    "available": True,
                    "status_code": status_code,
                    "data": None,
                    "error": "Malformed JSON returned by audit service",
                }
    except urllib.error.HTTPError as err:
        try:
            error_body = err.read().decode("utf-8")
            data = json.loads(error_body)
        except Exception:
            data = None
        return {
            "available": True,
            "status_code": err.code,
            "data": data,
            "error": f"HTTP {err.code}: {err.reason}",
        }
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as err:
        return {
            "available": False,
            "status_code": None,
            "data": None,
            "error": f"Service unavailable ({err})",
        }
    except Exception as err:
        return {
            "available": False,
            "status_code": None,
            "data": None,
            "error": f"Unexpected error querying audit service ({err})",
        }


@tool
def get_blockchain_audit(transaction_id: str = "TX1001") -> str:
    """
    Retrieve audit information for a financial transaction.

    Use this tool when the user asks about:
    blockchain records, audit history, transaction verification,
    or recorded financial decisions.
    """

    mock_data = load_data()
    mock_audit = mock_data.get("audit", {}).get(transaction_id)

    live_res = query_live_audit_service(transaction_id)

    # 1. LIVE SERVICE RETURNED A VALID RESPONSE
    if live_res["available"] and live_res["data"] is not None:
        payload = live_res["data"]
        verified = bool(payload.get("verified", False))
        status = str(payload.get("status", "UNKNOWN")).strip().upper()
        details = payload.get("details") or {}
        message = str(payload.get("message", ""))

        # Case 1A: Record Not Found on Live Bridge
        if status == "NOT_FOUND" or (not verified and "not found" in message.lower()):
            if mock_audit:
                return f"""
BLOCKCHAIN AUDIT RECORD — FALLBACK DATA

Data Source:
Local mock baseline (live audit service responded: record not found).

Transaction ID:
{transaction_id}

Audit ID:
{mock_audit.get("audit_id", f"AUDIT-{transaction_id}")}

Decision:
{mock_audit.get("decision", "RECORDED_BLOCKED")}

Recorded At:
{mock_audit.get("timestamp", "N/A")}

Block Number:
{mock_audit.get("block_number", 0)}

Record Status:
NOT_FOUND_ON_LIVE_BRIDGE

FALLBACK NOTICE:
The live audit bridge at {AUDIT_SERVICE_URL} was queried, but no audit record was found for {transaction_id}.
The record above is local mock baseline data and is NOT a live lookup result. Independent blockchain verification is not established.
"""
            return f"""
BLOCKCHAIN AUDIT RECORD — NOT FOUND

Transaction ID:
{transaction_id}

Record Status:
NOT_FOUND

STATUS:
The live audit service at {AUDIT_SERVICE_URL} confirmed that no audit record exists for transaction {transaction_id}.
No fallback mock baseline is available for this transaction ID.
"""

        # Case 1B: Integrity Verification Failed (Tampering or Hash Mismatch)
        is_tampered = bool(details.get("isTampered", False))
        if not verified or status == "INTEGRITY_FAILED" or is_tampered:
            stored_hash = str(details.get("storedOffChainHash") or "N/A")
            calculated_hash = str(details.get("calculatedCurrentHash") or "N/A")
            blockchain_hash = str(details.get("blockchainOnChainHash") or "N/A")
            canonical = details.get("canonicalPayload") or {}
            on_chain = details.get("onChainData") or {}

            decision = str(canonical.get("finalDecision") or canonical.get("aiDecision") or on_chain.get("decision") or "UNKNOWN")
            timestamp = str(canonical.get("timestamp") or on_chain.get("timestamp") or "N/A")
            audit_id = str(on_chain.get("blockchainTxId") or details.get("blockchainTransactionId") or f"AUDIT-{transaction_id}")
            block_number = on_chain.get("blockNumber") or details.get("blockNumber") or 0

            return f"""
BLOCKCHAIN AUDIT RECORD — INTEGRITY VERIFICATION FAILED

Data Source:
Live audit bridge ({AUDIT_SERVICE_URL}).

Transaction ID:
{transaction_id}

Audit ID:
{audit_id}

Decision:
{decision}

Recorded At:
{timestamp}

Block Number:
{block_number}

Record Status:
INTEGRITY_FAILED

INTEGRITY ALERT:
Cryptographic integrity verification failed for transaction {transaction_id}.
The calculated SHA-256 hash does not match the blockchain commitment.
Tampering Detected: {is_tampered}
Stored Off-Chain Hash: {stored_hash}
Calculated Hash: {calculated_hash}
Blockchain Hash: {blockchain_hash}
"""

        # Case 1C: Verified Record
        if verified and status == "VERIFIED":
            on_chain = details.get("onChainData") or {}
            canonical = details.get("canonicalPayload") or {}
            is_mock = bool(on_chain.get("isMock", True))
            network = str(on_chain.get("network") or ("Configured Blockchain" if not is_mock else "Simulated Local Blockchain Adapter"))
            stored_hash = str(details.get("storedOffChainHash") or on_chain.get("recordHash") or "N/A")

            decision = str(on_chain.get("decision") or canonical.get("finalDecision") or canonical.get("aiDecision") or "VERIFIED")
            timestamp = str(canonical.get("timestamp") or on_chain.get("timestamp") or "N/A")
            audit_id = str(on_chain.get("blockchainTxId") or details.get("blockchainTransactionId") or f"AUDIT-{transaction_id}")
            block_number = on_chain.get("blockNumber") or details.get("blockNumber") or 1

            if is_mock:
                return f"""
BLOCKCHAIN AUDIT RECORD — SIMULATED LEDGER

Data Source:
Live audit service ({AUDIT_SERVICE_URL}) — Simulated/Mock Ledger Adapter.

Transaction ID:
{transaction_id}

Audit ID:
{audit_id}

Decision:
{decision}

Recorded At:
{timestamp}

Block Number:
{block_number}

Record Status:
VERIFIED

Ledger Type:
Simulated Mock Ledger Adapter (isMock: true)

Cryptographic Verification:
Record SHA-256 hash matches the simulated ledger commitment:
{stored_hash}

LIMITATION:
The record's cryptographic integrity was verified against the audit service's local simulated ledger.
Independent real-world blockchain verification is not established.
"""
            else:
                return f"""
BLOCKCHAIN AUDIT RECORD — CONFIGURED BLOCKCHAIN

Data Source:
Live audit service ({AUDIT_SERVICE_URL}) — Configured Blockchain ({network}).

Transaction ID:
{transaction_id}

Audit ID:
{audit_id}

Decision:
{decision}

Recorded At:
{timestamp}

Block Number:
{block_number}

Record Status:
VERIFIED

Ledger Type:
Configured Blockchain ({network}, isMock: false)

Cryptographic Verification:
Record SHA-256 hash matches the on-chain smart contract commitment:
{stored_hash}

NOTICE:
Verification established against the configured blockchain network. Independent external mainnet verification is not established.
"""

    # 2. LIVE SERVICE UNAVAILABLE, TIMED OUT, OR RETURNED UNEXPECTED/MALFORMED ERROR
    reason = live_res["error"] or "Unknown service error"

    if mock_audit:
        return f"""
BLOCKCHAIN AUDIT RECORD — FALLBACK DATA

Data Source:
Local mock baseline (live audit service at {AUDIT_SERVICE_URL} was unavailable: {reason}).

Transaction ID:
{transaction_id}

Audit ID:
{mock_audit.get("audit_id", f"AUDIT-{transaction_id}")}

Decision:
{mock_audit.get("decision", "RECORDED_BLOCKED")}

Recorded At:
{mock_audit.get("timestamp", "N/A")}

Block Number:
{mock_audit.get("block_number", 0)}

Record Status:
{mock_audit.get("status", "VERIFIED")}

FALLBACK NOTICE:
This record is local mock baseline data because the live audit service was unreachable ({reason}).
This is NOT a live lookup result. Independent blockchain verification is not established.
"""

    return f"""
BLOCKCHAIN AUDIT RECORD — SERVICE UNAVAILABLE

Transaction ID:
{transaction_id}

Record Status:
UNAVAILABLE

STATUS:
The live blockchain audit service at {AUDIT_SERVICE_URL} was unreachable ({reason}).
No fallback mock baseline is available for transaction ID {transaction_id}.
"""