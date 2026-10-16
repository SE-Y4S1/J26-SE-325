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
    clean_id = (
        str(transaction_id).strip().strip("'\"")
        if transaction_id and str(transaction_id).strip().lower() not in ("none", "null", "")
        else "TX1001"
    )
    quoted_id = urllib.parse.quote(clean_id, safe="")
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
    Retrieve and verify blockchain audit information for a financial transaction.

    Parameters:
    - transaction_id: Transaction identifier to verify on the blockchain audit bridge
      (e.g., 'PP1-DEMO-001', 'TX1001'). If the user mentions a specific transaction ID in their
      message, pass that exact transaction ID. If no transaction ID is mentioned, use 'TX1001'.
    """
    target_id = (
        str(transaction_id).strip().strip("'\"")
        if transaction_id and str(transaction_id).strip().lower() not in ("none", "null", "")
        else "TX1001"
    )

    mock_data = load_data()
    mock_audit = mock_data.get("audit", {}).get(target_id)

    live_res = query_live_audit_service(target_id)

    status_code = live_res.get("status_code")
    is_404 = status_code == 404
    payload = live_res.get("data") if isinstance(live_res.get("data"), dict) else {}
    verified = bool(payload.get("verified", False))
    status = str(payload.get("status", "NOT_FOUND" if is_404 else "UNKNOWN")).strip().upper()
    details = payload.get("details") or {}
    message = str(payload.get("message", ""))

    # 1. LIVE SERVICE RETURNED A VALID RESPONSE (OR EXPLICIT 404 NOT FOUND)
    if is_404 or (live_res.get("available") and live_res.get("data") is not None):
        # Case 1A: Record Not Found on Live Bridge
        if is_404 or status == "NOT_FOUND" or (not verified and "not found" in message.lower()):
            if mock_audit:
                return f"""
BLOCKCHAIN AUDIT RECORD — FALLBACK DATA

Data Source:
Local mock baseline (live audit service responded: record not found).

Transaction ID:
{target_id}

Audit ID:
{mock_audit.get("audit_id", f"AUDIT-{target_id}")}

Decision:
{mock_audit.get("decision", "RECORDED_BLOCKED")}

Recorded At:
{mock_audit.get("timestamp", "N/A")}

Block Number:
{mock_audit.get("block_number", 0)}

Record Status:
NOT_FOUND_ON_LIVE_BRIDGE

FALLBACK NOTICE:
The live audit bridge at {AUDIT_SERVICE_URL} was queried, but no audit record was found for {target_id}.
The record above is local mock baseline data and is NOT a live lookup result. Independent blockchain verification is not established.
"""
            return f"""
BLOCKCHAIN AUDIT RECORD — NOT FOUND

Transaction ID:
{target_id}

Record Status:
NOT_FOUND

STATUS:
The live audit service at {AUDIT_SERVICE_URL} confirmed that no audit record exists for transaction {target_id}.
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
            audit_id = str(on_chain.get("blockchainTxId") or details.get("blockchainTransactionId") or f"AUDIT-{target_id}")
            block_number = on_chain.get("blockNumber") or details.get("blockNumber") or 0

            return f"""
BLOCKCHAIN AUDIT RECORD — INTEGRITY VERIFICATION FAILED

Data Source:
Live audit bridge ({AUDIT_SERVICE_URL}).

Transaction ID:
{target_id}

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
Cryptographic integrity verification failed for transaction {target_id}.
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
            audit_id = str(on_chain.get("blockchainTxId") or details.get("blockchainTransactionId") or f"AUDIT-{target_id}")
            block_number = on_chain.get("blockNumber") or details.get("blockNumber") or 1

            if is_mock:
                return f"""
BLOCKCHAIN AUDIT RECORD — SIMULATED LEDGER

Data Source:
Live audit service ({AUDIT_SERVICE_URL}) — Simulated/Mock Ledger Adapter.

Transaction ID:
{target_id}

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
{target_id}

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
{target_id}

Audit ID:
{mock_audit.get("audit_id", f"AUDIT-{target_id}")}

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
{target_id}

Record Status:
UNAVAILABLE

STATUS:
The live blockchain audit service at {AUDIT_SERVICE_URL} was unreachable ({reason}).
No fallback mock baseline is available for transaction ID {target_id}.
"""