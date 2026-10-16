import json
import unittest
from unittest.mock import patch, MagicMock
import urllib.error

from tools.audit_tools import query_live_audit_service, get_blockchain_audit
from evidence_validator import validate_evidence_claims


class TestAuditTools(unittest.TestCase):

    def test_missing_record_live_bridge_pp1_demo_001(self):
        """
        When Component 3 returns status: 'NOT_FOUND' for PP1-DEMO-001,
        get_blockchain_audit must report NOT_FOUND honestly without fabricating fallback data.
        """
        mock_response = {
            "verified": False,
            "status": "NOT_FOUND",
            "message": "Audit record for transaction PP1-DEMO-001 was not found in off-chain database."
        }

        with patch("tools.audit_tools.query_live_audit_service") as mock_query:
            mock_query.return_value = {
                "available": True,
                "status_code": 200,
                "data": mock_response,
                "error": None,
            }

            result = get_blockchain_audit.invoke({"transaction_id": "PP1-DEMO-001"})

            self.assertIn("BLOCKCHAIN AUDIT RECORD — NOT FOUND", result)
            self.assertIn("Transaction ID:\nPP1-DEMO-001", result)
            self.assertIn("Record Status:\nNOT_FOUND", result)
            self.assertIn("confirmed that no audit record exists", result)
            self.assertNotIn("FALLBACK DATA", result)
            self.assertNotIn("VERIFIED", result)

    def test_missing_record_with_mock_fallback_tx1001(self):
        """
        When Component 3 returns NOT_FOUND for TX1001 (which has local baseline data),
        it returns clearly labeled fallback data with status NOT_FOUND_ON_LIVE_BRIDGE.
        """
        mock_response = {
            "verified": False,
            "status": "NOT_FOUND",
            "message": "Audit record for transaction TX1001 was not found in off-chain database."
        }

        with patch("tools.audit_tools.query_live_audit_service") as mock_query:
            mock_query.return_value = {
                "available": True,
                "status_code": 200,
                "data": mock_response,
                "error": None,
            }

            result = get_blockchain_audit.invoke({"transaction_id": "TX1001"})

            self.assertIn("BLOCKCHAIN AUDIT RECORD — FALLBACK DATA", result)
            self.assertIn("Record Status:\nNOT_FOUND_ON_LIVE_BRIDGE", result)
            self.assertIn("NOT a live lookup result", result)
            self.assertIn("AUDIT-2026-001", result)

    def test_explicit_404_handled_as_not_found(self):
        """
        An HTTP 404 from the audit bridge must be treated as NOT_FOUND, not SERVICE UNAVAILABLE.
        """
        with patch("tools.audit_tools.query_live_audit_service") as mock_query:
            mock_query.return_value = {
                "available": True,
                "status_code": 404,
                "data": {"message": "Transaction not found"},
                "error": "HTTP 404: Not Found",
            }

            result = get_blockchain_audit.invoke({"transaction_id": "PP1-DEMO-001"})

            self.assertIn("BLOCKCHAIN AUDIT RECORD — NOT FOUND", result)
            self.assertIn("Record Status:\nNOT_FOUND", result)
            self.assertNotIn("SERVICE UNAVAILABLE", result)

    def test_verified_record_simulated_ledger(self):
        """
        When Component 3 returns a VERIFIED record on the simulated mock ledger,
        it is labeled as SIMULATED LEDGER with appropriate limitations disclosed.
        """
        mock_response = {
            "verified": True,
            "status": "VERIFIED",
            "message": "Audit record integrity successfully verified against blockchain commitment.",
            "details": {
                "transactionId": "PP1-DEMO-001",
                "storedOffChainHash": "0xabc123hash",
                "calculatedCurrentHash": "0xabc123hash",
                "blockchainOnChainHash": "0xabc123hash",
                "isTampered": False,
                "canonicalPayload": {
                    "finalDecision": "APPROVE",
                    "timestamp": "2026-10-16T12:00:00Z"
                },
                "onChainData": {
                    "isMock": True,
                    "network": "Simulated Local Blockchain Adapter",
                    "blockchainTxId": "0xmocktx999",
                    "blockNumber": 4050
                }
            }
        }

        with patch("tools.audit_tools.query_live_audit_service") as mock_query:
            mock_query.return_value = {
                "available": True,
                "status_code": 200,
                "data": mock_response,
                "error": None,
            }

            result = get_blockchain_audit.invoke({"transaction_id": "PP1-DEMO-001"})

            self.assertIn("BLOCKCHAIN AUDIT RECORD — SIMULATED LEDGER", result)
            self.assertIn("Transaction ID:\nPP1-DEMO-001", result)
            self.assertIn("Audit ID:\n0xmocktx999", result)
            self.assertIn("Block Number:\n4050", result)
            self.assertIn("Record Status:\nVERIFIED", result)
            self.assertIn("Simulated Mock Ledger Adapter (isMock: true)", result)
            self.assertIn("0xabc123hash", result)
            self.assertIn("Independent real-world blockchain verification is not established", result)

    def test_verified_record_configured_blockchain(self):
        """
        When Component 3 returns a VERIFIED record on a real/configured network,
        it is labeled as CONFIGURED BLOCKCHAIN.
        """
        mock_response = {
            "verified": True,
            "status": "VERIFIED",
            "message": "Audit record integrity successfully verified against blockchain commitment.",
            "details": {
                "transactionId": "PP1-DEMO-001",
                "storedOffChainHash": "0xrealonchainhash",
                "calculatedCurrentHash": "0xrealonchainhash",
                "blockchainOnChainHash": "0xrealonchainhash",
                "isTampered": False,
                "canonicalPayload": {
                    "finalDecision": "REJECT",
                    "timestamp": "2026-10-16T12:15:00Z"
                },
                "onChainData": {
                    "isMock": False,
                    "network": "Hardhat Local Network (ChainId: 31337)",
                    "blockchainTxId": "0x001122334455",
                    "blockNumber": 8821
                }
            }
        }

        with patch("tools.audit_tools.query_live_audit_service") as mock_query:
            mock_query.return_value = {
                "available": True,
                "status_code": 200,
                "data": mock_response,
                "error": None,
            }

            result = get_blockchain_audit.invoke({"transaction_id": "PP1-DEMO-001"})

            self.assertIn("BLOCKCHAIN AUDIT RECORD — CONFIGURED BLOCKCHAIN", result)
            self.assertIn("Decision:\nREJECT", result)
            self.assertIn("Block Number:\n8821", result)
            self.assertIn("Audit ID:\n0x001122334455", result)
            self.assertIn("Configured Blockchain (Hardhat Local Network (ChainId: 31337), isMock: false)", result)

    def test_integrity_failed_tampering_detected(self):
        """
        When Component 3 detects a hash mismatch or tampering,
        it alerts with INTEGRITY_FAILED and discloses hash values.
        """
        mock_response = {
            "verified": False,
            "status": "INTEGRITY_FAILED",
            "message": "ALERT: Cryptographic integrity failure detected!",
            "details": {
                "transactionId": "PP1-DEMO-001",
                "storedOffChainHash": "0xoriginalhash",
                "calculatedCurrentHash": "0xtamperedhash",
                "blockchainOnChainHash": "0xoriginalhash",
                "isTampered": True,
                "canonicalPayload": {
                    "finalDecision": "APPROVE",
                    "timestamp": "2026-10-16T10:00:00Z"
                },
                "onChainData": {
                    "blockchainTxId": "0xtx_tampered",
                    "blockNumber": 2210
                }
            }
        }

        with patch("tools.audit_tools.query_live_audit_service") as mock_query:
            mock_query.return_value = {
                "available": True,
                "status_code": 200,
                "data": mock_response,
                "error": None,
            }

            result = get_blockchain_audit.invoke({"transaction_id": "PP1-DEMO-001"})

            self.assertIn("BLOCKCHAIN AUDIT RECORD — INTEGRITY VERIFICATION FAILED", result)
            self.assertIn("Record Status:\nINTEGRITY_FAILED", result)
            self.assertIn("Tampering Detected: True", result)
            self.assertIn("Stored Off-Chain Hash: 0xoriginalhash", result)
            self.assertIn("Calculated Hash: 0xtamperedhash", result)

    def test_service_unavailable_error(self):
        """
        When Component 3 is offline, get_blockchain_audit reports SERVICE UNAVAILABLE
        for un-mocked transactions such as PP1-DEMO-001.
        """
        with patch("tools.audit_tools.query_live_audit_service") as mock_query:
            mock_query.return_value = {
                "available": False,
                "status_code": None,
                "data": None,
                "error": "Connection refused",
            }

            result = get_blockchain_audit.invoke({"transaction_id": "PP1-DEMO-001"})

            self.assertIn("BLOCKCHAIN AUDIT RECORD — SERVICE UNAVAILABLE", result)
            self.assertIn("Record Status:\nUNAVAILABLE", result)
            self.assertIn("Connection refused", result)

    def test_transaction_id_sanitization(self):
        """
        Ensures transaction ID formatting (whitespace, quotes, None) is handled cleanly.
        """
        with patch("tools.audit_tools.urllib.request.urlopen") as mock_urlopen:
            mock_cm = MagicMock()
            mock_cm.read.return_value = b'{"verified": false, "status": "NOT_FOUND"}'
            mock_cm.status = 200
            mock_urlopen.return_value.__enter__.return_value = mock_cm

            res = query_live_audit_service("  'PP1-DEMO-001'  ")
            self.assertTrue(res["available"])
            self.assertEqual(res["status_code"], 200)

            # Check that query string had clean ID
            req = mock_urlopen.call_args[0][0]
            self.assertIn("/api/audit/PP1-DEMO-001/verify", req.full_url)

    def test_evidence_validator_on_not_found_audit(self):
        """
        When evidence_validator catches an ungrounded claim on a NOT_FOUND audit response,
        it must NOT fabricate a verified audit record or mention TX1001 fraud score.
        """
        not_found_tool_result = (
            "BLOCKCHAIN AUDIT RECORD — NOT FOUND\n"
            "Transaction ID:\nPP1-DEMO-001\n"
            "Record Status:\nNOT_FOUND\n"
            "STATUS:\nThe live audit service confirmed that no audit record exists for transaction PP1-DEMO-001.\n"
        )
        evidence = [{"tool": "get_blockchain_audit", "result": not_found_tool_result}]

        # User LLM answer that made an invalid independent verification claim
        answer_with_issue = "The transaction PP1-DEMO-001 provides an independent verification on chain."

        validation = validate_evidence_claims(answer_with_issue, evidence)

        self.assertFalse(validation["passed"])
        self.assertIn("Unsupported blockchain independent verification claim", validation["issues"])
        # Crucially: it must NOT output fake TX1001 fraud or fake AUDIT-2026-001 verified data
        self.assertNotIn("TX1001", validation["answer"])
        self.assertNotIn("0.92", validation["answer"])
        self.assertNotIn("AUDIT-2026-001", validation["answer"])
        self.assertIn("PP1-DEMO-001", validation["answer"])
        self.assertIn("no audit record exists", validation["answer"].lower())


if __name__ == "__main__":
    unittest.main()
