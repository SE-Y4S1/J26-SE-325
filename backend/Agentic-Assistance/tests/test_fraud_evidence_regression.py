"""
Regression tests for Component 4 FinTrust Fraud Evidence grounding with Component 2.

Covers all 8 core regression requirements:
1. Unknown transaction ID never returns TX1001 mock data.
2. Live lookup failure never produces a fabricated score or explanation.
3. A real transaction returns its own recorded decision.
4. Unauthorized users cannot retrieve another user's transaction.
5. Missing device/location signals are not described as verified facts.
6. Mock mode works only when explicitly enabled and is visibly labelled.
7. SHAP prototype output is never represented as the production model's explanation.
8. "Why was my transaction blocked?" asks for clarification when no specific transaction can be resolved.
"""

import io
import json
import unittest
import urllib.error
from unittest.mock import patch, MagicMock

from tools.fraud_tools import get_fraud_analysis
from preferences_context import (
    set_request_demo_mode,
    reset_request_demo_mode,
    set_request_transaction_id,
    reset_request_transaction_id,
)
from request_auth import set_caller_authorization, reset_caller_authorization
from agent import classify_domain_gate, call_llm
from evidence_validator import validate_evidence_claims


class TestFraudEvidenceRegression(unittest.TestCase):
    def setUp(self):
        self._demo_token = set_request_demo_mode(False)
        self._tx_token = set_request_transaction_id(None)
        self._auth_token = set_caller_authorization(None)

    def tearDown(self):
        reset_request_demo_mode(self._demo_token)
        reset_request_transaction_id(self._tx_token)
        reset_caller_authorization(self._auth_token)

    @patch("urllib.request.urlopen")
    def test_1_unknown_transaction_never_returns_tx1001_mock_data(self, mock_urlopen):
        """1. Unknown transaction ID never returns TX1001 mock data."""
        mock_err = urllib.error.HTTPError(
            url="http://localhost:8001/v1/transactions/TX9999",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=io.BytesIO(b'{"detail": "No decision is held for this transaction."}'),
        )
        mock_urlopen.side_effect = mock_err

        result = get_fraud_analysis.invoke({"transaction_id": "TX9999"})

        self.assertIn("TX9999", result)
        self.assertIn("RECORD NOT FOUND", result)
        self.assertNotIn("TX1001", result)
        self.assertNotIn("0.92", result)
        self.assertNotIn("LKR 250,000", result)
        self.assertNotIn("Recorded Status: BLOCKED", result)

    @patch("urllib.request.urlopen")
    def test_2_live_lookup_failure_never_produces_fabricated_score(self, mock_urlopen):
        """2. Live lookup failure never produces a fabricated score or explanation."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused to Component 2")

        result = get_fraud_analysis.invoke({"transaction_id": "TX5555"})

        self.assertIn("SERVICE UNAVAILABLE", result)
        self.assertIn("TX5555", result)
        self.assertNotIn("0.92", result)
        self.assertNotIn("LKR 250,000", result)
        self.assertNotIn("Recorded Status: BLOCKED", result)
        self.assertNotIn("new device, unusual location", result)

    @patch("urllib.request.urlopen")
    def test_3_real_transaction_returns_own_recorded_decision(self, mock_urlopen):
        """3. A real transaction returns its own recorded decision."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({
            "transaction_id": "TX-LIVE-789",
            "fraud_score": 0.42,
            "decision": "REVIEW",
            "model_version": "fraud-detector-v2.1",
            "benchmark_model": True,
            "explanation": {
                "rules": ["Amount exceeds typical pattern"],
                "top_factors": [{"name": "TransactionAmt", "contribution": 0.42}],
                "streams": {
                    "lstm": 0.55,
                    "gnn": 0.38,
                    "fused": 0.42,
                },
                "notes": ["Decision held in Component 2 Gateway"],
            },
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        result = get_fraud_analysis.invoke({"transaction_id": "TX-LIVE-789"})

        self.assertIn("LIVE COMPONENT 2 AUDIT RECORD", result)
        self.assertIn("TX-LIVE-789", result)
        self.assertIn("0.4200", result)
        self.assertIn("REVIEW", result)
        self.assertIn("fraud-detector-v2.1", result)
        self.assertIn("trained bundle validated on a public benchmark", result)
        self.assertIn("0.5500", result)
        self.assertIn("0.3800", result)
        self.assertNotIn("TX1001", result)
        self.assertNotIn("0.92", result)

    @patch("tools.execution_tools.query_execution_trace")
    def test_4_unauthorized_user_cannot_retrieve_another_users_transaction(self, mock_trace):
        """4. Unauthorized users cannot retrieve another user's execution trace / transaction."""
        set_caller_authorization("Bearer bearer-token-for-user-123")
        mock_trace.return_value = {
            "available": True,
            "trace": None,
            "reason": "No execution with this ID belongs to the signed-in user",
        }

        result = get_fraud_analysis.invoke({"transaction_id": "EXE-OTHER-999"})

        self.assertIn("AUTHORIZATION DENIED", result)
        self.assertIn("EXE-OTHER-999", result)
        self.assertNotIn("0.92", result)

    def test_5_missing_device_location_signals_not_described_as_verified_facts(self):
        """5. Missing device/location signals are not described as verified facts."""
        live_evidence_without_telemetry = (
            "COMPONENT 2 TRANSACTION AUDIT RECORD\n"
            "Transaction ID: TX-AUDIT-101\n"
            "Recorded Fraud Score: 0.15\n"
            "Recorded Status: ALLOW\n"
            "TELEMETRY DATA STATUS:\n"
            "- Device Signals: Unavailable (device fingerprint not collected in gateway)\n"
            "- Location Signals: Unavailable (IP and location not collected in gateway)\n"
        )

        fabricated_claim = "The system detected that you used a new device."
        validation = validate_evidence_claims(
            fabricated_claim,
            [{"tool": "get_fraud_analysis", "result": live_evidence_without_telemetry}],
        )

        issues = validation.get("issues", [])
        self.assertGreater(len(issues), 0)
        self.assertTrue(
            any("device" in issue.lower() or "unverified" in issue.lower() for issue in issues)
        )

    @patch("urllib.request.urlopen")
    def test_6_mock_mode_works_only_when_explicitly_enabled(self, mock_urlopen):
        """6. Mock mode works only when explicitly enabled and is visibly labelled."""
        mock_err = urllib.error.HTTPError(
            url="http://localhost:8001/v1/transactions/TX1001",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=io.BytesIO(b'{"detail": "No decision is held for this transaction."}'),
        )
        mock_urlopen.side_effect = mock_err

        # Without demo mode -> 404 yields RECORD NOT FOUND, never fallback mock
        result_without_demo = get_fraud_analysis.invoke({"transaction_id": "TX1001"})
        self.assertIn("RECORD NOT FOUND", result_without_demo)
        self.assertNotIn("DEMO / SYNTHETIC SPECIFICATION", result_without_demo)

        # With demo mode explicitly enabled -> labeled demo specification
        set_request_demo_mode(True)
        result_with_demo = get_fraud_analysis.invoke({"transaction_id": "TX1001"})
        self.assertIn("DEMO / SYNTHETIC SPECIFICATION (EXPLICIT DEMO MODE)", result_with_demo)
        self.assertIn("DEMO NOTICE", result_with_demo)
        self.assertIn("SYNTHETIC DEMO EVIDENCE", result_with_demo)

    @patch("urllib.request.urlopen")
    def test_7_shap_prototype_suppressed_for_live_decisions(self, mock_urlopen):
        """7. SHAP prototype output is never represented as the production model's explanation."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({
            "transaction_id": "TX-PROD-001",
            "fraud_score": 0.88,
            "decision": "BLOCKED",
            "model_version": "fraud-detector-v2.1",
            "explanation": {
                "top_factors": ["High behavioral anomaly"],
                "streams": {"lstm": 0.91, "gnn": 0.84, "fused": 0.88},
            },
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        result = get_fraud_analysis.invoke({"transaction_id": "TX-PROD-001"})

        # Live records must suppress local prototype SHAP and disclaim it
        self.assertIn("Component 4's local synthetic Random Forest prototype is NOT used for this live transaction", result)
        self.assertNotIn("LOCAL PROTOTYPE SHAP ATTRIBUTION", result)

    def test_8_why_was_my_transaction_blocked_asks_clarification(self):
        """8. 'Why was my transaction blocked?' asks for clarification when no specific transaction can be resolved."""
        from langchain_core.messages import HumanMessage
        from agent import agent

        query = "Why was my transaction blocked?"
        gate = classify_domain_gate(query, history=[])
        self.assertEqual(gate.get("category"), "AMBIGUOUS_CLARIFICATION")

        result = agent.invoke({"messages": [HumanMessage(content=query)]})
        response = result["messages"][-1].content
        self.assertIn("transaction ID", response)
        self.assertNotIn("TX1001", response)
        self.assertNotIn("0.92", response)
        self.assertNotIn("250,000", response)


if __name__ == "__main__":
    unittest.main()
