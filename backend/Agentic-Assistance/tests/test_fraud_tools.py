"""
Focused unit tests for Component 4 live fraud detection integration:
- Safe read-only lookup via Component 2 audit log (GET /audit?limit=200)
- Missing transaction in live audit log -> fallback to mock data
- Service timeout -> graceful fallback
- Malformed response -> graceful fallback
- Service unavailable -> graceful fallback
- Missing from both live and mock -> graceful error string
- Verification that POST /score is never called
- Verification that mock data is never presented as live
"""

import io
import json
import os
import sys
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

# Ensure backend/Agentic-Assistance is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from tools.fraud_tools import (
    FRAUD_SERVICE_URL,
    get_fraud_analysis,
    query_live_fraud_service,
)


class TestFraudToolsIntegration(unittest.TestCase):

    def test_successful_live_lookup(self):
        """Test successful retrieval of an existing transaction from Component 2 audit log."""
        live_audit_payload = {
            "count": 1,
            "entries": [
                {
                    "timestamp": "2026-10-14T18:30:00Z",
                    "transaction_id": "TX1001",
                    "user_id": "u_live_user_42",
                    "behavioral_score": 0.8750,
                    "graph_score": 0.9120,
                    "risk_score": 0.9410,
                    "decision": "BLOCK",
                    "reason": "Relational risk corroborated by anomalous device and location",
                    "attack_type": None,
                }
            ],
        }

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps(live_audit_payload).encode("utf-8")
        mock_response.__enter__.return_value = mock_response

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            output = get_fraud_analysis.invoke({"transaction_id": "TX1001"})

            # Verify safe read-only URL and GET method used
            mock_urlopen.assert_called_once()
            called_req = mock_urlopen.call_args[0][0]
            self.assertIn("/audit?limit=200", called_req.full_url)
            self.assertNotIn("/score", called_req.full_url)

            # Verify provenance and live data mapping
            self.assertIn("FRAUD ANALYSIS — LIVE COMPONENT 2 AUDIT RECORD", output)
            self.assertIn("Live Component 2 Gateway Audit Log", output)
            self.assertIn("Transaction ID:\nTX1001", output)
            self.assertIn("User ID:\nu_live_user_42", output)
            self.assertIn("Recorded Fraud Score:\n0.9410", output)
            self.assertIn("Recorded Status:\nBLOCK", output)
            self.assertIn("Recorded Risk Level:\nHIGH", output)
            self.assertIn("Relational risk corroborated", output)
            self.assertIn("Behavioral Stream Anomaly Score", output)
            self.assertIn("0.8750", output)
            self.assertIn("Relational Stream Risk Score", output)
            self.assertIn("0.9120", output)

            # Verify Random Forest SHAP is NOT claimed to explain Component 2 model
            self.assertIn("Component 4's local synthetic Random Forest prototype is NOT used for this live transaction", output)
            self.assertIn("Probability calibration not established", output)
            self.assertNotIn("FALLBACK NOTICE", output)

    def test_missing_transaction_live_fallback(self):
        """Test when live audit service is active but doesn't have the requested transaction."""
        live_audit_payload = {
            "count": 1,
            "entries": [
                {
                    "timestamp": "2026-10-14T18:25:00Z",
                    "transaction_id": "TX9999",
                    "user_id": "u_other",
                    "behavioral_score": 0.2,
                    "graph_score": 0.1,
                    "risk_score": 0.15,
                    "decision": "ALLOW",
                    "reason": "Normal transaction",
                    "attack_type": None,
                }
            ],
        }

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps(live_audit_payload).encode("utf-8")
        mock_response.__enter__.return_value = mock_response

        with patch("urllib.request.urlopen", return_value=mock_response):
            output = get_fraud_analysis.invoke({"transaction_id": "TX1001"})

            # Must fall back to mock data with explicit labeling
            self.assertIn("FRAUD ANALYSIS — FALLBACK MOCK DATA", output)
            self.assertIn("FALLBACK NOTICE", output)
            self.assertIn("This record is NOT a live lookup result from Component 2", output)
            self.assertIn("not found in Component 2 live audit log", output)

            # Mock data values preserved
            self.assertIn("Transaction ID:\nTX1001", output)
            self.assertIn("LKR 250,000.00", output)
            self.assertIn("Recorded Fraud Score:\n0.92", output)
            self.assertIn("Recorded Risk Level:\nHIGH", output)
            self.assertIn("Recorded Status:\nBLOCKED", output)

            # Prototype SHAP disclaimers preserved
            self.assertIn("LOCAL SHAP PROTOTYPE — NOT THE TEAM MODEL", output)
            self.assertIn("LIMITATION:", output)

    def test_timeout_fallback(self):
        """Test that timeout errors trigger graceful fallback to labeled mock data."""
        with patch("urllib.request.urlopen", side_effect=TimeoutError("Request timed out")):
            output = get_fraud_analysis.invoke({"transaction_id": "TX1001"})

            self.assertIn("FRAUD ANALYSIS — FALLBACK MOCK DATA", output)
            self.assertIn("FALLBACK NOTICE", output)
            self.assertIn("Request timed out", output)
            self.assertIn("Recorded Fraud Score:\n0.92", output)

    def test_malformed_response_fallback(self):
        """Test that invalid/non-JSON responses trigger graceful fallback."""
        mock_response = MagicMock()
        mock_response.status = 502
        mock_response.read.return_value = b"<html><head><title>502 Bad Gateway</title></head></html>"
        mock_response.__enter__.return_value = mock_response

        with patch("urllib.request.urlopen", return_value=mock_response):
            output = get_fraud_analysis.invoke({"transaction_id": "TX1001"})

            self.assertIn("FRAUD ANALYSIS — FALLBACK MOCK DATA", output)
            self.assertIn("FALLBACK NOTICE", output)
            self.assertIn("non-JSON payload", output)
            self.assertIn("Recorded Fraud Score:\n0.92", output)

    def test_service_unavailable_fallback(self):
        """Test that connection failure (e.g. port 8001 not running) falls back cleanly."""
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            output = get_fraud_analysis.invoke({"transaction_id": "TX1001"})

            self.assertIn("FRAUD ANALYSIS — FALLBACK MOCK DATA", output)
            self.assertIn("FALLBACK NOTICE", output)
            self.assertIn("Cannot connect to Component 2", output)
            self.assertIn("Recorded Fraud Score:\n0.92", output)

    def test_missing_from_both_live_and_mock(self):
        """Test clean message when transaction ID is unknown in both live audit and mock data."""
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            output = get_fraud_analysis.invoke({"transaction_id": "UNKNOWN_9999"})

            self.assertIn("No transaction was found for ID 'UNKNOWN_9999'", output)
            self.assertIn("either live Component 2 audit records or fallback mock data", output)

    def test_never_calls_score_mutation(self):
        """Verify that POST /score is NEVER called under any circumstance."""
        called_urls = []

        def fake_urlopen(req, timeout=None):
            called_urls.append((req.full_url, req.get_method()))
            raise urllib.error.URLError("Simulated offline")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            get_fraud_analysis.invoke({"transaction_id": "TX1001"})

        for url, method in called_urls:
            self.assertNotIn("/score", url)
            self.assertEqual(method, "GET")


if __name__ == "__main__":
    unittest.main()
