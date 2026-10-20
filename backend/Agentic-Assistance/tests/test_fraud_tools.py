"""
Focused unit tests for Component 4 live fraud detection integration:
- Safe read-only lookup via Component 2 decision store (GET /v1/transactions/{id})
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
        """Test successful retrieval of an existing transaction from Component 2's decision store."""
        engine_payload = {
            "transaction_id": "TX1001",
            "fraud_score": 0.9410,
            "decision": "BLOCK",
            "model_version": "ieee-cis-dev-20k",
            "benchmark_model": True,
            "explanation": {
                "rules": ["Relational risk corroborated by anomalous device and location"],
                "streams": {"lstm": 0.8750, "gnn": 0.9120, "fused": 0.9410},
                "top_factors": [
                    {"name": "TransactionAmt", "contribution": 1.25},
                    {"name": "freq__card1", "contribution": -0.40},
                ],
                "notes": [],
            },
        }

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps(engine_payload).encode("utf-8")
        mock_response.__enter__.return_value = mock_response

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            output = get_fraud_analysis.invoke({"transaction_id": "TX1001"})

            # Verify safe read-only URL and GET method used
            mock_urlopen.assert_called_once()
            called_req = mock_urlopen.call_args[0][0]
            self.assertIn("/v1/transactions/TX1001", called_req.full_url)
            self.assertNotIn("/score", called_req.full_url)
            self.assertEqual(called_req.get_method(), "GET")

            # Verify provenance and live data mapping
            self.assertIn("FRAUD ANALYSIS — LIVE COMPONENT 2 AUDIT RECORD", output)
            self.assertIn("Live Component 2 Gateway Decision Store", output)
            self.assertIn("Transaction ID:\nTX1001", output)
            self.assertIn("Recorded Fraud Score:\n0.9410", output)
            self.assertIn("Recorded Status:\nBLOCK", output)
            self.assertIn("Recorded Risk Level:\nHIGH", output)
            self.assertIn("Relational risk corroborated", output)
            self.assertIn("Behavioral Stream Anomaly Score", output)
            self.assertIn("0.8750", output)
            self.assertIn("Relational Stream Risk Score", output)
            self.assertIn("0.9120", output)
            self.assertIn("ieee-cis-dev-20k", output)
            self.assertIn("TransactionAmt: contribution +1.2500 (increases fraud risk)", output)
            self.assertIn("freq__card1: contribution -0.4000 (decreases fraud risk)", output)

            # Verify Random Forest SHAP is NOT claimed to explain Component 2 model
            self.assertIn("Component 4's local synthetic Random Forest prototype is NOT used for this live transaction", output)
            self.assertIn("Probability calibration not established", output)
            self.assertNotIn("FALLBACK NOTICE", output)

    def test_step_up_maps_to_moderate(self):
        """The engine spells STEP_UP; it must still map to a MODERATE risk level."""
        engine_payload = {
            "transaction_id": "EXE-1", "fraud_score": 0.6, "decision": "STEP_UP",
            "model_version": "heuristic-untrained", "benchmark_model": False,
            "explanation": {"rules": [], "streams": {"heuristic": 0.6}, "top_factors": [], "notes": ["n"]},
        }
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps(engine_payload).encode("utf-8")
        mock_response.__enter__.return_value = mock_response

        with patch("urllib.request.urlopen", return_value=mock_response):
            output = get_fraud_analysis.invoke({"transaction_id": "EXE-1"})

        self.assertIn("Recorded Risk Level:\nMODERATE", output)
        self.assertIn("NOT a benchmark model", output)

    def test_missing_transaction_live_fallback(self):
        """Test when the live service is up but holds no decision for the transaction: returns NOT FOUND with no mock data."""
        not_found = urllib.error.HTTPError(
            "http://localhost:8001/v1/transactions/TX1001", 404, "Not Found", {},
            io.BytesIO(b'{"detail": "No decision is held for this transaction."}'),
        )

        with patch("urllib.request.urlopen", side_effect=not_found):
            output = get_fraud_analysis.invoke({"transaction_id": "TX1001"})

            # Must return RECORD NOT FOUND with unavailable states and 0 fabricated score
            self.assertIn("FRAUD ANALYSIS — RECORD NOT FOUND / SERVICE UNAVAILABLE", output)
            self.assertIn("RECORD_NOT_FOUND", output)
            self.assertIn("Recorded Fraud Score:\nUnavailable", output)
            self.assertIn("Recorded Status:\nUnavailable", output)
            self.assertIn("Recorded Risk Level:\nUnavailable", output)
            self.assertIn("does not substitute mock fallback data or fabricate fraud scores", output)
            self.assertNotIn("0.92", output)
            self.assertNotIn("LKR 250,000.00", output)

    def test_timeout_fallback(self):
        """Test that timeout errors return SERVICE UNAVAILABLE with no fabricated scores."""
        with patch("urllib.request.urlopen", side_effect=TimeoutError("Request timed out")):
            output = get_fraud_analysis.invoke({"transaction_id": "TX1001"})

            self.assertIn("FRAUD ANALYSIS — RECORD NOT FOUND / SERVICE UNAVAILABLE", output)
            self.assertIn("SERVICE_UNAVAILABLE", output)
            self.assertIn("Request timed out", output)
            self.assertIn("Recorded Fraud Score:\nUnavailable", output)
            self.assertNotIn("0.92", output)

    def test_malformed_response_fallback(self):
        """Test that invalid/non-JSON responses return SERVICE UNAVAILABLE with no fabricated scores."""
        mock_response = MagicMock()
        mock_response.status = 502
        mock_response.read.return_value = b"<html><head><title>502 Bad Gateway</title></head></html>"
        mock_response.__enter__.return_value = mock_response

        with patch("urllib.request.urlopen", return_value=mock_response):
            output = get_fraud_analysis.invoke({"transaction_id": "TX1001"})

            self.assertIn("FRAUD ANALYSIS — RECORD NOT FOUND / SERVICE UNAVAILABLE", output)
            self.assertIn("SERVICE_UNAVAILABLE", output)
            self.assertIn("non-JSON payload", output)
            self.assertIn("Recorded Fraud Score:\nUnavailable", output)
            self.assertNotIn("0.92", output)

    def test_service_unavailable_fallback(self):
        """Test that connection failure (e.g. port 8001 not running) returns clean unavailable result."""
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            output = get_fraud_analysis.invoke({"transaction_id": "TX1001"})

            self.assertIn("FRAUD ANALYSIS — RECORD NOT FOUND / SERVICE UNAVAILABLE", output)
            self.assertIn("SERVICE_UNAVAILABLE", output)
            self.assertIn("Cannot connect to Component 2", output)
            self.assertIn("Recorded Fraud Score:\nUnavailable", output)
            self.assertNotIn("0.92", output)

    def test_explicit_demo_mode_returns_labeled_demo_data(self):
        """Test that demo mode works ONLY when explicitly enabled and is visibly labelled."""
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            output = get_fraud_analysis.invoke({"transaction_id": "TX1001", "demo_mode": True})

            self.assertIn("DEMO / SYNTHETIC SPECIFICATION (EXPLICIT DEMO MODE)", output)
            self.assertIn("DEMO NOTICE", output)
            self.assertIn("EXPLICIT DEMO MODE", output)
            self.assertIn("Recorded Fraud Score:\n0.92", output)
            self.assertIn("LOCAL SHAP PROTOTYPE — NOT THE PRODUCTION MODEL", output)

    def test_missing_from_both_live_and_mock(self):
        """Test clean message when transaction ID is unknown in both live audit and mock data."""
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            output = get_fraud_analysis.invoke({"transaction_id": "UNKNOWN_9999", "demo_mode": True})

            self.assertIn("No demo transaction was found for ID 'UNKNOWN_9999'", output)

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
