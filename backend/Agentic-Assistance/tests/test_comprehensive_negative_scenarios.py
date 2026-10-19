"""
Comprehensive Negative and Safety Tests for PP1 Component 4:
- Empty or malformed chat requests
- Invalid or missing authentication where authentication is required
- Unauthorized governance access
- Platform, fraud service, audit service unavailability (500, timeout, connection refused)
- Request timeouts and malformed upstream responses
- Empty portfolio results (distinct from outages)
- Partial portfolio data and missing prices (marked Unavailable, not invented)
- Missing transaction and audit records (not treated as proof of fraud or verification)
- Failed governance persistence (asserted error, not false success)
- Duplicate confirmation or rejection submissions (idempotent duplicate detection)
- Rejected actions attempting to proceed (policy enforcement notice)
- Disabled personalized financial context (immediate suppression, no API call)
- Disabled explanation preferences (feature attributions suppressed)
"""

import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import urllib.error

import jwt
from fastapi.testclient import TestClient

from governance_store import GovernanceStore, default_governance_store
from main import app
from preferences_context import reset_current_preferences, set_current_preferences
from tools.audit_tools import get_blockchain_audit, query_live_audit_service
from tools.execution_tools import get_execution_explanation
from tools.fraud_tools import get_fraud_analysis, query_live_fraud_service
from tools.portfolio_tools import (
    get_portfolio_analysis,
    query_live_portfolio_service,
    _execute_portfolio_analysis,
)


class TestComprehensiveNegativeScenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.jwt_secret = "test-secret-at-least-32-chars-long-0123456789"

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_store_file = Path(self.temp_dir) / "negative_gov_test.jsonl"
        self.orig_store_file = default_governance_store.file_path
        default_governance_store.file_path = self.test_store_file
        default_governance_store._ensure_directory()

    def tearDown(self):
        default_governance_store.file_path = self.orig_store_file
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _mint_token(self, sub: str = "1", email: str = "test@example.com", exp_minutes: int = 60) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            "sub": str(sub),
            "email": email,
            "iat": now,
            "exp": now + timedelta(minutes=exp_minutes),
            "iss": "j26-se-325-platform",
        }
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")

    # -------------------------------------------------------------------------
    # 1. Empty or malformed chat requests
    # -------------------------------------------------------------------------
    def test_empty_chat_message_returns_safe_greeting(self):
        res = self.client.post("/assistant/chat", json={"message": ""})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("FinAgent", data["answer"])
        self.assertEqual(data["agent_execution"]["number_of_tools"], 0)

    def test_whitespace_chat_message_returns_safe_greeting(self):
        res = self.client.post("/assistant/chat", json={"message": "    \n\t  "})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("FinAgent", data["answer"])
        self.assertEqual(data["agent_execution"]["number_of_tools"], 0)

    def test_missing_message_field_returns_422(self):
        res = self.client.post("/assistant/chat", json={"preferences": {}})
        self.assertEqual(res.status_code, 422)

    def test_invalid_json_payload_returns_422(self):
        res = self.client.post(
            "/assistant/chat",
            content="not a valid json payload",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res.status_code, 422)

    # -------------------------------------------------------------------------
    # 2. Authentication and Authorization Enforcement
    # -------------------------------------------------------------------------
    @patch.dict(os.environ, {"AUTH_REQUIRED": "true", "JWT_SECRET": "test-secret-at-least-32-chars-long-0123456789"})
    def test_missing_auth_header_rejected_when_auth_required(self):
        res = self.client.post(
            "/assistant/governance/decision",
            json={"decision_type": "confirmed", "reference_id": "TX1001"},
        )
        self.assertEqual(res.status_code, 401)
        self.assertIn("Not authenticated", res.json()["detail"])

    @patch.dict(os.environ, {"AUTH_REQUIRED": "true", "JWT_SECRET": "test-secret-at-least-32-chars-long-0123456789"})
    def test_invalid_signature_rejected(self):
        bad_token = jwt.encode(
            {"sub": "1", "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
            "completely-wrong-secret-0123456789",
            algorithm="HS256",
        )
        res = self.client.post(
            "/assistant/governance/decision",
            json={"decision_type": "confirmed", "reference_id": "TX1001"},
            headers={"Authorization": f"Bearer {bad_token}"},
        )
        self.assertEqual(res.status_code, 401)

    @patch.dict(os.environ, {"AUTH_REQUIRED": "true", "JWT_SECRET": "test-secret-at-least-32-chars-long-0123456789"})
    def test_expired_token_rejected(self):
        expired_token = self._mint_token(sub="1", exp_minutes=-10)
        res = self.client.post(
            "/assistant/governance/decision",
            json={"decision_type": "confirmed", "reference_id": "TX1001"},
            headers={"Authorization": f"Bearer {expired_token}"},
        )
        self.assertEqual(res.status_code, 401)
        self.assertIn("Token expired", res.json()["detail"])

    # -------------------------------------------------------------------------
    # 3. Platform, Fraud, and Audit Service Unavailability (500, Timeout, Malformed)
    # -------------------------------------------------------------------------
    @patch("urllib.request.urlopen")
    def test_platform_500_error_reported_as_service_error_not_empty_portfolio(self, mock_urlopen):
        # Explicit safety test: an outage MUST NOT be treated as an empty portfolio!
        err = urllib.error.HTTPError(
            url="http://localhost:8100/portfolios",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=MagicMock(read=lambda: b'{"detail": "Database crashed"}'),
        )
        mock_urlopen.side_effect = err

        res = query_live_portfolio_service(user_id="1")
        self.assertTrue(res["available"])
        self.assertEqual(res["status_code"], 500)
        self.assertFalse(res["empty_list"])
        self.assertIn("HTTP 500", res["error"])

        analysis = _execute_portfolio_analysis(user_id="1")
        self.assertIn("FALLBACK DATA (NO LIVE RECORD RETRIEVED)", analysis)
        self.assertIn("HTTP 500", analysis)
        self.assertIn("The user does NOT have a verified balance", analysis)

    @patch("urllib.request.urlopen")
    def test_platform_timeout_not_treated_as_empty_portfolio(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError("Connection timed out after 1.5s")

        res = query_live_portfolio_service(user_id="1")
        self.assertFalse(res["available"])
        self.assertIsNone(res["status_code"])
        self.assertFalse(res["empty_list"])
        self.assertIn("Service unavailable", res["error"])

        analysis = _execute_portfolio_analysis(user_id="1")
        self.assertIn("This does not indicate an empty portfolio", analysis)
        self.assertIn("connection failure or timeout", analysis)

    @patch("urllib.request.urlopen")
    def test_platform_malformed_json_response_handled_gracefully(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"<html>502 Bad Gateway Nginx</html>"
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        res = query_live_portfolio_service(user_id="1")
        self.assertTrue(res["available"])
        self.assertIn("Malformed JSON returned by platform service", res["error"])

    @patch("urllib.request.urlopen")
    def test_fraud_service_outage_reported_safely_not_as_fraud(self, mock_urlopen):
        # Missing record / outage must NOT be treated as proof of fraud
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        res = query_live_fraud_service("TX1001")
        self.assertFalse(res["available"])
        self.assertIn("Cannot connect to Component 2", res["reason"])

        text = get_fraud_analysis.invoke({"transaction_id": "TX1001"})
        self.assertIn("FALLBACK MOCK DATA", text)
        self.assertIn("Cannot connect to Component 2", text)
        self.assertIn("These results are prototype evidence, not verified production output", text)

    @patch("urllib.request.urlopen")
    def test_audit_service_outage_reported_safely_not_as_verified(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        res = query_live_audit_service("TX1001")
        self.assertFalse(res["available"])
        self.assertIn("Service unavailable", res["error"])

        text = get_blockchain_audit.invoke({"transaction_id": "TX1001"})
        self.assertIn("FALLBACK DATA", text)
        self.assertIn("This is NOT a live lookup result", text)
        self.assertIn("Independent blockchain verification is not established", text)

    # -------------------------------------------------------------------------
    # 4. Empty Portfolio Results vs Partial Portfolio Data
    # -------------------------------------------------------------------------
    @patch("urllib.request.urlopen")
    def test_empty_portfolio_list_reported_clearly(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"[]"
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        res = query_live_portfolio_service(user_id="1")
        self.assertTrue(res["available"])
        self.assertEqual(res["status_code"], 200)
        self.assertTrue(res["empty_list"])

        analysis = _execute_portfolio_analysis(user_id="1")
        self.assertIn("No portfolios registered for user on platform service", analysis)
        self.assertIn("Live holdings: Unverified (No live portfolio registered on platform)", analysis)

    @patch("urllib.request.urlopen")
    def test_partial_portfolio_data_missing_prices_marked_unavailable(self, mock_urlopen):
        def fake_urlopen(req, timeout=None):
            url = req.full_url
            mock = MagicMock()
            mock.status = 200
            mock.__enter__.return_value = mock
            if url.endswith("/portfolios"):
                mock.read.return_value = json.dumps([{"id": 42, "name": "Partial Port"}]).encode("utf-8")
                return mock
            elif "/portfolios/42" in url:
                partial_data = {
                    "id": 42,
                    "name": "Partial Port",
                    "base_currency": "USD",
                    "total_value": None,
                    "holdings": [
                        {"symbol": "AAPL", "quantity": 10.0, "current_price": None, "cost_basis": 150.0},
                        {"symbol": "MSFT", "quantity": None, "current_price": 400.0, "cost_basis": None},
                    ],
                }
                mock.read.return_value = json.dumps(partial_data).encode("utf-8")
                return mock
            raise ValueError(f"Unexpected url {url}")

        mock_urlopen.side_effect = fake_urlopen
        analysis = _execute_portfolio_analysis(user_id="1", portfolio_id="42")
        self.assertIn("LIVE PLATFORM RECORD", analysis)
        self.assertIn("Unavailable", analysis)
        self.assertIn("Notice: Some fields (such as valuation, price, or cost basis) are not recorded", analysis)

    # -------------------------------------------------------------------------
    # 5. Missing Records
    # -------------------------------------------------------------------------
    def test_missing_transaction_not_treated_as_fraud(self):
        text = get_fraud_analysis.invoke({"transaction_id": "TX_COMPLETELY_NONEXISTENT_99999"})
        self.assertIn("No transaction was found for ID 'TX_COMPLETELY_NONEXISTENT_99999'", text)
        self.assertNotIn("PROVED FRAUD", text)

    def test_missing_audit_record_not_treated_as_verified(self):
        with patch("tools.audit_tools.query_live_audit_service") as mock_query:
            mock_query.return_value = {
                "available": True,
                "status_code": 404,
                "data": {"status": "NOT_FOUND"},
                "error": None,
            }
            text = get_blockchain_audit.invoke({"transaction_id": "TX_NONEXISTENT_888"})
            self.assertIn("BLOCKCHAIN AUDIT RECORD — NOT FOUND", text)
            self.assertIn("Record Status:\nNOT_FOUND", text)
            self.assertNotIn("Record Status:\nVERIFIED", text)

    # -------------------------------------------------------------------------
    # 6. Governance Persistence Failures and Duplicate Protection
    # -------------------------------------------------------------------------
    def test_failed_governance_persistence_raises_error_not_success(self):
        # Safety requirement: A failed save must not be presented as success!
        bad_store = GovernanceStore(file_path=Path(self.temp_dir) / "nonexistent_dir" / "file.jsonl")
        # Ensure parent does not exist and is unwritable
        with patch("builtins.open", side_effect=OSError("Disk write error")):
            with self.assertRaises(RuntimeError) as ctx:
                bad_store.record_decision(
                    decision_type="confirmed",
                    reference_id="TX1001",
                )
            self.assertIn("Governance persistence failed", str(ctx.exception))

    def test_duplicate_decision_submission_idempotency(self):
        store = default_governance_store
        first = store.record_decision("confirmed", "TX1001")
        self.assertFalse(first["is_duplicate"])
        self.assertEqual(first["status"], "recorded")

        second = store.record_decision("confirmed", "TX1001")
        self.assertTrue(second["is_duplicate"])
        self.assertEqual(second["status"], "already_recorded")
        self.assertEqual(second["decision_id"], first["decision_id"])

    # -------------------------------------------------------------------------
    # 7. Rejected Actions Attempting to Proceed
    # -------------------------------------------------------------------------
    def test_rejected_action_enforces_blocking_policy(self):
        store = default_governance_store
        store.record_decision(decision_type="rejected", reference_id="TX1001")

        fraud_text = get_fraud_analysis.invoke({"transaction_id": "TX1001"})
        self.assertIn("HUMAN OVERSIGHT GOVERNANCE STATUS:", fraud_text)
        self.assertIn("ACTION REJECTED BY USER — Financial execution is strictly blocked; do not execute real financial transactions.", fraud_text)

        exec_text = get_execution_explanation.invoke({"execution_id": "EXE-TX1001"})
        # Even if trace is missing, governance status shows rejected
        store.record_decision(decision_type="rejected", reference_id="EXE-TX1001")
        exec_text2 = get_execution_explanation.invoke({"execution_id": "EXE-TX1001"})
        self.assertIn("EXECUTION BLOCKED — Human oversight rejected this plan; executing real transactions is strictly prohibited.", exec_text2)

    # -------------------------------------------------------------------------
    # 8. Disabled Preferences
    # -------------------------------------------------------------------------
    def test_disabled_personalized_context_blocks_portfolio_lookup(self):
        token = set_current_preferences({"personalized_context": False})
        try:
            with patch("tools.portfolio_tools._http_get") as mock_http:
                res = query_live_portfolio_service(user_id="1")
                self.assertFalse(res["available"])
                self.assertIn("Personalized financial context is disabled", res["error"])
                mock_http.assert_not_called()

                analysis = _execute_portfolio_analysis(user_id="1")
                self.assertIn("PORTFOLIO ANALYSIS — PERSONALIZED CONTEXT DISABLED", analysis)
                self.assertIn("No anonymous or fallback portfolio data has been substituted", analysis)
        finally:
            reset_current_preferences(token)

    def test_disabled_explanation_suppresses_shap_and_rule_attributions(self):
        token = set_current_preferences({"explanation_enabled": False})
        try:
            fraud_text = get_fraud_analysis.invoke({"transaction_id": "TX1001"})
            self.assertIn("Optional SHAP feature-attribution details were suppressed based on user preference", fraud_text)
            self.assertNotIn("SHAP CONTRIBUTIONS:", fraud_text)
        finally:
            reset_current_preferences(token)


if __name__ == "__main__":
    unittest.main()
