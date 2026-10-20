import unittest
from unittest.mock import patch, MagicMock

from agent import (
    resolve_transaction_id,
    extract_transaction_ids,
    is_fraud_or_blocked_transaction_query,
    get_deterministic_fast_path,
)
from preferences_context import (
    set_request_transaction_id,
    get_request_transaction_id,
    set_current_preferences,
)
from tools.fraud_tools import get_fraud_analysis
from tools.audit_tools import get_blockchain_audit


class TestFraudHandoffRegression(unittest.TestCase):
    def setUp(self):
        set_request_transaction_id(None)
        set_current_preferences({"explanation_enabled": True, "demo_mode": False})

    def tearDown(self):
        set_request_transaction_id(None)

    def test_extract_transaction_ids_case_preservation(self):
        """ui- prefixed transaction IDs must preserve exact casing; others uppercase."""
        tokens = extract_transaction_ids("Check transaction ui-syn-1500k and tx1001 and exe-abc")
        self.assertIn("ui-syn-1500k", tokens)
        self.assertIn("TX1001", tokens)
        self.assertIn("EXE-ABC", tokens)

    def test_resolve_transaction_id_with_selected_request_context(self):
        """Selected request context is resolved when user does not mention an explicit ID."""
        set_request_transaction_id("ui-test-1500000")
        resolved, clar = resolve_transaction_id(
            "Why was my $1,500,000 transaction blocked?",
            explicit_id=get_request_transaction_id(),
        )
        self.assertEqual(resolved, "ui-test-1500000")
        self.assertIsNone(clar)

    def test_resolve_transaction_id_no_selection_requests_clarification(self):
        """Missing transaction ID results in clarification prompt, never defaults to TX1001."""
        resolved, clar = resolve_transaction_id(
            "Why was my $1,500,000 transaction blocked?",
            history=[],
            explicit_id=None,
        )
        self.assertIsNone(resolved)
        self.assertIsNotNone(clar)
        self.assertIn("provide the transaction ID", clar)
        self.assertNotIn("TX1001", str(resolved))

    def test_explicit_id_in_message_overrides_selected_context(self):
        """If user explicitly asks about a different ID, the message ID takes precedence over selection."""
        set_request_transaction_id("ui-selected-old")
        resolved, clar = resolve_transaction_id(
            "Why was transaction TX2002 blocked?",
            explicit_id=get_request_transaction_id(),
        )
        self.assertEqual(resolved, "TX2002")
        self.assertIsNone(clar)

    def test_followup_question_preserves_selected_context(self):
        """Follow-up questions preserve the active selected transaction context."""
        set_request_transaction_id("ui-test-1500000")
        resolved, clar = resolve_transaction_id(
            "What factors contributed most to this decision?",
            explicit_id=get_request_transaction_id(),
        )
        self.assertEqual(resolved, "ui-test-1500000")
        self.assertIsNone(clar)

    def test_is_fraud_or_blocked_transaction_query_matches_amount_and_followups(self):
        """Broadened fraud query regex matches queries with currency amounts and follow-ups."""
        queries = [
            "Why was my $1,500,000 transaction blocked?",
            "Why was my USD 1,500,000 transaction blocked?",
            "Why was my transaction of $1,500,000 blocked?",
            "Why was this transaction blocked?",
            "What factors contributed most to this decision?",
            "Explain the factors behind this block",
        ]
        for q in queries:
            with self.subTest(query=q):
                self.assertTrue(
                    is_fraud_or_blocked_transaction_query(q),
                    f"Query failed to match is_fraud_or_blocked_transaction_query: {q}",
                )

    def test_deterministic_fast_path_clarification_when_no_tx_selected(self):
        """get_deterministic_fast_path asks for clarification when no transaction is selected."""
        set_request_transaction_id(None)
        res = get_deterministic_fast_path("Why was my $1,500,000 transaction blocked?", history=[])
        self.assertIsNotNone(res)
        self.assertIn("provide the transaction ID", res)
        self.assertNotIn("TX1001", res)

    @patch("tools.fraud_tools.query_live_fraud_service")
    def test_get_fraud_analysis_retrieves_exact_live_record(self, mock_query):
        """get_fraud_analysis passes exact ID to live Component 2 service and formats verified decision."""
        mock_query.return_value = {
            "available": True,
            "status_code": 200,
            "entry": {
                "transaction_id": "ui-test-1500000",
                "risk_score": 1.0,
                "decision": "BLOCK",
                "reason": "amount_hard_block, score_above_block_threshold",
                "behavioral_score": 1.0,
                "graph_score": 0.0,
                "model_version": "benchmark-v1",
                "benchmark_model": True,
                "top_factors": [{"name": "amount", "contribution": 1.0}],
            },
            "error": None,
            "reason": "Record found in Component 2 live decision store",
        }

        output = get_fraud_analysis.invoke({"transaction_id": "ui-test-1500000"})
        self.assertIn("ui-test-1500000", output)
        self.assertIn("1.0000", output)
        self.assertIn("BLOCK", output)
        self.assertIn("amount_hard_block", output)
        mock_query.assert_called_once_with("ui-test-1500000")

    @patch("tools.fraud_tools.query_live_fraud_service")
    def test_get_fraud_analysis_unknown_id_record_not_found(self, mock_query):
        """Unknown transaction ID produces RECORD NOT FOUND and never falls back to TX1001."""
        mock_query.return_value = {
            "available": True,
            "status_code": 404,
            "entry": None,
            "error": None,
            "reason": "Transaction 'ui-unknown-888' not found in Component 2 live decision store",
        }

        output = get_fraud_analysis.invoke({"transaction_id": "ui-unknown-888"})
        self.assertIn("RECORD NOT FOUND", output)
        self.assertIn("ui-unknown-888", output)
        self.assertNotIn("TX1001", output)
        self.assertNotIn("0.92", output)

    @patch("request_auth.get_caller_authorization", return_value=None)
    @patch("tools.execution_tools.query_execution_trace")
    def test_authorization_check_prevents_unauthorized_execution_access(self, mock_trace, mock_auth):
        """Querying an execution ID without ownership authorization is blocked."""
        mock_trace.return_value = {
            "available": True,
            "trace": None,
            "reason": "No execution belongs to this user",
        }

        output = get_fraud_analysis.invoke({"transaction_id": "EXE-OTHER-USER-123"})
        self.assertIn("AUTHORIZATION DENIED", output)
        self.assertIn("does not belong to the authenticated user", output)

    @patch("tools.audit_tools.query_live_audit_service")
    def test_get_blockchain_audit_respects_request_transaction_id(self, mock_audit):
        """Blockchain audit tool uses request transaction ID context when active."""
        set_request_transaction_id("ui-audit-selected")
        mock_audit.return_value = {
            "available": True,
            "status_code": 200,
            "data": {
                "transaction_id": "ui-audit-selected",
                "status": "RECORDED",
                "block_number": 192837,
                "verified": True,
            },
            "error": None,
        }

        output = get_blockchain_audit.invoke({"transaction_id": ""})
        mock_audit.assert_called_once_with("ui-audit-selected")
        self.assertIn("ui-audit-selected", output)


if __name__ == "__main__":
    unittest.main()
