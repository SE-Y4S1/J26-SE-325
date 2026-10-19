"""
Unit and integration tests for Component 4 Responsible AI preference enforcement:
1. Preference contract validation & backward-compatibility
2. AI explanation suppression (SHAP suppressed, fraud decision/score preserved)
3. Personalized financial context gating (blocks portfolio tools, no network call, no fabricated data)
4. Data-sharing consent enforcement & accurate reporting (research telemetry suppressed, no false upstream claims)
5. Essential safety controls remaining active under all preference configurations
"""

import unittest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from preferences_context import (
    get_current_preferences,
    set_current_preferences,
    reset_current_preferences,
)
from main import app, ChatRequest, ResponsibleAIPreferences
from responsible_ai import run_responsible_ai_checks, record_research_telemetry
from tools.fraud_tools import get_fraud_analysis
from tools.portfolio_tools import query_live_portfolio_service, get_portfolio_analysis
from evidence_validator import validate_evidence_claims


class TestResponsibleAIPreferencesContract(unittest.TestCase):
    """Test preference model parsing, default values, camelCase/snake_case, and malformed payloads."""

    def test_default_preferences_are_all_true(self):
        prefs = ResponsibleAIPreferences()
        self.assertTrue(prefs.explanation_enabled)
        self.assertTrue(prefs.personalized_context)
        self.assertTrue(prefs.data_sharing_consent)

    def test_chat_request_backward_compatibility_without_preferences(self):
        req = ChatRequest(message="What is my risk score?")
        self.assertIsNotNone(req.preferences)
        self.assertTrue(req.preferences.explanation_enabled)
        self.assertTrue(req.preferences.personalized_context)
        self.assertTrue(req.preferences.data_sharing_consent)

    def test_camel_case_preference_keys_from_frontend(self):
        payload = {
            "message": "Check transaction",
            "preferences": {
                "explanationEnabled": False,
                "personalizedContext": False,
                "dataSharingConsent": False,
            },
        }
        req = ChatRequest(**payload)
        self.assertFalse(req.preferences.explanation_enabled)
        self.assertFalse(req.preferences.personalized_context)
        self.assertFalse(req.preferences.data_sharing_consent)

    def test_snake_case_preference_keys(self):
        payload = {
            "message": "Check transaction",
            "preferences": {
                "explanation_enabled": False,
                "personalized_context": True,
                "data_sharing_consent": False,
            },
        }
        req = ChatRequest(**payload)
        self.assertFalse(req.preferences.explanation_enabled)
        self.assertTrue(req.preferences.personalized_context)
        self.assertFalse(req.preferences.data_sharing_consent)

    def test_malformed_preferences_handled_safely(self):
        # String instead of object
        req1 = ChatRequest.model_validate({"message": "test", "preferences": "malformed_string"})
        self.assertTrue(req1.preferences.explanation_enabled)
        self.assertTrue(req1.preferences.personalized_context)
        self.assertTrue(req1.preferences.data_sharing_consent)

        # None instead of object
        req2 = ChatRequest.model_validate({"message": "test", "preferences": None})
        self.assertTrue(req2.preferences.explanation_enabled)

        # Non-boolean values in dictionary
        req3 = ChatRequest.model_validate({
            "message": "test",
            "preferences": {
                "explanationEnabled": "false",
                "personalizedContext": "0",
                "dataSharingConsent": "invalid_value",
            },
        })
        self.assertFalse(req3.preferences.explanation_enabled)
        self.assertFalse(req3.preferences.personalized_context)
        self.assertTrue(req3.preferences.data_sharing_consent)

    def test_extra_unknown_fields_safely_ignored(self):
        payload = {
            "message": "Hello",
            "preferences": {
                "explanationEnabled": True,
                "unknown_custom_field": 9999,
            },
        }
        req = ChatRequest(**payload)
        self.assertTrue(req.preferences.explanation_enabled)


class TestExplanationPreferenceEnforcement(unittest.TestCase):
    """Test AI explanation disabled preference enforcement on fraud tools and explanations."""

    def test_explanation_enabled_true_includes_shap_evidence(self):
        token = set_current_preferences({"explanation_enabled": True})
        try:
            output = get_fraud_analysis.invoke({"transaction_id": "TX1001"})
            self.assertIn("Recorded Fraud Score:\n0.92", output)
            self.assertIn("Recorded Status:\nBLOCKED", output)
            self.assertIn("LOCAL SHAP PROTOTYPE", output)
            self.assertIn("SHAP CONTRIBUTIONS:", output)
            self.assertNotIn("AI EXPLANATION NOTICE", output)
        finally:
            reset_current_preferences(token)

    def test_explanation_enabled_false_suppresses_shap_details(self):
        token = set_current_preferences({"explanation_enabled": False})
        try:
            with patch("tools.fraud_tools.explain_transaction") as mock_explain:
                output = get_fraud_analysis.invoke({"transaction_id": "TX1001"})

                # explain_transaction must NOT even be called
                mock_explain.assert_not_called()

                # Core fraud decision and recorded score must be strictly preserved
                self.assertIn("Recorded Fraud Score:\n0.92", output)
                self.assertIn("Recorded Risk Level:\nHIGH", output)
                self.assertIn("Recorded Status:\nBLOCKED", output)
                self.assertIn("RECORDED CONTRIBUTING FACTORS:", output)
                self.assertIn("new device", output.lower())

                # SHAP details must be suppressed
                self.assertNotIn("LOCAL SHAP PROTOTYPE", output)
                self.assertNotIn("SHAP CONTRIBUTIONS:", output)
                self.assertIn("AI EXPLANATION NOTICE:", output)
                self.assertIn("Optional SHAP feature-attribution details were suppressed", output)
        finally:
            reset_current_preferences(token)

    def test_evidence_validator_with_suppressed_shap_evidence(self):
        evidence = [
            {
                "tool": "get_fraud_analysis",
                "result": (
                    "Transaction ID: TX1001\n"
                    "Recorded Fraud Score: 0.92\n"
                    "Recorded Risk Level: HIGH\n"
                    "Recorded Status: BLOCKED\n"
                    "AI EXPLANATION NOTICE:\n"
                    "Optional SHAP feature-attribution details were suppressed based on user preference."
                ),
            }
        ]
        # Answer with unsupported certainty claim
        risky_answer = "This transaction was blocked because of 100% chance of fraud and the models share training data."
        val = validate_evidence_claims(risky_answer, evidence)

        self.assertFalse(val["passed"])
        self.assertIn("Detailed SHAP feature attributions and prototype model metrics were suppressed", val["answer"])
        self.assertIn("0.92", val["answer"])
        self.assertIn("BLOCKED", val["answer"])
        self.assertNotIn("1.0000", val["answer"])


class TestPersonalizedContextPreferenceEnforcement(unittest.TestCase):
    """Test personalized financial context disabled preference enforcement on portfolio queries."""

    def test_query_live_portfolio_service_skipped_when_personalization_disabled(self):
        token = set_current_preferences({"personalized_context": False})
        try:
            with patch("tools.portfolio_tools._http_get") as mock_http:
                res = query_live_portfolio_service(user_id="1", portfolio_id="1")

                # No network call to Platform should be made
                mock_http.assert_not_called()

                self.assertFalse(res["available"])
                self.assertIsNone(res["status_code"])
                self.assertIsNone(res["data"])
                self.assertIn("Personalized financial context is disabled", res["error"])
        finally:
            reset_current_preferences(token)

    def test_get_portfolio_analysis_explains_disabled_without_fabricating_data(self):
        token = set_current_preferences({"personalized_context": False})
        try:
            with patch("tools.portfolio_tools._http_get") as mock_http:
                output = get_portfolio_analysis.invoke({"user_id": "1"})

                # HTTP call must not happen
                mock_http.assert_not_called()

                self.assertIn("PERSONALIZED CONTEXT DISABLED", output)
                self.assertIn("Personalized financial context is disabled by user preference", output)
                self.assertIn("No anonymous or fallback portfolio data has been substituted", output)

                # Must NOT return fallback balance of LKR 1,000,000 or mock asset allocations
                self.assertNotIn("LKR 1,000,000", output)
                self.assertNotIn("Illustrative Baseline Valuation", output)
                self.assertNotIn("Technology: 35%", output)
        finally:
            reset_current_preferences(token)

    def test_evidence_validator_handles_disabled_personalization_without_mock_balance(self):
        evidence = [
            {
                "tool": "get_portfolio_analysis",
                "result": "PORTFOLIO ANALYSIS — PERSONALIZED CONTEXT DISABLED\nPersonalized financial context is disabled by user preference.",
            }
        ]
        # Answer with risky claim
        risky_answer = "Your actual portfolio has a value of LKR 1,000,000 and your risk score is 45."
        val = validate_evidence_claims(risky_answer, evidence)

        self.assertFalse(val["passed"])
        self.assertIn("Personalized financial context is disabled by user preference", val["answer"])
        self.assertNotIn("LKR 1,000,000", val["answer"])

    def test_regression_exact_portfolio_query_with_personalization_disabled(self):
        """Regression test for user reproduction scenario:
        Query: 'Analyze my investment portfolio. Show my current holdings and total valuation.'
        With personalization OFF:
        - Must not invoke get_portfolio_analysis or query platform
        - Must not return mock portfolio metrics or LKR 1,000,000
        - Must explain that personalized context was disabled
        """
        token = set_current_preferences({"personalized_context": False})
        try:
            with patch("tools.portfolio_tools._http_get") as mock_http, \
                 patch("tools.portfolio_tools.load_mock_data") as mock_mock_data:
                from agent import call_llm
                from langchain_core.messages import HumanMessage

                state = {
                    "messages": [
                        HumanMessage(content="Analyze my investment portfolio. Show my current holdings and total valuation.")
                    ]
                }
                res = call_llm(state)
                ai_msg = res["messages"][0]
                text = ai_msg.content

                # Neither network nor mock data should be queried
                mock_http.assert_not_called()
                mock_mock_data.assert_not_called()

                # Must not contain tool calls
                self.assertFalse(hasattr(ai_msg, "tool_calls") and len(ai_msg.tool_calls) > 0)

                # Must clearly explain that personalized context was disabled
                self.assertIn("Personalized financial context is disabled", text)
                self.assertNotIn("1,000,000", text)
                self.assertNotIn("LKR 1,000,000", text)
                self.assertNotIn("Illustrative Baseline Valuation", text)
        finally:
            reset_current_preferences(token)


class TestDataSharingConsentEnforcement(unittest.TestCase):
    """Test data-sharing consent reporting and empirical research telemetry logging."""

    def test_research_telemetry_suppressed_when_consent_withdrawn(self):
        logged = record_research_telemetry(
            event={"test": "data"},
            consent_granted=False,
        )
        self.assertFalse(logged)

    def test_research_telemetry_allowed_when_consent_granted(self):
        logged = record_research_telemetry(
            event={"test": "data"},
            consent_granted=True,
        )
        self.assertTrue(logged)

    def test_consent_status_accurately_reported_when_withdrawn(self):
        results = run_responsible_ai_checks(
            user_message="Hello",
            response="Hello there!",
            preferences={"data_sharing_consent": False},
        )
        data_sharing = results["data_sharing"]
        self.assertEqual(data_sharing["status"], "opted_out")
        self.assertIn("User opted out", data_sharing["message"])
        # Explicit disclaimer that upstream audit logging is not claimed to be disabled
        self.assertIn("does not disable upstream gateway audit logs", data_sharing["message"])

    def test_consent_status_accurately_reported_when_granted(self):
        results = run_responsible_ai_checks(
            user_message="Hello",
            response="Hello there!",
            preferences={"data_sharing_consent": True},
        )
        data_sharing = results["data_sharing"]
        self.assertEqual(data_sharing["status"], "consented")
        self.assertIn("Consent granted", data_sharing["message"])
        self.assertIn("Upstream transaction and audit trail logging remain governed", data_sharing["message"])


class TestEssentialSafetyControlsPreserved(unittest.TestCase):
    """Test that core privacy, safety, and fairness controls remain fully active regardless of preferences."""

    def test_essential_safety_active_with_all_preferences_disabled(self):
        all_disabled = {
            "explanation_enabled": False,
            "personalized_context": False,
            "data_sharing_consent": False,
        }

        # Response that attempts to guarantee profits (should trigger safety warning)
        unsafe_response = "I offer a guaranteed return of 200% on this investment."
        checks = run_responsible_ai_checks(
            user_message="How do I get rich?",
            response=unsafe_response,
            preferences=all_disabled,
        )

        # Essential safety check must actively trigger
        self.assertEqual(checks["safety"]["status"], "warning")
        self.assertIn("guaranteed return", checks["safety"]["detected"])

        # Privacy and fairness checks must be present
        self.assertIn("privacy", checks)
        self.assertIn("fairness", checks)

        # User control must report enforced status
        self.assertEqual(checks["user_control"]["status"], "enforced")
        self.assertEqual(checks["transparency"]["status"], "suppressed_by_preference")
        self.assertEqual(checks["personalized_context"]["status"], "disabled_by_preference")
        self.assertEqual(checks["data_sharing"]["status"], "opted_out")


class TestChatEndpointIntegrationWithPreferences(unittest.TestCase):
    """End-to-end integration tests on FastAPI /assistant/chat endpoint with preference objects."""

    def setUp(self):
        self.client = TestClient(app)

    @patch("main.agent.invoke")
    def test_chat_without_preferences_backward_compatible(self, mock_invoke):
        mock_msg = MagicMock()
        mock_msg.content = "Hello! I can help you with financial analysis."
        mock_msg.type = "ai"
        mock_invoke.return_value = {"messages": [mock_msg]}

        response = self.client.post("/assistant/chat", json={"message": "Hello"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        self.assertIn("responsible_ai", data)
        self.assertEqual(data["responsible_ai"]["transparency"]["status"], "enabled")
        self.assertEqual(data["responsible_ai"]["personalized_context"]["status"], "enabled")
        self.assertEqual(data["responsible_ai"]["data_sharing"]["status"], "consented")

    @patch("main.agent.invoke")
    def test_chat_with_all_preferences_disabled(self, mock_invoke):
        mock_msg = MagicMock()
        mock_msg.content = "Notice: Personalized financial context is disabled by user preference."
        mock_msg.type = "ai"
        mock_invoke.return_value = {"messages": [mock_msg]}

        payload = {
            "message": "Hi, what can you do?",
            "preferences": {
                "explanationEnabled": False,
                "personalizedContext": False,
                "dataSharingConsent": False,
            },
        }
        response = self.client.post("/assistant/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["responsible_ai"]["transparency"]["status"], "suppressed_by_preference")
        self.assertEqual(data["responsible_ai"]["personalized_context"]["status"], "disabled_by_preference")
        self.assertEqual(data["responsible_ai"]["data_sharing"]["status"], "opted_out")
        self.assertEqual(data["responsible_ai"]["user_control"]["status"], "enforced")

    def test_regression_chat_endpoint_exact_portfolio_reproduction(self):
        """Test exact reproduction payload via FastAPI endpoint with personalization disabled."""
        payload = {
            "message": "Analyze my investment portfolio. Show my current holdings and total valuation.",
            "preferences": {
                "explanationEnabled": True,
                "personalizedContext": False,
                "dataSharingConsent": True,
            },
        }
        response = self.client.post("/assistant/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Tools used must be empty
        self.assertEqual(data["agent_execution"]["tools_used"], [])
        self.assertEqual(data["agent_execution"]["number_of_tools"], 0)
        self.assertEqual(data["evidence"], [])

        # Answer must explain disabled and never contain 1,000,000
        self.assertIn("Personalized financial context is disabled", data["answer"])
        self.assertNotIn("1,000,000", data["answer"])
        self.assertNotIn("LKR 1,000,000", data["answer"])

        # Responsible AI reporting
        self.assertEqual(data["responsible_ai"]["personalized_context"]["status"], "disabled_by_preference")
        self.assertEqual(data["responsible_ai"]["transparency"]["status"], "enabled")


if __name__ == "__main__":
    unittest.main()
