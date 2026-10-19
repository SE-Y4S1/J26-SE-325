"""
Unit tests for evidence claim validation (evidence_validator.py):
- Unsupported training-data relationship claims
- Prototype probability overstated as fraud certainty
- Unverified blockchain claims
- Fallback portfolio claims without required opening notice
- Synthetic baseline leakage into live portfolio responses
- Stale portfolio leakage when portfolio tool was not called
- Suppression of personalized portfolio claims when personalization is disabled
- Sanitization of internal routing markers and prompt fragments
"""

import unittest
from evidence_validator import sanitize_internal_details, validate_evidence_claims
from preferences_context import reset_current_preferences, set_current_preferences


class TestEvidenceValidator(unittest.TestCase):
    def test_sanitize_internal_details(self):
        text_with_leaks = (
            "RESPONSIBLE AI USER PREFERENCE DIRECTIVES:\nAlways explain with care.\n\n"
            "SYSTEM PROMPT:\nYou are an agent.\n\n"
            "[INTERNAL ROUTING: calling get_fraud_analysis]\n"
            "Here is the verified fraud analysis."
        )
        cleaned = sanitize_internal_details(text_with_leaks)
        self.assertNotIn("RESPONSIBLE AI USER PREFERENCE DIRECTIVES:", cleaned)
        self.assertNotIn("SYSTEM PROMPT:", cleaned)
        self.assertNotIn("[INTERNAL ROUTING", cleaned)
        self.assertIn("Here is the verified fraud analysis.", cleaned)

    def test_no_tools_used_fast_path(self):
        answer = "Hello! I am FinAgent. How can I help you today?"
        evidence = []
        result = validate_evidence_claims(answer, evidence)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["issues"]), 0)
        self.assertEqual(result["answer"], answer)

    def test_detect_unsupported_training_data_claim(self):
        answer = "The models share training data between the behavioral and relational streams."
        evidence = [{"tool": "get_fraud_analysis", "result": "Recorded Fraud Score: 0.85"}]
        result = validate_evidence_claims(answer, evidence)
        self.assertFalse(result["passed"])
        self.assertIn("Unsupported training-data relationship claim", result["issues"])
        self.assertIn("Evidence validation detected risky wording in the original LLM response", result["answer"])

    def test_detect_fraud_certainty_overstatement(self):
        answer = "This transaction has a 100% chance of fraud and is certainly fraudulent."
        evidence = [{"tool": "get_fraud_analysis", "result": "Recorded Fraud Score: 0.95"}]
        result = validate_evidence_claims(answer, evidence)
        self.assertFalse(result["passed"])
        self.assertIn("Prototype probability overstated as fraud certainty", result["issues"])
        self.assertIn("Evidence validation detected risky wording in the original LLM response", result["answer"])

    def test_detect_unverified_blockchain_claim(self):
        answer = "This decision can be independently verified on the blockchain."
        evidence = [{"tool": "get_blockchain_audit", "result": "Record Status: VERIFIED\nLedger Type: Simulated Mock Ledger Adapter"}]
        result = validate_evidence_claims(answer, evidence)
        self.assertFalse(result["passed"])
        self.assertIn("Unsupported blockchain independent verification claim", result["issues"])

    def test_detect_fallback_portfolio_without_opening_notice(self):
        answer = "Your portfolio is valued at LKR 1,000,000 with 35% in Technology."
        evidence = [{"tool": "get_portfolio_analysis", "result": "PORTFOLIO ANALYSIS — FALLBACK DATA (NO LIVE RECORD RETRIEVED)\nLocal mock baseline"}]
        result = validate_evidence_claims(answer, evidence)
        self.assertFalse(result["passed"])
        self.assertIn("Fallback portfolio described as actual user portfolio", result["issues"])
        self.assertIn("Missing required opening notice that no live portfolio was retrieved", result["issues"])

    def test_detect_synthetic_leak_in_live_portfolio_response(self):
        answer = "Here is your live portfolio: you have 1,000,000 LKR in Tech with a Diversification Score: 74."
        evidence = [{"tool": "get_portfolio_analysis", "result": "PORTFOLIO ANALYSIS — LIVE PLATFORM RECORD\nTotal Portfolio Value: USD 45,000"}]
        result = validate_evidence_claims(answer, evidence)
        self.assertFalse(result["passed"])
        self.assertIn("Synthetic baseline values substituted into live portfolio response", result["issues"])

    def test_detect_stale_portfolio_leak_when_portfolio_tool_not_called(self):
        answer = "Your asset allocation has a baseline valuation of 1,000,000."
        evidence = [{"tool": "get_fraud_analysis", "result": "Recorded Fraud Score: 0.15"}]
        result = validate_evidence_claims(answer, evidence)
        self.assertFalse(result["passed"])
        self.assertIn("Unrelated portfolio claims in non-portfolio response", result["issues"])

    def test_personalized_context_disabled_suppression(self):
        token = set_current_preferences({"personalized_context": False})
        try:
            answer = "Your portfolio has a baseline valuation of LKR 1,000,000 with 35% in Technology."
            evidence = [{"tool": "get_portfolio_analysis", "result": "PORTFOLIO ANALYSIS — PERSONALIZED CONTEXT DISABLED"}]
            result = validate_evidence_claims(answer, evidence)
            self.assertIn("Personalized financial context is disabled by user preference", result["answer"])
            self.assertNotIn("1,000,000", result["answer"])
        finally:
            reset_current_preferences(token)


if __name__ == "__main__":
    unittest.main()
