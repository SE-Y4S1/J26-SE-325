"""
Test suite covering FinTrust Contextual Intelligence & Reasoning (PP1).
Validates all 22 required test cases specified in Section 12:
1. "What is volatility?"
2. "What is the difference between VaR and CVaR?"
3. "Explain diversification in simple English."
4. "What does amount_hard_block mean?"
5. A genuinely unrelated question receives an appropriate short redirection.
6. "Why did you give me this answer?"
7. "Which data sources did you use?"
8. "Which parts are calculated and which are assumptions?"
9. "What does that score mean?" after a fraud result.
10. "Which transaction signals were unavailable?"
11. "Did you use SHAP?" when the live response did not compute SHAP.
12. Multi-turn context does not mix two different transaction IDs.
13. A fraud follow-up receives a focused answer rather than the full report every time.
14. A calibrated-probability question respects the recorded calibration status.
15. A live audit lookup returning not found is not reported as verified.
16. Mock fallback remains explicitly labeled.
17. An audit question does not append unrelated fraud evidence.
18. A tool error is not presented as successful verification.
19. A portfolio question uses the correct authorized user's data.
20. A risk metric is not fabricated when the required model/data is unavailable.
21. A hypothetical 10% decline is clearly labeled as a scenario.
22. Existing Evidence Validator safeguards still reject unsupported claims.
"""

import json
import unittest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from agent import (
    agent,
    classify_domain_gate,
    get_deterministic_fast_path,
    synthesize_fraud_response,
    synthesize_audit_response,
    synthesize_portfolio_response,
    render_hypothetical_calc,
    resolve_transaction_id,
    OUT_OF_DOMAIN_RESPONSE,
)
from evidence_validator import validate_evidence_claims
from tools.audit_tools import get_blockchain_audit
from tools.fraud_tools import get_fraud_analysis
from tools.portfolio_tools import get_portfolio_analysis, resolve_user_id
from preferences_context import set_current_preferences, reset_current_preferences, DEFAULT_PREFERENCES
from request_auth import set_caller_authorization, reset_caller_authorization


class TestContextualIntelligence(unittest.TestCase):
    def setUp(self):
        self.pref_token = set_current_preferences(dict(DEFAULT_PREFERENCES))
        self.auth_token = set_caller_authorization(None)

    def tearDown(self):
        reset_current_preferences(self.pref_token)
        reset_caller_authorization(self.auth_token)

    # ---------------------------------------------------------
    # Category 1: Scope and General Finance (Cases 1 - 5)
    # ---------------------------------------------------------

    def test_01_what_is_volatility(self):
        """1. 'What is volatility?' must receive a clear conceptual answer, not a generic rejection."""
        gate = classify_domain_gate("What is volatility?")
        self.assertEqual(gate["category"], "IN_DOMAIN_FINANCIAL")
        reply = get_deterministic_fast_path("What is volatility?")
        self.assertIsNotNone(reply)
        self.assertNotIn(OUT_OF_DOMAIN_RESPONSE, reply)
        self.assertIn("Volatility", reply)
        self.assertIn("standard deviation", reply.lower())

    def test_02_var_vs_cvar(self):
        """2. 'What is the difference between VaR and CVaR?' must receive a comparative risk explanation."""
        gate = classify_domain_gate("What is the difference between VaR and CVaR?")
        self.assertEqual(gate["category"], "IN_DOMAIN_FINANCIAL")
        reply = get_deterministic_fast_path("What is the difference between VaR and CVaR?")
        self.assertIsNotNone(reply)
        self.assertNotIn(OUT_OF_DOMAIN_RESPONSE, reply)
        self.assertIn("Value at Risk (VaR)", reply)
        self.assertIn("Conditional Value at Risk (CVaR", reply)

    def test_03_explain_diversification(self):
        """3. 'Explain diversification in simple English.' must be answered plainly."""
        gate = classify_domain_gate("Explain diversification in simple English.")
        self.assertEqual(gate["category"], "IN_DOMAIN_FINANCIAL")
        reply = get_deterministic_fast_path("Explain diversification in simple English.")
        self.assertIsNotNone(reply)
        self.assertNotIn(OUT_OF_DOMAIN_RESPONSE, reply)
        self.assertIn("Diversification", reply)
        self.assertIn("basket", reply.lower())

    def test_04_amount_hard_block_meaning(self):
        """4. 'What does amount_hard_block mean?' must explain the gateway rule directly."""
        gate = classify_domain_gate("What does amount_hard_block mean?")
        self.assertEqual(gate["category"], "IN_DOMAIN_FINANCIAL")
        reply = get_deterministic_fast_path("What does amount_hard_block mean?")
        self.assertIsNotNone(reply)
        self.assertNotIn(OUT_OF_DOMAIN_RESPONSE, reply)
        self.assertIn("amount_hard_block", reply)
        self.assertIn("security gateway rule", reply.lower())

    def test_05_unrelated_question_redirection(self):
        """5. A genuinely unrelated question receives an appropriate short redirection."""
        clean = "How do I bake sourdough bread?"
        gate = classify_domain_gate(clean)
        self.assertEqual(gate["category"], "OUT_OF_DOMAIN")
        fast = get_deterministic_fast_path(clean)
        self.assertIn(OUT_OF_DOMAIN_RESPONSE, fast)

    # ---------------------------------------------------------
    # Category 2: Follow-up and Explanation (Cases 6 - 12)
    # ---------------------------------------------------------

    def test_06_why_did_you_give_me_this_answer(self):
        """6. 'Why did you give me this answer?' explains the decision rationale."""
        gate = classify_domain_gate("Why did you give me this answer?")
        self.assertEqual(gate["category"], "IN_DOMAIN_FINANCIAL")
        reply = get_deterministic_fast_path("Why did you give me this answer?")
        self.assertIsNotNone(reply)
        self.assertNotIn(OUT_OF_DOMAIN_RESPONSE, reply)
        self.assertIn("grounded in verified tool evidence", reply)
        self.assertIn("factual data", reply.lower())

    def test_07_which_data_sources_did_you_use(self):
        """7. 'Which data sources did you use?' identifies verified stores vs general knowledge."""
        gate = classify_domain_gate("Which data sources did you use?")
        self.assertEqual(gate["category"], "IN_DOMAIN_FINANCIAL")
        reply = get_deterministic_fast_path("Which data sources did you use?")
        self.assertIsNotNone(reply)
        self.assertNotIn(OUT_OF_DOMAIN_RESPONSE, reply)
        self.assertIn("Component 2", reply)
        self.assertIn("Blockchain Audit Bridge", reply)

    def test_08_calculated_vs_assumptions(self):
        """8. 'Which parts are calculated and which are assumptions?' separates facts from assumptions."""
        gate = classify_domain_gate("Which parts are calculated and which are assumptions?")
        self.assertEqual(gate["category"], "IN_DOMAIN_FINANCIAL")
        reply = get_deterministic_fast_path("Which parts are calculated and which are assumptions?")
        self.assertIsNotNone(reply)
        self.assertNotIn(OUT_OF_DOMAIN_RESPONSE, reply)
        self.assertIn("Recorded Facts", reply)
        self.assertIn("Actual Calculations", reply)
        self.assertIn("Assumptions", reply)

    def test_09_what_does_that_score_mean(self):
        """9. 'What does that score mean?' after a fraud result interprets the score."""
        gate = classify_domain_gate("What does that score mean?")
        self.assertEqual(gate["category"], "IN_DOMAIN_FINANCIAL")
        history = [
            {"role": "user", "content": "Check fraud for TX1001"},
            {"role": "assistant", "content": "Transaction TX1001 recorded fraud score 0.87 (HIGH risk)."},
        ]
        reply = get_deterministic_fast_path("What does that score mean?", history=history)
        self.assertIsNotNone(reply)
        self.assertNotIn(OUT_OF_DOMAIN_RESPONSE, reply)
        self.assertIn("Component 2", reply)
        self.assertIn("0.87", reply)
        self.assertIn("anomaly risk rating", reply)

    def test_10_which_transaction_signals_were_unavailable(self):
        """10. 'Which transaction signals were unavailable?' reports missing telemetry truthfully."""
        gate = classify_domain_gate("Which transaction signals were unavailable?")
        self.assertEqual(gate["category"], "IN_DOMAIN_FINANCIAL")
        reply = get_deterministic_fast_path("Which transaction signals were unavailable?")
        self.assertIsNotNone(reply)
        self.assertNotIn(OUT_OF_DOMAIN_RESPONSE, reply)
        self.assertIn("Device Signals: Raw device fingerprints", reply)
        self.assertIn("Location Signals", reply)

    def test_11_did_you_use_shap(self):
        """11. 'Did you use SHAP?' clarifies that SHAP was not executed on live models."""
        gate = classify_domain_gate("Did you use SHAP?")
        self.assertEqual(gate["category"], "IN_DOMAIN_FINANCIAL")
        reply = get_deterministic_fast_path("Did you use SHAP?")
        self.assertIsNotNone(reply)
        self.assertNotIn(OUT_OF_DOMAIN_RESPONSE, reply)
        self.assertIn("No, SHAP (SHapley Additive exPlanations) was not applied to the live Component 2 fraud model", reply)

    def test_12_multi_turn_context_does_not_mix_transactions(self):
        """12. Multi-turn context does not mix two different transaction IDs."""
        history = [
            {"role": "user", "content": "Review transaction TX1001"},
            {"role": "assistant", "content": "Transaction TX1001 was blocked with score 0.87."},
            {"role": "user", "content": "Now check transaction PP1-DEMO-002"},
            {"role": "assistant", "content": "Transaction PP1-DEMO-002 is under review."},
        ]
        # Ambiguous follow-up without an explicit ID does not arbitrarily guess TX1001
        res_id, clar = resolve_transaction_id("Why was it blocked?", history=history)
        self.assertIsNone(res_id)
        self.assertIn("multiple transactions", clar.lower())
        self.assertIn("TX1001", clar)
        self.assertIn("PP1-DEMO-002", clar)

        # Single transaction history resolves unambiguously to that exact ID
        single_hist = [
            {"role": "user", "content": "Review transaction PP1-DEMO-002"},
            {"role": "assistant", "content": "Transaction PP1-DEMO-002 is under review."},
        ]
        res_single, _ = resolve_transaction_id("Why was it blocked?", history=single_hist)
        self.assertEqual(res_single, "PP1-DEMO-002")

        # Explicitly mentioning TX1001 resolves strictly to TX1001
        res_id2, _ = resolve_transaction_id("What about TX1001?", history=history)
        self.assertEqual(res_id2, "TX1001")

    # ---------------------------------------------------------
    # Category 3: Fraud and Audit (Cases 13 - 18)
    # ---------------------------------------------------------

    def test_13_fraud_followup_focused_answer(self):
        """13. A fraud follow-up receives a focused answer rather than the full 30-line report."""
        tool_content = """
FRAUD ANALYSIS — LIVE COMPONENT 2 AUDIT RECORD
Transaction ID: TX999
Recorded Fraud Score: 0.8500
Recorded Risk Level: HIGH
Recorded Status: BLOCKED
Model Version: v1.0.0 (trained benchmark)
Behavioral Stream Anomaly Score (LSTM): 0.8200
Relational Stream Risk Score (GNN): 0.8800
DECISION RULES & REASONING:
- amount_hard_block (Transaction amount 5000000.0 >= 1000000.0)
FEATURE ATTRIBUTIONS & CONTRIBUTING FACTORS:
- TransactionAmt: +0.6500
GENUINE SIGNALS & TELEMETRY STATUS:
- Device Signals: Unavailable
- Location Signals: Unavailable
"""
        # User asks specifically why it was blocked
        focused_reply = synthesize_fraud_response(
            tool_content,
            user_query="Why was my transaction blocked?"
        )
        self.assertIn("Transaction Block Analysis (TX999):", focused_reply)
        self.assertIn("BLOCKED", focused_reply)
        self.assertIn("amount_hard_block", focused_reply)
        # Should NOT dump the generic full report header
        self.assertNotIn("Here is the verified fraud analysis for transaction", focused_reply)

    def test_14_calibrated_probability_status(self):
        """14. A calibrated-probability question respects the recorded calibration status."""
        tool_content = """
FRAUD ANALYSIS — LIVE COMPONENT 2 AUDIT RECORD
Transaction ID: TX888
Recorded Fraud Score: 0.1000
Recorded Risk Level: LOW
Recorded Status: APPROVED
Model Version: v1.0.0 (trained benchmark)
Behavioral Stream Anomaly Score (LSTM): 0.1000
Relational Stream Risk Score (GNN): 0.1000
DECISION RULES & REASONING:
- Passed standard thresholds
"""
        reply = synthesize_fraud_response(
            tool_content,
            user_query="Is this fraud score a calibrated probability?"
        )
        self.assertIn("NOT a calibrated probability", reply)
        self.assertIn("A score of 0.1000 must not be equated to a 10.0% mathematical likelihood", reply)

    def test_15_live_audit_not_found_not_reported_as_verified(self):
        """15. A live audit lookup returning not found is not reported as verified."""
        with patch("tools.audit_tools.query_live_audit_service") as mock_live, \
             patch("tools.audit_tools.load_data") as mock_data:
            mock_live.return_value = {
                "available": True,
                "status_code": 404,
                "data": None,
                "error": "HTTP 404: Not Found",
            }
            mock_data.return_value = {}

            tool_output = get_blockchain_audit.invoke({"transaction_id": "TX-MISSING-99"})
            self.assertIn("BLOCKCHAIN AUDIT RECORD — NOT FOUND", tool_output)
            self.assertIn("Record Status:\nNOT_FOUND", tool_output)
            self.assertNotIn("VERIFIED", tool_output)

            synth = synthesize_audit_response(tool_output, user_query="Is TX-MISSING-99 verified?")
            self.assertIn("Record Status: NOT_FOUND", synth)
            self.assertNotIn("VERIFIED on simulated adapter", synth)

    def test_16_mock_fallback_explicitly_labeled(self):
        """16. Mock fallback remains explicitly labeled with status NOT_FOUND_ON_LIVE_BRIDGE."""
        with patch("tools.audit_tools.query_live_audit_service") as mock_live, \
             patch("tools.audit_tools.load_data") as mock_data:
            mock_live.return_value = {
                "available": True,
                "status_code": 404,
                "data": None,
                "error": "Not found",
            }
            mock_data.return_value = {
                "audit": {
                    "PP1-DEMO-001": {
                        "audit_id": "AUDIT-PP1-DEMO-001",
                        "status": "VERIFIED",
                        "decision": "RECORDED_BLOCKED",
                        "block_number": 101,
                        "timestamp": "2026-03-29T10:00:00Z"
                    }
                }
            }

            tool_output = get_blockchain_audit.invoke({"transaction_id": "PP1-DEMO-001"})
            self.assertIn("BLOCKCHAIN AUDIT RECORD — FALLBACK DATA", tool_output)
            self.assertIn("Record Status:\nNOT_FOUND_ON_LIVE_BRIDGE", tool_output)
            self.assertIn("FALLBACK NOTICE:", tool_output)

            synth = synthesize_audit_response(tool_output)
            self.assertIn("Fallback Baseline", synth)
            self.assertIn("Verification Status: NOT_FOUND_ON_LIVE_BRIDGE", synth)
            self.assertIn("Independent blockchain verification is NOT established", synth)

    def test_17_audit_question_does_not_append_fraud_evidence(self):
        """17. An audit question does not append unrelated fraud evidence in evidence validator."""
        audit_tool_text = "BLOCKCHAIN AUDIT RECORD: TX1001 verified on simulated adapter Block #1."
        fraud_tool_text = "FRAUD ANALYSIS: TX1001 recorded score 0.87 HIGH risk BLOCKED."
        combined_tools = f"{audit_tool_text}\n\n{fraud_tool_text}"

        answer = "Transaction TX1001 is independently verified on the blockchain."
        evidence = [
            {"tool": "get_blockchain_audit", "result": audit_tool_text},
            {"tool": "get_fraud_analysis", "result": fraud_tool_text},
        ]
        result = validate_evidence_claims(
            answer=answer,
            evidence=evidence,
        )
        self.assertFalse(result["passed"])
        self.assertIn("blockchain audit", result["answer"].lower())
        self.assertNotIn("0.87", result["answer"])

    def test_18_tool_error_not_presented_as_verification(self):
        """18. A tool error (e.g. service offline) is not presented as successful verification."""
        tool_content = """
BLOCKCHAIN AUDIT RECORD — SERVICE UNAVAILABLE

Transaction ID: TX777
Record Status: NOT_FOUND_ON_LIVE_BRIDGE
STATUS:
The live blockchain audit bridge at http://localhost:8002 could not be reached.
"""
        synth = synthesize_audit_response(tool_content)
        self.assertIn("Notice: The live blockchain audit bridge is currently unreachable or offline", synth)
        self.assertIn("Verification Status: Unverified", synth)
        self.assertIn("does not present connection failures or offline services as successful verifications", synth)

    # ---------------------------------------------------------
    # Category 4: Portfolio and Evidence (Cases 19 - 22)
    # ---------------------------------------------------------

    def test_19_portfolio_authorized_user_data(self):
        """19. A portfolio question uses the correct authorized user's data from context."""
        from request_auth import set_caller_authorization
        import jwt
        # Mint token for user 42
        token = jwt.encode(
            {"sub": "42", "iat": 1000, "exp": 9999999999, "iss": "j26-se-325-platform"},
            "test-secret-at-least-32-chars-long-0123456789",
            algorithm="HS256"
        )
        set_caller_authorization(f"Bearer {token}")
        with patch.dict("os.environ", {"JWT_SECRET": "test-secret-at-least-32-chars-long-0123456789"}):
            uid = resolve_user_id(None)
            self.assertEqual(uid, "42")

    def test_20_risk_metric_not_fabricated(self):
        """20. A risk metric is not fabricated when the required model/data is unavailable."""
        tool_content = """
PORTFOLIO ANALYSIS — LIVE PLATFORM RECORD
Portfolio Name: Growth Portfolio
Recorded Holdings Count: 4
Base Currency: LKR
Total Market Value: LKR 10,000,000.00
RECORDED HOLDINGS:
- JKH.N0000: 25,000 units @ LKR 200.00 = LKR 5,000,000.00 (50.0%)
- COMB.N0000: 25,000 units @ LKR 100.00 = LKR 2,500,000.00 (25.0%)
- AAPL: 50 units @ USD 150.00 = LKR 1,500,000.00 (15.0%)
- SPY: 20 units @ USD 500.00 = LKR 1,000,000.00 (10.0%)
"""
        with patch("tools.portfolio_tools.query_component1_health") as mock_c1:
            mock_c1.return_value = {
                "reachable": False,
                "error": "Connection refused",
                "model_version": "unavailable",
                "available_forecasters": [],
            }
            synth = synthesize_portfolio_response(
                tool_content,
                user_query="What is my portfolio VaR and risk score?"
            )
            self.assertIn("cannot calculate a 95% one-day Value at Risk (VaR), CVaR, portfolio volatility, expected return, or portfolio-level risk score", synth)
            self.assertIn("No hypothetical or synthetic risk score (such as 38/100) has been fabricated", synth)

    def test_21_hypothetical_decline_labeled_as_scenario(self):
        """21. A hypothetical 10% decline is clearly labeled as a scenario on recorded holdings."""
        holdings = [
            {"symbol": "JKH.N0000", "quantity": 1000.0, "price": 200.0, "value": 200000.0, "weight": 66.7},
            {"symbol": "COMB.N0000", "quantity": 1000.0, "price": 100.0, "value": 100000.0, "weight": 33.3},
        ]
        calc = render_hypothetical_calc(
            query="What would happen if my largest holding fell by 10%?",
            holdings=holdings,
            currency="LKR",
            total_val_num=300000.0,
        )
        self.assertIn("Hypothetical Scenario Calculation (10% Price Fall on JKH.N0000):", calc)
        self.assertIn("Total Estimated Loss: LKR 20,000.00", calc)
        self.assertIn("Resulting Portfolio Value: LKR 280,000.00", calc)
        self.assertIn("Deterministic calculation based on recorded platform holdings; not a market forecast or predictive risk model output", calc)

    def test_22_evidence_validator_rejects_unsupported_claims(self):
        """22. Existing Evidence Validator safeguards still reject unsupported claims."""
        tool_output = "PORTFOLIO ANALYSIS — FALLBACK DATA: Baseline Valuation: LKR 1,000,000. Expected Return: 11.8%."
        hallucinated_answer = "Your verified portfolio is worth LKR 1,000,000 and has an expected annual return of 11.8%."
        evidence = [{"tool": "get_portfolio_analysis", "result": tool_output}]
        result = validate_evidence_claims(
            answer=hallucinated_answer,
            evidence=evidence,
        )
        self.assertFalse(result["passed"])
        self.assertTrue(len(result["issues"]) > 0)
        self.assertIn("Missing required opening notice that no live portfolio was retrieved", result["issues"])


if __name__ == "__main__":
    unittest.main()
