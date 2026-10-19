"""
Automated Regression Tests for Component 4 Portfolio Analysis Data Selection and Fallback Logic.

Verifies the six audit scenarios:
1. Live portfolio exists: Live data must be used; synthetic baseline values must not be substituted.
2. No portfolio exists: API returns empty portfolio result; reports zero holdings; fallback labels synthetic values.
3. Platform unavailable: API times out/errors; reports data could not be retrieved; does not claim zero holdings or present mock as live.
4. Partial portfolio data: Missing fields marked as Unavailable; no invented values.
5. Personalized context disabled: Preference blocks portfolio tool and suppresses mock portfolio values.
6. Evidence & Trust consistency: Distinguishes live, fallback demo, missing data, service error, and disabled context.
"""

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

from agent import call_llm, synthesize_portfolio_response
from evidence_validator import validate_evidence_claims
from preferences_context import reset_current_preferences, set_current_preferences
from tools.portfolio_tools import (
    get_portfolio_analysis,
    query_live_portfolio_service,
)


class TestPortfolioScenarios(unittest.TestCase):

    def setUp(self):
        # Reset preferences to defaults before each test
        self.pref_token = set_current_preferences({
            "explanation_enabled": True,
            "personalized_context": True,
            "strict_grounding": True,
        })

    def tearDown(self):
        reset_current_preferences(self.pref_token)

    # --------------------------------------------------------------------------
    # Scenario 1: Live portfolio exists
    # --------------------------------------------------------------------------
    def test_scenario_1_live_portfolio_uses_real_data_and_omits_synthetic_baseline(self):
        """When platform API returns valid portfolio and holdings, assistant must use live data without synthetic baseline values."""
        live_payload = {
            "id": 101,
            "name": "Enterprise Alpha Fund",
            "base_currency": "USD",
            "total_value": 450000.0,
            "holdings": [
                {
                    "symbol": "AAPL",
                    "quantity": 1000.0,
                    "current_price": 250.0,
                    "cost_basis": 200.0,
                    "avg_daily_volume": 40000000.0,
                },
                {
                    "symbol": "MSFT",
                    "quantity": 500.0,
                    "current_price": 400.0,
                    "cost_basis": 320.0,
                    "avg_daily_volume": 25000000.0,
                },
            ],
        }

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(live_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            raw_evidence = get_portfolio_analysis.invoke({"user_id": "1"})

        # 1. Raw tool evidence verification
        self.assertIn("PORTFOLIO ANALYSIS — LIVE PLATFORM RECORD", raw_evidence)
        self.assertIn("Enterprise Alpha Fund", raw_evidence)
        self.assertIn("Total Portfolio Value:\nUSD 450,000.00", raw_evidence)
        self.assertIn("AAPL", raw_evidence)
        self.assertIn("MSFT", raw_evidence)
        self.assertNotIn("FALLBACK DATA", raw_evidence)
        self.assertNotIn("LKR 1,000,000", raw_evidence)
        self.assertNotIn("11.8%", raw_evidence)

        # 2. Synthesized response verification
        response = synthesize_portfolio_response(raw_evidence)
        self.assertIn("Enterprise Alpha Fund", response)
        self.assertIn("USD 450,000.00", response)
        self.assertIn("AAPL", response)
        self.assertIn("MSFT", response)
        # Verify synthetic baseline values are NOT substituted
        self.assertNotIn("LKR 1,000,000", response)
        self.assertNotIn("11.8%", response)
        self.assertNotIn("Risk Score: 38", response)
        self.assertNotIn("Diversification Score: 74", response)

        # 3. Evidence validator enforcement: checks live_portfolio_verified and rejects synthetic values
        evidence = [{"tool": "get_portfolio_analysis", "result": raw_evidence}]
        val_clean = validate_evidence_claims(response, evidence)
        self.assertTrue(val_clean["passed"])
        self.assertIn("live_portfolio_verified", val_clean["checks_applied"])

        # If a contaminated response has synthetic baseline figures, validator must catch and fix it
        polluted_response = "Here is your portfolio Enterprise Alpha Fund valued at LKR 1,000,000 with 11.8% return."
        val_polluted = validate_evidence_claims(polluted_response, evidence)
        self.assertIn("Synthetic baseline values substituted into live portfolio response", val_polluted["issues"])
        # Corrected answer must restore the live data
        self.assertIn("Enterprise Alpha Fund", val_polluted["answer"])
        self.assertNotIn("LKR 1,000,000", val_polluted["answer"])

    # --------------------------------------------------------------------------
    # Scenario 2: No portfolio exists
    # --------------------------------------------------------------------------
    def test_scenario_2_no_portfolio_exists_reports_zero_holdings_and_labels_mock_data(self):
        """When API returns empty list or portfolio with 0 holdings, report zero holdings and label synthetic metrics."""
        # Case A: API returns empty portfolio list []
        mock_resp_empty = MagicMock()
        mock_resp_empty.status = 200
        mock_resp_empty.read.return_value = json.dumps([]).encode("utf-8")
        mock_resp_empty.__enter__.return_value = mock_resp_empty

        with patch("urllib.request.urlopen", return_value=mock_resp_empty):
            raw_evidence = get_portfolio_analysis.invoke({"user_id": "1"})

        self.assertIn("PORTFOLIO ANALYSIS — FALLBACK DATA", raw_evidence)
        self.assertIn("Empty portfolio list", raw_evidence)
        self.assertIn("account has no registered portfolios", raw_evidence)

        # Every synthetic metric must be explicitly labeled
        self.assertIn("Illustrative Baseline Valuation: LKR 1,000,000 (Mock Baseline Demonstration Only)", raw_evidence)
        self.assertIn("Illustrative Expected Annual Return: 11.8% (Synthetic Estimate)", raw_evidence)
        self.assertIn("Illustrative Risk Score: 38/100 (Synthetic Estimate)", raw_evidence)
        self.assertIn("Mock Baseline Example Only", raw_evidence)

        # Synthesized response
        response = synthesize_portfolio_response(raw_evidence)
        self.assertIn("No live portfolio was retrieved for this account from the platform service", response)
        self.assertIn("zero verified live holdings on record", response)
        self.assertIn("Mock Baseline Demonstration Only", response)
        self.assertIn("illustrative mock data for demonstration only", response)

        # Evidence validator enforcement
        evidence = [{"tool": "get_portfolio_analysis", "result": raw_evidence}]
        val = validate_evidence_claims(response, evidence)
        self.assertTrue(val["passed"])

        # Case B: Portfolio exists on platform but has 0 holdings
        empty_holdings_payload = {
            "id": 102,
            "name": "Empty Cash Vault",
            "base_currency": "USD",
            "total_value": 0.0,
            "holdings": [],
        }
        mock_resp_zero = MagicMock()
        mock_resp_zero.status = 200
        mock_resp_zero.read.return_value = json.dumps(empty_holdings_payload).encode("utf-8")
        mock_resp_zero.__enter__.return_value = mock_resp_zero

        with patch("urllib.request.urlopen", return_value=mock_resp_zero):
            zero_evidence = get_portfolio_analysis.invoke({"user_id": "1"})

        self.assertIn("PORTFOLIO ANALYSIS — LIVE PLATFORM RECORD", zero_evidence)
        self.assertIn("Active live portfolio record retrieved (0 holdings recorded)", zero_evidence)
        self.assertIn("Live API confirms 0 active positions", zero_evidence)
        self.assertIn("Total Portfolio Value:\nUSD 0.00", zero_evidence)

    # --------------------------------------------------------------------------
    # Scenario 3: Platform unavailable
    # --------------------------------------------------------------------------
    def test_scenario_3_platform_unavailable_does_not_claim_empty_portfolio(self):
        """When platform API times out or errors, report retrieval failure without claiming zero holdings or presenting mock data as live."""
        # Case A: Timeout
        with patch("urllib.request.urlopen", side_effect=TimeoutError("Request timed out connecting to platform:8100")):
            timeout_evidence = get_portfolio_analysis.invoke({"user_id": "1"})

        self.assertIn("Request timed out", timeout_evidence)
        self.assertIn("This does not indicate an empty portfolio; live holdings could not be retrieved due to a platform service connection failure or timeout", timeout_evidence)

        response_timeout = synthesize_portfolio_response(timeout_evidence)
        self.assertIn("Live portfolio data could not be retrieved because the platform service is unreachable or timed out", response_timeout)
        self.assertIn("This does not indicate an empty portfolio", response_timeout)
        # Must NOT claim user has 0 holdings
        self.assertNotIn("The user has zero verified live holdings on record", response_timeout)

        # Case B: HTTP 500 Internal Server Error
        err_500 = urllib.error.HTTPError(
            url="http://localhost:8100/portfolios",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=MagicMock(read=lambda: b'{"detail":"Database unreachable"}'),
        )
        with patch("urllib.request.urlopen", side_effect=err_500):
            err_evidence = get_portfolio_analysis.invoke({"user_id": "1"})

        response_500 = synthesize_portfolio_response(err_evidence)
        self.assertIn("Live portfolio data could not be retrieved because the platform service is unreachable or timed out", response_500)
        self.assertIn("This does not indicate an empty portfolio", response_500)
        self.assertNotIn("The user has zero verified live holdings on record", response_500)

        # Case C: Network unreachable
        err_network = urllib.error.URLError("Connection refused by host")
        with patch("urllib.request.urlopen", side_effect=err_network):
            net_evidence = get_portfolio_analysis.invoke({"user_id": "1"})

        response_net = synthesize_portfolio_response(net_evidence)
        self.assertIn("Live portfolio data could not be retrieved because the platform service is unreachable or timed out", response_net)
        self.assertIn("This does not indicate an empty portfolio", response_net)

        # Evidence validation: must pass without claiming actual portfolio
        evidence = [{"tool": "get_portfolio_analysis", "result": timeout_evidence}]
        val = validate_evidence_claims(response_timeout, evidence)
        self.assertTrue(val["passed"])

    # --------------------------------------------------------------------------
    # Scenario 4: Partial portfolio data
    # --------------------------------------------------------------------------
    def test_scenario_4_partial_portfolio_data_marks_unavailable_fields_honestly(self):
        """When some fields are missing from platform record, use only verified fields and mark missing as Unavailable."""
        partial_payload = {
            "id": 103,
            "name": "Partial Data Portfolio",
            "base_currency": "USD",
            "total_value": None,  # Missing valuation
            "holdings": [
                {
                    "symbol": "AMZN",
                    "quantity": 100.0,
                    "current_price": None,  # Missing price!
                    "cost_basis": 150.0,
                    "avg_daily_volume": None,
                },
                {
                    "symbol": "GOOGL",
                    "quantity": 50.0,
                    "current_price": 180.0,
                    "cost_basis": None,  # Missing cost basis
                    "avg_daily_volume": 20000000.0,
                },
            ],
        }

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(partial_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            raw_evidence = get_portfolio_analysis.invoke({"user_id": "1"})

        # 1. Total Portfolio Value must be marked Unavailable, not invented 0.00
        self.assertIn("Total Portfolio Value:\nUnavailable / Not recorded", raw_evidence)
        self.assertIn("Active live portfolio record retrieved (partial fields marked)", raw_evidence)
        self.assertIn("AMZN: 100.00 units @ Price: Unavailable = Value: Unavailable (Weight: Unavailable)", raw_evidence)
        self.assertIn("GOOGL: 50.00 units @USD 180.00 = USD 9,000.00 (Weight: Unavailable)", raw_evidence)
        self.assertIn("Notice: Some fields (such as valuation, price, or cost basis) are not recorded on the platform and are explicitly marked as Unavailable", raw_evidence)

        # 2. Synthesized response must show partial notice
        response = synthesize_portfolio_response(raw_evidence)
        self.assertIn("Total Market Value: Unavailable / Not recorded", response)
        self.assertIn("Notice: Only verified fields from the live platform record are presented. Unavailable fields are explicitly marked as unavailable rather than estimated", response)
        self.assertIn("AMZN", response)
        self.assertIn("GOOGL", response)
        # Must not contain invented baseline figures
        self.assertNotIn("LKR 1,000,000", response)
        self.assertNotIn("11.8%", response)

        # 3. Evidence validator verification
        evidence = [{"tool": "get_portfolio_analysis", "result": raw_evidence}]
        val = validate_evidence_claims(response, evidence)
        self.assertTrue(val["passed"])

    # --------------------------------------------------------------------------
    # Scenario 5: Personalized context disabled
    # --------------------------------------------------------------------------
    def test_scenario_5_personalized_context_disabled_blocks_tool_and_suppresses_mock_data(self):
        """When Responsible AI preference disables personalized context, portfolio tool is not invoked and no mock values are exposed."""
        pref_token = set_current_preferences({"personalized_context": False})
        try:
            # 1. Direct tool invocation returns clean disabled notice without mock baseline data
            direct_output = get_portfolio_analysis.invoke({"user_id": "1"})
            self.assertIn("PORTFOLIO ANALYSIS — PERSONALIZED CONTEXT DISABLED", direct_output)
            self.assertIn("Personalized financial context is disabled by user preference", direct_output)
            self.assertIn("No anonymous or fallback portfolio data has been substituted", direct_output)
            self.assertNotIn("LKR 1,000,000", direct_output)
            self.assertNotIn("11.8%", direct_output)
            self.assertNotIn("Risk Score: 38", direct_output)

            # 2. Agent chat routing test: portfolio request must return notice immediately without tool binding
            state = {
                "messages": [
                    MagicMock(
                        type="human",
                        content="Can you analyze my portfolio allocation and total risk score?",
                    )
                ]
            }
            agent_result = call_llm(state)
            self.assertIn("messages", agent_result)
            ai_msg = agent_result["messages"][0]
            msg_content = str(getattr(ai_msg, "content", ""))
            self.assertIn("Personalized financial context is disabled by user preference", msg_content)
            self.assertIn("Account-specific portfolio lookup and asset holdings analysis are blocked", msg_content)
            # Ensure no mock portfolio numbers are exposed
            self.assertNotIn("LKR 1,000,000", msg_content)
            self.assertNotIn("11.8%", msg_content)
            self.assertNotIn("Risk Score: 38", msg_content)
            # Ensure no tool calls are attached
            self.assertEqual(getattr(ai_msg, "tool_calls", []), [])

            # 3. Evidence validator enforcement
            evidence = [{"tool": "get_portfolio_analysis", "result": direct_output}]
            val = validate_evidence_claims(msg_content, evidence)
            self.assertTrue(val["passed"])
            self.assertNotIn("LKR 1,000,000", val["answer"])
        finally:
            reset_current_preferences(pref_token)

    # --------------------------------------------------------------------------
    # Scenario 6: Evidence & Trust Panel distinction
    # --------------------------------------------------------------------------
    def test_scenario_6_evidence_and_trust_distinguishes_all_five_conditions(self):
        """Ensures that responses and evidence structures maintain strict provenance distinction across all conditions."""
        # 1. Live portfolio
        live_text = "PORTFOLIO ANALYSIS — LIVE PLATFORM RECORD\nTotal Portfolio Value:\nUSD 500,000.00"
        synth_live = synthesize_portfolio_response(live_text)
        self.assertIn("recorded portfolio summary from the live platform service", synth_live)

        # 2. Fallback empty portfolio
        empty_text = "PORTFOLIO ANALYSIS — FALLBACK DATA (NO LIVE RECORD RETRIEVED)\nPlatform service note: No portfolios registered for user on platform service."
        synth_empty = synthesize_portfolio_response(empty_text)
        self.assertIn("zero verified live holdings on record", synth_empty)
        self.assertIn("Mock Baseline Demonstration Only", synth_empty)

        # 3. Service outage
        outage_text = "PORTFOLIO ANALYSIS — FALLBACK DATA (NO LIVE RECORD RETRIEVED)\nPlatform service note: Request timed out (connection error)."
        synth_outage = synthesize_portfolio_response(outage_text)
        self.assertIn("Live portfolio data could not be retrieved because the platform service is unreachable or timed out", synth_outage)
        self.assertIn("This does not indicate an empty portfolio", synth_outage)

        # 4. Partial data
        partial_text = "PORTFOLIO ANALYSIS — LIVE PLATFORM RECORD\nTotal Portfolio Value:\nUnavailable / Not recorded\nRECORDED HOLDINGS:\n- AAPL: 10.00 units @ Price: Unavailable = Value: Unavailable"
        synth_partial = synthesize_portfolio_response(partial_text)
        self.assertIn("Unavailable / Not recorded", synth_partial)
        self.assertIn("Unavailable fields are explicitly marked as unavailable", synth_partial)

        # 5. Disabled
        disabled_text = "PORTFOLIO ANALYSIS — PERSONALIZED CONTEXT DISABLED\nPersonalized financial context is disabled"
        synth_disabled = synthesize_portfolio_response(disabled_text)
        self.assertIn("Personalized financial context is disabled by user preference", synth_disabled)


if __name__ == "__main__":
    unittest.main()
