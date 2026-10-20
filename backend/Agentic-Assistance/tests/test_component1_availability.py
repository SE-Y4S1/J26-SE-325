"""
Tests for Component 1 Availability, Internal Service URL, and Portfolio Risk-Model Limitations.
Verifies that:
1. Component 4 dynamically resolves the internal service URL (PORTFOLIO_OPTIMIZATION_URL).
2. Service failure or missing models NEVER produce fabricated VaR, CVaR, volatility, expected return, or risk scores.
3. Available forecaster status (e.g. 'baseline_lstm') and unavailable-metric disclosures are accurately rendered.
4. Factual concentration analysis from recorded holdings is preserved regardless of Component 1 status.
"""

import os
import sys
import unittest
from unittest.mock import patch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from agent import (
    analyze_portfolio_request,
    synthesize_portfolio_response,
)
from preferences_context import reset_current_preferences, set_current_preferences
from tools.portfolio_tools import (
    get_component1_url,
    query_component1_health,
)


class TestComponent1AvailabilityAndRiskLimits(unittest.TestCase):

    def setUp(self):
        self.pref_token = set_current_preferences({
            "explanation_enabled": True,
            "personalized_context": True,
            "strict_grounding": True,
        })
        self.sample_live_evidence = (
            "PORTFOLIO ANALYSIS — LIVE PLATFORM RECORD\n\n"
            "Data Source:\n"
            "Live platform portfolio store (http://platform:8100/portfolios).\n\n"
            "Portfolio Name: Core Growth Portfolio\n"
            "Base Currency: LKR\n"
            "Total Portfolio Value: LKR 6,746,660.00\n"
            "Holding Count: 4\n\n"
            "RECORDED HOLDINGS:\n"
            "- JKH: 20,000.00 units @ LKR 184.00 = LKR 3,680,000.00 (54.5% of portfolio), Cost: LKR 170.00, ADV: 850,000\n"
            "- COMB: 20,000.00 units @ LKR 150.00 = LKR 3,000,000.00 (44.5% of portfolio), Cost: LKR 145.00, ADV: 620,000\n"
            "- AAPL: 100.00 units @ LKR 336.60 = LKR 33,660.00 (0.5% of portfolio), Cost: LKR 320.00, ADV: 48,000,000\n"
            "- SPY: 100.00 units @ LKR 330.00 = LKR 33,000.00 (0.5% of portfolio), Cost: LKR 310.00, ADV: 55,000,000\n\n"
            "OBSERVATIONS:\n"
            "- Live holdings retrieved directly from the platform portfolio store."
        )

    def tearDown(self):
        reset_current_preferences(self.pref_token)

    def test_component1_url_resolution(self):
        """Verify get_component1_url resolves environment overrides properly."""
        with patch.dict(os.environ, {"PORTFOLIO_OPTIMIZATION_URL": "http://component1:8000"}):
            self.assertEqual(get_component1_url(), "http://component1:8000")

        with patch.dict(os.environ, {"PORTFOLIO_OPTIMIZATION_URL": "", "COMPONENT1_SERVICE_URL": "http://custom-c1:9000"}):
            self.assertEqual(get_component1_url(), "http://custom-c1:9000")

    def test_component1_unreachable_discloses_failure_and_refuses_to_invent_risk_metrics(self):
        """When Component 1 is unreachable, report service failure without inventing any risk metrics."""
        mock_unreachable = {
            "reachable": False,
            "error": "<urlopen error [Errno 111] Connection refused>",
            "model_version": "unavailable",
            "available_forecasters": [],
            "kafka_connected": False,
            "url": "http://component1:8000",
        }
        with patch("tools.portfolio_tools.query_component1_health", return_value=mock_unreachable):
            query = "What is my portfolio Value at Risk (VaR), CVaR, and overall risk score?"
            resp = synthesize_portfolio_response(self.sample_live_evidence, user_query=query)

            # Must accurately report service status and url
            self.assertIn("Service unreachable at http://component1:8000", resp)
            self.assertIn("Connection refused", resp)

            # Must disclose unavailability of risk metrics
            self.assertIn("cannot calculate a 95% one-day Value at Risk (VaR)", resp)
            self.assertIn("The system has NOT executed a portfolio-level risk model", resp)

            # Must strictly forbid fabricating VaR, CVaR, volatility, or risk score
            self.assertIn("No synthetic or estimated VaR figure is fabricated", resp)
            self.assertIn("No hypothetical or synthetic risk score (such as 38/100) has been fabricated", resp)
            self.assertNotIn("Risk Score: 38/100", resp)
            self.assertNotIn("Expected Annual Return: 11.8%", resp)

    def test_component1_reachable_with_only_baseline_lstm_discloses_model_limits(self):
        """When Component 1 is reachable but has only individual baseline_lstm forecaster, report that limitation."""
        mock_healthy_lstm_only = {
            "reachable": True,
            "status": "ok",
            "model_version": "unregistered",
            "available_forecasters": ["baseline_lstm"],
            "kafka_connected": True,
            "url": "http://component1:8000",
        }
        with patch("tools.portfolio_tools.query_component1_health", return_value=mock_healthy_lstm_only):
            query = "Calculate my portfolio's 95% one-day VaR and portfolio risk score using Component 1."
            resp = synthesize_portfolio_response(self.sample_live_evidence, user_query=query)

            # Must report reachable with exact model information
            self.assertIn("Reachable at http://component1:8000", resp)
            self.assertIn("Available forecasters: ['baseline_lstm']", resp)
            self.assertIn("baseline_lstm", resp)

            # Must disclose that no portfolio-level risk model has been run
            self.assertIn("The system has NOT executed a portfolio-level risk model", resp)
            self.assertIn("Component 1 currently registers only an individual asset price forecaster ('baseline_lstm')", resp)

            # Responsible AI disclosure
            self.assertIn("No synthetic or estimated VaR figure is fabricated", resp)
            self.assertIn("No hypothetical or synthetic risk score (such as 38/100) has been fabricated", resp)

    def test_volatility_query_routes_to_risk_inquiry_without_hallucinating_volatility(self):
        """User asks 'What is my portfolio volatility?' -> must report volatility model requirements and unavailability."""
        mock_healthy = {
            "reachable": True,
            "status": "ok",
            "model_version": "unregistered",
            "available_forecasters": ["baseline_lstm"],
            "kafka_connected": True,
            "url": "http://component1:8000",
        }
        with patch("tools.portfolio_tools.query_component1_health", return_value=mock_healthy):
            query = "What is my portfolio volatility?"
            plan = analyze_portfolio_request(query, [])
            self.assertIn("risk_model_inquiry", plan["tasks"])

            resp = synthesize_portfolio_response(self.sample_live_evidence, user_query=query)
            self.assertIn("Portfolio Volatility Evaluation", resp)
            self.assertIn("cannot calculate a 95% one-day Value at Risk (VaR), CVaR, portfolio volatility", resp)
            self.assertIn("Variance-Covariance Matrix", resp)
            self.assertIn("No synthetic or estimated VaR figure is fabricated", resp)

    def test_expected_return_query_routes_to_risk_inquiry_without_hallucinating_return(self):
        """User asks 'What is my expected return?' -> must disclose that expected return model has not been run."""
        mock_healthy = {
            "reachable": True,
            "status": "ok",
            "model_version": "unregistered",
            "available_forecasters": ["baseline_lstm"],
            "kafka_connected": True,
            "url": "http://component1:8000",
        }
        with patch("tools.portfolio_tools.query_component1_health", return_value=mock_healthy):
            query = "What is my expected return on this portfolio?"
            plan = analyze_portfolio_request(query, [])
            self.assertIn("risk_model_inquiry", plan["tasks"])

            resp = synthesize_portfolio_response(self.sample_live_evidence, user_query=query)
            self.assertIn("Expected Return Evaluation", resp)
            self.assertIn("cannot calculate a 95% one-day Value at Risk (VaR), CVaR, portfolio volatility, expected return", resp)
            self.assertNotIn("11.8%", resp)

    def test_cvar_query_routes_to_risk_inquiry_without_hallucination(self):
        """User asks 'Calculate my portfolio CVaR' -> must disclose that CVaR model has not been executed."""
        mock_healthy = {
            "reachable": True,
            "status": "ok",
            "model_version": "unregistered",
            "available_forecasters": ["baseline_lstm"],
            "kafka_connected": True,
            "url": "http://component1:8000",
        }
        with patch("tools.portfolio_tools.query_component1_health", return_value=mock_healthy):
            query = "Calculate my portfolio CVaR."
            plan = analyze_portfolio_request(query, [])
            self.assertIn("risk_model_inquiry", plan["tasks"])

            resp = synthesize_portfolio_response(self.sample_live_evidence, user_query=query)
            self.assertIn("Conditional Value at Risk (CVaR) Evaluation", resp)
            self.assertIn("cannot calculate a 95% one-day Value at Risk (VaR), CVaR", resp)

    def test_concentration_analysis_preserved_during_component1_outage(self):
        """Component 1 outage must NOT break or prevent concentration analysis derived from recorded holdings."""
        mock_down = {
            "reachable": False,
            "error": "Connection timed out",
            "model_version": "unavailable",
            "available_forecasters": [],
            "kafka_connected": False,
            "url": "http://component1:8000",
        }
        with patch("tools.portfolio_tools.query_component1_health", return_value=mock_down):
            query = "What is my portfolio concentration risk across the top two positions?"
            resp = synthesize_portfolio_response(self.sample_live_evidence, user_query=query)

            # Concentration calculation must be completely preserved
            self.assertIn("JKH", resp)
            self.assertIn("COMB", resp)
            self.assertIn("99.01%", resp)
            self.assertIn("LKR 6,680,000", resp)
            self.assertIn("54.5%", resp)
            self.assertIn("44.5%", resp)


if __name__ == "__main__":
    unittest.main()
