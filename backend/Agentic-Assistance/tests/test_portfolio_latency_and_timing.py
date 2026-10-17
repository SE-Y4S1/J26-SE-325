"""
Unit and regression tests for Component 4 portfolio latency optimizations,
timing telemetry, structured logging, request cache isolation, and Responsible AI safeguards.
"""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

from agent import call_llm, synthesize_portfolio_response
from evidence_validator import sanitize_internal_details, validate_evidence_claims
from preferences_context import (
    reset_current_preferences,
    set_current_preferences,
)
from timing_context import (
    RequestTimingContext,
    end_request_timing,
    get_current_timing,
    start_request_timing,
)
from tools.portfolio_tools import (
    clear_request_portfolio_cache,
    get_portfolio_analysis,
    init_request_portfolio_cache,
    query_live_portfolio_service,
)


class TestTimingContextAndStructuredLogging(unittest.TestCase):
    """Test monotonic timing, structured telemetry, and correlated request IDs."""

    def test_request_timing_context_lifecycle(self):
        ctx, token = start_request_timing(request_id="req-test-1234")
        try:
            self.assertEqual(ctx.request_id, "req-test-1234")
            self.assertIs(get_current_timing(), ctx)

            # Record Ollama call
            ctx.record_ollama(stage="initial_tool_selection", duration_ms=120.5, success=True)
            ctx.record_ollama(stage="final_synthesis", duration_ms=45.2, success=True)

            # Record tool call
            ctx.record_tool(tool_name="get_portfolio_analysis", duration_ms=12.3, success=True)
            # Record repeated tool call
            ctx.record_tool(tool_name="get_portfolio_analysis", duration_ms=8.1, success=True)

            # Record HTTP call
            ctx.record_http(
                endpoint="/portfolios",
                duration_ms=7.5,
                status_code=200,
                retries=0,
                is_timeout=False,
            )

            summary = ctx.finish()
            self.assertEqual(summary["request_id"], "req-test-1234")
            self.assertEqual(summary["initial_tool_selection_ms"], 120.5)
            self.assertEqual(summary["final_synthesis_ms"], 45.2)
            self.assertAlmostEqual(summary["total_ollama_inference_ms"], 165.7, places=1)
            self.assertEqual(summary["tool_call_count"], 2)
            self.assertEqual(summary["repeated_tool_calls"], 1)
            self.assertEqual(summary["timeouts_count"], 0)
            self.assertEqual(summary["retries_count"], 0)
            self.assertEqual(len(summary["http_calls"]), 1)
            self.assertEqual(summary["http_calls"][0]["status_code"], 200)
        finally:
            end_request_timing(token)
            self.assertIsNone(get_current_timing())

    def test_no_sensitive_secrets_or_jwt_in_timing_logs(self):
        """Ensure HTTP and tool call records do not capture JWTs or secrets."""
        ctx, token = start_request_timing()
        try:
            ctx.record_http(
                endpoint="/portfolios",
                duration_ms=10.0,
                status_code=200,
                retries=0,
            )
            summary = ctx.finish()
            summary_str = json.dumps(summary)
            self.assertNotIn("Bearer", summary_str)
            self.assertNotIn("eyJ", summary_str)
            self.assertNotIn("secret", summary_str.lower())
        finally:
            end_request_timing(token)


class TestRequestScopedPortfolioCache(unittest.TestCase):
    """Test prevention of redundant portfolio API requests within a single request."""

    def test_request_cache_prevents_duplicate_http_calls(self):
        cache_token = init_request_portfolio_cache()
        try:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.read.return_value = json.dumps([]).encode("utf-8")
            mock_resp.__enter__.return_value = mock_resp

            with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
                # First call
                res1 = query_live_portfolio_service(user_id="1")
                self.assertTrue(res1["available"])
                self.assertEqual(mock_urlopen.call_count, 1)

                # Second identical call within the same request lifecycle
                res2 = query_live_portfolio_service(user_id="1")
                self.assertTrue(res2["available"])
                # Must NOT invoke urlopen again (cached in request context)
                self.assertEqual(mock_urlopen.call_count, 1)
        finally:
            clear_request_portfolio_cache(cache_token)

    def test_request_cache_isolation_across_requests(self):
        """Cache reset ensures subsequent requests perform fresh lookups."""
        cache_token1 = init_request_portfolio_cache()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps([]).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            query_live_portfolio_service(user_id="1")
            self.assertEqual(mock_urlopen.call_count, 1)

        clear_request_portfolio_cache(cache_token1)

        # New request context
        cache_token2 = init_request_portfolio_cache()
        try:
            with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
                query_live_portfolio_service(user_id="1")
                # New request must fetch freshly
                self.assertEqual(mock_urlopen.call_count, 1)
        finally:
            clear_request_portfolio_cache(cache_token2)


class TestDeterministicPortfolioSynthesis(unittest.TestCase):
    """Test deterministic evidence-backed response generation for portfolio queries."""

    def test_live_portfolio_synthesis_passes_validation(self):
        tool_content = """PORTFOLIO ANALYSIS — LIVE PLATFORM RECORD

Data Source:
Live platform portfolio store (http://platform:8100).

Portfolio ID:
1

Portfolio Name:
Main Growth Portfolio

Base Currency:
USD

Total Portfolio Value:
USD 750,000.00

Holding Count:
2

RECORDED HOLDINGS:
- AAPL: 1,500.00 units @ USD 220.00 = USD 330,000.00 (44.0% of portfolio)
- GOOGL: 2,500.00 units @ USD 168.00 = USD 420,000.00 (56.0% of portfolio)

OBSERVATIONS:
- Live holdings retrieved directly from platform.

METRIC NOTICE:
The platform portfolio service stores live asset holdings and valuation.
Optimization metrics require running Component 1's analytical models.
"""
        response = synthesize_portfolio_response(tool_content)
        self.assertIn("Main Growth Portfolio", response)
        self.assertIn("USD 750,000.00", response)
        self.assertIn("AAPL", response)
        self.assertIn("GOOGL", response)
        self.assertIn("Optimization metrics", response)

        evidence = [{"tool": "get_portfolio_analysis", "result": tool_content}]
        val = validate_evidence_claims(response, evidence)
        self.assertTrue(val["passed"])
        self.assertEqual(len(val["issues"]), 0)

    def test_fallback_portfolio_synthesis_passes_validation(self):
        tool_content = """PORTFOLIO ANALYSIS — FALLBACK DATA (NO LIVE RECORD RETRIEVED)

Data Source:
Local mock baseline (live platform service at http://platform:8100: No portfolios registered for user).

STATUS:
NO LIVE PORTFOLIO RECORD RETRIEVED for user ID '1'.
Platform service note: No portfolios registered for user on platform service.

ILLUSTRATIVE MOCK BASELINE ESTIMATES (NOT ACTUAL USER PORTFOLIO):
- Illustrative Baseline Valuation: LKR 1,000,000 (Mock Baseline Demonstration Only)
- Illustrative Expected Annual Return: 11.8% (Synthetic Example)
- Risk Score: 38/100 (Synthetic Example)
"""
        response = synthesize_portfolio_response(tool_content)
        self.assertTrue(response.startswith("Notice: No live portfolio was retrieved for this account from the platform service"))
        self.assertIn("zero verified live holdings on record", response)
        self.assertIn("Mock Baseline Demonstration Only", response)

        evidence = [{"tool": "get_portfolio_analysis", "result": tool_content}]
        val = validate_evidence_claims(response, evidence)
        self.assertTrue(val["passed"])
        self.assertEqual(len(val["issues"]), 0)

    def test_personalized_context_disabled_synthesis(self):
        tool_content = """PORTFOLIO ANALYSIS — PERSONALIZED CONTEXT DISABLED

Personalized financial context is disabled by user preference.
"""
        response = synthesize_portfolio_response(tool_content)
        self.assertIn("Personalized financial context is disabled by user preference", response)
        self.assertNotIn("LKR 1,000,000", response)

        evidence = [{"tool": "get_portfolio_analysis", "result": tool_content}]
        pref_token = set_current_preferences({"personalized_context": False})
        try:
            val = validate_evidence_claims(response, evidence)
            self.assertTrue(val["passed"])
            self.assertNotIn("LKR 1,000,000", val["answer"])
        finally:
            reset_current_preferences(pref_token)


class TestInternalDetailsSanitization(unittest.TestCase):
    """Ensure internal system prompts and routing tags are never leaked."""

    def test_sanitize_internal_details_cleans_prompts(self):
        raw = (
            "RESPONSIBLE AI USER PREFERENCE DIRECTIVES:\n- User preference is False\n\n"
            "SYSTEM PROMPT: You are FinAgent\n\n"
            "[INTERNAL ROUTING: portfolio]\n"
            "Here is your factual financial response."
        )
        cleaned = sanitize_internal_details(raw)
        self.assertNotIn("SYSTEM PROMPT", cleaned)
        self.assertNotIn("USER PREFERENCE DIRECTIVES", cleaned)
        self.assertNotIn("[INTERNAL ROUTING", cleaned)
        self.assertEqual(cleaned, "Here is your factual financial response.")


if __name__ == "__main__":
    unittest.main()
