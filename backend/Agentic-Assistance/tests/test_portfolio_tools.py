"""
Focused unit tests for Component 4 live portfolio service integration:
- Valid credential minting matching Platform validation requirements
- Rejection of missing, placeholder, or short JWT secrets
- Direct token override (PORTFOLIO_AUTH_TOKEN)
- Handling of expired/invalid token and HTTP 401
- Handling of HTTP 404 (authenticated but portfolio not found)
- Successful live portfolio record mapping
- Bounded timeout and service unavailability fallback
- Verification against live Platform service if reachable
"""

import json
import os
import sys
import unittest
import urllib.error
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import jwt

# Ensure backend/Agentic-Assistance is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from tools.portfolio_tools import (
    DEFAULT_PORTFOLIO_ID,
    DEFAULT_USER_ID,
    EXAMPLE_SECRET,
    MIN_SECRET_LENGTH,
    PLATFORM_ALGORITHM,
    PLATFORM_ISSUER,
    PLATFORM_SERVICE_URL,
    get_auth_header,
    get_portfolio_analysis,
    query_live_portfolio_service,
    resolve_user_id,
    validate_platform_secret,
)

VALID_TEST_SECRET = "abcdefghijklmnopqrstuvwxyz0123456789_VALID_SECRET_32_PLUS_CHARS"


class TestPortfolioAuthAndIntegration(unittest.TestCase):

    def test_secret_validation_rules(self):
        """Platform invariants: secret must be non-empty, not the example string, and >= 32 chars."""
        self.assertFalse(validate_platform_secret(None))
        self.assertFalse(validate_platform_secret(""))
        self.assertFalse(validate_platform_secret(EXAMPLE_SECRET))
        self.assertFalse(validate_platform_secret("short-secret-under-32-chars"))
        self.assertTrue(validate_platform_secret(VALID_TEST_SECRET))

    def test_valid_credentials_minting(self):
        """Valid secret mints a token strictly satisfying platform decode requirements."""
        with patch.dict(os.environ, {"JWT_SECRET": VALID_TEST_SECRET, "PORTFOLIO_AUTH_TOKEN": ""}):
            header = get_auth_header("1")
            self.assertIn("Authorization", header)
            self.assertTrue(header["Authorization"].startswith("Bearer "))

            raw_token = header["Authorization"].split(" ")[1]
            decoded = jwt.decode(
                raw_token,
                VALID_TEST_SECRET,
                algorithms=[PLATFORM_ALGORITHM],
                issuer=PLATFORM_ISSUER,
                options={"require": ["exp", "iat", "sub"]},
            )
            self.assertEqual(decoded["sub"], "1")
            self.assertEqual(decoded["iss"], PLATFORM_ISSUER)
            self.assertIn("exp", decoded)
            self.assertIn("iat", decoded)

    def test_missing_secret_returns_empty_auth(self):
        """Missing JWT_SECRET returns empty auth header without crashing."""
        with patch.dict(os.environ, {"JWT_SECRET": "", "PORTFOLIO_AUTH_TOKEN": ""}):
            header = get_auth_header("1")
            self.assertEqual(header, {})

    def test_example_placeholder_secret_refused(self):
        """Example placeholder secret from .env.example must never mint tokens."""
        with patch.dict(os.environ, {"JWT_SECRET": EXAMPLE_SECRET, "PORTFOLIO_AUTH_TOKEN": ""}):
            header = get_auth_header("1")
            self.assertEqual(header, {})

    def test_direct_token_override(self):
        """PORTFOLIO_AUTH_TOKEN takes precedence if explicitly provided."""
        with patch.dict(os.environ, {"PORTFOLIO_AUTH_TOKEN": "direct_bearer_token_123"}):
            header = get_auth_header("1")
            self.assertEqual(header, {"Authorization": "Bearer direct_bearer_token_123"})

    def test_http_401_unauthorized_triggers_labeled_fallback(self):
        """Platform returning HTTP 401 triggers labeled mock fallback with authentication notice."""
        mock_err = urllib.error.HTTPError(
            url="http://localhost:8100/portfolios/1",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=MagicMock(read=lambda: b'{"detail":"Not authenticated"}'),
        )

        with patch("urllib.request.urlopen", side_effect=mock_err):
            output = get_portfolio_analysis.invoke({"user_id": "1"})

            self.assertIn("PORTFOLIO ANALYSIS — FALLBACK DATA", output)
            self.assertIn("STATUS:\nNO LIVE PORTFOLIO RECORD RETRIEVED", output)
            self.assertIn("FALLBACK NOTICE:", output)
            self.assertIn("Platform service authentication required", output)
            self.assertIn("HTTP 401: Unauthorized", output)
            self.assertIn("LKR 1,000,000", output)
            self.assertIn("Mock Baseline Demonstration Only", output)

    def test_expired_token_handled_gracefully(self):
        """Expired token returning 401 is trapped cleanly without crashing the agent."""
        mock_err = urllib.error.HTTPError(
            url="http://localhost:8100/portfolios/1",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=MagicMock(read=lambda: b'{"detail":"Token expired"}'),
        )

        with patch("urllib.request.urlopen", side_effect=mock_err):
            output = get_portfolio_analysis.invoke({"user_id": "1"})

            self.assertIn("FALLBACK DATA", output)
            self.assertIn("STATUS:\nNO LIVE PORTFOLIO RECORD RETRIEVED", output)
            self.assertIn("FALLBACK NOTICE:", output)
            self.assertIn("Platform service authentication required", output)

    def test_http_404_not_found_triggers_labeled_fallback(self):
        """Authenticated request for non-existent portfolio returns 404, caught as clean fallback."""
        mock_err = urllib.error.HTTPError(
            url="http://localhost:8100/portfolios/1",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=MagicMock(read=lambda: b'{"detail":"Portfolio not found"}'),
        )

        with patch("urllib.request.urlopen", side_effect=mock_err):
            output = get_portfolio_analysis.invoke({"user_id": "1"})

            self.assertIn("PORTFOLIO ANALYSIS — FALLBACK DATA", output)
            self.assertIn("STATUS:\nNO LIVE PORTFOLIO RECORD RETRIEVED", output)
            self.assertIn("FALLBACK NOTICE:", output)
            self.assertIn("Portfolio not found on platform service", output)
            self.assertIn("NOT the user's actual portfolio", output)
            self.assertIn("synthetic baseline estimates for demonstration only", output)
            self.assertIn("Live holdings: Unverified", output)
            self.assertIn("Live market value: Unverified (No live portfolio record)", output)
            self.assertIn("Resolved User Identity:\n1", output)

    def test_fallback_labels_all_synthetic_metrics_as_mock_baseline(self):
        """All metrics and allocations in fallback must be explicitly marked as illustrative mock examples."""
        mock_err = urllib.error.HTTPError(
            url="http://localhost:8100/portfolios/1",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=MagicMock(read=lambda: b'{"detail":"Portfolio not found"}'),
        )

        with patch("urllib.request.urlopen", side_effect=mock_err):
            output = get_portfolio_analysis.invoke({"user_id": "USER001"})

            self.assertIn("ILLUSTRATIVE MOCK BASELINE ESTIMATES (NOT ACTUAL USER PORTFOLIO):", output)
            self.assertIn("Illustrative Baseline Valuation: LKR 1,000,000 (Mock Baseline Demonstration Only)", output)
            self.assertIn("Illustrative Expected Annual Return: 11.8% (Synthetic Estimate)", output)
            self.assertIn("Illustrative Risk Score: 38/100 (Synthetic Estimate)", output)
            self.assertIn("Illustrative Diversification Score: 74/100 (Synthetic Estimate)", output)
            self.assertIn("Illustrative Liquidity: High (Synthetic Estimate)", output)
            self.assertIn("ILLUSTRATIVE CURRENT ALLOCATION (Mock Baseline Example Only):", output)
            self.assertIn("ILLUSTRATIVE OBSERVATIONS (Sample Demonstration Only):", output)
            self.assertIn("Do not describe these synthetic metrics, asset allocations, or observations as the user's actual portfolio.", output)
            self.assertIn("Resolved User Identity:\n1 (Requested: 'USER001')", output)

    def test_live_portfolio_success_mapping(self):
        """Successful 200 response from platform maps live holdings and valuations accurately."""
        live_portfolio_data = {
            "id": 1,
            "name": "Main Growth Portfolio",
            "base_currency": "USD",
            "total_value": 750000.0,
            "holdings": [
                {
                    "symbol": "AAPL",
                    "quantity": 1500.0,
                    "current_price": 220.0,
                    "cost_basis": 180.0,
                    "avg_daily_volume": 45000000.0,
                },
                {
                    "symbol": "GOOGL",
                    "quantity": 2500.0,
                    "current_price": 168.0,
                    "cost_basis": 140.0,
                    "avg_daily_volume": 20000000.0,
                },
            ],
        }

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(live_portfolio_data).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            output = get_portfolio_analysis.invoke({"user_id": "1"})

            self.assertIn("PORTFOLIO ANALYSIS — LIVE PLATFORM RECORD", output)
            self.assertIn("Main Growth Portfolio", output)
            self.assertIn("Total Portfolio Value:\nUSD 750,000.00", output)
            self.assertIn("AAPL", output)
            self.assertIn("GOOGL", output)
            self.assertIn("METRIC NOTICE:", output)
            self.assertNotIn("FALLBACK DATA", output)

    def test_timeout_fallback(self):
        """Timeout connecting to platform falls back to labeled mock data."""
        with patch("urllib.request.urlopen", side_effect=TimeoutError("Request timed out")):
            output = get_portfolio_analysis.invoke({"user_id": "1"})

            self.assertIn("PORTFOLIO ANALYSIS — FALLBACK DATA", output)
            self.assertIn("STATUS:\nNO LIVE PORTFOLIO RECORD RETRIEVED", output)
            self.assertIn("FALLBACK NOTICE:", output)
            self.assertIn("Request timed out", output)

    def test_evidence_validator_enforces_portfolio_fallback_notice(self):
        """Validator requires portfolio fallback responses to begin with a notice and avoid actual ownership claims."""
        from evidence_validator import validate_evidence_claims

        mock_err = urllib.error.HTTPError(
            url="http://localhost:8100/portfolios/1",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=MagicMock(read=lambda: b'{"detail":"Portfolio not found"}'),
        )
        with patch("urllib.request.urlopen", side_effect=mock_err):
            tool_output = get_portfolio_analysis.invoke({"user_id": "1"})

        evidence = [{"tool": "get_portfolio_analysis", "result": tool_output}]

        # Case A: LLM claims user actually has LKR 1,000,000 without starting notice
        bad_answer = "Your portfolio is valued at LKR 1,000,000 with an expected return of 12%."
        val_bad = validate_evidence_claims(bad_answer, evidence)
        self.assertFalse(val_bad["passed"])
        self.assertIn("Fallback portfolio described as actual user portfolio", val_bad["issues"])
        self.assertTrue(val_bad["answer"].startswith("Notice: No live portfolio was retrieved for this account from the platform service"))

        # Case B: LLM provides properly grounded response starting with clear notice
        good_answer = (
            "Notice: No live portfolio record was retrieved from the platform service for your account. "
            "The following figures are illustrative mock baseline examples for demonstration only: "
            "Illustrative valuation of LKR 1,000,000 with synthetic expected annual return of 12%."
        )
        val_good = validate_evidence_claims(good_answer, evidence)
        self.assertTrue(val_good["passed"])
        self.assertEqual(len(val_good["issues"]), 0)

    def test_placeholder_user_id_resolution(self):
        """Placeholders like 'your_user_id' and 'USER001' resolve to configured default user ID."""
        self.assertEqual(resolve_user_id("your_user_id"), "1")
        self.assertEqual(resolve_user_id("USER001"), "1")
        self.assertEqual(resolve_user_id("user_id"), "1")
        self.assertEqual(resolve_user_id(""), "1")
        self.assertEqual(resolve_user_id(None), "1")
        self.assertEqual(resolve_user_id("not_a_number"), "1")
        self.assertEqual(resolve_user_id("42"), "42")

    def test_jwt_subject_selection_with_placeholders(self):
        """Tokens minted for placeholder user IDs use resolved default user ID as subject."""
        with patch.dict(os.environ, {"JWT_SECRET": VALID_TEST_SECRET, "PORTFOLIO_AUTH_TOKEN": ""}):
            header = get_auth_header("your_user_id")
            token = header["Authorization"].split(" ")[1]
            decoded = jwt.decode(token, VALID_TEST_SECRET, algorithms=[PLATFORM_ALGORITHM], issuer=PLATFORM_ISSUER)
            self.assertEqual(decoded["sub"], "1")

            header_2 = get_auth_header("USER001")
            token_2 = header_2["Authorization"].split(" ")[1]
            decoded_2 = jwt.decode(token_2, VALID_TEST_SECRET, algorithms=[PLATFORM_ALGORITHM], issuer=PLATFORM_ISSUER)
            self.assertEqual(decoded_2["sub"], "1")

    def test_empty_portfolio_list_fallback(self):
        """Empty list [] from GET /portfolios triggers clean fallback and does not claim zero holdings."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps([]).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            output = get_portfolio_analysis.invoke({"user_id": "your_user_id"})

            self.assertIn("PORTFOLIO ANALYSIS — FALLBACK DATA", output)
            self.assertIn("Empty portfolio list", output)
            self.assertIn("Live holdings: Unverified (No live portfolio registered on platform)", output)
            self.assertNotIn("Live holdings: 0 verified assets", output)
            self.assertIn("Resolved User Identity:\n1 (Requested: 'your_user_id')", output)
            self.assertIn("Illustrative Baseline Valuation: LKR 1,000,000 (Mock Baseline Demonstration Only)", output)

    def test_portfolio_list_retrieval_and_detail_fetch(self):
        """Non-empty list from GET /portfolios selects the portfolio ID and fetches detailed record."""
        portfolio_list_json = json.dumps([{"id": 5, "name": "Tech Growth", "base_currency": "USD"}]).encode("utf-8")
        portfolio_detail_json = json.dumps({
            "id": 5,
            "user_id": 1,
            "name": "Tech Growth",
            "base_currency": "USD",
            "total_value": 120000.0,
            "holdings": [
                {
                    "symbol": "NVDA",
                    "quantity": 1000.0,
                    "current_price": 120.0,
                    "cost_basis": 90.0,
                    "avg_daily_volume": 50000000.0,
                }
            ],
        }).encode("utf-8")

        def fake_urlopen(req, timeout=1.5):
            url = getattr(req, "full_url", str(req))
            resp = MagicMock()
            resp.status = 200
            if "/portfolios/5" in url:
                resp.read.return_value = portfolio_detail_json
            else:
                resp.read.return_value = portfolio_list_json
            resp.__enter__.return_value = resp
            return resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            output = get_portfolio_analysis.invoke({"user_id": "1"})

            self.assertIn("PORTFOLIO ANALYSIS — LIVE PLATFORM RECORD", output)
            self.assertIn("Portfolio ID:\n5", output)
            self.assertIn("Tech Growth", output)
            self.assertIn("NVDA", output)
            self.assertIn("USD 120,000.00", output)
            self.assertNotIn("FALLBACK DATA", output)

    def test_existing_portfolio_with_zero_holdings(self):
        """Active portfolio with empty holdings is reported as live record with 0 positions, not fallback."""
        portfolio_list_json = json.dumps([{"id": 8, "name": "Empty Cash Fund", "base_currency": "USD"}]).encode("utf-8")
        portfolio_detail_json = json.dumps({
            "id": 8,
            "user_id": 1,
            "name": "Empty Cash Fund",
            "base_currency": "USD",
            "total_value": 0.0,
            "holdings": [],
        }).encode("utf-8")

        def fake_urlopen(req, timeout=1.5):
            url = getattr(req, "full_url", str(req))
            resp = MagicMock()
            resp.status = 200
            if "/portfolios/8" in url:
                resp.read.return_value = portfolio_detail_json
            else:
                resp.read.return_value = portfolio_list_json
            resp.__enter__.return_value = resp
            return resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            output = get_portfolio_analysis.invoke({"user_id": "1"})

            self.assertIn("PORTFOLIO ANALYSIS — LIVE PLATFORM RECORD", output)
            self.assertIn("Portfolio ID:\n8", output)
            self.assertIn("Holding Count:\n0", output)
            self.assertIn("Live API confirms 0 active positions", output)
            self.assertIn("Active live portfolio record retrieved (0 holdings recorded)", output)
            self.assertNotIn("FALLBACK DATA", output)

    def test_llm_supplied_portfolio_id_with_empty_portfolio_list(self):
        """LLM passing portfolio_id='1' when user has no portfolios returns clean empty-list fallback without calling /portfolios/1."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps([]).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            output = get_portfolio_analysis.invoke({"user_id": "1", "portfolio_id": "1"})

            # Verify the only HTTP call was to /portfolios, not /portfolios/1
            self.assertEqual(mock_urlopen.call_count, 1)
            req = mock_urlopen.call_args[0][0]
            req_url = getattr(req, "full_url", str(req))
            self.assertTrue(req_url.endswith("/portfolios"))
            self.assertFalse(req_url.endswith("/portfolios/1"))

            self.assertIn("PORTFOLIO ANALYSIS — FALLBACK DATA", output)
            self.assertIn("Empty portfolio list", output)
            self.assertIn("Live holdings: Unverified (No live portfolio registered on platform)", output)
            self.assertIn("Resolved User Identity:\n1", output)

    def test_valid_portfolio_selection_from_portfolio_list(self):
        """Explicit portfolio_id matching user's owned portfolios retrieves details successfully."""
        portfolio_list_json = json.dumps([
            {"id": 3, "name": "Income Fund", "base_currency": "USD"},
            {"id": 7, "name": "Growth Fund", "base_currency": "USD"},
        ]).encode("utf-8")
        portfolio_detail_json = json.dumps({
            "id": 7,
            "user_id": 1,
            "name": "Growth Fund",
            "base_currency": "USD",
            "total_value": 75000.0,
            "holdings": [{"symbol": "GOOGL", "quantity": 500.0, "current_price": 150.0}],
        }).encode("utf-8")

        def fake_urlopen(req, timeout=1.5):
            url = getattr(req, "full_url", str(req))
            resp = MagicMock()
            resp.status = 200
            if "/portfolios/7" in url:
                resp.read.return_value = portfolio_detail_json
            else:
                resp.read.return_value = portfolio_list_json
            resp.__enter__.return_value = resp
            return resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            output = get_portfolio_analysis.invoke({"user_id": "1", "portfolio_id": "7"})

            self.assertIn("PORTFOLIO ANALYSIS — LIVE PLATFORM RECORD", output)
            self.assertIn("Portfolio ID:\n7", output)
            self.assertIn("Growth Fund", output)
            self.assertIn("GOOGL", output)
            self.assertNotIn("FALLBACK DATA", output)

    def test_explicit_portfolio_id_cross_user_forbidden(self):
        """Explicit portfolio_id not present in user's portfolio list is rejected as not owned."""
        portfolio_list_json = json.dumps([
            {"id": 3, "name": "User 1 Fund", "base_currency": "USD"}
        ]).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = portfolio_list_json
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            output = get_portfolio_analysis.invoke({"user_id": "1", "portfolio_id": "99"})

            # Verify only GET /portfolios was called, not /portfolios/99
            self.assertEqual(mock_urlopen.call_count, 1)
            self.assertIn("FALLBACK DATA", output)
            self.assertIn("Portfolio 99 not owned by user 1", output)

    def test_malformed_json_fallback(self):
        """Platform returning malformed JSON is caught safely as fallback."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"<!DOCTYPE html><html><body>Error</body></html>"
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            output = get_portfolio_analysis.invoke({"user_id": "1"})

            self.assertIn("FALLBACK DATA", output)
            self.assertIn("Malformed JSON", output)


if __name__ == "__main__":
    unittest.main()
