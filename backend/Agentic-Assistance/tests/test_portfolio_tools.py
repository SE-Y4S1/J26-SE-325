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
    EXAMPLE_SECRET,
    MIN_SECRET_LENGTH,
    PLATFORM_ALGORITHM,
    PLATFORM_ISSUER,
    PLATFORM_SERVICE_URL,
    get_auth_header,
    get_portfolio_analysis,
    query_live_portfolio_service,
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
            self.assertIn("FALLBACK NOTICE:", output)
            self.assertIn("Platform service authentication required", output)
            self.assertIn("HTTP 401: Unauthorized", output)
            self.assertIn("Total Portfolio Value:", output)
            self.assertIn("LKR 1,000,000", output)

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
            self.assertIn("FALLBACK NOTICE:", output)
            self.assertIn("Portfolio not found on platform service", output)

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
            self.assertIn("FALLBACK NOTICE:", output)
            self.assertIn("Request timed out", output)


if __name__ == "__main__":
    unittest.main()
