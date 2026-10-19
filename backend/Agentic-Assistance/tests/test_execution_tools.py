"""
Unit tests for get_execution_explanation, the cross-component execution trace:
- Reads the platform trace only with the caller's own bearer token, never a minted one
- Joins Component 1's plan, Component 2's decision and Component 3's verification by ID
- Honours the explanation preference for Component 1's rule trace
- Refuses identifiers that are not execution IDs
"""

import json
import os
import sys
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from preferences_context import reset_current_preferences, set_current_preferences
from request_auth import reset_caller_authorization, set_caller_authorization
from tools.execution_tools import get_execution_explanation

EXE_ID = "EXE-0123456789abcdef"

TRACE = {
    "id": EXE_ID, "portfolio_id": 1, "kind": "withdrawal", "status": "EXECUTED",
    "status_reason": "Screened by Component 2 and anchored by Component 3.",
    "amount": 9980.0, "currency": "USD", "source_model_version": "c1-v7",
    "plan_hash": "0x" + "ab" * 32,
    "plan": {"fuzzy_rule_trace": [{"rule": "R3", "strength": 0.8}], "agent_reasoning_trace": []},
    "orders": [{"symbol": "SPY", "side": "SELL", "quantity": 20.0, "price": 499.0}],
    "fraud": {}, "audit": {},
}
ENGINE = {
    "transaction_id": EXE_ID, "fraud_score": 0.12, "decision": "ALLOW",
    "model_version": "heuristic-untrained", "benchmark_model": False,
    "explanation": {"rules": [], "streams": {"heuristic": 0.12}, "top_factors": [
        {"name": "amount", "contribution": 0.12}], "notes": []},
}
VERIFY = {
    "verified": True, "status": "VERIFIED_SIMULATED",
    "details": {
        "blockchainOnChainHash": "0x" + "cd" * 32,
        "canonicalPayload": {"finalDecision": "APPROVE", "upstreamDecision": "ALLOW",
                             "planHash": "0x" + "ab" * 32},
        "onChainData": {"isMock": True},
    },
}


def _response(body: dict) -> MagicMock:
    mock = MagicMock()
    mock.status = 200
    mock.read.return_value = json.dumps(body).encode("utf-8")
    mock.__enter__.return_value = mock
    return mock


class TestExecutionExplanation(unittest.TestCase):

    def setUp(self):
        self.requests = []

    def _fake_urlopen(self, req, timeout=None):
        self.requests.append(req)
        url = req.full_url
        if "/executions/" in url:
            return _response(TRACE)
        if "/v1/transactions/" in url:
            return _response(ENGINE)
        if "/api/audit/" in url:
            return _response(VERIFY)
        raise urllib.error.URLError("unexpected url")

    def _invoke(self, authorization=None, explanation_enabled=True):
        auth_token = set_caller_authorization(authorization)
        pref_token = set_current_preferences({"explanation_enabled": explanation_enabled})
        try:
            with patch("urllib.request.urlopen", side_effect=self._fake_urlopen):
                return get_execution_explanation.invoke({"execution_id": EXE_ID})
        finally:
            reset_current_preferences(pref_token)
            reset_caller_authorization(auth_token)

    def test_joins_all_three_components_with_the_callers_token(self):
        output = self._invoke("Bearer caller-token")

        platform = [r for r in self.requests if "/executions/" in r.full_url]
        self.assertEqual(len(platform), 1)
        self.assertEqual(platform[0].get_header("Authorization"), "Bearer caller-token")
        self.assertIn("Execution Status: EXECUTED", output)
        self.assertIn("SELL 20.0000 SPY at 499.0000", output)
        self.assertIn('"rule": "R3"', output)
        self.assertIn("Decision: ALLOW", output)
        self.assertIn("(NOT a benchmark model)", output)
        self.assertIn("Integrity Verified: True", output)
        self.assertIn("Upstream Decision Anchored: ALLOW", output)
        self.assertIn("simulated adapter", output)
        for request in self.requests:
            self.assertEqual(request.get_method(), "GET")

    def test_without_a_caller_token_the_platform_is_not_queried(self):
        with patch.dict(os.environ, {"JWT_SECRET": "x" * 64}):
            output = self._invoke(None)

        self.assertFalse(any("/executions/" in r.full_url for r in self.requests))
        self.assertIn("needs sign-in", output)
        # Components 2 and 3 need no user token and are still explained.
        self.assertIn("Decision: ALLOW", output)
        self.assertIn("Integrity Verified: True", output)

    def test_explanation_preference_suppresses_rule_trace(self):
        output = self._invoke("Bearer caller-token", explanation_enabled=False)

        self.assertNotIn('"rule": "R3"', output)
        self.assertIn("suppressed by user preference", output)

    def test_non_execution_id_is_refused_without_any_request(self):
        with patch("urllib.request.urlopen", side_effect=self._fake_urlopen):
            output = get_execution_explanation.invoke({"execution_id": "TX1001"})

        self.assertEqual(self.requests, [])
        self.assertIn("not one", output)


if __name__ == "__main__":
    unittest.main()
