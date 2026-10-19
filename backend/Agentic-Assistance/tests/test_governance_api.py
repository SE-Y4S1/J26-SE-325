import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import jwt
from fastapi.testclient import TestClient

from governance_store import GovernanceStore, default_governance_store
from main import app
from tools.execution_tools import get_execution_explanation
from tools.fraud_tools import get_fraud_analysis


class TestGovernanceAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.jwt_secret = "test-secret-at-least-32-chars-long-abc-123"

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_store_file = Path(self.temp_dir) / "gov_test.jsonl"
        self.original_file_path = default_governance_store.file_path
        default_governance_store.file_path = self.test_store_file
        default_governance_store._ensure_directory()

    def tearDown(self):
        default_governance_store.file_path = self.original_file_path
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_token(self, sub: str = "42", email: str = "user@example.com", expires_minutes: int = 60) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            "sub": str(sub),
            "email": email,
            "iat": now,
            "exp": now + timedelta(minutes=expires_minutes),
            "iss": "j26-se-325-platform",
        }
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")

    def test_record_confirmed_decision_success(self):
        res = self.client.post(
            "/assistant/governance/decision",
            json={
                "decision_type": "confirmed",
                "reference_id": "TX1001",
                "reference_type": "transaction",
                "evidence_reference": {"fraud_score": 0.85, "provenanceType": "live"},
                "rationale": "Audited and confirmed",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "recorded")
        self.assertFalse(data["is_duplicate"])
        self.assertEqual(data["decision"]["decision_type"], "confirmed")
        self.assertEqual(data["decision"]["reference_id"], "TX1001")
        self.assertTrue(data["decision"]["decision_id"].startswith("gov-"))

    def test_record_rejected_decision_success(self):
        res = self.client.post(
            "/assistant/governance/decision",
            json={
                "decision_type": "rejected",
                "reference_id": "TX9999",
                "reference_type": "transaction",
                "evidence_reference": {"fraud_score": 0.98},
                "rationale": "High fraud probability rejected",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "recorded")
        self.assertEqual(data["decision"]["decision_type"], "rejected")

    def test_validation_errors(self):
        # Invalid decision_type
        res = self.client.post(
            "/assistant/governance/decision",
            json={
                "decision_type": "maybe",
                "reference_id": "TX1001",
            },
        )
        self.assertEqual(res.status_code, 422)

        # Empty reference_id
        res2 = self.client.post(
            "/assistant/governance/decision",
            json={
                "decision_type": "confirmed",
                "reference_id": "   ",
            },
        )
        self.assertEqual(res2.status_code, 422)

    def test_duplicate_submission_handling(self):
        payload = {
            "decision_type": "confirmed",
            "reference_id": "TX1001",
            "evidence_reference": {"fraud_score": 0.5},
        }
        res1 = self.client.post("/assistant/governance/decision", json=payload)
        self.assertEqual(res1.status_code, 200)
        self.assertFalse(res1.json()["is_duplicate"])
        id_1 = res1.json()["decision"]["decision_id"]

        # Resubmit identical decision
        res2 = self.client.post("/assistant/governance/decision", json=payload)
        self.assertEqual(res2.status_code, 200)
        self.assertTrue(res2.json()["is_duplicate"])
        self.assertEqual(res2.json()["status"], "already_recorded")
        self.assertEqual(res2.json()["decision"]["decision_id"], id_1)

    def test_get_decision_endpoints(self):
        # When missing
        get_res1 = self.client.get("/assistant/governance/decision/TX-UNKNOWN")
        self.assertEqual(get_res1.status_code, 200)
        self.assertFalse(get_res1.json()["found"])
        self.assertIsNone(get_res1.json()["decision"])

        # Record one
        self.client.post(
            "/assistant/governance/decision",
            json={"decision_type": "confirmed", "reference_id": "TX-RECORDED-1"},
        )

        get_res2 = self.client.get("/assistant/governance/decision/TX-RECORDED-1")
        self.assertEqual(get_res2.status_code, 200)
        self.assertTrue(get_res2.json()["found"])
        self.assertEqual(get_res2.json()["decision"]["decision_type"], "confirmed")

    def test_list_decisions(self):
        self.client.post("/assistant/governance/decision", json={"decision_type": "confirmed", "reference_id": "T1"})
        self.client.post("/assistant/governance/decision", json={"decision_type": "rejected", "reference_id": "T2"})

        res = self.client.get("/assistant/governance/decisions")
        self.assertEqual(res.status_code, 200)
        decisions = res.json()["decisions"]
        self.assertGreaterEqual(len(decisions), 2)
        self.assertEqual(decisions[0]["reference_id"], "T2")

    @patch.dict(os.environ, {"AUTH_REQUIRED": "true", "JWT_SECRET": "test-secret-at-least-32-chars-long-abc-123"})
    def test_auth_enforcement_when_auth_required(self):
        # 1. No token -> 401
        res_no_auth = self.client.post(
            "/assistant/governance/decision",
            json={"decision_type": "confirmed", "reference_id": "TX-AUTH-1"},
        )
        self.assertEqual(res_no_auth.status_code, 401)

        # 2. Invalid token -> 401
        res_invalid = self.client.post(
            "/assistant/governance/decision",
            json={"decision_type": "confirmed", "reference_id": "TX-AUTH-1"},
            headers={"Authorization": "Bearer not-a-valid-jwt"},
        )
        self.assertEqual(res_invalid.status_code, 401)

        # 3. Valid token -> 200 with extracted user identity
        token = self._create_token(sub="99", email="auditor@university.edu")
        res_valid = self.client.post(
            "/assistant/governance/decision",
            json={"decision_type": "confirmed", "reference_id": "TX-AUTH-1"},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(res_valid.status_code, 200)
        record = res_valid.json()["decision"]
        self.assertEqual(record["user_id"], "99")
        self.assertEqual(record["user_email"], "auditor@university.edu")

    def test_workflow_respects_rejection_in_tool_output(self):
        # Record human rejection for TX1001
        self.client.post(
            "/assistant/governance/decision",
            json={"decision_type": "rejected", "reference_id": "TX1001"},
        )

        output = get_fraud_analysis.invoke({"transaction_id": "TX1001"})
        self.assertIn("HUMAN OVERSIGHT GOVERNANCE STATUS", output)
        self.assertIn("REJECTED", output)
        self.assertIn("Financial execution is strictly blocked", output)

        # Record human rejection for an execution plan
        self.client.post(
            "/assistant/governance/decision",
            json={"decision_type": "rejected", "reference_id": "EXE-123"},
        )
        with patch("tools.execution_tools.query_execution_trace") as mock_trace, \
             patch("tools.execution_tools.query_live_fraud_service") as mock_fraud, \
             patch("tools.execution_tools.query_live_audit_service") as mock_audit:
            mock_trace.return_value = {"available": True, "trace": {"status": "HELD", "status_reason": "Held for screening"}}
            mock_fraud.return_value = {"entry": None, "reason": "none"}
            mock_audit.return_value = {"available": False, "error": "none"}

            exec_output = get_execution_explanation.invoke({"execution_id": "EXE-123"})
            self.assertIn("HUMAN OVERSIGHT GOVERNANCE STATUS", exec_output)
            self.assertIn("EXECUTION BLOCKED", exec_output)
