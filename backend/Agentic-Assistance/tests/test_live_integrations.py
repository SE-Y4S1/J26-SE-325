"""
Live Integration Tests for PP1 Component 4:
Exercises real running services across configured ports:
- Component 4 Assistant (port 8003)
- Platform Portfolio Service (port 8100)
- Fraud Detection Gateway (port 8001)
- Blockchain Audit Bridge (port 8002)
- Frontend Web Server (port 3000)
- Governance Decision Persistence and Retrieval
- Chat fast-path and pipeline timing telemetry

Distinguishes live service integrations from fallbacks and simulated responses.
"""

import json
import urllib.request
import urllib.error
import unittest
import uuid


def _is_service_reachable(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status in (200, 204, 401, 404)
    except urllib.error.HTTPError as err:
        return err.code in (200, 204, 401, 404)
    except Exception:
        return False


class TestLiveIntegrations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c4_url = "http://localhost:8003"
        cls.platform_url = "http://localhost:8100"
        cls.fraud_url = "http://localhost:8001"
        cls.audit_url = "http://localhost:8002"
        cls.frontend_url = "http://localhost:3000"

        cls.c4_live = _is_service_reachable(f"{cls.c4_url}/health")
        cls.platform_live = _is_service_reachable(f"{cls.platform_url}/health")
        cls.fraud_live = _is_service_reachable(f"{cls.fraud_url}/health")
        cls.audit_live = _is_service_reachable(f"{cls.audit_url}/api/audit/PP1-DEMO-001/verify")
        cls.frontend_live = _is_service_reachable(cls.frontend_url)

    def test_01_component4_health(self):
        if not self.c4_live:
            self.skipTest("Component 4 container is not running on port 8003")
        req = urllib.request.urlopen(f"{self.c4_url}/health", timeout=3)
        self.assertEqual(req.status, 200)
        data = json.loads(req.read().decode("utf-8"))
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("component"), "agentic-assistance")

    def test_02_platform_health_contract(self):
        if not self.platform_live:
            self.skipTest("Platform container is not running on port 8100")
        req = urllib.request.urlopen(f"{self.platform_url}/health", timeout=3)
        self.assertEqual(req.status, 200)
        data = json.loads(req.read().decode("utf-8"))
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("service"), "platform")
        self.assertTrue(data.get("database_ready"))

    def test_03_fraud_detection_health_contract(self):
        if not self.fraud_live:
            self.skipTest("Fraud detection container is not running on port 8001")
        req = urllib.request.urlopen(f"{self.fraud_url}/health", timeout=3)
        self.assertEqual(req.status, 200)
        data = json.loads(req.read().decode("utf-8"))
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("component"), "fraud-detection")
        # Confirms model version and provenance from live service
        self.assertIn("model_version", data)

    def test_04_blockchain_audit_verification_contract(self):
        if not self.audit_live:
            self.skipTest("Blockchain audit container is not running on port 8002")
        # Query known demo record on live bridge
        url = f"{self.audit_url}/api/audit/PP1-DEMO-001/verify"
        req = urllib.request.urlopen(url, timeout=3)
        self.assertEqual(req.status, 200)
        data = json.loads(req.read().decode("utf-8"))
        self.assertIn("verified", data)
        self.assertIn("status", data)

    def test_05_frontend_web_server_live(self):
        if not self.frontend_live:
            self.skipTest("Frontend container is not running on port 3000")
        req = urllib.request.urlopen(self.frontend_url, timeout=3)
        self.assertEqual(req.status, 200)
        html = req.read().decode("utf-8", errors="ignore")
        self.assertTrue(len(html) > 100)

    def test_06_live_governance_decision_lifecycle(self):
        if not self.c4_live:
            self.skipTest("Component 4 is not running on port 8003")

        unique_ref = f"TX-INT-{uuid.uuid4().hex[:8]}"
        post_data = json.dumps({
            "decision_type": "confirmed",
            "reference_id": unique_ref,
            "reference_type": "transaction",
            "evidence_reference": {"source": "live_integration_test"},
            "rationale": "Automated integration test confirmation",
        }).encode("utf-8")

        from tools.portfolio_tools import get_auth_header
        auth_hdr = get_auth_header("1")
        req_headers = {"Content-Type": "application/json", "Accept": "application/json"}
        req_headers.update(auth_hdr)

        req = urllib.request.Request(
            f"{self.c4_url}/assistant/governance/decision",
            data=post_data,
            headers=req_headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            res_body = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(res_body["status"], "recorded")
            self.assertFalse(res_body["is_duplicate"])
            decision = res_body["decision"]
            self.assertEqual(decision["reference_id"], unique_ref)
            self.assertEqual(decision["decision_type"], "confirmed")
            decision_id = decision["decision_id"]

        # GET by reference_id to verify immediate persistence
        get_url = f"{self.c4_url}/assistant/governance/decision/{unique_ref}"
        get_req = urllib.request.Request(get_url, headers=req_headers)
        with urllib.request.urlopen(get_req, timeout=5) as get_resp:
            self.assertEqual(get_resp.status, 200)
            get_body = json.loads(get_resp.read().decode("utf-8"))
            self.assertTrue(get_body["found"])
            self.assertEqual(get_body["decision"]["decision_id"], decision_id)

        # GET list to verify presence in audit log
        list_url = f"{self.c4_url}/assistant/governance/decisions?limit=10"
        list_req = urllib.request.Request(list_url, headers=req_headers)
        with urllib.request.urlopen(list_req, timeout=5) as list_resp:
            self.assertEqual(list_resp.status, 200)
            list_body = json.loads(list_resp.read().decode("utf-8"))
            decisions = list_body.get("decisions", [])
            matching = [d for d in decisions if d.get("reference_id") == unique_ref]
            self.assertTrue(len(matching) >= 1)

    def test_07_live_chat_fast_path_and_telemetry(self):
        if not self.c4_live:
            self.skipTest("Component 4 is not running on port 8003")

        post_data = json.dumps({
            "message": "Hello FinAgent! What can you help me with?",
            "preferences": {
                "explanation_enabled": True,
                "personalized_context": True,
                "data_sharing_consent": False,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.c4_url}/assistant/chat",
            data=post_data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, 200)
            res_body = json.loads(resp.read().decode("utf-8"))
            self.assertIn("FinAgent", res_body["answer"])
            self.assertTrue(res_body["evidence_validation"]["passed"])
            self.assertEqual(res_body["agent_execution"]["number_of_tools"], 0)
            self.assertEqual(res_body["agent_execution"]["status"], "no_tools_required")
            timing = res_body["agent_execution"].get("timing")
            self.assertIsNotNone(timing)
            self.assertIn("total_duration_ms", timing)


if __name__ == "__main__":
    unittest.main()
