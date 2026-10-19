import shutil
import tempfile
import unittest
from pathlib import Path

from governance_store import GovernanceStore


class TestGovernanceStore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data_file = Path(self.temp_dir) / "test_governance.jsonl"
        self.store = GovernanceStore(file_path=self.data_file)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_record_confirmed_decision(self):
        record = self.store.record_decision(
            decision_type="confirmed",
            reference_id="TX1001",
            reference_type="transaction",
            user_id="42",
            user_email="evaluator@university.edu",
            evidence_reference={"fraud_score": 0.85, "provenance": "live"},
            rationale="Verified with customer via phone",
        )

        self.assertEqual(record["decision_type"], "confirmed")
        self.assertEqual(record["reference_id"], "TX1001")
        self.assertEqual(record["user_id"], "42")
        self.assertEqual(record["user_email"], "evaluator@university.edu")
        self.assertEqual(record["status"], "recorded")
        self.assertFalse(record["is_duplicate"])
        self.assertTrue(record["decision_id"].startswith("gov-"))
        self.assertIn("timestamp", record)
        self.assertFalse(self.store.is_rejected("TX1001"))

    def test_record_rejected_decision(self):
        record = self.store.record_decision(
            decision_type="rejected",
            reference_id="TX9999",
            reference_type="transaction",
            evidence_reference={"fraud_score": 0.99},
            rationale="Confirmed fraudulent device and IP mismatch",
        )

        self.assertEqual(record["decision_type"], "rejected")
        self.assertEqual(record["reference_id"], "TX9999")
        self.assertTrue(self.store.is_rejected("TX9999"))

    def test_duplicate_submission_prevention(self):
        first = self.store.record_decision(
            decision_type="confirmed",
            reference_id="TX1001",
            evidence_reference={"fraud_score": 0.5},
        )
        self.assertFalse(first["is_duplicate"])

        # Identical decision submitted again
        second = self.store.record_decision(
            decision_type="confirmed",
            reference_id="TX1001",
            evidence_reference={"fraud_score": 0.5},
        )
        self.assertTrue(second["is_duplicate"])
        self.assertEqual(second["status"], "already_recorded")
        self.assertEqual(second["decision_id"], first["decision_id"])

    def test_state_change_creates_new_record(self):
        # First confirmed
        first = self.store.record_decision(
            decision_type="confirmed",
            reference_id="TX1001",
        )
        self.assertFalse(first["is_duplicate"])
        self.assertFalse(self.store.is_rejected("TX1001"))

        # Later rejected
        second = self.store.record_decision(
            decision_type="rejected",
            reference_id="TX1001",
        )
        self.assertFalse(second["is_duplicate"])
        self.assertNotEqual(second["decision_id"], first["decision_id"])
        self.assertTrue(self.store.is_rejected("TX1001"))

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            self.store.record_decision(decision_type="unknown_type", reference_id="TX1001")

        with self.assertRaises(ValueError):
            self.store.record_decision(decision_type="confirmed", reference_id="")

    def test_persistence_across_instances(self):
        self.store.record_decision(
            decision_type="confirmed",
            reference_id="TX2002",
            user_id="101",
        )

        # Create new store instance pointing to same file
        new_store = GovernanceStore(file_path=self.data_file)
        latest = new_store.get_latest_decision("TX2002")
        self.assertIsNotNone(latest)
        self.assertEqual(latest["reference_id"], "TX2002")
        self.assertEqual(latest["user_id"], "101")
        self.assertEqual(latest["decision_type"], "confirmed")

    def test_list_decisions(self):
        self.store.record_decision("confirmed", "TX1")
        self.store.record_decision("rejected", "TX2")
        self.store.record_decision("confirmed", "TX3")

        decisions = self.store.list_decisions(limit=10)
        self.assertEqual(len(decisions), 3)
        self.assertEqual(decisions[0]["reference_id"], "TX3")  # Most recent first
