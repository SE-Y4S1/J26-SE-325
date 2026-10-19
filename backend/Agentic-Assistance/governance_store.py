"""
Persistent Governance Decision Store for Component 4 (Human-in-the-Loop Oversight).

Maintains an append-only, thread-safe JSON Lines audit log of human oversight
decisions (Confirm Decision, Reject Action) associated with transaction/recommendation
references and available tool evidence.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_DATA_PATH = DEFAULT_DATA_DIR / "governance_decisions.jsonl"


class GovernanceStore:
    def __init__(self, file_path: Path | str | None = None):
        self.file_path = Path(file_path or os.getenv("GOVERNANCE_DATA_PATH") or DEFAULT_DATA_PATH)
        self._lock = threading.Lock()
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.file_path.exists():
                self.file_path.touch(exist_ok=True)
        except OSError as err:
            logger.error("Failed to ensure governance storage directory: %s", err)

    def record_decision(
        self,
        decision_type: str,
        reference_id: str,
        reference_type: str = "transaction",
        user_id: str | int | None = None,
        user_email: str | None = None,
        evidence_reference: dict[str, Any] | None = None,
        rationale: str | None = None,
    ) -> dict[str, Any]:
        """
        Record a human oversight decision.
        If an identical decision (same reference_id and decision_type) is already the latest
        record, returns the existing record to prevent accidental duplicate entries.
        """
        cleaned_type = decision_type.strip().lower()
        if cleaned_type not in {"confirmed", "rejected"}:
            raise ValueError(f"Invalid decision_type '{decision_type}'. Must be 'confirmed' or 'rejected'.")

        cleaned_ref = str(reference_id or "").strip()
        if not cleaned_ref:
            raise ValueError("reference_id cannot be empty.")

        with self._lock:
            # Check latest existing decision for this reference to avoid accidental duplicate records
            latest = self._get_latest_decision_unlocked(cleaned_ref)
            if latest and latest.get("decision_type") == cleaned_type:
                logger.info(
                    "Duplicate decision submission for %s with state %s; returning existing record %s",
                    cleaned_ref,
                    cleaned_type,
                    latest.get("decision_id"),
                )
                return {**latest, "is_duplicate": True, "status": "already_recorded"}

            decision_id = f"gov-{uuid.uuid4().hex[:12]}"
            now_iso = datetime.now(timezone.utc).isoformat()

            record: dict[str, Any] = {
                "decision_id": decision_id,
                "decision_type": cleaned_type,
                "reference_id": cleaned_ref,
                "reference_type": str(reference_type or "transaction").strip().lower(),
                "user_id": str(user_id) if user_id is not None else None,
                "user_email": str(user_email) if user_email is not None else None,
                "timestamp": now_iso,
                "evidence_reference": evidence_reference or {},
                "rationale": rationale.strip() if isinstance(rationale, str) and rationale.strip() else None,
                "status": "recorded",
            }

            try:
                with open(self.file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
            except OSError as err:
                logger.error("Failed to write governance decision to %s: %s", self.file_path, err)
                raise RuntimeError(f"Governance persistence failed: {err}") from err

            logger.info(
                "Governance decision %s persisted for %s (%s)",
                decision_id,
                cleaned_ref,
                cleaned_type,
            )
            return {**record, "is_duplicate": False}

    def _get_latest_decision_unlocked(self, reference_id: str) -> dict[str, Any] | None:
        target = reference_id.strip()
        latest_match = None
        if not self.file_path.exists():
            return None

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get("reference_id") == target:
                            latest_match = entry
                    except json.JSONDecodeError:
                        continue
        except OSError as err:
            logger.warning("Error reading governance file %s: %s", self.file_path, err)

        return latest_match

    def get_latest_decision(self, reference_id: str) -> dict[str, Any] | None:
        """Get the latest recorded decision for a given reference_id."""
        with self._lock:
            return self._get_latest_decision_unlocked(reference_id)

    def is_rejected(self, reference_id: str) -> bool:
        """Return True if the latest recorded decision for reference_id is 'rejected'."""
        decision = self.get_latest_decision(reference_id)
        return bool(decision and decision.get("decision_type") == "rejected")

    def list_decisions(self, limit: int = 50) -> list[dict[str, Any]]:
        """List recorded decisions in reverse chronological order."""
        with self._lock:
            if not self.file_path.exists():
                return []
            records = []
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            except OSError as err:
                logger.warning("Error reading governance file %s: %s", self.file_path, err)
                return []

            records.reverse()
            return records[: max(1, limit)]

    def clear(self) -> None:
        """Clear all stored decisions. Used primarily for tests."""
        with self._lock:
            try:
                with open(self.file_path, "w", encoding="utf-8") as f:
                    f.truncate(0)
            except OSError as err:
                logger.error("Failed to clear governance store %s: %s", self.file_path, err)


# Global default instance
default_governance_store = GovernanceStore()
