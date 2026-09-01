"""
audit_log.py
============
Append-only audit trail for every gateway decision.

Regulators do not accept "the model said so". Each entry records the score,
the enforced action, the reason and the policy version, so any decision can be
reconstructed months later. Entries are kept in memory (for the /audit
endpoint and the demo UI) and appended to a JSON-Lines file on disk.

Logging failures are swallowed deliberately: a full disk must never stop the
gateway from protecting a transaction.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional

from config import SETTINGS, AuditConfig


class AuditLog:
    """Thread-safe, append-only decision log."""

    def __init__(self, config: Optional[AuditConfig] = None, directory: Optional[Path] = None):
        self.cfg = config or SETTINGS.audit
        base = directory or Path(__file__).resolve().parent
        self.path = base / self.cfg.log_file
        self._entries: Deque[Dict] = deque(maxlen=self.cfg.memory_limit)
        self._lock = threading.Lock()
        self._write_errors = 0

    def record(
        self,
        *,
        transaction_id: str,
        user_id: str,
        behavioral_score: float,
        graph_score: float,
        risk_score: float,
        decision: str,
        reason: str,
        attack_type: Optional[str] = None,
    ) -> Dict:
        """Write one decision to the trail and return the stored entry."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "transaction_id": transaction_id,
            "user_id": user_id,
            "behavioral_score": round(float(behavioral_score), 4),
            "graph_score": round(float(graph_score), 4),
            "risk_score": round(float(risk_score), 4),
            "decision": decision,
            "reason": reason,
            "attack_type": attack_type,
        }

        with self._lock:
            self._entries.append(entry)
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry) + "\n")
            except OSError:
                # Never let a disk problem break the security decision path.
                self._write_errors += 1

        return entry

    def recent(self, limit: int = 25) -> List[Dict]:
        """Most recent entries first."""
        with self._lock:
            return list(self._entries)[-limit:][::-1]

    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        """Clear the in-memory view. The file on disk is intentionally kept."""
        with self._lock:
            self._entries.clear()
