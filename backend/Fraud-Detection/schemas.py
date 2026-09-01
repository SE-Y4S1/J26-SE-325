"""
schemas.py
==========
Pydantic request/response models. This is the validation boundary of the
service: anything that reaches the model or the gateway has already been
range-checked and normalised here.
"""

from __future__ import annotations

import time
import uuid
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from config import SETTINGS

# Navigation labels the UI offers. Unknown labels are accepted but mapped to
# "unknown" (mid risk) instead of rejecting the request, so a live demo never
# dies on a typo.
KNOWN_NAVIGATION_PATTERNS = tuple(SETTINGS.behavioral.navigation_risk.keys())

AttackType = Literal["camouflage", "slow_drift", "structuring"]


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
class Transaction(BaseModel):
    """A single transaction submitted for scoring."""

    transaction_id: Optional[str] = Field(
        default=None, description="Client id; generated automatically if omitted."
    )
    user_id: str = Field(min_length=1, max_length=64, examples=["u_1001"])
    amount: float = Field(ge=0, le=1_000_000_000, examples=[1200.0])
    location: str = Field(min_length=1, max_length=64, examples=["Colombo, LK"])
    device_id: str = Field(min_length=1, max_length=64, examples=["dev_trusted_01"])

    # --- behavioural biometrics / device fingerprinting -------------------
    device_change: bool = Field(
        default=False, description="Device fingerprint differs from the known one."
    )
    typing_speed: float = Field(
        default=240.0, ge=0, le=2000, description="Keystrokes per minute."
    )
    navigation_pattern: str = Field(
        default="normal",
        description=f"One of {KNOWN_NAVIGATION_PATTERNS}; anything else -> 'unknown'.",
    )
    transaction_frequency: int = Field(
        default=3, ge=0, le=1000, description="Transactions by this user in the last 24h."
    )

    # --- relational context ------------------------------------------------
    beneficiary_change: bool = Field(
        default=False, description="Money is going to a newly added payee."
    )
    beneficiary_id: Optional[str] = Field(
        default=None, max_length=64, description="Optional payee id for graph edges."
    )
    ip_address: Optional[str] = Field(
        default=None, max_length=64, description="Optional IP for relational analysis."
    )

    # --- account context ---------------------------------------------------
    previous_transaction_amount: float = Field(default=0.0, ge=0, le=1_000_000_000)
    account_age: int = Field(default=365, ge=0, le=50_000, description="Account age in days.")

    timestamp: Optional[float] = Field(
        default=None, description="Unix epoch seconds; defaults to now."
    )

    @field_validator("navigation_pattern", mode="before")
    @classmethod
    def _normalise_navigation(cls, value: object) -> str:
        """Lower-case the label and fall back to 'unknown' if unrecognised."""
        if not isinstance(value, str):
            return "unknown"
        cleaned = value.strip().lower().replace("-", "_").replace(" ", "_")
        return cleaned if cleaned in KNOWN_NAVIGATION_PATTERNS else "unknown"

    @field_validator("user_id", "device_id", "location", mode="before")
    @classmethod
    def _strip_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    def normalised(self) -> "Transaction":
        """Return a copy with the auto-generated fields filled in."""
        return self.model_copy(
            update={
                "transaction_id": self.transaction_id or f"txn_{uuid.uuid4().hex[:12]}",
                "timestamp": self.timestamp if self.timestamp is not None else time.time(),
                "beneficiary_id": self.beneficiary_id or f"ben_default_{self.user_id}",
            }
        )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
class SignalOut(BaseModel):
    """One explainable contribution to a stream score."""

    name: str
    value: float = Field(description="Normalised strength of the signal, 0..1.")
    weight: float
    message: str


class ScoreResponse(BaseModel):
    """Everything the UI needs to explain a single decision."""

    transaction_id: str
    user_id: str
    timestamp: float

    # --- the three numbers the presentation talks about --------------------
    behavioral_score: float = Field(description="LSTM-autoencoder-like anomaly score, 0..1.")
    graph_score: float = Field(description="GNN-like relational risk score, 0..1.")
    risk_score: float = Field(description="Fused final fraud risk, 0..1.")

    # --- the deterministic gateway output ----------------------------------
    decision: Literal["ALLOW", "STEP-UP", "BLOCK"]
    reason: str = Field(description="Single-line summary reason.")
    reasons: List[str] = Field(default_factory=list, description="All human-readable reasons.")
    rules_fired: List[str] = Field(default_factory=list)
    thresholds: Dict[str, float] = Field(default_factory=dict)
    policy_version: str

    # --- transparency -------------------------------------------------------
    behavioral_signals: List[SignalOut] = Field(default_factory=list)
    graph_signals: List[SignalOut] = Field(default_factory=list)
    model_mode: str = Field(description="SIMULATED_PROTOTYPE - no trained network is used.")
    attack_type: Optional[str] = None


class ScoreRequest(Transaction):
    """POST /score body. Adds a switch to keep the demo state clean."""

    persist_state: bool = Field(
        default=True,
        description="If false, the transaction is scored without updating user/graph memory.",
    )
    attack_type: Optional[str] = Field(
        default=None, description="Free-text tag written to the audit log."
    )


class AttackRequest(BaseModel):
    """POST /simulate-attack body."""

    attack_type: AttackType
    steps: int = Field(default=8, ge=2, le=25, description="Number of simulated stages.")


class AttackStep(BaseModel):
    step: int
    label: str
    amount: float
    behavioral_score: float
    graph_score: float
    risk_score: float
    decision: str
    reason: str


class AttackResponse(BaseModel):
    attack_type: str
    title: str
    description: str
    baseline: AttackStep
    steps: List[AttackStep]
    summary: str
    detected: bool
    single_stream_would_have_missed: bool = Field(
        default=False,
        description="True when a behaviour-only model would have allowed the attack.",
    )
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_mode: str
    uptime_seconds: float
    transactions_scored: int
    policy_version: str
    thresholds: Dict[str, float]


class AuditEntry(BaseModel):
    timestamp: str
    transaction_id: str
    user_id: str
    behavioral_score: float
    graph_score: float
    risk_score: float
    decision: str
    reason: str
    attack_type: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    detail: str
