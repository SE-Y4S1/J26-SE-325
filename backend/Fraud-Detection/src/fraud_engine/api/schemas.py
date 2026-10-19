"""HTTP schemas. Extra personal fields are rejected."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SimulatedSignals(BaseModel):
    model_config = ConfigDict(extra="forbid")
    origin: Literal["simulated"]
    typing_interval_ms: float | None = None
    pointer_jitter: float | None = None
    session_duration_s: float | None = None


class TransactionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transaction_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:\-]+$")
    amount: float = Field(ge=0, le=1_000_000_000_000)
    currency: str = Field(default="USD", pattern=r"^[A-Za-z]{3}$")
    product_code: str | None = Field(default=None, max_length=8)
    account_key: str | None = Field(default=None, max_length=128)
    device_key: str | None = Field(default=None, max_length=128)
    ip_reputation: float | None = Field(default=None, ge=0, le=1)
    features: dict[str, float | int | None] = Field(default_factory=dict)
    categories: dict[str, str] = Field(default_factory=dict)
    simulated_signals: SimulatedSignals | None = None


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transaction_id: str = Field(min_length=1, max_length=128)
    fraud_score: float
    label: int = Field(ge=0, le=1)
