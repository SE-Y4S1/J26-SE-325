"""Request/response models for the platform service.

`HoldingModel` deliberately mirrors Component 1's `Holding` field-for-field
(service/contracts.py in backend/Portfolio-Optimization). The frontend takes a stored
portfolio and posts its holdings straight to /portfolio/withdraw; any renaming here would
force a mapping layer, and a mapping layer is where field drift hides between two contracts
that are supposed to be the same shape.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# --------------------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: EmailStr
    display_name: str = Field(min_length=1, max_length=120)
    # 8 is the floor, not a recommendation. Argon2 imposes no upper bound, so long
    # passphrases are encouraged rather than truncated as they would be under bcrypt.
    password: str = Field(min_length=8, max_length=256)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int                     # seconds, so the client can refresh before expiry
    user: "UserResponse"


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    display_name: str
    created_at: datetime


# --------------------------------------------------------------------------------------
# Portfolios
# --------------------------------------------------------------------------------------

class HoldingModel(BaseModel):
    """One position. Field names match Component 1's Holding exactly."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str = Field(min_length=1, max_length=20)
    quantity: float = Field(ge=0)
    current_price: float = Field(gt=0)
    avg_daily_volume: float = Field(ge=0)
    cost_basis: float | None = None


class PortfolioCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    base_currency: str = Field(default="USD", min_length=3, max_length=3)
    holdings: list[HoldingModel] = Field(default_factory=list)


class PortfolioUpdate(BaseModel):
    """Every field optional -- this is a PUT that behaves as a partial update.

    `holdings=None` means "leave them alone"; `holdings=[]` means "remove them all". Without
    that distinction there is no way to rename a portfolio without resending every holding.
    """

    name: str | None = Field(default=None, min_length=1, max_length=120)
    base_currency: str | None = Field(default=None, min_length=3, max_length=3)
    holdings: list[HoldingModel] | None = None


class PortfolioResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    base_currency: str
    holdings: list[HoldingModel]
    total_value: float
    created_at: datetime
    updated_at: datetime


class PortfolioSummary(BaseModel):
    """List view: enough for a picker without shipping every holding."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    base_currency: str
    holding_count: int
    total_value: float
    updated_at: datetime


# --------------------------------------------------------------------------------------
# Executions: Component 1 plan -> Component 2 fraud gateway -> Component 3 audit anchor
# --------------------------------------------------------------------------------------

ExecutionKind = Literal["withdrawal", "rebalance"]
ExecutionStatus = Literal["EXECUTED", "BLOCKED", "PENDING_REVIEW", "HELD", "REJECTED"]


class ExecutionRequest(BaseModel):
    """A Component 1 response, submitted for execution as returned.

    The orders are derived here from `plan`, never accepted from the client: a withdrawal
    from `assets_to_sell`, a rebalance from `recommended_weights` against current holdings.
    That way what is screened and anchored is what Component 1 actually produced.
    """

    kind: ExecutionKind
    plan: dict
    # An opaque, pseudonymous device identifier for Component 2's relational stream. Never
    # a fingerprint, IP address or user-agent string.
    device_key: str | None = Field(default=None, max_length=128, pattern=r"^[A-Za-z0-9_.:\-]+$")


class Order(BaseModel):
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: float = Field(gt=0)
    price: float = Field(gt=0)


class ExecutionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    portfolio_id: int | None
    kind: ExecutionKind
    status: ExecutionStatus
    status_reason: str
    amount: float
    currency: str
    source_model_version: str
    plan_hash: str
    plan: dict
    orders: list[Order]
    fraud: dict | None
    audit: dict | None
    created_at: datetime
    updated_at: datetime


class ExecutionSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    portfolio_id: int | None
    kind: ExecutionKind
    status: ExecutionStatus
    amount: float
    currency: str
    created_at: datetime


class ExecutionConfirm(BaseModel):
    """Step-up for a PENDING_REVIEW execution: the account holder re-authenticates.

    A password re-entry is the weakest acceptable step-up -- an OTP or device biometric is
    what Component 2's STEP_UP actually calls for -- but it is a real second check rather
    than a button anyone holding the session token could press.
    """

    password: str = Field(min_length=1, max_length=256)
    approve: bool


class HealthResponse(BaseModel):
    status: str
    service: str
    database_ready: bool


TokenResponse.model_rebuild()
