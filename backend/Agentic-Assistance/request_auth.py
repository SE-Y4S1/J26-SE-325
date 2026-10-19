"""
Request-scoped caller credentials and JWT caller identity verification.

The chat and governance endpoints store and verify the caller's bearer token so:
1. Tools can read the caller's records from the platform service AS that caller.
2. Governance decisions associate with the authenticated user ID and email.
"""

from __future__ import annotations

import contextvars
import logging
import os
from typing import Tuple

import jwt
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"
ISSUER = "j26-se-325-platform"

_caller_authorization: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_caller_authorization", default=None
)


def auth_required() -> bool:
    """Whether JWT authentication is strictly required."""
    return os.getenv("AUTH_REQUIRED", "false").strip().lower() in {"1", "true", "yes", "on"}


def set_caller_authorization(header: str | None):
    # Not a str when chat() is called directly rather than over HTTP: the parameter then
    # holds FastAPI's Header() default object.
    value = header.strip() if isinstance(header, str) else ""
    return _caller_authorization.set(value if value.lower().startswith("bearer ") else None)


def get_caller_authorization() -> str | None:
    return _caller_authorization.get()


def reset_caller_authorization(token) -> None:
    _caller_authorization.reset(token)


def decode_caller_identity(
    header: str | None,
    enforce_auth: bool | None = None,
) -> Tuple[str | None, str | None]:
    """
    Extract (user_id, email) from Bearer Authorization header.
    If enforce_auth is True (or AUTH_REQUIRED is on), raises HTTPException(401) on missing/invalid token.
    Otherwise returns (None, None) if no valid credentials provided.
    """
    required = auth_required() if enforce_auth is None else enforce_auth

    if not isinstance(header, str) or not header.strip().lower().startswith("bearer "):
        if required:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return (None, None)

    raw_token = header.strip()[7:].strip()
    if not raw_token:
        if required:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return (None, None)

    secret = os.getenv("JWT_SECRET", "").strip()
    if not secret:
        if required:
            logger.error("AUTH_REQUIRED is on but JWT_SECRET is unset; refusing request")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server auth is misconfigured",
            )
        return (None, None)

    try:
        payload = jwt.decode(
            raw_token,
            secret,
            algorithms=[ALGORITHM],
            issuer=ISSUER,
            options={"require": ["exp", "iat", "sub"]},
        )
        user_id = str(payload.get("sub"))
        email = payload.get("email")
        return (user_id, email)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None
    except jwt.InvalidTokenError as exc:
        logger.info("Rejected invalid token in governance endpoint: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None
