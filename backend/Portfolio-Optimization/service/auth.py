"""Optional JWT verification for Component 1's endpoints.

Tokens are ISSUED by the platform service (backend/Platform) and merely VERIFIED here, using
the same `JWT_SECRET`. That is the point of a shared secret: this service does not call back
to the platform on every request, so a slow or restarted platform service cannot take
Component 1 down with it.

This file is written to be copied unchanged into Components 2-4.

WHY IT IS OPTIONAL
------------------
`AUTH_REQUIRED` gates enforcement, defaulting to OFF. The existing 338 tests call these
endpoints with no credentials and must keep passing untouched; compose turns it on. A flag
that defaults to "secure" would have meant editing every one of those tests, which is how a
security control ends up disabled wholesale instead of scoped.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"
ISSUER = "j26-se-325-platform"

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class CallerIdentity:
    """Who the request is from, as far as the token claims."""

    user_id: int
    email: str | None = None


def auth_required() -> bool:
    """Whether tokens are enforced. Off unless explicitly enabled."""
    return os.getenv("AUTH_REQUIRED", "false").strip().lower() in {"1", "true", "yes", "on"}


def _unauthenticated() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> CallerIdentity | None:
    """Verify the platform service's token.

    Returns None when AUTH_REQUIRED is off, so endpoints can accept the dependency
    unconditionally and simply receive nothing in unauthenticated mode.
    """
    if not auth_required():
        return None

    if credentials is None or not credentials.credentials:
        raise _unauthenticated()

    secret = os.getenv("JWT_SECRET", "").strip()
    if not secret:
        # Enforcement was switched on without a secret. Refusing the request is the only safe
        # answer -- falling back to "allow" would silently disable the control that was just
        # asked for. 500, not 401: the fault is the deployment's, not the caller's.
        logger.error("AUTH_REQUIRED is on but JWT_SECRET is unset; refusing all requests")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server auth is misconfigured",
        )

    import jwt

    try:
        payload = jwt.decode(
            credentials.credentials,
            secret,
            # Pinned to one algorithm: accepting a caller-controlled list is how the
            # "alg: none" and RS256->HS256 confusion attacks work.
            algorithms=[ALGORITHM],
            issuer=ISSUER,
            options={"require": ["exp", "iat", "sub"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None
    except jwt.InvalidTokenError as exc:
        logger.info("rejected an invalid token: %s", exc)
        raise _unauthenticated() from None

    try:
        return CallerIdentity(user_id=int(payload["sub"]), email=payload.get("email"))
    except (KeyError, TypeError, ValueError):
        logger.warning("token carried an unusable subject: %r", payload.get("sub"))
        raise _unauthenticated() from None
