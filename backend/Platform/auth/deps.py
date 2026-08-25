"""FastAPI dependencies for authentication."""

from __future__ import annotations

import logging

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from auth.security import decode_access_token
from store.database import get_session
from store.models import User

logger = logging.getLogger(__name__)

# auto_error=False so a missing header reaches our handler and produces a consistent
# WWW-Authenticate response, rather than FastAPI's bare 403.
_bearer = HTTPBearer(auto_error=False)

_UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    session: Session = Depends(get_session),
) -> User:
    """Resolve the caller from their bearer token, or raise 401.

    Every failure path returns the same 401 with the same message. Distinguishing "expired"
    from "malformed" from "user deleted" in the response would tell an attacker which tokens
    are real; the detail goes to the log instead.
    """
    if credentials is None or not credentials.credentials:
        raise _UNAUTHENTICATED

    try:
        payload = decode_access_token(credentials.credentials)
    except jwt.ExpiredSignatureError:
        logger.info("rejected an expired token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None
    except jwt.InvalidTokenError as exc:
        logger.info("rejected an invalid token: %s", exc)
        raise _UNAUTHENTICATED from None

    subject = payload.get("sub")
    try:
        user_id = int(subject)
    except (TypeError, ValueError):
        logger.warning("token carried a non-numeric subject: %r", subject)
        raise _UNAUTHENTICATED from None

    user = session.get(User, user_id)
    if user is None:
        # A validly-signed token for a since-deleted account. Still 401: the credential no
        # longer identifies anyone.
        logger.info("token referenced a user that no longer exists: %s", user_id)
        raise _UNAUTHENTICATED

    return user
