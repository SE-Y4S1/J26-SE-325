"""Password hashing and JWT issue/verify.

Argon2 rather than bcrypt. Argon2id won the Password Hashing Competition and is the current
OWASP recommendation, and it has no length ceiling -- bcrypt silently truncates at 72 bytes,
so a long passphrase is quietly weakened with no error anywhere.

The JWT secret is shared with Component 1 (and later 2-4) so each service can verify a token
the platform service issued, without calling back here on every request.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from passlib.context import CryptContext

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 12       # a working day; this is a demo platform

# The value shipped in .env.example. Refusing to boot on this exact string is what stops a
# deployment from running with a secret that is public in the repository.
EXAMPLE_SECRET = "change-me-to-a-long-random-string"

_pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


def get_jwt_secret() -> str:
    """Read JWT_SECRET, refusing to run without a real one.

    Failing loudly at startup is deliberate. The usual failure mode is a hardcoded fallback
    that works everywhere including production, so nobody ever notices the tokens are
    forgeable by anyone who has read the source.
    """
    secret = os.getenv("JWT_SECRET", "").strip()

    if not secret:
        raise RuntimeError(
            "JWT_SECRET is not set. Generate one with:\n"
            "  python -c \"import secrets; print(secrets.token_urlsafe(48))\"\n"
            "and put it in the repo-root .env, shared with every backend service."
        )
    if secret == EXAMPLE_SECRET:
        raise RuntimeError(
            "JWT_SECRET is still the placeholder from .env.example. Replace it -- that value "
            "is committed to the repository and anyone could forge tokens with it."
        )
    if len(secret) < 32:
        raise RuntimeError(
            f"JWT_SECRET is only {len(secret)} characters. Use at least 32; a short secret "
            "is brute-forceable offline against any captured token."
        )
    return secret


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Check a password. Returns False on a malformed hash rather than raising.

    A corrupt or foreign hash in the database should read as "wrong password", not crash the
    login endpoint -- otherwise one bad row takes down authentication for everyone.
    """
    try:
        return _pwd_context.verify(password, password_hash)
    except Exception as exc:  # noqa: BLE001
        logger.warning("password verification failed on a malformed hash: %s", exc)
        return False


def create_access_token(
    *, subject: str, extra_claims: dict[str, Any] | None = None,
    expires_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES,
) -> str:
    """Issue a signed access token. `subject` is the user id as a string, per JWT convention."""
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
        "iss": "j26-se-325-platform",
        **(extra_claims or {}),
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    """Verify and decode a token.

    Raises jwt.InvalidTokenError (which ExpiredSignatureError subclasses) on anything wrong.
    `algorithms` is pinned to a single value on purpose: accepting a list the caller controls
    is how the classic "alg: none" and RS256->HS256 confusion attacks work.
    """
    return jwt.decode(
        token,
        get_jwt_secret(),
        algorithms=[ALGORITHM],
        issuer="j26-se-325-platform",
        options={"require": ["exp", "iat", "sub"]},
    )
