"""
Request-scoped caller credentials.

The chat endpoint stores the caller's own bearer token here so a tool can read the caller's
records from the platform service AS that caller. Tools must not mint a token for some
other user to fill the gap: without a caller token they report that the data needs sign-in.
"""

import contextvars

_caller_authorization: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_caller_authorization", default=None
)


def set_caller_authorization(header: str | None):
    # Not a str when chat() is called directly rather than over HTTP: the parameter then
    # holds FastAPI's Header() default object.
    value = header.strip() if isinstance(header, str) else ""
    return _caller_authorization.set(value if value.lower().startswith("bearer ") else None)


def get_caller_authorization() -> str | None:
    return _caller_authorization.get()


def reset_caller_authorization(token) -> None:
    _caller_authorization.reset(token)
