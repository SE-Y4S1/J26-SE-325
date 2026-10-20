"""
Context variable management for Responsible AI user preferences.
Ensures thread-safe, request-scoped preference propagation across tools and LLM workflows.
"""

import contextvars
from typing import TypedDict


class PreferenceDict(TypedDict, total=False):
    explanation_enabled: bool
    personalized_context: bool
    data_sharing_consent: bool
    demo_mode: bool
    transaction_id: str | None


DEFAULT_PREFERENCES: PreferenceDict = {
    "explanation_enabled": True,
    "personalized_context": True,
    "data_sharing_consent": True,
    "demo_mode": False,
    "transaction_id": None,
}

_current_preferences: contextvars.ContextVar[PreferenceDict] = contextvars.ContextVar(
    "_current_preferences", default=DEFAULT_PREFERENCES
)

_current_transaction_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_transaction_id", default=None
)

_current_demo_mode: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_current_demo_mode", default=False
)


def get_current_preferences() -> PreferenceDict:
    return _current_preferences.get()


def set_current_preferences(prefs: PreferenceDict):
    return _current_preferences.set(prefs)


def reset_current_preferences(token):
    _current_preferences.reset(token)


def get_request_transaction_id() -> str | None:
    return _current_transaction_id.get()


def set_request_transaction_id(tx_id: str | None):
    return _current_transaction_id.set(tx_id.strip() if tx_id else None)


def reset_request_transaction_id(token):
    _current_transaction_id.reset(token)


def get_request_demo_mode() -> bool:
    return _current_demo_mode.get()


def set_request_demo_mode(enabled: bool):
    return _current_demo_mode.set(bool(enabled))


def reset_request_demo_mode(token):
    _current_demo_mode.reset(token)

