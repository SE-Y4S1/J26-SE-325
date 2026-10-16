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


DEFAULT_PREFERENCES: PreferenceDict = {
    "explanation_enabled": True,
    "personalized_context": True,
    "data_sharing_consent": True,
}

_current_preferences: contextvars.ContextVar[PreferenceDict] = contextvars.ContextVar(
    "_current_preferences", default=DEFAULT_PREFERENCES
)


def get_current_preferences() -> PreferenceDict:
    return _current_preferences.get()


def set_current_preferences(prefs: PreferenceDict):
    return _current_preferences.set(prefs)


def reset_current_preferences(token):
    _current_preferences.reset(token)
