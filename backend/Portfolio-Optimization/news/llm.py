"""Local LLM access via Ollama (keyword expansion; sentiment fallback).

The model is chosen at call time, not hard-coded: OLLAMA_MODEL if that model is pulled,
otherwise the first model that is. The code previously assumed `gemma4-e4b`, and on a
machine with only `llama3.2:3b` every Ollama path silently reported "unavailable".
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
PREFERRED_MODEL = os.getenv("OLLAMA_MODEL", "gemma4-e4b")

_TAGS_TTL_SECONDS = 60
_tags_cache: tuple[float, list[str]] | None = None


def installed_models(base_url: str = OLLAMA_BASE_URL) -> list[str]:
    """Pulled model names ('llama3.2:3b'), or [] when the daemon is down. Cached briefly."""
    global _tags_cache
    now = time.time()
    if _tags_cache and now - _tags_cache[0] < _TAGS_TTL_SECONDS:
        return _tags_cache[1]
    try:
        response = requests.get(base_url.replace("/v1", "/api/tags"), timeout=3)
        response.raise_for_status()
        names = [m.get("name", "") for m in response.json().get("models", []) if m.get("name")]
    except Exception:  # noqa: BLE001 - daemon down is a normal state
        names = []
    _tags_cache = (now, names)
    return names


def resolve_model(preferred: str | None = None) -> str | None:
    """The preferred model if pulled (with or without a tag), else any pulled model."""
    preferred = preferred or PREFERRED_MODEL
    names = installed_models()
    if not names:
        return None
    for name in names:
        if name == preferred or name.split(":")[0] == preferred.split(":")[0]:
            return name
    logger.info("Ollama model %s is not pulled; using %s instead", preferred, names[0])
    return names[0]


def chat_json(system: str, user: str, *, model: str | None = None, timeout: float = 240.0) -> dict[str, Any]:
    """One JSON-constrained completion. Raises RuntimeError when no model is available."""
    from openai import OpenAI

    chosen = model or resolve_model()
    if chosen is None:
        raise RuntimeError("Ollama is not running or has no models pulled")

    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=timeout)
    response = client.chat.completions.create(
        model=chosen,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"model returned non-JSON output: {content[:200]!r}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("model returned JSON that is not an object")
    return payload
