"""One JSON-over-HTTP helper for every exchange adapter.

Centralised for two reasons:
  * Consistent failure behaviour. Every adapter gets the same timeouts and the same bounded
    retry on transient errors, instead of each one inventing its own and one of them
    retrying forever against a rate limit.
  * One seam for tests. The adapter tests monkeypatch `request_json` with recorded responses
    (tests/fixtures/markets/), so the suite never touches the network.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 2.0

# Statuses worth retrying. 4xx other than 429 means the request itself is wrong, and
# repeating it only spends rate limit.
_RETRYABLE = {429, 500, 502, 503, 504}

_session: requests.Session | None = None


class UpstreamError(RuntimeError):
    """An exchange or data provider could not answer. Carries the provider for the message."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider


def _get_session() -> requests.Session:
    """Reuse one connection pool; a fresh TLS handshake per call is slow on a thin link."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = "j26-se-325-component1/0.1"
    return _session


def _request(
    provider: str,
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> requests.Response:
    """A 200 response, retrying transient failures; UpstreamError otherwise."""
    last_error = "no attempt made"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = _get_session().request(
                method, url, params=params, data=data, timeout=timeout
            )
        except requests.RequestException as exc:
            last_error = str(exc)
        else:
            if response.status_code == 200:
                return response
            if response.status_code not in _RETRYABLE:
                raise UpstreamError(provider, f"HTTP {response.status_code} from {url}")
            last_error = f"HTTP {response.status_code}"

        if attempt < MAX_ATTEMPTS:
            delay = BACKOFF_SECONDS * attempt
            logger.info("%s %s failed (%s); retry %d in %.0fs", provider, url, last_error, attempt, delay)
            time.sleep(delay)

    raise UpstreamError(provider, f"{url} unavailable after {MAX_ATTEMPTS} attempts: {last_error}")


def request_json(
    provider: str,
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Any:
    """Issue a request and return the decoded JSON body, retrying transient failures.

    Raises UpstreamError when the provider stays unavailable, so callers can degrade (serve
    cached history, mark a symbol `partial`) rather than surface a raw requests exception.
    """
    response = _request(provider, method, url, params=params, data=data, timeout=timeout)
    try:
        return response.json()
    except ValueError:
        # CSE answers unknown paths with an HTML page and status 200 on its CDN.
        raise UpstreamError(provider, f"non-JSON response from {url}") from None


def request_bytes(provider: str, url: str, *, timeout: float = 600) -> bytes:
    """GET a binary body (report PDFs). Long default timeout: a 1 MB PDF on a thin link
    takes minutes."""
    return _request(provider, "GET", url, timeout=timeout).content
