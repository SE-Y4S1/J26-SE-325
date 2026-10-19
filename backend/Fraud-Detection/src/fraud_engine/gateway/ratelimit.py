"""In-memory rate limit. One process, explicit timestamps, no personal data."""

from __future__ import annotations

from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, limit: int, window_seconds: float, block_multiplier: float = 3.0) -> None:
        if limit < 1:
            raise ValueError("rate limit must be at least 1")
        self.limit = limit
        self.window_seconds = window_seconds
        self.block_multiplier = block_multiplier
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def observe(self, key: str | None, now: float) -> str:
        """Record one event and return ok, step_up, or block."""
        if not key:
            return "ok"
        bucket = self._events[key]
        cutoff = now - self.window_seconds
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        bucket.append(now)
        count = len(bucket)
        if count > self.limit * self.block_multiplier:
            return "block"
        if count > self.limit:
            return "step_up"
        return "ok"
