"""Reputation is an input score, not a live lookup of a person's network address."""

from __future__ import annotations


def reputation_band(reputation: float | None, step_below: float, block_below: float) -> str | None:
    if reputation is None:
        return None
    if not 0.0 <= reputation <= 1.0:
        raise ValueError("ip_reputation must be between 0 and 1, or omitted.")
    if reputation <= block_below:
        return "block"
    if reputation <= step_below:
        return "step_up"
    return "allow"
