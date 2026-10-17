"""Causal transaction graphs.

An edge always points from an earlier transaction to a later one that shares an
account or device key. The model therefore cannot read the future when it scores
the current transaction.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def _clean_key(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def causal_edges(
    groups: list[np.ndarray],
    times: np.ndarray,
    max_neighbors: int = 5,
) -> np.ndarray:
    """Return an array of shape [2, E] with directed past-to-present edges."""
    if max_neighbors < 1:
        raise ValueError("max_neighbors must be at least 1")
    pairs: set[tuple[int, int]] = set()
    count = len(times)
    for group in groups:
        if len(group) != count:
            raise ValueError("Each grouping column must have one value per transaction.")
        buckets: dict[str, list[int]] = defaultdict(list)
        for index, raw in enumerate(group):
            key = _clean_key(raw)
            if key is not None:
                buckets[key].append(index)
        for indexes in buckets.values():
            ordered = sorted(indexes, key=lambda item: (times[item], item))
            for position, destination in enumerate(ordered):
                start = max(0, position - max_neighbors)
                for source in ordered[start:position]:
                    if times[source] < times[destination]:
                        pairs.add((source, destination))
                    elif times[source] == times[destination] and source < destination:
                        # Equal timestamps stay in file order and never point backwards.
                        pairs.add((source, destination))
    if not pairs:
        return np.zeros((2, 0), dtype=np.int64)
    edges = np.array(sorted(pairs), dtype=np.int64).T
    return edges
