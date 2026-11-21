"""Small statistics helpers for the evaluation notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def paired_block_bootstrap(
    a,
    b,
    *,
    groups,
    n_boot: int = 2000,
    seed: int = 42,
    min_groups: int = 10,
) -> tuple[float, float, float]:
    """Mean of (a - b) with a 95% interval, resampling whole GROUPS (e.g. dates).

    Test rows are not independent draws. Every holding on a date shares that day's market
    move, and a label that looks ~30 days ahead shares a price path with its neighbours.
    Resampling individual rows treats thousands of correlated rows as thousands of
    independent observations, so the interval comes out far too narrow and "significant"
    verdicts follow from chance. Resampling whole dates keeps each date's rows together and
    makes the effective sample size the number of dates.

    The statistic on each resample is the mean over all rows drawn (a ratio of the drawn
    groups' sums to their row counts), so unequal group sizes are handled naturally and the
    point estimate equals the plain mean difference.

    Returns (nan, nan, nan) with fewer than `min_groups` groups: too few clusters for any
    interval to mean something.
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    keys = np.asarray(pd.Series(np.asarray(groups)).astype(str))
    if diff.shape != keys.shape:
        raise ValueError(f"a, b and groups must be the same length, got {diff.shape} and {keys.shape}")

    frame = pd.DataFrame({"d": diff, "g": keys}).dropna(subset=["d"])
    per_group = frame.groupby("g")["d"].agg(["sum", "count"])
    n_groups = len(per_group)
    if n_groups < min_groups:
        return float("nan"), float("nan"), float("nan")

    sums = per_group["sum"].to_numpy()
    counts = per_group["count"].to_numpy().astype(float)

    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, n_groups, size=(n_boot, n_groups))
    boots = sums[drawn].sum(axis=1) / counts[drawn].sum(axis=1)
    low, high = np.percentile(boots, [2.5, 97.5])
    return float(sums.sum() / counts.sum()), float(low), float(high)
