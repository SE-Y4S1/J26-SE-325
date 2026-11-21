"""Joint return scenarios for the CVaR objective.

CVaR of a portfolio depends on how assets move TOGETHER. Scenarios drawn independently per
asset give every pair a correlation of zero, so a book of SPY, QQQ and AAPL (really ~0.8
correlated) looks diversified, and the optimizer concentrates in whichever names have the
lowest standalone volatility. Every scenario here is a single joint draw across assets.

Two sources, in order of preference:

  * historical -- rows of the symbols' own aligned daily returns, re-centred on the supplied
    expected returns so the forecast (not the sample mean of a noisy history) sets the drift
  * assumed_correlation -- a multivariate normal with a constant pairwise correlation, used
    only when there is not enough aligned history. Its correlation is an assumption and the
    caller is told so via the returned source label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Equity/crypto/FX books are positively correlated on the whole; 0.5 is a deliberately
# non-zero default for the no-history case. It is an assumption, surfaced as such.
DEFAULT_ASSUMED_CORRELATION = 0.5
DEFAULT_N_SCENARIOS = 500
MIN_HISTORY_OBSERVATIONS = 60


def bootstrap_scenarios(
    history: np.ndarray,
    mu: np.ndarray,
    *,
    n_scenarios: int = DEFAULT_N_SCENARIOS,
    seed: int = 42,
) -> np.ndarray:
    """Resample whole rows of `history` (T, n_assets), centred on `mu`.

    Resampling rows rather than columns is what keeps the cross-asset dependence intact.
    """
    history = np.asarray(history, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if history.ndim != 2 or history.shape[1] != mu.size:
        raise ValueError(f"history {history.shape} does not match {mu.size} assets")

    centred = history - history.mean(axis=0)
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, centred.shape[0], size=n_scenarios)
    return centred[rows] + mu


def correlated_normal_scenarios(
    mu: np.ndarray,
    volatility: np.ndarray,
    *,
    correlation: float = DEFAULT_ASSUMED_CORRELATION,
    n_scenarios: int = DEFAULT_N_SCENARIOS,
    seed: int = 42,
) -> np.ndarray:
    """Multivariate normal with a constant pairwise correlation."""
    mu = np.asarray(mu, dtype=float)
    vol = np.asarray(volatility, dtype=float)
    n = mu.size
    if not -1.0 / max(n - 1, 1) < correlation < 1.0:
        raise ValueError(f"correlation {correlation} is not a valid constant correlation for {n} assets")

    corr = np.full((n, n), correlation)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)
    z = np.random.default_rng(seed).standard_normal((n_scenarios, n))
    return mu + (z @ chol.T) * vol


def joint_scenarios(
    mu: np.ndarray,
    volatility: np.ndarray,
    *,
    history: np.ndarray | None = None,
    n_scenarios: int = DEFAULT_N_SCENARIOS,
    seed: int = 42,
    correlation: float = DEFAULT_ASSUMED_CORRELATION,
    min_observations: int = MIN_HISTORY_OBSERVATIONS,
) -> tuple[np.ndarray, str]:
    """Scenarios plus where they came from: "historical" or "assumed_correlation"."""
    mu = np.asarray(mu, dtype=float)
    if history is not None:
        history = np.asarray(history, dtype=float)
        if (
            history.ndim == 2
            and history.shape[1] == mu.size
            and history.shape[0] >= min_observations
            and np.isfinite(history).all()
        ):
            return bootstrap_scenarios(history, mu, n_scenarios=n_scenarios, seed=seed), "historical"

    return (
        correlated_normal_scenarios(
            mu, volatility, correlation=correlation, n_scenarios=n_scenarios, seed=seed
        ),
        "assumed_correlation",
    )


def returns_matrix(
    bars_by_symbol: dict[str, pd.DataFrame],
    *,
    min_observations: int = MIN_HISTORY_OBSERVATIONS,
) -> pd.DataFrame | None:
    """Aligned daily simple returns, one column per symbol, on dates ALL symbols traded.

    Inner-joined on date: a symbol that listed later shortens everyone's window rather than
    being padded with zeros, which would fake a low correlation. Returns None when fewer
    than `min_observations` common returns exist.
    """
    if not bars_by_symbol:
        return None

    closes = {}
    for symbol, bars in bars_by_symbol.items():
        if bars is None or bars.empty or "close" not in bars or "timestamp" not in bars:
            return None
        series = bars.drop_duplicates("timestamp").set_index("timestamp")["close"].sort_index()
        closes[symbol] = series.astype(float)

    prices = pd.DataFrame(closes).dropna()
    returns = prices.pct_change().dropna()
    if len(returns) < min_observations:
        return None
    return returns
