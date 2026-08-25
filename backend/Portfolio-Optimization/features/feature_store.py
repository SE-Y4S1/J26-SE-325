"""Joins price + indicators + sentiment into one wide table keyed on (symbol, timestamp).

MISSING-SENTIMENT POLICY (decay, not zero-fill)
-----------------------------------------------
Most symbols have no news on most days. Forward-filling the last score unchanged implies a
week-old headline still carries full weight; zero-filling implies "neutral", which is a
*different claim* from "no information" and is indistinguishable from genuinely neutral
news -- it injects a fake signal on the majority of rows.

We instead decay the last observed score toward neutral:

    s_t = s_{t-1} * exp(-dt / TAU)      TAU = 3 trading days

so a headline's influence halves in ~2 days and is negligible inside a fortnight. TAU=3 is
chosen to sit between the 1-day horizon (where news should dominate) and the 21-day horizon
(where it should not). `sentiment_volume` is zero-filled -- "no headlines" genuinely is a
count of zero -- and `days_since_news` is exposed so a model can learn to discount stale
sentiment itself rather than trusting the decay blindly.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SENTIMENT_DECAY_TAU_DAYS = 3.0

SENTIMENT_COLUMNS = ["mean_sentiment", "sentiment_volume", "sentiment_momentum"]

# Keys, raw inputs and forward-looking targets are never model inputs.
_NON_FEATURE_COLUMNS = {
    "timestamp", "symbol", "asset_class",
    "open", "high", "low", "close", "volume",
    "target_return", "target_volatility",
}


def decay_fill_sentiment(
    frame: pd.DataFrame, *, tau: float = SENTIMENT_DECAY_TAU_DAYS
) -> pd.DataFrame:
    """Fill sentiment gaps by exponential decay toward neutral. Adds `days_since_news`.

    Operates per symbol. Rows before a symbol's first-ever headline get 0.0 sentiment and a
    sentinel `days_since_news`, since there is no prior score to decay from -- that is a
    genuine absence of information, not a decayed signal.
    """
    if frame.empty:
        return frame

    out = frame.sort_values(["symbol", "timestamp"]).copy()
    filled: list[pd.DataFrame] = []

    for symbol, group in out.groupby("symbol", sort=False):
        group = group.copy()
        observed = group["mean_sentiment"].notna()

        # Trading days since the most recent observed headline.
        position = np.arange(len(group))
        last_observed_pos = pd.Series(np.where(observed, position, np.nan), index=group.index).ffill()
        days_since = position - last_observed_pos.to_numpy()

        decayed = group["mean_sentiment"].ffill() * np.exp(-days_since / tau)

        group["mean_sentiment"] = decayed.fillna(0.0)
        group["days_since_news"] = pd.Series(days_since, index=group.index).fillna(-1.0)
        group["sentiment_volume"] = group["sentiment_volume"].fillna(0.0)
        # Momentum is a difference of levels; with no news there is no change to report.
        group["sentiment_momentum"] = group["sentiment_momentum"].fillna(0.0)
        filled.append(group)

    return pd.concat(filled).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def build_feature_table(
    bars_by_symbol: dict[str, pd.DataFrame],
    indicators: pd.DataFrame,
    sentiment: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Left-join indicators and sentiment onto the price spine.

    Price is the spine because it defines the trading calendar; sentiment timestamps do not
    align to it (weekend news belongs to the next session). Left-joining the other way would
    invent rows on non-trading days that no price exists for.
    """
    price_frames = []
    for symbol, bars in bars_by_symbol.items():
        if bars.empty:
            continue
        frame = bars.copy()
        frame["symbol"] = symbol
        price_frames.append(frame)

    if not price_frames:
        return pd.DataFrame()

    spine = pd.concat(price_frames, ignore_index=True)
    spine["timestamp"] = pd.to_datetime(spine["timestamp"]).dt.normalize()

    table = spine
    if indicators is not None and not indicators.empty:
        ind = indicators.copy()
        ind["timestamp"] = pd.to_datetime(ind["timestamp"]).dt.normalize()
        table = table.merge(ind, on=["symbol", "timestamp"], how="left")

    if sentiment is not None and not sentiment.empty:
        sent = sentiment.copy()
        # Sentiment is aggregated to a date; normalize both sides so the join keys match.
        sent["timestamp"] = pd.to_datetime(sent["date"] if "date" in sent.columns else sent["timestamp"]).dt.normalize()
        keep = ["symbol", "timestamp", *[c for c in SENTIMENT_COLUMNS if c in sent.columns]]
        table = table.merge(sent[keep], on=["symbol", "timestamp"], how="left")
    else:
        for column in SENTIMENT_COLUMNS:
            table[column] = np.nan

    table = decay_fill_sentiment(table)

    # Indicator warm-up leaves leading NaNs. Dropping them here (rather than imputing) keeps
    # the burn-in rows out of training entirely -- imputed indicator values are fabricated
    # data, and a model that trains on them learns a relationship that does not exist.
    indicator_cols = [c for c in table.columns if c not in _NON_FEATURE_COLUMNS and c not in SENTIMENT_COLUMNS]
    before = len(table)
    table = table.dropna(subset=[c for c in indicator_cols if c != "days_since_news"], how="any")
    logger.info("dropped %d warm-up/incomplete rows (%d remain)", before - len(table), len(table))

    return table.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def add_targets(table: pd.DataFrame, *, horizon: int) -> pd.DataFrame:
    """Attach forward-looking training targets.

    Computed with a NEGATIVE shift per symbol, which is the only place look-ahead is
    intentional -- these are labels, not features. feature_columns() excludes them so they
    can never be fed back in as inputs.
    """
    out = table.sort_values(["symbol", "timestamp"]).copy()
    grouped = out.groupby("symbol", sort=False)["close"]

    out["target_return"] = grouped.transform(lambda s: s.shift(-horizon) / s - 1.0)
    out["target_volatility"] = grouped.transform(
        lambda s: np.log(s / s.shift(1)).rolling(horizon).std().shift(-horizon)
    )
    return out


def assert_no_leakage(table: pd.DataFrame) -> None:
    """Guard that no row carries information dated after its own timestamp.

    Cheap to check here and catches the single most common way a backtest silently becomes
    meaningless. Phase 7 enforces the same property across folds.
    """
    if table.empty:
        return

    if not table.groupby("symbol", sort=False)["timestamp"].is_monotonic_increasing.all():
        raise ValueError("feature table is not sorted by timestamp within each symbol")

    leaked = [c for c in table.columns if c.startswith("target_") and c in table.columns]
    for column in leaked:
        # Targets are forward-looking by construction, so the LAST `horizon` rows per symbol
        # must be NaN. If they are not, the shift was applied in the wrong direction.
        tail_all_null = table.groupby("symbol", sort=False)[column].apply(lambda s: s.iloc[-1:].isna().all())
        if not bool(tail_all_null.all()):
            raise ValueError(
                f"{column}: final row per symbol is populated, which means the target was "
                "shifted the wrong way and leaks future data"
            )


def feature_columns(table: pd.DataFrame) -> list[str]:
    """Model-input columns only -- excludes keys, raw OHLCV and forward-looking targets."""
    return [
        c for c in table.columns
        if c not in _NON_FEATURE_COLUMNS and not c.startswith("target_")
    ]
