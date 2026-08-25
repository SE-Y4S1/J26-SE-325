"""Technical indicators via pandas-ta-openbb.

MACD, RSI, MFI and ATR are named explicitly in the TAF task list, so all four are
mandatory. Bollinger Band width is added from the build brief as a volatility-regime
feature the fuzzy layer consumes.

Uses pandas-ta-openbb rather than upstream pandas-ta: the classic 0.3.14b0 was pulled from
PyPI, and the current release hard-pins numba==0.61.2, which boxes numpy into [2.2.6, 2.3)
and conflicts with the rest of the stack. The import name is unchanged (`pandas_ta`).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard lookbacks, kept explicit so the dissertation can cite them rather than referring
# to library defaults that may drift between versions.
DEFAULT_PARAMS: dict[str, float] = {
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "rsi_length": 14,
    "mfi_length": 14,
    "atr_length": 14,
    "bb_length": 20,
    "bb_std": 2.0,
}

INDICATOR_COLUMNS = [
    "macd", "macd_signal", "macd_hist",
    "rsi", "mfi", "atr", "atr_pct", "bb_width",
]


def warmup_periods(params: dict | None = None) -> int:
    """Rows to discard at the head of a series before indicators are trustworthy.

    Driven by the longest lookback (MACD slow EMA + signal EMA). The backtest harness uses
    this to offset its first fold so no fold starts on burn-in values, which would otherwise
    leak the previous fold's data through the EMA state.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    return int(max(p["macd_slow"] + p["macd_signal"], p["bb_length"], p["rsi_length"])) + 1


def compute_indicators(bars: pd.DataFrame, *, params: dict | None = None) -> pd.DataFrame:
    """Indicator table for one symbol, one row per timestamp.

    `bars` needs columns open/high/low/close/volume indexed by timestamp. Returns MACD
    (line, signal, histogram), RSI, MFI, ATR (absolute and %-of-price) and Bollinger width.

    Forex caveat: MFI is volume-weighted and yfinance reports FX volume as 0, so MFI is NaN
    for FX symbols. That is handled here (left as NaN, then dropped from the FX feature set)
    rather than leaking a fake 0 into the feature store, which a model would read as an
    extreme oversold reading.
    """
    import pandas_ta as ta

    if bars.empty:
        return pd.DataFrame(columns=["timestamp", *INDICATOR_COLUMNS])

    p = {**DEFAULT_PARAMS, **(params or {})}
    frame = bars.sort_values("timestamp").reset_index(drop=True)
    high, low, close, volume = frame["high"], frame["low"], frame["close"], frame["volume"]

    out = pd.DataFrame({"timestamp": frame["timestamp"]})

    macd = ta.macd(close, fast=p["macd_fast"], slow=p["macd_slow"], signal=p["macd_signal"])
    if macd is not None and not macd.empty:
        # pandas-ta names columns MACD_12_26_9 / MACDh_.. / MACDs_..; match by prefix so a
        # parameter change does not silently produce all-NaN columns.
        out["macd"] = _column_by_prefix(macd, "MACD_")
        out["macd_hist"] = _column_by_prefix(macd, "MACDh_")
        out["macd_signal"] = _column_by_prefix(macd, "MACDs_")
    else:
        out["macd"] = out["macd_hist"] = out["macd_signal"] = np.nan

    out["rsi"] = _as_series(ta.rsi(close, length=p["rsi_length"]), len(frame))
    out["atr"] = _as_series(ta.atr(high=high, low=low, close=close, length=p["atr_length"]), len(frame))

    # ATR as a fraction of price: comparable across a $5 stock and a $500 one, which the raw
    # ATR is not. This is the input the fuzzy layer's `market_volatility` uses.
    out["atr_pct"] = out["atr"] / close.replace(0, np.nan)

    if (volume > 0).any():
        out["mfi"] = _as_series(
            ta.mfi(high=high, low=low, close=close, volume=volume, length=p["mfi_length"]),
            len(frame),
        )
    else:
        logger.debug("zero volume throughout; MFI left as NaN (expected for FX)")
        out["mfi"] = np.nan

    bbands = ta.bbands(close, length=p["bb_length"], std=p["bb_std"])
    if bbands is not None and not bbands.empty:
        upper = _column_by_prefix(bbands, "BBU_")
        lower = _column_by_prefix(bbands, "BBL_")
        middle = _column_by_prefix(bbands, "BBM_")
        # Normalized width -- the standard volatility-squeeze measure. Raw (upper-lower) is
        # price-scale dependent and not comparable across symbols.
        out["bb_width"] = (upper - lower) / middle.replace(0, np.nan)
    else:
        out["bb_width"] = np.nan

    return out[["timestamp", *INDICATOR_COLUMNS]]


def _column_by_prefix(frame: pd.DataFrame, prefix: str) -> pd.Series:
    """First column whose name starts with `prefix`, else an all-NaN series.

    pandas-ta encodes parameters into column names, so a hard-coded name breaks the moment a
    lookback changes. Matching by prefix keeps this robust.
    """
    for column in frame.columns:
        if column.startswith(prefix):
            return frame[column]
    logger.warning("no pandas-ta column matching prefix %r; columns=%s", prefix, list(frame.columns))
    return pd.Series(np.nan, index=frame.index)


def _as_series(value: object, length: int) -> pd.Series:
    """Coerce a pandas-ta return value to a Series of the expected length."""
    if value is None:
        return pd.Series(np.nan, index=range(length))
    if isinstance(value, pd.DataFrame):
        return value.iloc[:, 0]
    return value


def compute_universe_indicators(
    bars_by_symbol: dict[str, pd.DataFrame],
    *,
    params: dict | None = None,
) -> pd.DataFrame:
    """Indicators for every symbol, concatenated and keyed on (symbol, timestamp)."""
    frames = []
    for symbol, bars in bars_by_symbol.items():
        if bars.empty:
            logger.warning("no bars for %s; skipping indicators", symbol)
            continue
        indicators = compute_indicators(bars, params=params)
        indicators.insert(1, "symbol", symbol)
        frames.append(indicators)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "symbol", *INDICATOR_COLUMNS])
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
