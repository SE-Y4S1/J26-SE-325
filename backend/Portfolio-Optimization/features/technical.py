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


# --------------------------------------------------------------------------------------
# Technical analysis snapshot (Phase 2c)
# --------------------------------------------------------------------------------------
# The indicator table above feeds the forecaster row by row. The snapshot below is the
# analyst's view of ONE symbol today -- trend, momentum, volume confirmation, where price sits
# against its range and against its market -- and is what the liquidation analysis (and the
# Laya technical questions) read. It deliberately needs no opening price: CSE publishes none,
# so nothing here may depend on one rather than fabricate it.

SNAPSHOT_PARAMS = {"sma_short": 20, "sma_mid": 50, "sma_long": 200, "adx_length": 14,
                   "range_sessions": 60, "high_52w_sessions": 252, "rs_sessions": 63,
                   "obv_sessions": 20, "cross_lookback": 20}


def _last(series: pd.Series | None) -> float | None:
    if series is None or series.empty:
        return None
    value = series.iloc[-1]
    return float(value) if pd.notna(value) else None


def _ret(close: pd.Series, sessions: int) -> float | None:
    if len(close) <= sessions:
        return None
    return float(close.iloc[-1] / close.iloc[-1 - sessions] - 1)


def technical_snapshot(bars: pd.DataFrame, benchmark: pd.Series | None = None) -> dict:
    """Trend, momentum, volume and range read-out for one symbol's daily bars.

    `benchmark` is the market's daily close (ASPI for CSE, SPY for US, BTC for crypto),
    indexed by date. Every field is None when there is not enough history for it.
    """
    import importlib.metadata  # noqa: F401 - pandas-ta-openbb reads it without importing it

    import pandas_ta as ta

    p = SNAPSHOT_PARAMS
    if bars.empty:
        return {"sessions": 0}
    frame = bars.sort_values("timestamp").reset_index(drop=True)
    close, high, low, volume = frame["close"], frame["high"], frame["low"], frame["volume"]
    last = float(close.iloc[-1])

    sma = {n: close.rolling(n).mean() for n in (p["sma_short"], p["sma_mid"], p["sma_long"])}
    s20, s50, s200 = (_last(sma[p["sma_short"]]), _last(sma[p["sma_mid"]]), _last(sma[p["sma_long"]]))

    if s50 is not None and s200 is not None:
        trend = "bullish" if last > s50 > s200 else "bearish" if last < s50 < s200 else "neutral"
    elif s20 is not None and s50 is not None:
        trend = "bullish" if last > s20 > s50 else "bearish" if last < s20 < s50 else "neutral"
    else:
        trend = None

    cross = None
    if s200 is not None:
        diff = (sma[p["sma_mid"]] - sma[p["sma_long"]]).dropna().tail(p["cross_lookback"] + 1)
        signs = np.sign(diff.to_numpy())
        if len(signs) > 1 and signs[0] != signs[-1] and signs[-1] != 0:
            cross = "golden" if signs[-1] > 0 else "death"

    indicators = compute_indicators(frame)

    pct_b = None
    if len(close) >= 20:
        mid = close.rolling(20).mean().iloc[-1]
        std = close.rolling(20).std(ddof=0).iloc[-1]
        if std and np.isfinite(std):
            pct_b = float((last - (mid - 2 * std)) / (4 * std))

    adx = None
    if len(close) > 2 * p["adx_length"]:
        adx_frame = ta.adx(high, low, close, length=p["adx_length"])
        if adx_frame is not None and not adx_frame.empty:
            adx = _last(_column_by_prefix(adx_frame, "ADX_"))

    obv_trend = None
    if (volume > 0).any() and len(close) > p["obv_sessions"]:
        obv = ta.obv(close, volume)
        avg_volume = volume.tail(p["obv_sessions"]).mean()
        if obv is not None and avg_volume > 0:
            # OBV change over the window in units of average daily volume: +N means N days'
            # worth of volume net-accumulated, so it is comparable across symbols.
            obv_trend = float((obv.iloc[-1] - obv.iloc[-1 - p["obv_sessions"]]) / (avg_volume * p["obv_sessions"]))

    window_52w = close.tail(p["high_52w_sessions"])
    range_window = frame.tail(p["range_sessions"])
    support, resistance = float(range_window["low"].min()), float(range_window["high"].max())

    relative_strength = None
    benchmark_return = None
    own_return = _ret(close, p["rs_sessions"])
    if benchmark is not None and not benchmark.empty and own_return is not None:
        start_date = frame["timestamp"].iloc[-1 - p["rs_sessions"]]
        end_date = frame["timestamp"].iloc[-1]
        bench = benchmark.sort_index()
        start_level = bench[bench.index <= start_date]
        end_level = bench[bench.index <= end_date]
        if not start_level.empty and not end_level.empty and start_level.index[-1] != end_level.index[-1]:
            benchmark_return = float(end_level.iloc[-1] / start_level.iloc[-1] - 1)
            relative_strength = own_return - benchmark_return

    def r(value: float | None, digits: int = 4) -> float | None:
        return round(value, digits) if value is not None and np.isfinite(value) else None

    return {
        "sessions": int(len(frame)),
        "as_of": str(pd.Timestamp(frame["timestamp"].iloc[-1]).date()),
        "close": r(last, 6),
        "trend": trend,
        "sma_20": r(s20, 6), "sma_50": r(s50, 6), "sma_200": r(s200, 6),
        "price_vs_sma_50": r(last / s50 - 1) if s50 else None,
        "price_vs_sma_200": r(last / s200 - 1) if s200 else None,
        "ma_cross_recent": cross,
        "rsi": r(_last(indicators["rsi"]), 2),
        "macd_hist": r(_last(indicators["macd_hist"]), 6),
        "mfi": r(_last(indicators["mfi"]), 2),
        "atr_pct": r(_last(indicators["atr_pct"])),
        "bollinger_pct_b": r(pct_b),
        "adx": r(adx, 2),
        "obv_trend": r(obv_trend),
        "return_20d": r(_ret(close, 20)),
        "return_63d": r(own_return),
        "drawdown_52w": r(last / float(window_52w.max()) - 1),
        "support": r(support, 6), "resistance": r(resistance, 6),
        "dist_to_support": r(last / support - 1) if support else None,
        "dist_to_resistance": r(resistance / last - 1) if last else None,
        "benchmark_return_63d": r(benchmark_return),
        "relative_strength_63d": r(relative_strength),
    }
