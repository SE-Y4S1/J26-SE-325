"""Per-instrument liquidity and risk metrics from daily bars.

These replace the constants the service used to stamp on every holding
(`daily_volatility=0.0126`, `volatility_pct=0.5`), which made every asset identical to the
GA's timing-risk term and pinned the fuzzy layer's `market_volatility` input at "medium".

  adv_value          mean of close x volume over the last ADV_WINDOW sessions, in the
                     instrument's native currency. The GA's participation cap and square-
                     root impact model divide position size by this.
  daily_volatility   sample stdev of daily log returns over VOL_WINDOW sessions.
  volatility_pct     where today's ATR% sits within its own trailing year, in [0, 1] -- the
                     "ATR percentile" the fuzzy layer's market_volatility variable expects.
  zero_volume_share  share of recent sessions with no trades.

CSE CAVEAT: the CSE chart endpoint lists only sessions with trades, so a thinly traded
name's silent days are missing rather than zero. adv_value is therefore an average over
TRADED sessions and overstates true daily liquidity; `sessions_per_week` exposes the gap so
the optimizer and the UI can see how often the name actually trades.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

ADV_WINDOW = 20
VOL_WINDOW = 60
PERCENTILE_WINDOW = 252
MIN_RETURNS_FOR_VOL = 10

# Used only when history is too short to estimate anything. Matches the GA's default.
FALLBACK_DAILY_VOLATILITY = 0.0126
FALLBACK_VOLATILITY_PCT = 0.5


@dataclass(frozen=True)
class LiquidityMetrics:
    last_close: float
    as_of: str
    adv_value: float
    adv_units: float
    daily_volatility: float
    volatility_pct: float
    atr_pct: float | None
    zero_volume_share: float
    sessions_per_week: float | None
    history_days: int
    # True when daily_volatility / volatility_pct are fallbacks, not estimates.
    estimated: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) or math.isinf(value) else value


def compute_liquidity_metrics(bars: pd.DataFrame) -> LiquidityMetrics:
    """Metrics from one instrument's daily bars (OHLCV_COLUMNS, any order)."""
    if bars.empty:
        raise ValueError("no bars to compute liquidity metrics from")

    frame = bars.sort_values("timestamp").reset_index(drop=True)
    close = pd.to_numeric(frame["close"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)

    recent = frame.tail(ADV_WINDOW)
    recent_close = close.tail(ADV_WINDOW)
    recent_volume = volume.tail(ADV_WINDOW)
    adv_value = float((recent_close * recent_volume).mean())
    adv_units = float(recent_volume.mean())
    zero_volume_share = float((recent_volume <= 0).mean())

    stamps = pd.to_datetime(recent["timestamp"])
    span_days = (stamps.max() - stamps.min()).days if len(stamps) > 1 else 0
    sessions_per_week = (len(stamps) - 1) / span_days * 7 if span_days > 0 else None

    log_returns = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    window = log_returns.tail(VOL_WINDOW)
    estimated = len(window) >= MIN_RETURNS_FOR_VOL
    daily_volatility = float(window.std(ddof=1)) if estimated else FALLBACK_DAILY_VOLATILITY

    atr_pct, volatility_pct = _atr_percentile(frame)
    if volatility_pct is None:
        volatility_pct = FALLBACK_VOLATILITY_PCT
        estimated = False

    return LiquidityMetrics(
        last_close=float(close.iloc[-1]),
        as_of=pd.Timestamp(frame["timestamp"].iloc[-1]).date().isoformat(),
        adv_value=adv_value,
        adv_units=adv_units,
        daily_volatility=daily_volatility,
        volatility_pct=volatility_pct,
        atr_pct=_clean(atr_pct),
        zero_volume_share=zero_volume_share,
        sessions_per_week=_clean(sessions_per_week),
        history_days=len(frame),
        estimated=estimated,
    )


def _atr_percentile(frame: pd.DataFrame) -> tuple[float | None, float | None]:
    """Today's ATR% and its percentile rank within the trailing year."""
    from features.technical import compute_indicators

    try:
        indicators = compute_indicators(frame)
    except Exception:  # noqa: BLE001 - too short for pandas-ta is not an error here
        return None, None

    atr_pct = indicators["atr_pct"].dropna().tail(PERCENTILE_WINDOW)
    if len(atr_pct) < MIN_RETURNS_FOR_VOL:
        return None, None
    current = float(atr_pct.iloc[-1])
    # Fraction of the trailing window at or below today's reading: 1.0 = most volatile day
    # of the year, 0.0 = calmest.
    return current, float((atr_pct <= current).mean())
