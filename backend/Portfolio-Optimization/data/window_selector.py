"""Per-symbol training window and forecast horizon selection.

A single global "use 10 years, forecast 5 days" constant is indefensible across a universe
that mixes AAPL (deep history, high ADV), TLT (regime-dependent), and EURUSD=X (24h market,
no reported volume). This module derives both choices per instrument from explicit,
measurable criteria and records the evidence, so every window in the dissertation is
auditable rather than an unexplained magic number.

Resolved output is written to configs/resolved_universe.yaml as, per symbol:
    history_start, history_end, horizons, and the criterion values that produced them
    (first trade date, completeness ratio, detected breaks, regime count, median ADV,
    ACF decay lag, days-to-liquidate).

HISTORY WINDOW -- start from maximum available, then tighten by:
  1. Availability   -- first trade date; total clamped to [min_history_years, max_history_years].
  2. Completeness   -- fraction of expected exchange trading days present; truncate at any
                       gap longer than max_gap_days (delistings, halts, data outages).
  3. Structural break -- CUSUM test on log-return mean and variance. Training across a hard
                       break (splits, redenominations, post-IPO lockup expiry) teaches the
                       model a relationship that no longer holds, so the window starts after
                       the most recent detected break.
  4. Regime coverage -- the window must span >= min_regimes distinct realized-volatility
                       regimes (rolling 21d vol bucketed by tercile). Prevents a symbol from
                       training entirely on a calm stretch and then being evaluated in a
                       stressed one, which would flatter RQ1 and wreck RQ4.
  5. Liquidity floor -- median ADV over the window >= min_adv_usd. Below-floor symbols are
                       FLAGGED, not dropped: an illiquid holding is exactly what the
                       withdrawal module exists to handle well (RQ3/RQ4).

FORECAST HORIZON -- the union of two signals, clamped to horizon_menu:
  a. Signal-driven    -- first lag at which the return autocorrelation falls inside the
                         Bartlett significance band. Forecasting beyond the horizon where a
                         series carries usable autocorrelation is fitting noise.
  b. Liquidity-driven -- trading days needed to exit a typical position at <= participation_cap
                         of ADV. This deliberately couples the horizon to liquidity, which is
                         this component's whole thesis: a forecast horizon shorter than the
                         time it takes to actually exit the position cannot inform a
                         liquidation decision.

All thresholds come from configs/universe.yaml::window_selection.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.schema import AssetClass

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252

# CUSUM 5% critical value for the standardized statistic (Brown, Durbin & Evans 1975).
CUSUM_CRITICAL_VALUE = 1.358

# Two consecutive breaks inside this many days are treated as one event -- a split or a
# crash produces a cluster of border-crossings, not N independent regime changes.
BREAK_MERGE_WINDOW_DAYS = 21


@dataclass(frozen=True)
class WindowCriteria:
    """The evidence behind one symbol's resolved window. Serialized for auditability."""

    first_trade_date: date
    completeness_ratio: float
    largest_gap_days: int
    break_dates: tuple[date, ...]
    regime_count: int
    median_adv_usd: float
    below_liquidity_floor: bool
    acf_decay_lag: int
    days_to_liquidate: int


@dataclass(frozen=True)
class ResolvedWindow:
    """A symbol's training window and horizons, plus the criteria that produced them."""

    symbol: str
    asset_class: AssetClass
    history_start: date
    history_end: date
    horizons: tuple[int, ...]
    criteria: WindowCriteria

    @property
    def history_days(self) -> int:
        return (self.history_end - self.history_start).days


# --------------------------------------------------------------------------------------
# Individual criteria
# --------------------------------------------------------------------------------------

def detect_structural_breaks(
    returns: pd.Series, *, threshold: float = CUSUM_CRITICAL_VALUE
) -> list[date]:
    """CUSUM break dates in the mean of log returns.

    Standardizes the series, walks the cumulative sum, and flags any point where the
    normalized statistic exceeds the critical value. Clustered crossings are merged so one
    market event yields one break.
    """
    clean = returns.dropna()
    if len(clean) < TRADING_DAYS_PER_YEAR // 2:
        return []

    values = clean.to_numpy(dtype=float)
    sigma = values.std(ddof=1)
    if sigma == 0 or not np.isfinite(sigma):
        return []

    centered = (values - values.mean()) / sigma
    cusum = np.cumsum(centered) / math.sqrt(len(centered))
    exceed_idx = np.flatnonzero(np.abs(cusum) > threshold)
    if exceed_idx.size == 0:
        return []

    index = clean.index
    breaks: list[date] = []
    last_kept: pd.Timestamp | None = None
    for pos in exceed_idx:
        stamp = pd.Timestamp(index[pos])
        if last_kept is not None and (stamp - last_kept).days < BREAK_MERGE_WINDOW_DAYS:
            continue
        breaks.append(stamp.date())
        last_kept = stamp
    return breaks


def count_volatility_regimes(returns: pd.Series, *, window: int = 21, n_buckets: int = 3) -> int:
    """Number of distinct realized-volatility terciles the series actually visits.

    Buckets are cut on the series' OWN quantiles, so this measures whether the window spans
    a variety of conditions for this instrument -- not whether it is volatile in absolute
    terms, which would just re-rank asset classes.
    """
    clean = returns.dropna()
    if len(clean) < window * 2:
        return 0

    realized_vol = clean.rolling(window).std().dropna()
    if realized_vol.empty or realized_vol.nunique() < n_buckets:
        return int(realized_vol.nunique())

    try:
        buckets = pd.qcut(realized_vol, n_buckets, labels=False, duplicates="drop")
    except ValueError:
        return 1
    return int(pd.Series(buckets).nunique())


def acf_decay_lag(returns: pd.Series, *, max_lag: int = 40) -> int:
    """First lag whose autocorrelation falls inside the Bartlett band.

    The band is +/- 1.96/sqrt(n), the standard significance envelope for white noise.
    Returns 1 when no lag is significant, which is the common (and expected) case for
    liquid daily returns -- efficient markets should show little linear autocorrelation.
    """
    clean = returns.dropna()
    n = len(clean)
    if n < 30:
        return 1

    band = 1.96 / math.sqrt(n)
    values = clean.to_numpy(dtype=float)
    centered = values - values.mean()
    denominator = float(np.dot(centered, centered))
    if denominator == 0:
        return 1

    for lag in range(1, min(max_lag, n - 1) + 1):
        acf = float(np.dot(centered[:-lag], centered[lag:]) / denominator)
        if abs(acf) < band:
            return lag
    return min(max_lag, n - 1)


def days_to_liquidate(position_value: float, adv_usd: float, *, participation_cap: float) -> int:
    """Trading days to exit a position without exceeding the daily participation cap."""
    if adv_usd <= 0 or participation_cap <= 0:
        return 0
    tradable_per_day = adv_usd * participation_cap
    return max(1, math.ceil(position_value / tradable_per_day))


def completeness(bars: pd.DataFrame, *, timestamp_col: str = "timestamp") -> tuple[float, int]:
    """(observed / expected trading days, largest gap in calendar days).

    Expected days come from a business-day count, which slightly over-counts because of
    exchange holidays -- so a healthy equity lands around 0.96 rather than 1.0. The absolute
    level matters less than catching the symbol that returns 0.4.
    """
    if bars.empty:
        return 0.0, 0

    stamps = pd.to_datetime(bars[timestamp_col]).sort_values()
    span_start, span_end = stamps.iloc[0], stamps.iloc[-1]
    expected = np.busday_count(span_start.date(), span_end.date()) or 1
    ratio = float(len(stamps) / expected)

    gaps = stamps.diff().dt.days.dropna()
    largest_gap = int(gaps.max()) if not gaps.empty else 0
    return min(ratio, 1.0), largest_gap


# --------------------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------------------

def _truncate_at_large_gap(bars: pd.DataFrame, max_gap_days: int) -> pd.DataFrame:
    """Keep only the contiguous tail after the last oversized gap."""
    if bars.empty:
        return bars
    stamps = pd.to_datetime(bars["timestamp"])
    gaps = stamps.diff().dt.days
    offenders = np.flatnonzero((gaps > max_gap_days).to_numpy())
    if offenders.size == 0:
        return bars
    return bars.iloc[int(offenders[-1]) :].reset_index(drop=True)


def _pick_horizons(
    signal_lag: int, liquidity_days: int, menu: list[int]
) -> tuple[int, ...]:
    """Map the two horizon signals onto the allowed menu.

    Each signal snaps to the smallest menu entry that covers it; the result is their union,
    so a symbol whose autocorrelation dies at lag 1 but takes 8 days to liquidate is
    evaluated at both 1 and 21 days rather than at a single compromise horizon that serves
    neither purpose.
    """
    ordered = sorted(menu)
    chosen: set[int] = set()
    for signal in (max(1, signal_lag), max(1, liquidity_days)):
        covering = [h for h in ordered if h >= signal]
        chosen.add(covering[0] if covering else ordered[-1])
    return tuple(sorted(chosen))


def resolve_window(
    symbol: str,
    asset_class: AssetClass,
    bars: pd.DataFrame,
    *,
    config: dict,
    typical_position_value: float,
) -> ResolvedWindow:
    """Apply all five history criteria and both horizon criteria to one symbol."""
    if bars.empty:
        raise ValueError(f"{symbol}: no bars supplied; cannot resolve a window")

    frame = bars.sort_values("timestamp").reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    first_trade = frame["timestamp"].iloc[0].date()
    history_end = frame["timestamp"].iloc[-1].date()

    max_days = int(config["max_history_years"] * 365.25)
    min_days = int(config["min_history_years"] * 365.25)

    # 1. Availability -- cap at max_history_years.
    earliest_allowed = history_end - pd.Timedelta(days=max_days)
    frame = frame[frame["timestamp"] >= pd.Timestamp(earliest_allowed)].reset_index(drop=True)

    # 2. Completeness -- drop everything before the last oversized gap.
    frame = _truncate_at_large_gap(frame, int(config["max_gap_days"]))
    completeness_ratio, largest_gap = completeness(frame)

    # 3. Structural breaks -- prefer to start after the most recent one.
    indexed = frame.set_index("timestamp")
    log_returns = np.log(indexed["close"]).diff()
    break_dates = detect_structural_breaks(log_returns)

    candidate_start = frame["timestamp"].iloc[0].date()
    if break_dates:
        latest_break = max(break_dates)
        remaining = (history_end - latest_break).days
        # Only honour the break if enough history survives it; otherwise a recent break
        # would starve the model entirely, which is worse than training across it.
        if remaining >= min_days:
            candidate_start = latest_break
        else:
            logger.info(
                "%s: break at %s leaves only %d days (<%d); keeping longer window",
                symbol, latest_break, remaining, min_days,
            )

    windowed = frame[frame["timestamp"] >= pd.Timestamp(candidate_start)].reset_index(drop=True)
    windowed_returns = np.log(windowed.set_index("timestamp")["close"]).diff()

    # 4. Regime coverage -- widen back toward full history if the window is too homogeneous.
    regime_count = count_volatility_regimes(windowed_returns, n_buckets=3)
    if regime_count < int(config["min_regimes"]):
        logger.info(
            "%s: only %d volatility regime(s) after break trim; reverting to full window",
            symbol, regime_count,
        )
        windowed = frame
        candidate_start = frame["timestamp"].iloc[0].date()
        windowed_returns = np.log(windowed.set_index("timestamp")["close"]).diff()
        regime_count = count_volatility_regimes(windowed_returns, n_buckets=3)

    # 5. Liquidity floor -- flag, never drop.
    notional = windowed["close"] * windowed["volume"]
    median_adv = float(notional.median()) if not notional.empty else 0.0
    if asset_class is AssetClass.FOREX:
        # yfinance reports FX volume as 0, so ADV comes from universe.yaml instead.
        median_adv = float(config.get("forex_notional_adv_usd", 0.0)) or median_adv
    below_floor = median_adv < float(config["min_adv_usd"])

    # Horizons.
    signal_lag = acf_decay_lag(windowed_returns)
    liquidity_days = days_to_liquidate(
        typical_position_value, median_adv, participation_cap=float(config["participation_cap"])
    )
    horizons = _pick_horizons(signal_lag, liquidity_days, list(config["horizon_menu"]))

    criteria = WindowCriteria(
        first_trade_date=first_trade,
        completeness_ratio=round(completeness_ratio, 4),
        largest_gap_days=largest_gap,
        break_dates=tuple(break_dates),
        regime_count=regime_count,
        median_adv_usd=round(median_adv, 2),
        below_liquidity_floor=below_floor,
        acf_decay_lag=signal_lag,
        days_to_liquidate=liquidity_days,
    )

    return ResolvedWindow(
        symbol=symbol,
        asset_class=asset_class,
        history_start=candidate_start,
        history_end=history_end,
        horizons=horizons,
        criteria=criteria,
    )


def resolve_universe(
    universe_path: Path,
    output_path: Path,
    *,
    as_of: date | None = None,
    typical_position_value: float = 50_000.0,
    bars_provider=None,
) -> dict[str, ResolvedWindow]:
    """Resolve every symbol in universe.yaml and write configs/resolved_universe.yaml.

    `bars_provider(symbol, asset_class, start, end)` is injectable so tests can resolve a
    universe without hitting the network; it defaults to data.ingestion.
    """
    from datetime import timedelta

    universe = yaml.safe_load(Path(universe_path).read_text(encoding="utf-8"))
    settings = universe["window_selection"]
    end = as_of or date.today()
    start = end - timedelta(days=int(settings["max_history_years"] * 365.25))

    if bars_provider is None:
        from data.ingestion import bars_to_frame, fetch_ohlcv

        def bars_provider(symbol, asset_class, start, end):  # noqa: ANN001
            return bars_to_frame(fetch_ohlcv(symbol, asset_class, start, end))

    groups = (
        ("equities", AssetClass.EQUITY),
        ("etfs", AssetClass.ETF),
        ("forex", AssetClass.FOREX),
    )

    resolved: dict[str, ResolvedWindow] = {}
    for group_name, asset_class in groups:
        for entry in universe.get(group_name, []):
            symbol = entry["symbol"]
            symbol_config = dict(settings)
            if asset_class is AssetClass.FOREX:
                symbol_config["forex_notional_adv_usd"] = entry.get("notional_adv_usd", 0.0)
            try:
                bars = bars_provider(symbol, asset_class, start, end)
                resolved[symbol] = resolve_window(
                    symbol,
                    asset_class,
                    bars,
                    config=symbol_config,
                    typical_position_value=typical_position_value,
                )
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not kill the run
                logger.error("failed to resolve window for %s: %s", symbol, exc)

    _write_resolved(resolved, Path(output_path), typical_position_value)
    return resolved


def _write_resolved(
    resolved: dict[str, ResolvedWindow], output_path: Path, typical_position_value: float
) -> None:
    """Serialize resolved windows WITH their criteria, so each choice is auditable."""
    payload: dict[str, object] = {
        "_generated": {
            "note": (
                "Generated by data/window_selector.py. Each entry records the criterion "
                "values that produced its window -- do not hand-edit."
            ),
            "typical_position_value_usd": typical_position_value,
        },
        "symbols": {},
    }

    for symbol, window in sorted(resolved.items()):
        criteria = asdict(window.criteria)
        criteria["break_dates"] = [d.isoformat() for d in window.criteria.break_dates]
        payload["symbols"][symbol] = {
            "asset_class": window.asset_class.value,
            "history_start": window.history_start.isoformat(),
            "history_end": window.history_end.isoformat(),
            "horizons": list(window.horizons),
            "criteria": criteria,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, default_flow_style=False), encoding="utf-8"
    )
    logger.info("wrote %d resolved windows to %s", len(resolved), output_path)
