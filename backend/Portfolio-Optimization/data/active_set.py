"""The active set: symbols someone holds, and their preparation state.

Analysis is expensive (history backfill, keyword search across channels, forecasting,
financial-statement extraction), so it runs only for symbols in a portfolio. The frontend
registers a portfolio's symbols when it is saved; the withdraw/optimize/forecast endpoints
register theirs too. Symbols nobody has touched for IDLE_DAYS drop out.

PREPARATION PIPELINE
Each symbol moves preparing -> ready | partial | failed. Preparation is a list of named
steps; Phase 1 has one (history + liquidity metrics), and later phases append theirs
(keywords, news, forecast, fundamentals) without changing this module's shape. A step's
outcome is recorded per step, so `partial` says exactly what is missing.

Reads on the request path (metrics_for) touch only this SQLite file -- never the network.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from data.instruments import Instrument

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "active_symbols.sqlite"
IDLE_DAYS = 30
# Metrics older than this are not used to override request defaults.
METRICS_MAX_AGE = timedelta(days=7)

PREPARING, READY, PARTIAL, FAILED = "preparing", "ready", "partial", "failed"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS active_symbols (
    symbol        TEXT PRIMARY KEY,
    exchange      TEXT NOT NULL,
    asset_class   TEXT NOT NULL,
    currency      TEXT NOT NULL,
    name          TEXT NOT NULL,
    provider_id   TEXT,
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL,
    status        TEXT NOT NULL,
    prepared_at   TEXT,
    steps         TEXT NOT NULL DEFAULT '{}',
    metrics       TEXT,
    error         TEXT
);
"""

# A step takes the instrument and returns (ok, detail). Detail is merged into `metrics`
# under the step's name when ok, or recorded as the error when not.
Step = Callable[[Instrument], tuple[bool, dict[str, Any] | str]]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(_SCHEMA)
    return connection


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    """Commit and CLOSE. sqlite3's own context manager only commits, and an open handle
    keeps the file locked on Windows."""
    connection = _connect()
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def track(instruments: list[Instrument]) -> list[str]:
    """Register held instruments. Returns the symbols that need (re)preparation."""
    now = _now()
    needs_prep: list[str] = []
    with _db() as connection:
        for instrument in instruments:
            row = connection.execute(
                "SELECT status, prepared_at FROM active_symbols WHERE symbol = ?",
                (instrument.symbol,),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO active_symbols (symbol, exchange, asset_class, currency, name, "
                    "provider_id, first_seen, last_seen, status) VALUES (?,?,?,?,?,?,?,?,?)",
                    (instrument.symbol, instrument.exchange.value, instrument.asset_class.value,
                     instrument.currency, instrument.name, instrument.provider_id, now, now, PREPARING),
                )
                needs_prep.append(instrument.symbol)
                continue

            connection.execute(
                "UPDATE active_symbols SET last_seen = ? WHERE symbol = ?", (now, instrument.symbol)
            )
            if row["status"] in (FAILED, PARTIAL) or _is_stale(row["prepared_at"]):
                needs_prep.append(instrument.symbol)
    return needs_prep


def touch(symbols: list[str]) -> None:
    """Refresh last_seen for symbols already tracked. Request-path safe: never resolves or
    fetches, so an untracked symbol is simply ignored rather than looked up mid-request."""
    if not symbols:
        return
    try:
        with _db() as connection:
            connection.executemany(
                "UPDATE active_symbols SET last_seen = ? WHERE symbol = ?",
                [(_now(), symbol) for symbol in symbols],
            )
    except sqlite3.Error as exc:
        logger.warning("could not touch active set: %s", exc)


def _is_stale(prepared_at: str | None) -> bool:
    if not prepared_at:
        return True
    return datetime.now(timezone.utc) - datetime.fromisoformat(prepared_at) > timedelta(days=1)


def get_instrument(symbol: str) -> Instrument | None:
    with _db() as connection:
        row = connection.execute("SELECT * FROM active_symbols WHERE symbol = ?", (symbol,)).fetchone()
    if row is None:
        return None
    return Instrument(
        symbol=row["symbol"], name=row["name"], exchange=row["exchange"],
        asset_class=row["asset_class"], currency=row["currency"], provider_id=row["provider_id"],
    )


def status(symbols: list[str] | None = None) -> list[dict[str, Any]]:
    """Registry rows (all, or the named ones), with steps/metrics decoded."""
    with _db() as connection:
        if symbols:
            marks = ",".join("?" for _ in symbols)
            rows = connection.execute(
                f"SELECT * FROM active_symbols WHERE symbol IN ({marks}) ORDER BY symbol", symbols
            ).fetchall()
        else:
            rows = connection.execute("SELECT * FROM active_symbols ORDER BY symbol").fetchall()
    return [_decode(row) for row in rows]


def _decode(row: sqlite3.Row) -> dict[str, Any]:
    record = dict(row)
    record["steps"] = json.loads(record.get("steps") or "{}")
    record["metrics"] = json.loads(record["metrics"]) if record.get("metrics") else None
    return record


def metrics_for(symbol: str) -> dict[str, Any] | None:
    """Fresh liquidity metrics for a symbol, or None. Request-path safe: SQLite only."""
    try:
        with _db() as connection:
            row = connection.execute(
                "SELECT metrics, prepared_at FROM active_symbols WHERE symbol = ?", (symbol,)
            ).fetchone()
    except sqlite3.Error as exc:
        logger.warning("active-set registry unreadable: %s", exc)
        return None
    if row is None or not row["metrics"] or not row["prepared_at"]:
        return None
    if datetime.now(timezone.utc) - datetime.fromisoformat(row["prepared_at"]) > METRICS_MAX_AGE:
        return None
    return json.loads(row["metrics"]).get("liquidity")


# --------------------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------------------

def _history_step(instrument: Instrument) -> tuple[bool, dict[str, Any] | str]:
    """Phase 1: backfill/refresh daily bars and compute liquidity metrics."""
    from data.history import get_history
    from data.http import UpstreamError
    from features.liquidity import compute_liquidity_metrics

    try:
        history = get_history(instrument)
    except UpstreamError as exc:
        return False, str(exc)
    metrics = compute_liquidity_metrics(history.bars).to_dict()
    metrics["stale"] = history.stale
    return True, metrics


def _keywords_step(instrument: Instrument) -> tuple[bool, dict[str, Any] | str]:
    """Phase 2: generate (or load) the symbol's keyword profile."""
    from news.pipeline import keywords_step

    return keywords_step(instrument)


def _sentiment_step(instrument: Instrument) -> tuple[bool, dict[str, Any] | str]:
    """Phase 2: per-tier news/social sentiment from every available channel."""
    from news.pipeline import sentiment_step

    return sentiment_step(instrument)


def _technical_step(instrument: Instrument) -> tuple[bool, dict[str, Any] | str]:
    """Phase 2c: trend / momentum / volume / range / relative-strength snapshot. Reads the
    bars the liquidity step just stored; only the benchmark may touch the network."""
    from data.history import benchmark_close, load_stored
    from features.technical import technical_snapshot

    bars = load_stored(instrument)
    if bars.empty:
        return False, "no stored history (liquidity step failed)"
    return True, technical_snapshot(bars, benchmark_close(instrument))


def _fundamentals_step(instrument: Instrument) -> tuple[bool, dict[str, Any] | str]:
    """Phase 2b: statements, ratios and red flags (tokenomics for crypto)."""
    from data.http import UpstreamError
    from fundamentals.pipeline import fundamentals_step

    try:
        return fundamentals_step(instrument)
    except UpstreamError as exc:
        return False, str(exc)


def _forecast_step(instrument: Instrument) -> tuple[bool, dict[str, Any] | str]:
    """Phase 3: hybrid (TimesFM + LSTM head) forecast bands, or a labelled empirical band
    when no head/base can run here."""
    from forecasting.serving import forecast_step

    return forecast_step(instrument)


def _laya_analysis_step(instrument: Instrument, metrics: dict[str, Any]) -> tuple[bool, dict[str, Any] | str]:
    """Phase 4b: Laya Stage 1 (technical / fundamental / news questions) for this holding."""
    from decision.engine import analysis_step

    return analysis_step(instrument, metrics)


_laya_analysis_step.needs_metrics = True  # type: ignore[attr-defined]


# Ordered. Later phases append (name, step) pairs here. Fundamentals run after liquidity,
# whose last close they use for valuation multiples.
PIPELINE: list[tuple[str, Step]] = [
    ("liquidity", _history_step),
    ("keywords", _keywords_step),
    ("technical", _technical_step),
    ("sentiment", _sentiment_step),
    ("fundamentals", _fundamentals_step),
    # Its features include the sentiment and fundamentals the steps above just stored.
    ("forecast", _forecast_step),
    # Phase 4b: Laya Stage 1 reads every block above.
    ("laya_analysis", _laya_analysis_step),
]


def prepare(symbol: str) -> dict[str, Any]:
    """Run every pipeline step for one tracked symbol and record the outcome."""
    instrument = get_instrument(symbol)
    if instrument is None:
        raise KeyError(f"{symbol} is not in the active set; track it first")

    with _db() as connection:
        row = connection.execute("SELECT metrics FROM active_symbols WHERE symbol = ?", (symbol,)).fetchone()
        connection.execute("UPDATE active_symbols SET status = ? WHERE symbol = ?", (PREPARING, symbol))
    metrics: dict[str, Any] = json.loads(row["metrics"]) if row and row["metrics"] else {}

    steps: dict[str, str] = {}
    errors: list[str] = []
    for name, step in PIPELINE:
        try:
            # A step that analyses earlier steps' output (Laya Stage 1) declares
            # `needs_metrics` and receives this run's results so far.
            ok, detail = step(instrument, metrics) if getattr(step, "needs_metrics", False) else step(instrument)
        except Exception as exc:  # noqa: BLE001 - one failing step must not abort the rest
            logger.exception("step %s failed for %s", name, symbol)
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        if ok:
            steps[name] = "ok"
            metrics[name] = detail
        else:
            steps[name] = "failed"
            errors.append(f"{name}: {detail}")

    succeeded = sum(1 for outcome in steps.values() if outcome == "ok")
    final = READY if succeeded == len(steps) else (PARTIAL if succeeded else FAILED)

    with _db() as connection:
        connection.execute(
            "UPDATE active_symbols SET status = ?, prepared_at = ?, steps = ?, metrics = ?, error = ? "
            "WHERE symbol = ?",
            (final, _now(), json.dumps(steps), json.dumps(metrics, default=str),
             "; ".join(errors) or None, symbol),
        )
    logger.info("prepared %s: %s %s", symbol, final, steps)
    return status([symbol])[0]


def prepare_many(symbols: list[str]) -> None:
    """Background entry point. Never raises: it runs after the HTTP response is sent."""
    for symbol in symbols:
        try:
            prepare(symbol)
        except Exception:  # noqa: BLE001
            logger.exception("preparation of %s failed", symbol)


def prune(idle_days: int = IDLE_DAYS) -> int:
    """Drop symbols nobody has touched for `idle_days`. Returns how many were removed."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=idle_days)).isoformat()
    with _db() as connection:
        cursor = connection.execute("DELETE FROM active_symbols WHERE last_seen < ?", (cutoff,))
        return cursor.rowcount


def snapshot_for(symbol: str) -> dict[str, Any] | None:
    """Everything prepared for a symbol (all step blocks) plus status and age, or None.
    Request-path safe: SQLite only. Unlike metrics_for it does not drop stale data -- the
    caller shows the age instead, since a two-day-old forecast still beats none."""
    try:
        rows = status([symbol])
    except sqlite3.Error as exc:
        logger.warning("active-set registry unreadable: %s", exc)
        return None
    if not rows:
        return None
    row = rows[0]
    age_hours = None
    if row.get("prepared_at"):
        age_hours = (datetime.now(timezone.utc) - datetime.fromisoformat(row["prepared_at"])).total_seconds() / 3600
    return {"status": row["status"], "prepared_at": row.get("prepared_at"), "age_hours": age_hours,
            "steps": row.get("steps") or {}, "metrics": row.get("metrics") or {}}
