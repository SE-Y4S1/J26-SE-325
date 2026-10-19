"""Serving the hybrid forecaster: install, register, load, and forecast a held symbol.

LIFECYCLE
  1. Colab Part A trains one residual head per horizon over the TimesFM base and exports
     `hybrid_heads.zip` (head.pt + meta.json per horizon; a few hundred KB).
  2. `python -m forecasting.serving --install hybrid_heads.zip` copies each head to
     artifacts/checkpoints/hybrid/h<H>-<hash>/, registers it in the model registry as
     `hybrid_h<H>` and activates it. From then on responses carry a real `model_version`.
  3. The service loads the active head lazily, together with its base model.

WHAT IF THE BASE CANNOT RUN HERE
The head corrects TimesFM's forecast; it is meaningless without it. TimesFM's weights are
~821 MB and are not fetched implicitly. When they are absent, `forecast()` answers with an
EMPIRICAL band instead -- quantiles of the symbol's own overlapping h-day returns over the
last two years -- and says so (`method: "empirical"`, plus the reason). The liquidation
planner still gets per-asset drift and downside, and nobody mistakes it for the model.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
import threading
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecasting.base import DEFAULT_QUANTILES

logger = logging.getLogger(__name__)

CHECKPOINT_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "checkpoints" / "hybrid"
# Trading-day horizons the analysis pipeline forecasts for every held symbol: about a week
# (urgent withdrawals) and about a month (deadline-spread plans).
FORECAST_HORIZONS = (5, 21)
EMPIRICAL_LOOKBACK = 500
MIN_EMPIRICAL_SAMPLES = 40


def model_name(horizon: int) -> str:
    return f"hybrid_h{horizon}"


# --------------------------------------------------------------------------------------
# Install / locate
# --------------------------------------------------------------------------------------

def _head_dirs(root: Path) -> list[Path]:
    return sorted({p.parent for p in root.rglob("meta.json") if (p.parent / "head.pt").exists()})


def install(source: Path, *, activate: bool = True) -> list[str]:
    """Install every head found in a zip or directory. Returns registered versions."""
    from forecasting.model_registry import compute_content_hash, register

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(source)
        if root.is_file():
            with zipfile.ZipFile(root) as archive:
                archive.extractall(tmp)
            root = Path(tmp)
        heads = _head_dirs(root)
        if not heads:
            raise FileNotFoundError(f"no head.pt + meta.json pairs under {source}")
        versions = []
        for head_dir in heads:
            meta = json.loads((head_dir / "meta.json").read_text(encoding="utf-8"))
            horizon = int(meta["horizon"])
            digest = compute_content_hash(head_dir)
            dest = CHECKPOINT_ROOT / f"h{horizon}-{digest[:12]}"
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(head_dir, dest)
            train = meta.get("train_window") or {}
            record = register(
                model_name(horizon), dest,
                train_start=date.fromisoformat(train.get("start", "1970-01-01")),
                train_end=date.fromisoformat(train.get("end", "1970-01-01")),
                metrics={k: float(v) for k, v in (meta.get("metrics") or {}).items()},
                activate=activate, feature_columns=meta["feature_cols"],
            )
            versions.append(record.model_version)
            logger.info("installed %s -> %s", head_dir.name, record.model_version)
    reset()
    return versions


def active_head_dir(horizon: int) -> tuple[Path, str] | None:
    """(checkpoint dir, model_version) of the active head for a horizon, or None."""
    from forecasting.model_registry import get_active_version, get_record

    version = get_active_version(model_name(horizon))
    if version == "unregistered":
        return None
    record = get_record(version)
    directory = CHECKPOINT_ROOT / f"h{horizon}-{record.content_hash[:12]}" if record else None
    if directory is None or not (directory / "head.pt").exists():
        logger.warning("active %s has no checkpoint on disk at %s", version, directory)
        return None
    return directory, version


# --------------------------------------------------------------------------------------
# Features for one held symbol
# --------------------------------------------------------------------------------------

def symbol_features(bars: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """The same feature table training builds (indicators + enrichment), for one symbol."""
    from features.enrichment import add_enrichment
    from features.feature_store import build_feature_table
    from features.technical import compute_universe_indicators

    frame = bars.copy()
    frame["symbol"] = symbol
    by_symbol = {symbol: frame}
    table = build_feature_table(by_symbol, compute_universe_indicators(by_symbol))
    return add_enrichment(table)


# --------------------------------------------------------------------------------------
# Forecasting
# --------------------------------------------------------------------------------------

def empirical_quantiles(close: pd.Series, horizon: int,
                        quantiles: tuple[float, ...] = DEFAULT_QUANTILES) -> list[float] | None:
    """Quantiles of the symbol's own overlapping `horizon`-session simple returns."""
    series = close.dropna().astype(float).tail(EMPIRICAL_LOOKBACK + horizon)
    if len(series) < horizon + MIN_EMPIRICAL_SAMPLES:
        return None
    returns = (series.shift(-horizon) / series - 1.0).dropna()
    return [float(v) for v in np.quantile(returns.to_numpy(), quantiles)]


class ForecastService:
    """Loads active heads lazily (one per horizon) and serves forecasts for held symbols."""

    def __init__(self) -> None:
        self._heads: dict[int, tuple[Any, str] | None] = {}
        self._reasons: dict[int, str] = {}
        self._lock = threading.Lock()

    def head(self, horizon: int):  # noqa: ANN201 - (HybridForecaster, version) | None
        with self._lock:
            if horizon not in self._heads:
                self._heads[horizon] = self._load(horizon)
            return self._heads[horizon]

    def reason(self, horizon: int) -> str | None:
        return self._reasons.get(horizon)

    def _load(self, horizon: int):  # noqa: ANN202
        from forecasting.base import get_forecaster, usable_foundation_models
        from forecasting.hybrid_model import HybridForecaster

        located = active_head_dir(horizon)
        if located is None:
            self._reasons[horizon] = f"no trained head registered for h={horizon} (Colab Part A, then --install)"
            return None
        directory, version = located
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        base_name = meta["base_model"]
        family = {"timesfm": "timesfm", "chronos_bolt": "chronos"}.get(base_name)
        if family and not usable_foundation_models().get(family):
            self._reasons[horizon] = (f"{version} needs {base_name} weights, which are not cached locally "
                                      "(huggingface-cli download google/timesfm-2.5-200m-pytorch)")
            return None
        try:
            model = HybridForecaster.load(directory, get_forecaster(base_name))
        except Exception as exc:  # noqa: BLE001
            self._reasons[horizon] = f"{version} failed to load: {type(exc).__name__}: {exc}"
            logger.warning(self._reasons[horizon])
            return None
        self._reasons.pop(horizon, None)
        return model, version

    def available(self) -> bool:
        return any(self.head(h) is not None for h in FORECAST_HORIZONS)

    def forecast(self, symbol: str, bars: pd.DataFrame, horizon: int) -> dict[str, Any]:
        """Hybrid forecast when its head and base can run, else the empirical band."""
        loaded = self.head(horizon)
        out: dict[str, Any] = {"symbol": symbol, "horizon": horizon, "quantiles": list(DEFAULT_QUANTILES)}
        if loaded is not None:
            model, version = loaded
            try:
                result = model.predict_latest(symbol_features(bars, symbol), horizon=horizon)
                p10, p50, p90 = result["final"]
                return {**out, "method": "hybrid", "model_version": version, "p10": p10, "p50": p50, "p90": p90,
                        "base": result["base"], "residual": result["residual"],
                        "as_of": str(pd.Timestamp(result["as_of"]).date())}
            except Exception as exc:  # noqa: BLE001 - fall through to the empirical band
                logger.warning("hybrid forecast failed for %s h=%d: %s", symbol, horizon, exc)
                reason = f"hybrid failed: {type(exc).__name__}: {exc}"
        else:
            reason = self.reason(horizon)
        values = empirical_quantiles(bars.sort_values("timestamp")["close"], horizon)
        if values is None:
            return {**out, "method": "unavailable", "reason": reason or "not enough history"}
        p10, p50, p90 = values
        return {**out, "method": "empirical", "model_version": None, "p10": p10, "p50": p50, "p90": p90,
                "reason": reason, "as_of": str(pd.Timestamp(bars["timestamp"].max()).date())}


_service: ForecastService | None = None


def get_service() -> ForecastService:
    global _service
    if _service is None:
        _service = ForecastService()
    return _service


def reset() -> None:
    global _service
    _service = None


def forecast_step(instrument) -> tuple[bool, dict[str, Any] | str]:  # noqa: ANN001 - Instrument
    """Active-set step: forecasts at every FORECAST_HORIZONS for one held symbol."""
    from data.history import load_stored

    bars = load_stored(instrument)
    if bars.empty:
        return False, "no stored history (liquidity step failed)"
    service = get_service()
    block = {f"h{h}": service.forecast(instrument.symbol, bars, h) for h in FORECAST_HORIZONS}
    if all(entry["method"] == "unavailable" for entry in block.values()):
        return False, "; ".join(f"{k}: {v.get('reason')}" for k, v in block.items())
    return True, block


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install trained hybrid heads.")
    parser.add_argument("--install", type=Path, required=True, help="hybrid_heads.zip from Colab Part A")
    parser.add_argument("--no-activate", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    for version in install(args.install, activate=not args.no_activate):
        print("registered", version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
