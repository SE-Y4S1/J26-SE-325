"""The Forecaster interface every model implements, and the adapter registry.

Defined in Phase 3 -- before any foundation model is touched -- so that TimesFM,
Chronos-Bolt, the LSTM baseline and the hybrid are all swappable behind one type. This is
what lets RQ1 be a table rather than a rewrite, and what makes the Phase 0 gate's verdict
actionable: whichever foundation models imported get registered, the rest are skipped.

All forecasters are QUANTILE forecasters. A point forecast cannot express the downside tail
that Phase 5a's CVaR objective needs, and pinball loss (RQ1) is undefined without quantiles.
p50 doubles as the point forecast where one is wanted.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

# Single source of truth: forecasting/env_report.py also writes to this path.
from forecasting.env_report import ENV_REPORT_PATH

logger = logging.getLogger(__name__)

DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


# Baselines have no optional dependencies and are always available.
_ALWAYS_AVAILABLE = ("baseline_lstm",)


@dataclass(frozen=True)
class ForecastResult:
    """Quantile forecasts for one symbol over one horizon.

    `values` is (n_timestamps, n_quantiles), column order matching `quantiles`.
    """

    symbol: str
    horizon: int
    quantiles: tuple[float, ...]
    values: np.ndarray
    timestamps: pd.DatetimeIndex
    model_name: str
    model_version: str

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(f"values must be 2-D, got shape {self.values.shape}")
        if self.values.shape[1] != len(self.quantiles):
            raise ValueError(
                f"{self.values.shape[1]} value columns but {len(self.quantiles)} quantiles"
            )
        if len(self.timestamps) != self.values.shape[0]:
            raise ValueError(
                f"{len(self.timestamps)} timestamps but {self.values.shape[0]} value rows"
            )

    def point(self) -> np.ndarray:
        """The p50 column -- the point forecast."""
        try:
            median_idx = self.quantiles.index(0.5)
        except ValueError:
            # No exact median requested; fall back to the nearest quantile.
            median_idx = int(np.argmin(np.abs(np.array(self.quantiles) - 0.5)))
            logger.debug("no 0.5 quantile; using q=%s as the point forecast", self.quantiles[median_idx])
        return self.values[:, median_idx]

    def to_frame(self) -> pd.DataFrame:
        """Long-form frame with p10/p50/p90 columns, for the API and evaluation layers."""
        columns = {f"p{int(q * 100)}": self.values[:, i] for i, q in enumerate(self.quantiles)}
        return pd.DataFrame(
            {"timestamp": self.timestamps, "symbol": self.symbol, "horizon": self.horizon, **columns}
        )


@runtime_checkable
class Forecaster(Protocol):
    """Every model in forecasting/ satisfies this."""

    name: str
    version: str

    def fit(self, features: pd.DataFrame, *, horizon: int, **kwargs: object) -> None:
        """Train or fine-tune. Zero-shot adapters implement this as a no-op."""
        ...

    def predict_quantiles(
        self,
        features: pd.DataFrame,
        *,
        horizon: int,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> ForecastResult:
        """Forecast `horizon` steps ahead. Must never read rows dated at or after the
        target timestamp -- Phase 7 asserts this."""
        ...


def resolve_device(preference: str | None = None) -> "torch.device":  # noqa: F821
    """The device to use: an explicit preference, else the GPU when there is one.

    Shared by the adapters and the fine-tuner so a run picks a device ONCE. Letting each
    side decide for itself is what produced "mat1 is on cpu, different from other tensors on
    cuda:0": TimesFM's load_checkpoint() moves itself to cuda:0 whenever CUDA exists, while
    the DataLoader went on yielding CPU tensors. The quieter half of the same bug was
    Chronos defaulting to "cpu" and training there on a GPU runtime -- no error, just twenty
    epochs at the wrong speed.

    torch is imported lazily: this module is imported by the registry, which must stay
    usable on a machine where the optional forecasting stack is not installed.
    """
    import torch

    if preference is not None:
        return torch.device(preference)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def available_foundation_models() -> dict[str, bool]:
    """Which optional foundation forecasters survived the Phase 0 gate.

    Reads artifacts/env_report.json rather than importing, so a broken install cannot take
    down the whole registry at import time -- which would make the LSTM baseline and the
    entire optimization path unreachable because of an unrelated optional dependency.
    Missing report => nothing available.
    """
    if not ENV_REPORT_PATH.exists():
        logger.debug("no env_report.json; run `pytest tests/test_env.py` to generate it")
        return {}

    try:
        report = json.loads(ENV_REPORT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("unreadable env_report.json (%s); assuming no foundation models", exc)
        return {}

    return {
        name: bool(meta.get("available", False))
        for name, meta in report.get("optional_forecasters", {}).items()
    }


# repo ids whose weights each optional package needs at inference time.
FOUNDATION_REPO_IDS = {
    "timesfm": "google/timesfm-2.5-200m-pytorch",
    "chronos": "amazon/chronos-bolt-base",
}


def weights_cached(repo_id: str) -> bool:
    """Whether a HuggingFace checkpoint is genuinely present locally.

    Checks for a real weight file, not just the directory. An interrupted download leaves
    config.json plus zero-byte `.incomplete` placeholders, which would otherwise read as a
    cache hit and send the caller into a stalling multi-hundred-megabyte fetch.
    """
    cache = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_id.replace('/', '--')}"
    if not cache.exists():
        return False
    return any(
        f.is_file() and not f.name.endswith(".incomplete") and f.stat().st_size > 1_000_000
        for f in cache.rglob("*")
    )


def usable_foundation_models() -> dict[str, bool]:
    """Foundation models usable WITHOUT a download: package installed AND weights cached.

    Distinct from available_foundation_models(), which reports only whether the package
    imports. The difference matters: an installed package with absent weights looks available
    but costs hours on first use, so anything choosing a default must consult this instead.
    """
    available = available_foundation_models()
    return {
        name: bool(available.get(name)) and weights_cached(repo_id)
        for name, repo_id in FOUNDATION_REPO_IDS.items()
    }


def registered_forecasters(*, require_weights: bool = False) -> list[str]:
    """Names usable in this environment -- baselines plus whatever passed the gate.

    With require_weights=True, only models that can run offline right now are listed.
    """
    names = list(_ALWAYS_AVAILABLE)
    available = usable_foundation_models() if require_weights else available_foundation_models()

    if available.get("timesfm"):
        names.append("timesfm")
    if available.get("chronos"):
        names.append("chronos_bolt")
    if any(available.values()):
        names.append("hybrid")

    return names


def _foundation_unavailable(module: str) -> RuntimeError:
    """Explain why a foundation model is unavailable, rather than assuming it is absent.

    There are three distinct causes and they need different fixes, so reporting all of them
    as "not installed" sends people to reinstall a package that is already there. On a fresh
    Colab clone the cause is almost always the first one: artifacts/ is gitignored, so no
    report exists and every foundation model looks missing however the install went.
    """
    from forecasting.env_report import OPTIONAL_FORECASTERS

    dist = OPTIONAL_FORECASTERS.get(module, module)

    if not ENV_REPORT_PATH.exists():
        return RuntimeError(
            f"No capability report at {ENV_REPORT_PATH}, so no foundation model can be "
            f"registered. This says nothing about whether {module} is installed. "
            "Generate the report with:  python -m forecasting.env_report"
        )

    try:
        report = json.loads(ENV_REPORT_PATH.read_text(encoding="utf-8"))
        meta = report.get("optional_forecasters", {}).get(module, {})
    except Exception:  # noqa: BLE001 - fall through to the generic message
        meta = {}

    if error := meta.get("error"):
        return RuntimeError(
            f"{module} is present but failed to import: {error}. "
            "If a pip install replaced a pre-installed package (numpy is the usual one), "
            "restart the runtime and regenerate the report with: "
            "python -m forecasting.env_report"
        )

    return RuntimeError(
        f"{module} is not installed. Run: uv add --optional {module} '{dist}'. "
        "Then regenerate the report with: python -m forecasting.env_report"
    )


def get_forecaster(name: str, **kwargs: object) -> Forecaster:
    """Construct a registered forecaster by name.

    Raises a clear error naming the missing extra if an unavailable foundation model is
    requested, rather than a bare ImportError from three frames down.
    """
    available = available_foundation_models()

    if name == "baseline_lstm":
        from forecasting.baseline_lstm import BaselineLSTMForecaster, LSTMConfig

        config = kwargs.get("config")
        return BaselineLSTMForecaster(config if isinstance(config, LSTMConfig) else None)

    if name == "timesfm":
        if not available.get("timesfm"):
            raise _foundation_unavailable("timesfm")
        from forecasting.timesfm_adapter import TimesFMForecaster

        return TimesFMForecaster(**kwargs)  # type: ignore[arg-type]

    if name == "chronos_bolt":
        if not available.get("chronos"):
            raise _foundation_unavailable("chronos")
        from forecasting.chronos_adapter import ChronosBoltForecaster

        return ChronosBoltForecaster(**kwargs)  # type: ignore[arg-type]

    if name == "hybrid":
        base_name = str(kwargs.pop("base_model", None) or _preferred_base(available))
        from forecasting.hybrid_model import HybridForecaster

        return HybridForecaster(get_forecaster(base_name), **kwargs)  # type: ignore[arg-type]

    raise KeyError(
        f"unknown forecaster {name!r}. Available in this environment: {registered_forecasters()}"
    )


def _preferred_base(available: dict[str, bool]) -> str:
    """Pick a base model for the hybrid when the caller did not specify one.

    Prefers one whose weights are ALREADY CACHED, so an unqualified get_forecaster('hybrid')
    can never silently start an hours-long download. Among cached options TimesFM wins,
    because it is the model the TAF names -- keeping the proposal honest matters more than a
    marginal accuracy difference.
    """
    cached = usable_foundation_models()
    for name, adapter in (("timesfm", "timesfm"), ("chronos", "chronos_bolt")):
        if cached.get(name):
            return adapter

    installed = [n for n, ok in available.items() if ok]
    if installed:
        raise RuntimeError(
            f"Foundation package(s) {installed} are installed but their weights are not "
            "cached, so using them would trigger a multi-hundred-megabyte download. "
            "Pre-fetch with `huggingface-cli download "
            f"{FOUNDATION_REPO_IDS.get(installed[0], '<repo>')}`, run RQ1 in Colab, or use "
            "'baseline_lstm', which trains locally and needs no weights."
        )
    raise RuntimeError(
        "The hybrid needs a foundation model, but neither TimesFM nor Chronos-Bolt is "
        "installed. Install one, or use 'baseline_lstm'. See README 'Known risks'."
    )


def enforce_non_crossing(values: np.ndarray) -> np.ndarray:
    """Sort quantile columns so p10 <= p50 <= p90.

    Crossed quantiles make the CVaR objective incoherent (the "tail" would not be the tail),
    so this is applied at the interface rather than trusting every model to behave. Sorting
    is the standard post-hoc fix and cannot make the pinball loss worse.
    """
    return np.sort(np.asarray(values, dtype=float), axis=1)
