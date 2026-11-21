"""TimesFM adapter: TimesFM 3.0 (latest) and 2.5 behind one Forecaster.

TWO GENERATIONS, ONE PACKAGE
  * PyPI `timesfm` 3.0.x provides two modules: `timesfm` (the 2.5 model, class
    TimesFM_2p5_200M_torch) and `timesfm3` (the 3.0 model, class TimesFM3Forecaster, 330M
    parameters, 9 quantiles 0.1..0.9, native multivariate and covariates). Upgrading the package
    therefore keeps 2.5 working.
  * VERSION TRAP (2.5): there is no `timesfm==2.5` on PyPI. "2.5" is the MODEL generation, not
    the package version.

LICENCE -- READ BEFORE DEPLOYING
  The 3.0 weights are under the `timesfm-non-commercial-license-v1.0`: non-commercial and
  non-production use only; commercial/production use is permitted only through Google Cloud's
  own services (BigQuery ML). The 2.5 weights remain Apache-2.0. So the generation is an
  explicit, recorded choice:

      TimesFMForecaster(generation="3.0" | "2.5" | "auto")   # argument wins
      TIMESFM_GENERATION=3.0 | 2.5 | auto                    # then the environment
      default "auto": 3.0 when `timesfm3` is installed, else 2.5

  Research and Colab runs take the latest (3.0). A DEPLOYED service that self-hosts weights
  should pin TIMESFM_GENERATION=2.5.

LoRA is implemented and verified for 2.5 only (finetune_lora._timesfm_step mirrors the 2.5
prefill line for line, and a train/serve mismatch there would go unnoticed). An adapter therefore
implies 2.5, and asking for it on 3.0 is an error rather than a silent mismatch.

WINDOWS CONSTRAINT: `timesfm[xreg]` and `timesfm[flax]` both depend on jax[cuda], whose
jax-cuda12-plugin ships manylinux-only wheels -- they cannot install on Windows. We install
`timesfm[torch]` only, which means the 2.5 forecast_with_covariates() is unavailable (it routes
through the JAX-backed xreg_lib). That is by design: in this architecture the covariate path IS
the residual head (see hybrid_model.py), so TimesFM handles price dynamics and the head
conditions the correction on technical, sentiment and fundamental features.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecasting.base import DEFAULT_QUANTILES, ForecastResult, enforce_non_crossing

logger = logging.getLogger(__name__)

MODEL_CLASS_NAME = "TimesFM_2p5_200M_torch"          # the 2.5 model class
PACKAGE_PIN = "timesfm[torch]>=3.0.2"               # provides both generations

GENERATIONS = ("3.0", "2.5")
ENV_VAR = "TIMESFM_GENERATION"
REPO_IDS = {
    "3.0": "google/timesfm-3.0-pytorch",
    "2.5": "google/timesfm-2.5-200m-pytorch",
}
LICENCES = {
    "3.0": "TimesFM non-commercial licence v1.0: non-commercial, non-production use only",
    "2.5": "Apache-2.0",
}

# TimesFM emits a fixed quantile grid; we interpolate onto whatever the caller asked for.
TIMESFM_QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

_licence_warned = False


def _installed_generations() -> tuple[bool, bool]:
    """(3.0 code present, 2.5 code present), without importing either."""
    import importlib.util

    return (
        importlib.util.find_spec("timesfm3") is not None,
        importlib.util.find_spec("timesfm") is not None,
    )


def resolve_generation(requested: str | None, *, has_v3: bool, has_v25: bool, env: str | None) -> str:
    """Pick "3.0" or "2.5": an explicit request, else the environment, else the newest installed."""
    choice = str(requested or env or "auto").strip().lower()
    if choice not in ("auto", *GENERATIONS):
        raise ValueError(f"unknown TimesFM generation {choice!r}; expected 'auto', '3.0' or '2.5'")

    if choice == "auto":
        if has_v3:
            return "3.0"
        if has_v25:
            return "2.5"
        raise RuntimeError(f"TimesFM is not installed; install {PACKAGE_PIN}")
    if choice == "3.0" and not has_v3:
        raise RuntimeError(
            f"TimesFM 3.0 was requested but the timesfm3 module is not installed; install {PACKAGE_PIN}"
        )
    if choice == "2.5" and not has_v25:
        raise RuntimeError(f"TimesFM 2.5 was requested but timesfm is not installed; install {PACKAGE_PIN}")
    return choice


@dataclass
class TimesFMConfig:
    context_length: int = 512
    max_horizon: int = 128
    normalize_inputs: bool = True
    use_continuous_quantile_head: bool = True
    # TimesFM can enforce non-crossing quantiles itself. Enabled because a crossed interval
    # makes Phase 5a's CVaR incoherent; forecasting.base.enforce_non_crossing still runs as
    # a belt-and-braces check at the interface.
    fix_quantile_crossing: bool = True
    # 3.0 only: series forecast per forward pass.
    batch_size: int = 16


class TimesFMForecaster:
    """Zero-shot (3.0 or 2.5) or LoRA-fine-tuned (2.5) TimesFM behind the Forecaster protocol."""

    name = "timesfm"

    def __init__(
        self,
        config: TimesFMConfig | None = None,
        adapter_path: str | None = None,
        generation: str | None = None,
    ) -> None:
        """`adapter_path` loads LoRA weights from finetune_lora.py; None gives zero-shot.

        `generation` is "3.0", "2.5" or "auto"/None (see the module docstring). An adapter
        belongs to 2.5, so it pins that generation.
        """
        self.config = config or TimesFMConfig()
        self.adapter_path = adapter_path

        if adapter_path:
            if generation not in (None, "auto", "2.5"):
                raise ValueError(
                    f"a LoRA adapter was trained on TimesFM 2.5 and cannot be loaded onto {generation}: "
                    "LoRA is implemented and verified for 2.5 only"
                )
            generation = "2.5"

        has_v3, has_v25 = _installed_generations()
        self.generation = resolve_generation(
            generation, has_v3=has_v3, has_v25=has_v25, env=os.environ.get(ENV_VAR)
        )
        self.licence = LICENCES[self.generation]
        self.version = f"timesfm-{self.generation.replace('.', 'p')}{'-lora' if adapter_path else '-zeroshot'}"
        self._model = None

    def _load(self):
        """Lazily load and compile the model. Weights download on first call."""
        if self._model is not None:
            return self._model
        model = self._load_v3() if self.generation == "3.0" else self._load_v25()
        self._model = model
        return model

    def _load_v3(self):
        global _licence_warned
        from timesfm3 import ModelConfig, TimesFM3Forecaster

        from forecasting.base import resolve_device

        if not _licence_warned:
            logger.warning(
                "TimesFM 3.0 weights are under the %s. Fine for research and Colab runs; a deployed "
                "service that self-hosts them should set %s=2.5 (Apache-2.0) or use Google Cloud's "
                "BigQuery ML.", LICENCES["3.0"], ENV_VAR,
            )
            _licence_warned = True

        logger.info("loading TimesFM 3.0 (330M, torch) from %s", REPO_IDS["3.0"])
        return TimesFM3Forecaster(config=ModelConfig(
            checkpoint_path=REPO_IDS["3.0"],
            per_core_batch_size=self.config.batch_size,
            device=str(resolve_device()),
        ))

    def _load_v25(self):
        import timesfm

        model_class = timesfm.TimesFM_2p5_200M_torch
        # Use the class's own DEFAULT_REPO_ID rather than a hardcoded string, so a package
        # upgrade that repoints the weights does not silently load the wrong checkpoint.
        repo_id = getattr(model_class, "DEFAULT_REPO_ID", REPO_IDS["2.5"])
        logger.info("loading TimesFM 2.5 (200M, torch) from %s", repo_id)
        model = model_class.from_pretrained(repo_id)

        model.compile(
            timesfm.ForecastConfig(
                max_context=self.config.context_length,
                max_horizon=self.config.max_horizon,
                normalize_inputs=self.config.normalize_inputs,
                use_continuous_quantile_head=self.config.use_continuous_quantile_head,
                fix_quantile_crossing=self.config.fix_quantile_crossing,
            )
        )

        if self.adapter_path:
            from peft import PeftModel

            model.model = PeftModel.from_pretrained(model.model, self.adapter_path)
            logger.info("applied LoRA adapter from %s", self.adapter_path)
        return model

    def fit(self, features: pd.DataFrame, *, horizon: int, **kwargs: object) -> None:
        """No-op: fine-tuning lives in finetune_lora.py so both adapters share one loop."""
        logger.debug("TimesFMForecaster.fit is a no-op; see forecasting/finetune_lora.py")

    def predict_quantiles(
        self,
        features: pd.DataFrame,
        *,
        horizon: int,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        **kwargs: object,
    ) -> ForecastResult:
        """Quantile forecast from the close-price series alone. Covariates are the head's job.

        Forecasts RETURNS rather than price levels, matching the Chronos adapter and the LSTM
        baseline so RQ1 compares like with like.
        """
        return self.predict_quantiles_batch([features], horizon=horizon, quantiles=quantiles)[0]

    def _quantile_grids(self, model, contexts: list[np.ndarray], horizon: int) -> list[np.ndarray]:
        """One 9-level quantile grid (0.1..0.9) per series, at the last forecast step."""
        if self.generation == "3.0":
            outputs = list(model.predict_batch(
                contexts=contexts, horizon=horizon, return_quantiles=True,
                sort_quantiles=True, use_symmetric_averaging=False,
            ))
            if len(outputs) != len(contexts):
                raise ValueError(f"TimesFM 3.0 returned {len(outputs)} forecasts for {len(contexts)} series")
            grids = []
            for output in outputs:
                grid = np.asarray(output.quantiles)
                if grid.shape != (horizon, len(TIMESFM_QUANTILE_LEVELS)):
                    raise ValueError(
                        f"unexpected quantile grid shape {grid.shape} from TimesFM 3.0; "
                        f"expected ({horizon}, {len(TIMESFM_QUANTILE_LEVELS)})"
                    )
                grids.append(grid[-1])
            return grids

        _point, quantile_forecast = model.forecast(horizon=horizon, inputs=contexts)
        full = np.asarray(quantile_forecast)
        # (batch, horizon, n_levels); the leading level column is the mean, so the quantile grid
        # starts at index 1.
        return [full[index, -1, 1:] for index in range(len(contexts))]

    def predict_quantiles_batch(
        self,
        frames: list[pd.DataFrame],
        *,
        horizon: int,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        **kwargs: object,
    ) -> list[ForecastResult]:
        """Forecast many series in ONE forward pass.

        `forecast()` has always taken a list; passing a one-element list is what made the
        hybrid's walk-forward untenable -- roughly 5,200 single-item calls per fold, each a
        full pass over a 231M-parameter model, thirteen folds deep.

        Batching does not change any individual result: the model normalises each series
        against its own statistics, so a series forecast alongside others is forecast
        exactly as it would have been alone. A test asserts that rather than assuming it.
        """
        model = self._load()

        contexts, symbols, stamps = [], [], []
        for features in frames:
            frame = features.sort_values("timestamp")
            symbol = str(frame["symbol"].iloc[0]) if "symbol" in frame.columns else "unknown"
            closes = frame["close"].to_numpy(dtype=np.float32)
            if len(closes) < 2:
                raise ValueError(f"{symbol}: need at least 2 closes to form a return series")

            returns = np.log(closes[1:] / closes[:-1])
            contexts.append(returns[-self.config.context_length :])
            symbols.append(symbol)
            stamps.append(pd.to_datetime(frame["timestamp"].iloc[-1]))

        grids = self._quantile_grids(model, contexts, horizon)

        results = []
        for index, symbol in enumerate(symbols):
            values = np.interp(
                np.asarray(quantiles), np.asarray(TIMESFM_QUANTILE_LEVELS), grids[index]
            ).reshape(1, len(quantiles))
            results.append(
                ForecastResult(
                    symbol=symbol,
                    horizon=horizon,
                    quantiles=tuple(quantiles),
                    values=enforce_non_crossing(values),
                    timestamps=pd.DatetimeIndex([stamps[index]]),
                    model_name=self.name,
                    model_version=self.version,
                )
            )
        return results
