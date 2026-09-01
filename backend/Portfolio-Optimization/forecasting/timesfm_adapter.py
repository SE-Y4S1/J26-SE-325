"""TimesFM adapter.

VERSION TRAP: there is no `timesfm==2.5` on PyPI -- the package tops out at 2.0.2. "2.5" is
the MODEL generation, shipped inside 2.0.2 and exposed as the class TimesFM_2p5_200M_torch.
Pin the package at 2.0.2 and select the 2.5 model class. Verified against the installed
distribution: `timesfm` exports exactly ForecastConfig, TimesFM_2p5_200M_torch (and the flax
variant).

WINDOWS CONSTRAINT: `timesfm[xreg]` and `timesfm[flax]` both depend on jax[cuda], whose
jax-cuda12-plugin ships manylinux-only wheels -- they cannot install on Windows. We install
`timesfm[torch]` only, which means forecast_with_covariates() is unavailable.

Note the method `forecast_with_covariates` IS present on the torch class -- but it routes
through `timesfm.utils.xreg_lib`, which is JAX-backed, so calling it without jax installed
fails at import. The method existing is not the same as it being usable here.

That is by design, not a compromise: in this architecture the covariate path IS the residual
head (see hybrid_model.py). TimesFM handles pure price dynamics; technical and sentiment
features condition the correction on top. XReg would have been a second, redundant route for
the same information.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecasting.base import DEFAULT_QUANTILES, ForecastResult, enforce_non_crossing

logger = logging.getLogger(__name__)

MODEL_CLASS_NAME = "TimesFM_2p5_200M_torch"
PACKAGE_PIN = "timesfm[torch]==2.0.2"

# TimesFM emits a fixed quantile grid; we interpolate onto whatever the caller asked for.
TIMESFM_QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


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


class TimesFMForecaster:
    """Zero-shot or LoRA-fine-tuned TimesFM behind the Forecaster protocol."""

    name = "timesfm"

    def __init__(self, config: TimesFMConfig | None = None, adapter_path: str | None = None) -> None:
        """`adapter_path` loads LoRA weights from finetune_lora.py; None gives zero-shot."""
        self.config = config or TimesFMConfig()
        self.adapter_path = adapter_path
        self.version = f"timesfm-2p5{'-lora' if adapter_path else '-zeroshot'}"
        self._model = None

    def _load(self):
        """Lazily load and compile the model. Weights download on first call."""
        if self._model is not None:
            return self._model

        import timesfm

        model_class = timesfm.TimesFM_2p5_200M_torch
        # Use the class's own DEFAULT_REPO_ID rather than a hardcoded string, so a package
        # upgrade that repoints the weights does not silently load the wrong checkpoint.
        repo_id = getattr(model_class, "DEFAULT_REPO_ID", "google/timesfm-2.5-200m-pytorch")
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

        self._model = model
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

        _point, quantile_forecast = model.forecast(horizon=horizon, inputs=contexts)
        grids = np.asarray(quantile_forecast)

        results = []
        for index, symbol in enumerate(symbols):
            # (batch, horizon, n_levels); the leading level column is the mean, so the
            # quantile grid starts at index 1.
            grid = grids[index, -1, 1:]
            values = np.interp(
                np.asarray(quantiles), np.asarray(TIMESFM_QUANTILE_LEVELS), grid
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
