"""Chronos-Bolt adapter -- the second foundation model in the RQ1 benchmark.

Chronos-Bolt is PyTorch-native with no jax dependency, so unlike TimesFM it carries no
Windows install risk, and its encoder-decoder design gives direct multi-quantile output in a
single forward pass (notably faster on CPU than autoregressive sampling).

Having two foundation models behind one interface turns a dependency risk into a research
contribution: RQ1 reports a foundation-model comparison rather than a single-model result,
and the pipeline still ships if either one fails to install.

API note: the installed distribution is `chronos-forecasting` 2.x, which exposes
`BaseChronosPipeline.from_pretrained(...)` and `predict_quantiles(...)`. Weights download on
first use (~200MB for chronos-bolt-base) and are cached by huggingface_hub.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from forecasting.base import (
    DEFAULT_QUANTILES,
    ForecastResult,
    enforce_non_crossing,
    resolve_device,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "amazon/chronos-bolt-base"


@dataclass
class ChronosConfig:
    model_id: str = DEFAULT_MODEL_ID
    context_length: int = 512
    # None = use the GPU when there is one. This was pinned to "cpu" because the development
    # machine has no CUDA, which silently sent Colab's fine-tune to the CPU as well: not an
    # error, so nothing reported it, just twenty epochs at CPU speed on a GPU runtime.
    device: str | None = None
    # float32 deliberately: build_finetune_dataset produces float32, and a half-precision
    # model against float32 batches is the same class of mismatch as a device mismatch.
    torch_dtype: str = "float32"


class ChronosBoltForecaster:
    """Zero-shot or LoRA-fine-tuned Chronos-Bolt behind the Forecaster protocol."""

    name = "chronos_bolt"

    def __init__(self, config: ChronosConfig | None = None, adapter_path: str | None = None) -> None:
        self.config = config or ChronosConfig()
        self.adapter_path = adapter_path
        self.version = f"chronos-bolt{'-lora' if adapter_path else '-zeroshot'}"
        self._pipeline = None

    def _load(self):
        """Lazily load the pipeline. Weights download on first call and are cached."""
        if self._pipeline is not None:
            return self._pipeline

        from chronos import BaseChronosPipeline

        device = str(resolve_device(self.config.device))
        logger.info("loading %s on %s", self.config.model_id, device)
        self._pipeline = BaseChronosPipeline.from_pretrained(
            self.config.model_id,
            device_map=device,
            torch_dtype=getattr(torch, self.config.torch_dtype),
        )

        if self.adapter_path:
            from peft import PeftModel

            self._pipeline.model = PeftModel.from_pretrained(self._pipeline.model, self.adapter_path)
            logger.info("applied LoRA adapter from %s", self.adapter_path)

        return self._pipeline

    def fit(self, features: pd.DataFrame, *, horizon: int, **kwargs: object) -> None:
        """No-op: fine-tuning lives in finetune_lora.py so both adapters share one loop."""
        logger.debug("ChronosBoltForecaster.fit is a no-op; see forecasting/finetune_lora.py")

    def predict_quantiles(
        self,
        features: pd.DataFrame,
        *,
        horizon: int,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        **kwargs: object,
    ) -> ForecastResult:
        """Quantile forecast from the close-price series.

        Forecasts RETURNS, not prices: the target everywhere downstream is a return, and a
        price-level forecast would have to be differenced anyway -- doing it here keeps the
        comparison against the LSTM baseline like-for-like.
        """
        pipeline = self._load()
        frame = features.sort_values("timestamp")
        symbol = str(frame["symbol"].iloc[0]) if "symbol" in frame.columns else "unknown"

        closes = frame["close"].to_numpy(dtype=np.float32)
        if len(closes) < 2:
            raise ValueError(f"{symbol}: need at least 2 closes to form a return series")

        returns = np.diff(np.log(closes))
        context = torch.tensor(returns[-self.config.context_length :], dtype=torch.float32)

        quantile_preds, _mean = pipeline.predict_quantiles(
            context=context,
            prediction_length=horizon,
            quantile_levels=list(quantiles),
        )

        # (batch=1, horizon, n_quantiles) -> take the final step: the h-ahead forecast made
        # from the most recent context, which is what the evaluation harness aligns on.
        values = quantile_preds[0, -1, :].detach().cpu().numpy().reshape(1, len(quantiles))

        return ForecastResult(
            symbol=symbol,
            horizon=horizon,
            quantiles=tuple(quantiles),
            values=enforce_non_crossing(values),
            timestamps=pd.DatetimeIndex([pd.to_datetime(frame["timestamp"].iloc[-1])]),
            model_name=self.name,
            model_version=self.version,
        )
