"""LoRA fine-tuning, shared by both foundation adapters.

Adapted from google-research/timesfm's own example at
timesfm-forecasting/examples/finetuning/finetune_lora.py rather than written from scratch.
That example's key insight: the model computes its own loss when `future_values` is passed,
so fine-tuning needs nothing more than a standard PyTorch loop plus PEFT LoRA. Where a model
does not expose that, we fall back to an explicit quantile loss on its forecast head.

Saves ADAPTER WEIGHTS ONLY (a few MB), never the full model -- which also makes the
content-hash provenance in model_registry.py meaningful rather than hashing 200M frozen
parameters that never change.

WHERE THIS SHOULD RUN
---------------------
Colab, not this machine. The dev laptop has an AMD iGPU and no CUDA, so torch is the CPU
build; fine-tuning a 200M-parameter model on CPU is hours of work for a result a free T4
produces in minutes. The design accommodates that split deliberately: this script writes a
self-contained adapter directory, and `TimesFMForecaster(adapter_path=...)` /
`ChronosBoltForecaster(adapter_path=...)` load it locally for inference.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

CHECKPOINT_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "checkpoints"

# Attention projection names differ by architecture: Chronos-Bolt is T5-based ("q"/"v"),
# most decoder stacks use "q_proj"/"v_proj". Both are attempted before giving up, so the
# script works across the two adapters without per-model configuration.
CANDIDATE_TARGET_MODULES: tuple[tuple[str, ...], ...] = (
    ("q_proj", "v_proj"),
    ("q", "v"),
    ("query", "value"),
)


# TimesFM 2.5 patches its context into blocks of this size; a context that is not a
# whole number of them is truncated inside forward().
TIMESFM_PATCH_LEN = 32


@dataclass
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] | None = None   # None = auto-discover
    epochs: int = 20
    batch_size: int = 64
    lr: float = 5e-5
    seed: int = 42
    val_fraction: float = 0.2
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    max_grad_norm: float = 1.0
    patience: int = 5
    # None = use the GPU when there is one. An explicit "cpu" forces a CPU run, which is
    # useful for reproducing a GPU-only failure or for a machine whose GPU is busy.
    device: str | None = None
    # Optimizer steps are taken every `grad_accum_steps` batches, so halving batch_size and
    # doubling this leaves the effective batch -- and the optimisation -- unchanged while
    # holding a fraction of the activations. Raised automatically when a batch will not fit.
    grad_accum_steps: int = 1
    # A batch smaller than this is not worth retrying; the run is failing for another reason.
    min_batch_size: int = 4


class WindowedSeriesDataset(Dataset):
    """(context, future) pairs sliced from each symbol's return series ON DEMAND.

    Materialising every window is what the obvious implementation does, and it stores
    `context_length` floats per window when consecutive windows share all but one step. On
    the resolved universe that is 88,244 windows x 512 floats = 181MB of almost entirely
    duplicated data, built from 400KB of underlying returns -- a 450x blow-up, and it is
    held for the whole run rather than a batch at a time.

    Storing the series plus an index of (series, start) offsets instead costs about a
    megabyte, and __getitem__ slices the window when the DataLoader asks for it. The slices
    are views into contiguous 1-D arrays, so this trades no measurable time for the space.
    """

    def __init__(
        self,
        series: list[np.ndarray],
        index: np.ndarray,
        *,
        context_length: int,
        horizon: int,
    ) -> None:
        self.series = series
        self.index = index                      # (n, 2) int64: series id, window start
        self.context_length = context_length
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        series_id, start = self.index[item]
        returns = self.series[series_id]
        split = start + self.context_length
        return (
            torch.from_numpy(returns[start:split]),
            torch.from_numpy(returns[split : split + self.horizon]),
        )

    @property
    def nbytes(self) -> int:
        """What the dataset actually holds -- used by the memory report."""
        return sum(s.nbytes for s in self.series) + self.index.nbytes


def smoke_subset(
    features: pd.DataFrame,
    *,
    horizon: int,
    context_length: int = 512,
    windows: int = 80,
) -> tuple[pd.DataFrame, int]:
    """A slice small enough for a fast smoke run, plus a context length that fits it.

    Returns `(subset, context_length)`; pass both to `finetune`.

    `build_finetune_dataset` needs `context_length + horizon + 1` bars from a SINGLE symbol,
    and quietly skips any symbol with fewer. A naive `head(n)` across every symbol therefore
    produces nothing at all -- which is exactly how the first smoke check died on Colab with
    "no windows could be built: need at least 518 bars", having sliced to 400 rows per symbol
    while leaving context_length at its 512 default.

    One symbol is deliberate: the smoke check is testing the model's forward/backward path,
    and the cross-symbol boundary logic is already covered by unit tests. Taking the longest
    series makes the choice independent of universe order.
    """
    if "symbol" not in features.columns:
        raise ValueError("features must carry a 'symbol' column")

    counts = features.groupby("symbol", sort=False).size()
    symbol = counts.idxmax()
    available = int(counts.max())

    if available < context_length + horizon + 1:
        # Round DOWN to whole TimesFM patches: its forward() reshapes the context into
        # 32-step blocks, so anything ragged is truncated there regardless.
        context_length = ((available - horizon - 1) // TIMESFM_PATCH_LEN) * TIMESFM_PATCH_LEN

    if context_length < TIMESFM_PATCH_LEN:
        raise ValueError(
            f"longest series ({symbol}, {available} bars) is too short for a {horizon}-step "
            f"horizon plus one {TIMESFM_PATCH_LEN}-step patch of context"
        )

    subset = features[features["symbol"] == symbol].head(context_length + horizon + windows)
    return subset, context_length


def build_finetune_dataset(
    features: pd.DataFrame, *, context_length: int, horizon: int
) -> WindowedSeriesDataset:
    """Windowed (context, future) pairs.

    Windows never straddle a SYMBOL boundary -- a context mixing AAPL's history with MSFT's
    would teach the model a series that does not exist. Grouping by symbol is the only thing
    preventing it, exactly as in baseline_lstm.build_sequences.

    Operates on log RETURNS rather than price levels, matching what both adapters forecast
    at inference time. Fine-tuning on levels and predicting returns would be a train/serve
    mismatch that no test would catch.
    """
    if "close" not in features.columns:
        raise ValueError("features must carry a 'close' column")

    series: list[np.ndarray] = []
    index: list[tuple[int, int]] = []

    grouped = (
        features.groupby("symbol", sort=False)
        if "symbol" in features.columns
        else [("_", features)]
    )

    for symbol, group in grouped:
        closes = group.sort_values("timestamp")["close"].to_numpy(dtype=np.float64)
        if len(closes) < context_length + horizon + 1:
            logger.debug("%s: only %d bars, need %d; skipping",
                         symbol, len(closes), context_length + horizon + 1)
            continue

        # np.ascontiguousarray so every later slice is a contiguous view that
        # torch.from_numpy can wrap without copying.
        returns = np.ascontiguousarray(np.log(closes[1:] / closes[:-1]), dtype=np.float32)
        series_id = len(series)
        series.append(returns)
        # A window starting at `start` uses [start, start+context) and predicts the
        # `horizon` steps after it, so the last valid start is len - context - horizon.
        for start in range(0, len(returns) - context_length - horizon + 1):
            index.append((series_id, start))

    if not index:
        raise ValueError(
            f"no windows could be built: need at least {context_length + horizon + 1} bars "
            "for a single symbol"
        )

    dataset = WindowedSeriesDataset(
        series, np.asarray(index, dtype=np.int64),
        context_length=context_length, horizon=horizon,
    )
    logger.info(
        "built %d fine-tuning windows (context=%d, horizon=%d) holding %.1f MB; "
        "materialising them would have taken %.0f MB",
        len(dataset), context_length, horizon, dataset.nbytes / 1e6,
        len(dataset) * context_length * 4 / 1e6,
    )
    return dataset


def fit_batch_size(model, dataset, config: LoRAConfig, step_fn) -> tuple[int, int]:
    """Largest batch that survives one training step, and the accumulation to match it.

    A Colab OOM kills the session and takes the traceback with it, so discovering the limit
    on epoch 3 of 5 costs the whole run. One forward/backward up front costs seconds and
    turns a fatal crash into a smaller batch.

    Halving the batch while doubling accumulation keeps the effective batch -- and therefore
    the optimisation -- unchanged, so this is a memory decision rather than a training one.
    """
    if not torch.cuda.is_available():
        return config.batch_size, max(1, config.grad_accum_steps)

    batch_size = config.batch_size
    accum = max(1, config.grad_accum_steps)

    while batch_size >= config.min_batch_size:
        loader = DataLoader(dataset, batch_size=batch_size)
        context, future = next(iter(loader))
        device = _model_device(model)
        try:
            loss = step_fn(model, context.to(device), future.to(device), config)
            loss.backward()
            model.zero_grad(set_to_none=True)
            del loss
            _release_cuda()
            if batch_size != config.batch_size:
                logger.warning(
                    "batch %d did not fit; using %d with %d accumulation steps "
                    "(effective batch %d, unchanged)",
                    config.batch_size, batch_size, accum, batch_size * accum,
                )
            return batch_size, accum
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            _release_cuda()
            batch_size //= 2
            accum *= 2

    raise RuntimeError(
        f"even a batch of {config.min_batch_size} will not fit on this device. "
        f"{memory_note()}. Something other than batch size is consuming the GPU -- check "
        "whether an earlier cell is still holding a model."
    )


def _release_cuda() -> None:
    """Return freed blocks to the driver. The caching allocator keeps them otherwise, which
    looks like a leak and, worse, fragments the pool for the next allocation."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def memory_note() -> str:
    """Process RSS and, on CUDA, allocated / reserved / peak GPU memory.

    Included in the plan line and every epoch summary because a Colab OOM otherwise leaves
    nothing to diagnose from: the session dies and takes its output with it. Knowing which
    of the two limits was approached, and when, is the difference between fixing the cause
    and guessing at it.
    """
    parts = []
    try:
        import psutil

        process = psutil.Process()
        parts.append(f"rss {process.memory_info().rss / 1e9:.1f}G")
        # Peak RSS ever reached, which the point-in-time figure misses entirely: this is
        # sampled between folds, after a gc, so it records the trough while the spike
        # inside a fit goes unseen. VmHWM is the kernel's own high-water mark.
        try:
            hwm = [
                line for line in Path("/proc/self/status").read_text().splitlines()
                if line.startswith("VmHWM:")
            ]
            if hwm:
                parts.append(f"peak-rss {int(hwm[0].split()[1]) / 1e6:.1f}G")
        except Exception:  # noqa: BLE001 - Linux only; absent on Windows, and optional
            pass
    except Exception:  # noqa: BLE001 - psutil is optional; never fail a run over a log line
        pass
    if torch.cuda.is_available():
        parts.append(
            f"gpu {torch.cuda.memory_allocated() / 1e9:.1f}"
            f"/{torch.cuda.memory_reserved() / 1e9:.1f}"
            f"/peak {torch.cuda.max_memory_allocated() / 1e9:.1f}G"
        )
    return "  ".join(parts) or "memory unknown"


def _release(*objects) -> None:
    """Drop references and hand the memory back before the next model loads.

    finetune() is called once per architecture, and a 231M-parameter model plus its
    optimizer state stays resident until the allocator is told otherwise. Without this the
    second model starts with the first still occupying the device.
    """
    import gc

    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _model_device(model: torch.nn.Module) -> torch.device:
    """Where a model's parameters live.

    Read from the model rather than passed in, so the training loop is right however the
    model got there -- including when an adapter placed it somewhere before we were called.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:            # a model with no parameters at all
        return torch.device("cpu")


def _resolve_inner_model(forecaster) -> torch.nn.Module:
    """Find the torch Module inside an adapter, whatever it calls it."""
    for attribute in ("model", "_model", "pipeline", "_pipeline"):
        candidate = getattr(forecaster, attribute, None)
        if isinstance(candidate, torch.nn.Module):
            return candidate
        if candidate is not None:
            inner = getattr(candidate, "model", None)
            if isinstance(inner, torch.nn.Module):
                return inner
    raise TypeError(
        f"could not locate a torch.nn.Module inside {type(forecaster).__name__}; "
        "LoRA needs the underlying network"
    )


def _discover_target_modules(model: torch.nn.Module) -> tuple[str, ...]:
    """Pick Linear modules for LoRA to adapt, without needing to know the architecture.

    Guessing wrong is a silent failure mode: PEFT raises only if NOTHING matches, so a
    partially-matching guess attaches adapters to the wrong layers and trains something
    subtly useless. But a fixed candidate list is brittle in the other direction -- it fails
    outright on any architecture that names its projections differently, which is how
    fine-tuning dies on an unfamiliar foundation model.

    Three tiers, most specific first:
      1. A known attention-projection pair (q_proj/v_proj, q/v, query/value).
      2. Anything whose name looks like an attention projection.
      3. Every Linear layer. Adapting all linears is a standard, well-behaved LoRA strategy
         -- more trainable parameters than necessary, but correct.
    """
    linear_names = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    if not linear_names:
        raise ValueError(
            f"{type(model).__name__} exposes no torch.nn.Linear modules, so there is nothing "
            "for LoRA to adapt. Pass LoRAConfig(target_modules=(...)) explicitly."
        )

    for candidate in CANDIDATE_TARGET_MODULES:
        if all(part in linear_names for part in candidate):
            logger.info("LoRA targets (known attention pair): %s", candidate)
            return candidate

    attention_like = tuple(sorted(
        n for n in linear_names
        if any(tok in n.lower() for tok in ("q_", "k_", "v_", "query", "key", "value", "attn", "attention"))
    ))
    if attention_like:
        logger.info("LoRA targets (attention-like names): %s", attention_like)
        return attention_like

    all_linear = tuple(sorted(linear_names))
    logger.warning(
        "no attention-projection names recognised in %s; adapting ALL %d Linear module "
        "name(s): %s. This trains more parameters than necessary but is correct.",
        type(model).__name__, len(all_linear), all_linear,
    )
    return all_linear


def _quantile_loss(
    prediction: torch.Tensor, target: torch.Tensor, quantiles: tuple[float, ...]
) -> torch.Tensor:
    """Pinball loss, matching the metric RQ1 scores on."""
    if prediction.dim() == target.dim():
        return torch.nn.functional.l1_loss(prediction, target)

    target = target.unsqueeze(-1)
    errors = target - prediction
    q = torch.tensor(quantiles, dtype=prediction.dtype, device=prediction.device)
    return torch.maximum(q * errors, (q - 1.0) * errors).mean()


def _shim_embedding_introspection(model: torch.nn.Module) -> None:
    """Give peft the embedding accessor it assumes every model has.

    peft's `_check_tied_modules` calls `model.get_input_embeddings()` to find weights tied to
    the input embeddings, then does `set(... .parameters())` on the result. That assumption
    holds for language models and not for time-series ones: transformers raises
    NotImplementedError for ChronosBoltModelForForecasting, which has no token embeddings to
    tie anything to, and LoRA fine-tuning dies before it starts.

    An empty module answers the question correctly -- there are no tied embedding parameters
    -- and lets the tied-weight scan return an empty set instead of raising. It is only
    installed when the real accessor is missing or raises, so a model that implements it
    properly is left alone.
    """
    try:
        model.get_input_embeddings()
    except (NotImplementedError, AttributeError):
        model.get_input_embeddings = lambda: torch.nn.Module()  # type: ignore[method-assign]
        logger.info(
            "installed an empty get_input_embeddings() for %s: it has no token embeddings, "
            "and peft's tied-weight scan requires the accessor to exist",
            type(model).__name__,
        )


# --------------------------------------------------------------------------------------
# Per-architecture training steps
#
# There is no common fine-tuning interface across these models, and guessing at one was the
# bug: the loop tried a HuggingFace-style model(past_values=, future_values=), fell back to
# model(context), and reported "neither a loss nor a tensor" when both were wrong. Each
# architecture states how it computes a loss instead, so an unsupported one fails with its
# own reason rather than a generic message from three layers down.
# --------------------------------------------------------------------------------------


def _chronos_step(model, context, future, config: LoRAConfig) -> torch.Tensor:
    """ChronosBoltModelForForecasting computes its own quantile loss.

    Signature is forward(context, mask=None, target=None, target_mask=None). Given a target
    it normalises it with the same loc/scale as the context, pads it up to the model's fixed
    prediction_length with a zero mask so the padding does not contribute, and returns the
    pinball loss over its own quantile levels. Passing horizon < prediction_length is
    therefore fine and needs no padding from us.
    """
    output = model(context=context, target=future)
    if output.loss is None:
        raise RuntimeError(
            "Chronos returned no loss despite being given a target; the installed "
            "chronos-forecasting may have changed its forward() contract"
        )
    return output.loss


def _generic_step(model, context, future, config: LoRAConfig) -> torch.Tensor:
    """HuggingFace-style: the model computes its loss from future_values, or returns a
    tensor we score with an explicit pinball loss."""
    try:
        output = model(past_values=context, future_values=future)
    except TypeError as exc:
        raise RuntimeError(
            f"{type(model).__name__} does not accept past_values/future_values "
            f"({exc}). Add a step function for this architecture to STEP_FUNCTIONS."
        ) from exc

    loss = getattr(output, "loss", None)
    if loss is not None:
        return loss

    prediction = output if torch.is_tensor(output) else getattr(output, "logits", None)
    if prediction is None:
        raise RuntimeError(
            f"{type(model).__name__} returned neither a loss nor a tensor. Add a step "
            f"function for this architecture to STEP_FUNCTIONS."
        )
    return _quantile_loss(prediction, future, config.quantiles)


def _timesfm_step(model, context, future, config: LoRAConfig) -> torch.Tensor:
    """Loss for TimesFM 2.5, mirroring the prefill half of its own decode().

    decode() runs under torch.no_grad() and so cannot be trained through, but forward() is
    an ordinary differentiable module. For a horizon within one output patch,
    num_decode_steps = (horizon - 1) // output_patch_len is zero, so decode()'s
    autoregressive loop never runs and prefill IS the whole computation. Replicating prefill
    here -- patch, normalise, forward, denormalise -- gives the same numbers with gradients
    attached.

    The steps below deliberately mirror decode() line for line, because any divergence is a
    train/serve mismatch that nothing downstream would catch.
    """
    from timesfm.torch import util as timesfm_util

    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    patch_len, out_len, n_out = inner.p, inner.o, inner.q

    horizon = future.shape[-1]
    if horizon > out_len:
        raise RuntimeError(
            f"horizon {horizon} exceeds TimesFM's output patch length {out_len}. Beyond that "
            "decode() feeds its own output back autoregressively, which this step does not "
            "reproduce; fine-tune at a horizon of {out_len} or fewer steps."
        )

    batch = context.shape[0]
    # forward() patches by reshape, which needs an exact multiple of the patch length.
    # Keep the most RECENT whole patches: the tail is what the forecast depends on.
    usable = (context.shape[1] // patch_len) * patch_len
    if usable == 0:
        raise RuntimeError(
            f"context of {context.shape[1]} steps is shorter than one {patch_len}-step patch"
        )
    inputs = context[:, context.shape[1] - usable :]

    # Nothing is padded, so nothing is masked. decode() carries masks because a serving
    # batch may hold series of different lengths; a training batch never does.
    patched_inputs = inputs.reshape(batch, -1, patch_len)
    patched_masks = torch.zeros_like(patched_inputs, dtype=torch.bool)

    # Per-patch running statistics. These describe the input window rather than the model,
    # exactly like an instance-norm statistic, so they are computed without gradients --
    # what is being learned is the LoRA weights, not the normalisation of the data.
    with torch.no_grad():
        n = torch.zeros(batch, device=inputs.device)
        mu = torch.zeros(batch, device=inputs.device)
        sigma = torch.zeros(batch, device=inputs.device)
        mus, sigmas = [], []
        for i in range(patched_inputs.shape[1]):
            (n, mu, sigma), _ = timesfm_util.update_running_stats(
                n, mu, sigma, patched_inputs[:, i], patched_masks[:, i]
            )
            mus.append(mu)
            sigmas.append(sigma)
        context_mu = torch.stack(mus, dim=1)
        context_sigma = torch.stack(sigmas, dim=1)

    normed = timesfm_util.revin(patched_inputs, context_mu, context_sigma, reverse=False)
    normed = torch.where(patched_masks, torch.zeros_like(normed), normed)

    # decode_caches=None: forward expands it to one None per layer, which is the
    # no-cache path. Caching only matters for autoregressive decoding.
    (_, _, normed_outputs, _), _ = model(normed, patched_masks)

    renormed = timesfm_util.revin(normed_outputs, context_mu, context_sigma, reverse=True)
    renormed = renormed.reshape(batch, -1, out_len, n_out)

    # The LAST patch carries the forecast for the steps after the context.
    prediction = renormed[:, -1, :horizon, :]           # (batch, horizon, n_out)

    return _quantile_loss(prediction[..., _timesfm_quantile_indices(inner, config)],
                          future, config.quantiles)


def _timesfm_quantile_indices(inner, config: LoRAConfig) -> list[int]:
    """Column indices of the requested quantiles in TimesFM's output head.

    The head emits `len(model_quantiles) + 1` columns: index 0 is the MEAN, and the declared
    quantiles start at 1. Reading index 0 as the 0.1 quantile would train the model against
    the wrong column and still converge, which is the kind of error a loss curve hides.
    """
    model_quantiles = list(inner.config.quantiles)
    indices = []
    for q in config.quantiles:
        matches = [i for i, mq in enumerate(model_quantiles) if abs(mq - q) < 1e-9]
        if not matches:
            raise RuntimeError(
                f"TimesFM emits quantiles {model_quantiles}; {q} is not among them. Fine-tune "
                f"on a subset of the model's own quantiles, or add interpolation here."
            )
        indices.append(matches[0] + 1)   # +1 for the leading mean column
    return indices


STEP_FUNCTIONS = {
    "chronos_bolt": _chronos_step,
    "timesfm": _timesfm_step,
}

# Architectures that cannot be LoRA fine-tuned by this module, with the reason. Checked
# before anything is loaded, so the failure costs a second rather than a model download
# followed by a traceback from inside the training loop.
UNSUPPORTED_LORA: dict[str, str] = {
    # Architectures that cannot be LoRA fine-tuned by this module, with the reason. Checked
    # before anything loads, so an unsupported model costs a second rather than a download
    # followed by a traceback from inside the training loop.
    #
    # TimesFM was listed here on the grounds that decode() runs under torch.no_grad(). That
    # was wrong: decode() is not the only path. forward() is an ordinary differentiable
    # module, and for a horizon within one output patch decode()'s autoregressive loop never
    # runs, so prefill alone reproduces it -- see _timesfm_step.
}


def finetune(
    base_model_name: str,
    features: pd.DataFrame,
    *,
    horizon: int,
    config: LoRAConfig | None = None,
    output_dir: Path | None = None,
    context_length: int = 512,
    register: bool = True,
    log_to_mlflow: bool = True,
) -> Path:
    """Fine-tune one foundation model, log to MLflow, register the checkpoint.

    Works for both `timesfm` and `chronos_bolt`; the adapter supplies the model handle and
    its LoRA target module names. Returns the adapter directory.
    """
    from peft import LoraConfig, get_peft_model

    from forecasting.base import get_forecaster

    if base_model_name in UNSUPPORTED_LORA:
        raise RuntimeError(UNSUPPORTED_LORA[base_model_name])

    config = config or LoRAConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    output_dir = Path(output_dir or (CHECKPOINT_ROOT / f"{base_model_name}-lora-h{horizon}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    forecaster = get_forecaster(base_model_name)
    forecaster._load()                       # noqa: SLF001 - adapters expose loading here
    inner = _resolve_inner_model(forecaster)

    # Pin the device before wrapping. Each adapter loads its model wherever it likes --
    # TimesFM moves itself to cuda:0, Chronos honours its own config -- so without this the
    # training device is whatever the adapter happened to choose, and the two models can end
    # up on different ones in the same run.
    from forecasting.base import resolve_device

    device = resolve_device(config.device)
    inner = inner.to(device)
    logger.info("training %s on %s", base_model_name, device)

    targets = config.target_modules or _discover_target_modules(inner)
    _shim_embedding_introspection(inner)
    peft_model = get_peft_model(
        inner,
        LoraConfig(
            r=config.r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(targets),
            bias="none",
        ),
    )
    peft_model = peft_model.to(device)      # LoRA layers are created during wrapping
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info("LoRA: %d trainable of %d parameters (%.3f%%)",
                trainable, total, 100 * trainable / max(total, 1))

    dataset = build_finetune_dataset(features, context_length=context_length, horizon=horizon)

    # CHRONOLOGICAL split, never random: windows overlap heavily, so a shuffled split would
    # put near-duplicate windows on both sides and report a validation loss that is really a
    # training loss.
    split = int(len(dataset) * (1 - config.val_fraction))
    train_set = torch.utils.data.Subset(dataset, range(split))
    val_set = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    step_fn = STEP_FUNCTIONS.get(base_model_name, _generic_step)

    # Establish what fits BEFORE committing to a multi-epoch run.
    batch_size, accum = fit_batch_size(peft_model, train_set, config, step_fn)
    config = replace(config, batch_size=batch_size, grad_accum_steps=accum)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad], lr=config.lr
    )

    # State the size of the job before starting it. Without this the first sign of life is
    # the end of epoch 0, which on the full universe is many minutes of silence.
    steps_per_epoch = math.ceil(len(train_set) / config.batch_size)
    logger.info(
        "plan: %s on %s | %d train / %d val windows | batch %d x %d accum "
        "(effective %d) | %d steps/epoch | up to %d epochs (%d steps, patience %d)",
        base_model_name, device, len(train_set), len(val_set),
        config.batch_size, config.grad_accum_steps,
        config.batch_size * config.grad_accum_steps, steps_per_epoch,
        config.epochs, steps_per_epoch * config.epochs, config.patience,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    logger.info("memory at start: %s", memory_note())

    best_val, stagnant = float("inf"), 0
    run_started = time.perf_counter()
    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        peft_model.train()
        train_loss = _run_epoch(
            peft_model, train_loader, config, optimizer, step_fn, epoch=epoch, phase="train"
        )

        peft_model.eval()
        with torch.no_grad():
            val_loss = _run_epoch(
                peft_model, val_loader, config, None, step_fn, epoch=epoch, phase="val"
            )

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        done = time.perf_counter() - run_started
        logger.info(
            "epoch %d/%d: train %.6f  val %.6f  (%s elapsed, ~%s left)  %s",
            epoch, config.epochs - 1, train_loss, val_loss,
            _duration(done), _duration(done / (epoch + 1) * (config.epochs - epoch - 1)),
            memory_note(),
        )

        if val_loss < best_val - 1e-9:
            best_val, stagnant = val_loss, 0
            peft_model.save_pretrained(str(output_dir))       # adapter weights only
        else:
            stagnant += 1
            if stagnant >= config.patience:
                logger.info("early stopping at epoch %d (best val %.6f)", epoch, best_val)
                break

    metrics = {"val_loss": best_val, "trainable_params": float(trainable)}

    run_id = None
    if log_to_mlflow:
        run_id = _log_run(base_model_name, horizon, config, metrics, history, targets)

    if register:
        _register_checkpoint(base_model_name, output_dir, features, horizon, metrics, run_id)

    logger.info("saved LoRA adapter to %s  (%s)", output_dir, memory_note())
    _release(peft_model, inner, forecaster, optimizer, dataset, train_loader, val_loader)
    logger.info("released %s; %s", base_model_name, memory_note())
    return output_dir


def _run_epoch(
    model,
    loader,
    config: LoRAConfig,
    optimizer,
    step_fn,
    *,
    epoch: int | None = None,
    phase: str = "train",
) -> float:
    """One pass. `optimizer=None` means evaluation.

    How the loss is computed is the architecture's business, not this loop's -- see
    STEP_FUNCTIONS.

    Progress is logged DURING the epoch, not only at the end of it. On the full universe an
    epoch is over a thousand optimizer steps on a 231M-parameter model, so epoch-boundary
    logging alone leaves the cell silent for many minutes at a time -- indistinguishable
    from a hang, which is exactly how this looked on Colab.
    """
    total, seen = 0.0, 0
    accum = max(1, config.grad_accum_steps)
    # The loader always yields CPU tensors; the model may be anywhere.
    device = _model_device(model)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    try:
        n_batches = len(loader)
    except TypeError:                       # an iterable without a length
        n_batches = None
    # Around twenty updates per epoch: frequent enough to show movement, sparse enough not
    # to bury the epoch summaries.
    log_every = max(1, n_batches // 20) if n_batches else 0
    started = time.perf_counter()

    for index, (context, future) in enumerate(loader, start=1):
        context = context.to(device)
        future = future.to(device)

        loss = step_fn(model, context, future, config)

        if optimizer is not None:
            # Scale by the accumulation factor so the gradient matches what a single batch
            # of the effective size would have produced, rather than its sum.
            (loss / accum).backward()
            if index % accum == 0 or index == n_batches:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], config.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        total += loss.detach().item() * len(context)
        seen += len(context)
        del loss

        if log_every and index % log_every == 0:
            elapsed = time.perf_counter() - started
            remaining = elapsed / index * (n_batches - index)
            logger.info(
                "  %s epoch %s: batch %d/%d  loss %.6f  %s elapsed, ~%s left",
                phase,
                "?" if epoch is None else epoch,
                index,
                n_batches,
                total / max(seen, 1),
                _duration(elapsed),
                _duration(remaining),
            )

    return total / max(seen, 1)


def _duration(seconds: float) -> str:
    """Compact h/m/s, so an ETA reads at a glance rather than as four-digit seconds."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def _log_run(
    base_model_name: str,
    horizon: int,
    config: LoRAConfig,
    metrics: dict[str, float],
    history: list[dict[str, float]],
    targets: tuple[str, ...],
) -> str | None:
    """Log to MLflow. Never fatal -- a tracking outage must not lose a trained adapter."""
    try:
        import mlflow

        mlflow.set_tracking_uri("sqlite:///artifacts/mlflow.db")
        mlflow.set_experiment("rq1_forecast_quality")

        with mlflow.start_run(run_name=f"lora_{base_model_name}_h{horizon}") as run:
            mlflow.log_params(
                {
                    "model": base_model_name, "horizon": horizon, "lora_r": config.r,
                    "lora_alpha": config.lora_alpha, "lr": config.lr,
                    "batch_size": config.batch_size, "epochs": config.epochs,
                    "target_modules": ",".join(targets), "seed": config.seed,
                }
            )
            mlflow.log_metrics(metrics)
            for row in history:
                mlflow.log_metric("val_loss", row["val_loss"], step=int(row["epoch"]))
            return run.info.run_id
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow logging failed (adapter is still saved): %s", exc)
        return None


def _register_checkpoint(
    base_model_name: str,
    output_dir: Path,
    features: pd.DataFrame,
    horizon: int,
    metrics: dict[str, float],
    run_id: str | None,
) -> None:
    """Record the adapter in the provenance registry Component 3 anchors on-chain."""
    try:
        from features.feature_store import feature_columns
        from forecasting.model_registry import register as register_model

        stamps = pd.to_datetime(features["timestamp"])
        register_model(
            f"{base_model_name}-lora",
            output_dir,
            train_start=stamps.min().date(),
            train_end=stamps.max().date(),
            metrics=metrics,
            mlflow_run_id=run_id,
            activate=True,
            feature_columns=feature_columns(features),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("registry write failed (adapter is still saved): %s", exc)


def _default_train_window(features: pd.DataFrame) -> tuple[date, date]:
    stamps = pd.to_datetime(features["timestamp"])
    return stamps.min().date(), stamps.max().date()
