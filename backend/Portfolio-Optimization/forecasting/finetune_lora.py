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
from dataclasses import dataclass, field
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


class WindowedSeriesDataset(Dataset):
    """(context, future) pairs drawn from the return series of each symbol."""

    def __init__(self, contexts: np.ndarray, futures: np.ndarray) -> None:
        self.contexts = torch.tensor(contexts, dtype=torch.float32)
        self.futures = torch.tensor(futures, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.contexts[index], self.futures[index]


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

    contexts: list[np.ndarray] = []
    futures: list[np.ndarray] = []

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

        returns = np.log(closes[1:] / closes[:-1]).astype(np.float32)
        for end in range(context_length, len(returns) - horizon + 1):
            contexts.append(returns[end - context_length : end])
            futures.append(returns[end : end + horizon])

    if not contexts:
        raise ValueError(
            f"no windows could be built: need at least {context_length + horizon + 1} bars "
            "for a single symbol"
        )

    logger.info("built %d fine-tuning windows (context=%d, horizon=%d)",
                len(contexts), context_length, horizon)
    return WindowedSeriesDataset(np.stack(contexts), np.stack(futures))


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


STEP_FUNCTIONS = {
    "chronos_bolt": _chronos_step,
}

# Architectures that cannot be LoRA fine-tuned by this module, with the reason. Checked
# before anything is loaded, so the failure costs a second rather than a model download
# followed by a traceback from inside the training loop.
UNSUPPORTED_LORA = {
    "timesfm": (
        "TimesFM 2.5 cannot be LoRA fine-tuned by this module. Its "
        "forward(inputs, masks) takes tensors that are already patched into "
        "input_patch_len blocks and already normalised by a running-statistics pass, and "
        "it returns raw embeddings and projections rather than a prediction. The path that "
        "assembles a usable forecast, decode(), runs entirely under torch.no_grad(), so "
        "there is nothing to backpropagate through. Supporting it means reimplementing "
        "TimesFM's patching, running-stats normalisation, output-head selection and "
        "denormalisation against an undocumented internal contract -- a piece of work in "
        "its own right, not a configuration change. Chronos-Bolt supports LoRA and is the "
        "documented fallback for RQ1; TimesFM still contributes its zero-shot row."
    ),
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

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size)

    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad], lr=config.lr
    )

    step_fn = STEP_FUNCTIONS.get(base_model_name, _generic_step)

    best_val, stagnant = float("inf"), 0
    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        peft_model.train()
        train_loss = _run_epoch(peft_model, train_loader, config, optimizer, step_fn)

        peft_model.eval()
        with torch.no_grad():
            val_loss = _run_epoch(peft_model, val_loader, config, None, step_fn)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logger.info("epoch %d: train %.6f  val %.6f", epoch, train_loss, val_loss)

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

    logger.info("saved LoRA adapter to %s", output_dir)
    return output_dir


def _run_epoch(model, loader, config: LoRAConfig, optimizer, step_fn) -> float:
    """One pass.  means evaluation.

    How the loss is computed is the architecture's business, not this loop's -- see
    STEP_FUNCTIONS.
    """
    total, seen = 0.0, 0

    for context, future in loader:
        if optimizer is not None:
            optimizer.zero_grad()

        loss = step_fn(model, context, future, config)

        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], config.max_grad_norm
            )
            optimizer.step()

        total += loss.detach().item() * len(context)
        seen += len(context)

    return total / max(seen, 1)


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
