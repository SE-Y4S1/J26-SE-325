"""The LSTM/MLP residual head -- the learned half of the hybrid engine.

Predicts a CORRECTION to a foundation model's forecast rather than the return itself. It
sees what the foundation model structurally cannot: technical indicators, FinBERT sentiment,
asset class, and the base forecast itself (so it can learn *when* the base model is
unreliable, e.g. shrink the correction when the base p10-p90 band is already wide).

Asset class enters here as a learned categorical embedding rather than by fine-tuning three
separate foundation models. One shared model with an asset-class feature keeps the whole
training set together instead of splitting an already-small universe three ways, and lets
the head learn cross-asset structure (FX behaves unlike small-cap equity in a vol spike).
Tradeoff: a shared model cannot diverge as far per class as three specialists could.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from forecasting.base import DEFAULT_QUANTILES

# Asset classes, in a fixed order so embedding indices are stable across runs and
# checkpoints remain loadable.
ASSET_CLASS_ORDER = ("equity", "etf", "forex")


@dataclass
class ResidualHeadConfig:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    asset_class_embed_dim: int = 4
    lr: float = 1e-3
    epochs: int = 30
    batch_size: int = 64
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    seed: int = 42
    # Zero-init the output layer so the head starts as an exact no-op and the hybrid begins
    # at the base model's accuracy. Training can only improve on it from there, and RQ1's
    # "does the head help" question gets a clean answer rather than being confounded by a
    # randomly-initialised head that starts by making things worse.
    zero_init_output: bool = True


class ResidualHead(nn.Module):
    """LSTM over feature history + MLP over (base forecast, asset class) -> quantile deltas."""

    def __init__(self, n_features: int, n_asset_classes: int, config: ResidualHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.n_quantiles = len(config.quantiles)

        self.feature_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.asset_embedding = nn.Embedding(n_asset_classes, config.asset_class_embed_dim)

        fusion_input = config.hidden_size + config.asset_class_embed_dim + self.n_quantiles
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
        )
        self.output = nn.Linear(config.hidden_size // 2, self.n_quantiles)

        if config.zero_init_output:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(
        self,
        features: torch.Tensor,
        base_forecast: torch.Tensor,
        asset_class_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Returns per-quantile residual corrections, shape (batch, n_quantiles).

        `base_forecast` is fed in alongside the features precisely so the head can condition
        on the base model's own uncertainty -- a wide predicted interval is evidence that the
        base is unsure, which is information about how much to correct.
        """
        _, (hidden, _) = self.feature_lstm(features)
        encoded = hidden[-1]

        embedded = self.asset_embedding(asset_class_idx)
        fused = self.fusion(torch.cat([encoded, embedded, base_forecast], dim=1))
        return self.output(fused)


def asset_class_index(asset_class: str) -> int:
    """Stable index for an asset class string. Unknown classes map to equity."""
    try:
        return ASSET_CLASS_ORDER.index(str(asset_class).lower())
    except ValueError:
        return 0
