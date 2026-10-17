"""LSTM autoencoder for short transaction sequences.

The encoder is trained only on legitimate training sequences. A high reconstruction
error means the new sequence does not look like those legitimate sequences.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, n_features)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        _encoded, (state, _cell) = self.encoder(sequences)
        repeated = state[-1].unsqueeze(1).repeat(1, sequences.size(1), 1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)


def build_sequences(
    features: np.ndarray,
    entity_ids: np.ndarray,
    times: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return padded windows, masks, and the row index each window ends on."""
    buckets: dict[str, list[int]] = {}
    for index, raw in enumerate(entity_ids):
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            key = f"__row_{index}"
        else:
            key = str(raw)
        buckets.setdefault(key, []).append(index)
    windows = []
    masks = []
    owners = []
    width = features.shape[1]
    for indexes in buckets.values():
        ordered = sorted(indexes, key=lambda item: (times[item], item))
        for position, current in enumerate(ordered):
            chosen = ordered[max(0, position - seq_len + 1) : position + 1]
            window = np.zeros((seq_len, width), dtype=np.float32)
            mask = np.zeros(seq_len, dtype=np.float32)
            offset = seq_len - len(chosen)
            for step, row_index in enumerate(chosen):
                window[offset + step] = features[row_index]
                mask[offset + step] = 1.0
            windows.append(window)
            masks.append(mask)
            owners.append(current)
    if not windows:
        empty = np.zeros((0, seq_len, width), dtype=np.float32)
        return empty, np.zeros((0, seq_len), dtype=np.float32), np.zeros(0, dtype=np.int64)
    return np.stack(windows), np.stack(masks), np.asarray(owners, dtype=np.int64)


def train_autoencoder(
    model: LSTMAutoencoder,
    sequences: np.ndarray,
    masks: np.ndarray,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    patience: int,
    seed: int,
) -> list[float]:
    if len(sequences) == 0:
        raise ValueError("No legitimate sequences are available for the autoencoder.")
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    waiting = 0
    model.train()
    for _epoch in range(epochs):
        permutation = torch.randperm(len(sequences))
        total = 0.0
        seen = 0
        for start in range(0, len(sequences), batch_size):
            batch_index = permutation[start : start + batch_size]
            batch = torch.tensor(sequences[batch_index.numpy()], dtype=torch.float32)
            mask = torch.tensor(masks[batch_index.numpy()], dtype=torch.float32)
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = _masked_loss(reconstructed, batch, mask)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch_index)
            seen += len(batch_index)
        epoch_loss = total / max(seen, 1)
        losses.append(epoch_loss)
        if epoch_loss + 1e-6 < best_loss:
            best_loss = epoch_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            waiting = 0
        else:
            waiting += 1
            if waiting >= patience:
                break
    model.load_state_dict(best_state)
    return losses


def reconstruction_errors(model: LSTMAutoencoder, sequences: np.ndarray, masks: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    errors = np.zeros(len(sequences), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = torch.tensor(sequences[start : start + batch_size], dtype=torch.float32)
            mask = torch.tensor(masks[start : start + batch_size], dtype=torch.float32)
            reconstructed = model(batch)
            squared = ((reconstructed - batch) ** 2).mean(dim=2)
            weighted = (squared * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            errors[start : start + len(batch)] = weighted.cpu().numpy()
    return errors


def errors_to_scores(errors: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Map reconstruction error onto 0..1 using legitimate training percentiles."""
    low = float(np.quantile(reference, 0.05)) if len(reference) else 0.0
    high = float(np.quantile(reference, 0.95)) if len(reference) else 1.0
    if high - low < 1e-6:
        high = low + 1.0
    scores = np.clip((errors - low) / (high - low), 0.0, 1.0)
    return scores, low, high


def _masked_loss(reconstructed: torch.Tensor, original: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared = ((reconstructed - original) ** 2).mean(dim=2)
    weighted = (squared * mask).sum() / mask.sum().clamp_min(1.0)
    return weighted
