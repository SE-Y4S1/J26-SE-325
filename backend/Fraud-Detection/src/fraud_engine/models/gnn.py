"""Small graph neural networks written in PyTorch.

PyTorch Geometric is not required. Static GCN and GAT are comparison models.
GraphSAGE-style mean aggregation is the relational stream used by fusion because
the same neighbourhood can be built while a transaction is arriving.
Edges are directed from the past toward the transaction being scored.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _self_loops(edge_index: torch.Tensor, node_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    loops = torch.arange(node_count, dtype=torch.long)
    if edge_index.numel() == 0:
        return loops, loops
    source, target = edge_index
    return torch.cat([source, loops]), torch.cat([target, loops])


def _messages(values: torch.Tensor, source: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(values)
    output.index_add_(0, target, values[source] * weights.unsqueeze(1))
    return output


class GraphConvNet(nn.Module):
    """Two-layer graph convolution with causal, directed edges."""

    def __init__(self, n_features: int, hidden: int = 32) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.linear_1 = nn.Linear(n_features, hidden)
        self.linear_2 = nn.Linear(hidden, 1)

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        source, target, weights = self._normalized(edge_index, features.size(0))
        hidden = F.relu(self.linear_1(_messages(features, source, target, weights)))
        source, target, weights = self._normalized(edge_index, features.size(0))
        pooled = _messages(hidden, source, target, weights)
        return self.linear_2(pooled).squeeze(-1)

    @staticmethod
    def _normalized(edge_index: torch.Tensor, node_count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source, target = _self_loops(edge_index, node_count)
        in_degree = torch.bincount(target, minlength=node_count).float().clamp_min(1.0)
        out_degree = torch.bincount(source, minlength=node_count).float().clamp_min(1.0)
        weights = (out_degree[source] * in_degree[target]).sqrt().reciprocal()
        return source, target, weights


class GraphAttentionNet(nn.Module):
    """Two-layer, single-head graph attention. One head keeps the laptop run small."""

    def __init__(self, n_features: int, hidden: int = 32) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.linear_1 = nn.Linear(n_features, hidden)
        self.attention_1 = nn.Parameter(torch.empty(2 * hidden))
        self.linear_2 = nn.Linear(hidden, hidden)
        self.attention_2 = nn.Parameter(torch.empty(2 * hidden))
        self.output = nn.Linear(hidden, 1)
        nn.init.xavier_uniform_(self.attention_1.unsqueeze(0))
        nn.init.xavier_uniform_(self.attention_2.unsqueeze(0))

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = F.elu(self._attend(features, edge_index, self.linear_1, self.attention_1))
        hidden = self._attend(hidden, edge_index, self.linear_2, self.attention_2)
        return self.output(hidden).squeeze(-1)

    @staticmethod
    def _attend(
        features: torch.Tensor,
        edge_index: torch.Tensor,
        linear: nn.Linear,
        attention: torch.Tensor,
    ) -> torch.Tensor:
        transformed = linear(features)
        source, target = _self_loops(edge_index, transformed.size(0))
        paired = torch.cat([transformed[source], transformed[target]], dim=1)
        logits = F.leaky_relu((paired * attention).sum(dim=1), negative_slope=0.2)
        weights = _grouped_softmax(logits, target, transformed.size(0))
        return _messages(transformed, source, target, weights)


class MeanAggregatorNet(nn.Module):
    """Inductive mean aggregator used as the GNN-only stream and in fusion."""

    def __init__(self, n_features: int, hidden: int = 32) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.self_1 = nn.Linear(n_features, hidden)
        self.neighbor_1 = nn.Linear(n_features, hidden)
        self.self_2 = nn.Linear(hidden, hidden)
        self.neighbor_2 = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, 1)

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.self_1(features) + self.neighbor_1(_mean_aggregate(features, edge_index)))
        hidden = F.relu(self.self_2(hidden) + self.neighbor_2(_mean_aggregate(hidden, edge_index)))
        return self.output(hidden).squeeze(-1)


def _mean_aggregate(features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    source, target = _self_loops(edge_index, features.size(0))
    summed = torch.zeros_like(features)
    summed.index_add_(0, target, features[source])
    degree = torch.bincount(target, minlength=features.size(0)).float().clamp_min(1.0).unsqueeze(1)
    return summed / degree


def _grouped_softmax(logits: torch.Tensor, target: torch.Tensor, node_count: int) -> torch.Tensor:
    maximum = torch.full((node_count,), float("-inf"), device=logits.device)
    maximum.scatter_reduce_(0, target, logits, reduce="amax", include_self=True)
    # Isolated safety: amax of an empty group stays -inf. Self-loops prevent that.
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    exponent = torch.exp(logits - maximum[target])
    denominator = torch.zeros(node_count, device=logits.device).index_add_(0, target, exponent).clamp_min(1e-9)
    return exponent / denominator[target]


def train_classifier(
    model: nn.Module,
    features: np.ndarray,
    edge_index: np.ndarray,
    labels: np.ndarray,
    train_index: np.ndarray,
    val_index: np.ndarray,
    epochs: int,
    learning_rate: float,
    patience: int,
    seed: int,
    max_pos_weight: float,
    val_edge_index: np.ndarray | None = None,
) -> list[float]:
    """Train on `edge_index`. Early stopping may use a wider validation graph.

    Pass training edges that stay inside the training nodes. Pass validation edges
    that can point from training nodes into validation nodes, but not from the test set.
    """
    torch.manual_seed(seed)
    tensor_x = torch.tensor(features, dtype=torch.float32)
    tensor_edges = torch.tensor(edge_index, dtype=torch.long)
    validation_edges = tensor_edges if val_edge_index is None else torch.tensor(val_edge_index, dtype=torch.long)
    tensor_y = torch.tensor(labels, dtype=torch.float32)
    train_rows = torch.tensor(train_index, dtype=torch.long)
    positives = float(labels[train_index].sum())
    negatives = float(len(train_index) - positives)
    pos_weight_value = min(max_pos_weight, negatives / max(positives, 1.0))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_score = float("-inf")
    waiting = 0
    history: list[float] = []
    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(tensor_x, tensor_edges)
        loss = criterion(logits[train_rows], tensor_y[train_rows])
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
        score = _validation_auroc(model, tensor_x, validation_edges, tensor_y, val_index)
        if score > best_score + 1e-4:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            waiting = 0
        else:
            waiting += 1
            if waiting >= patience:
                break
    model.load_state_dict(best_state)
    return history


def predict_proba(model: nn.Module, features: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(edge_index, dtype=torch.long),
        )
        return torch.sigmoid(logits).cpu().numpy().astype(np.float64)


def _validation_auroc(
    model: nn.Module,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    val_index: np.ndarray,
) -> float:
    if len(val_index) == 0 or len(np.unique(labels[val_index].cpu().numpy())) < 2:
        return 0.0
    from sklearn.metrics import roc_auc_score

    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(features, edge_index))[torch.tensor(val_index, dtype=torch.long)]
    try:
        return float(roc_auc_score(labels[val_index].cpu().numpy(), scores.cpu().numpy()))
    except ValueError:
        return 0.0
