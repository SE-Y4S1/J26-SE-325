"""Model input and output shapes."""

from __future__ import annotations

import numpy as np
import torch

from fraud_engine.data.graphs import causal_edges
from fraud_engine.models.fusion import ScoreFusion
from fraud_engine.models.gnn import GraphAttentionNet, GraphConvNet, MeanAggregatorNet
from fraud_engine.models.lstm_ae import LSTMAutoencoder, build_sequences, reconstruction_errors


def test_edges_point_forward_in_time() -> None:
    times = np.array([1.0, 2.0, 3.0, 4.0])
    entities = np.array(["a", "a", "a", "b"])
    edges = causal_edges([entities], times, max_neighbors=2)
    assert edges.shape[0] == 2
    for source, target in edges.T:
        assert times[source] <= times[target]
        assert entities[source] == entities[target]
        assert source != target


def test_sequence_windows_end_on_each_row() -> None:
    features = np.arange(12, dtype=np.float32).reshape(4, 3)
    entities = np.array(["a", "a", "a", "a"])
    times = np.arange(4)
    sequences, masks, owners = build_sequences(features, entities, times, seq_len=3)
    assert sequences.shape == (4, 3, 3)
    assert masks.shape == (4, 3)
    assert owners.tolist() == [0, 1, 2, 3]
    assert masks[0].sum() == 1
    assert masks[-1].sum() == 3


def test_neural_models_emit_one_score_per_row() -> None:
    torch.manual_seed(0)
    features = torch.randn(5, 4)
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    for model in (GraphConvNet(4, 6), GraphAttentionNet(4, 6), MeanAggregatorNet(4, 6)):
        output = model(features, edges)
        assert output.shape == (5,)
        assert torch.isfinite(output).all()
    sequences = torch.randn(2, 4, 4)
    reconstructed = LSTMAutoencoder(4, 5)(sequences)
    assert reconstructed.shape == sequences.shape
    errors = reconstruction_errors(
        LSTMAutoencoder(4, 5),
        sequences.detach().numpy(),
        np.ones((2, 4), dtype=np.float32),
    )
    assert errors.shape == (2,)
    assert np.isfinite(errors).all()


def test_fusion_scores_are_probabilities() -> None:
    fusion = ScoreFusion(seed=0)
    first = np.array([0.1, 0.2, 0.8, 0.9])
    second = np.array([0.2, 0.1, 0.7, 0.95])
    labels = np.array([0, 0, 1, 1])
    fusion.fit(first, second, labels)
    scores = fusion.score(first, second)
    assert scores.shape == (4,)
    assert np.all((scores >= 0) & (scores <= 1))
