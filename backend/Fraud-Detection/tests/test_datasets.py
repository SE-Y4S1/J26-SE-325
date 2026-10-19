"""Dataset fixtures and the rule that synthetic output is not a benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_engine.data.behavior import add_label_free_simulated_biometrics
from fraud_engine.data.dgraphfin import load_dgraphfin
from fraud_engine.data.download import download_ieee, kaggle_credentials_present
from fraud_engine.data.elliptic import load_elliptic
from fraud_engine.data.ieee import load_ieee_frame
from fraud_engine.data.synthetic import make_synthetic_transactions
from fraud_engine.evaluation.benchmark import refuse_unearned_benchmark_flag


def test_simulated_biometrics_do_not_read_the_label() -> None:
    frame = make_synthetic_transactions(40, seed=3)
    first = add_label_free_simulated_biometrics(frame, seed=3)
    flipped = frame.copy()
    flipped["isFraud"] = 1 - flipped["isFraud"]
    second = add_label_free_simulated_biometrics(flipped, seed=3)
    assert first["simulated_signal_origin"].eq("simulated").all()
    assert np.allclose(first["simulated_typing_interval_ms"], second["simulated_typing_interval_ms"])


def test_synthetic_profile_cannot_be_called_a_benchmark() -> None:
    settings = {"dataset_name": "synthetic-fixture", "benchmark_valid": True}
    with pytest.raises(ValueError, match="research benchmark"):
        refuse_unearned_benchmark_flag(settings)


def test_download_stops_when_credentials_are_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fraud_engine.data.download.kaggle_credentials_present", lambda: False)
    with pytest.raises(RuntimeError, match="kaggle.json"):
        download_ieee()
    assert isinstance(kaggle_credentials_present(), bool)


def test_ieee_loader_joins_identity_and_records_checksums(tmp_path: Path) -> None:
    transactions = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "isFraud": [0, 1, 0, 1],
            "TransactionDT": [10, 20, 30, 40],
            "TransactionAmt": [5.0, 80.0, 6.0, 90.0],
            "card1": [11, 12, 11, 13],
            "ProductCD": ["W", "C", "W", "H"],
            "C1": [1, 4, 1, 5],
        }
    )
    identity = pd.DataFrame(
        {
            "TransactionID": [1, 2, 4],
            "DeviceType": ["mobile", "desktop", "mobile"],
            "DeviceInfo": ["dev-a", "dev-b", "dev-a"],
            "id_01": [-5, 0, -1],
        }
    )
    transactions.to_csv(tmp_path / "train_transaction.csv", index=False)
    identity.to_csv(tmp_path / "train_identity.csv", index=False)
    frame, metadata = load_ieee_frame(tmp_path, max_rows=None)
    assert list(frame["TransactionID"]) == [1, 2, 3, 4]
    assert frame.loc[frame["TransactionID"] == 3, "DeviceType"].isna().all()
    assert "DeviceInfo" in frame.columns
    assert metadata["files"][0]["sha256"]
    assert metadata["identity_joined"] is True
    assert "isFraud" in frame.columns


def test_elliptic_and_dgraph_loaders_read_local_fixtures(tmp_path: Path) -> None:
    features = tmp_path / "elliptic" / "elliptic_txs_features.csv"
    classes = tmp_path / "elliptic" / "elliptic_txs_classes.csv"
    edges = tmp_path / "elliptic" / "elliptic_txs_edgelist.csv"
    features.parent.mkdir()
    features.write_text("1,1,0.1,0.2\n2,31,0.3,0.4\n3,40,0.8,0.9\n", encoding="utf-8")
    classes.write_text("txId,class\n1,2\n2,1\n3,unknown\n", encoding="utf-8")
    edges.write_text("txId1,txId2\n1,2\n2,3\n", encoding="utf-8")
    elliptic = load_elliptic(tmp_path / "elliptic", max_nodes=None)
    assert elliptic["x"].shape[0] == 3
    assert elliptic["y"].tolist() == [0, 1, -1]
    assert elliptic["edge_index"].shape[0] == 2

    dgraph = tmp_path / "dgraph"
    dgraph.mkdir()
    np.savez(
        dgraph / "dgraphfin.npz",
        x=np.ones((6, 2), dtype=np.float32),
        y=np.array([0, 1, 0, 1, 0, 1]),
        edge_index=np.array([[0, 1], [1, 2], [2, 4], [3, 5]]),
        train_mask=np.array([0, 1]),
        valid_mask=np.array([2, 3]),
        test_mask=np.array([4, 5]),
    )
    loaded = load_dgraphfin(dgraph, max_nodes=None)
    assert loaded["x"].shape == (6, 2)
    assert set(loaded["split"].tolist()) == {0, 1, 2}
    assert loaded["dataset"] == "dgraphfin"
