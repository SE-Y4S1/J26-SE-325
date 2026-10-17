"""Preprocessing and leakage checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_engine.data.preprocess import TabularPreprocessor, assign_time_splits, splits_are_time_ordered
from fraud_engine.data.synthetic import make_synthetic_transactions
from fraud_engine.data.validate import profile_frame


def test_timestamps_are_not_shared_across_splits() -> None:
    times = np.array([1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10])
    split = assign_time_splits(times, 0.7, 0.15)
    assert set(split.tolist()) == {0, 1, 2}
    for left, right in ((0, 1), (1, 2), (0, 2)):
        assert set(times[split == left]).isdisjoint(set(times[split == right]))
    assert splits_are_time_ordered(times, split)


def test_label_and_clock_are_not_model_features() -> None:
    frame = make_synthetic_transactions(60, seed=1)
    split = assign_time_splits(frame["TransactionDT"].to_numpy(), 0.7, 0.15)
    model = TabularPreprocessor().fit(frame, split == 0)
    assert "isFraud" not in model.feature_names
    assert "TransactionDT" not in model.feature_names
    assert "TransactionID" not in model.feature_names
    transformed = model.transform(frame)
    assert transformed.shape == (len(frame), len(model.feature_names))
    assert np.isfinite(transformed).all()


def test_unseen_category_and_train_median_do_not_use_the_test_rows() -> None:
    frame = pd.DataFrame(
        {
            "TransactionAmt": [1.0, 2.0, 3.0, 1000.0],
            "ProductCD": ["W", "W", "C", "NEW"],
            "isFraud": [0, 0, 1, 1],
        }
    )
    train_mask = np.array([True, True, True, False])
    model = TabularPreprocessor().fit(frame, train_mask)
    transformed = model.transform(frame)
    amount_index = model.feature_names.index("TransactionAmt")
    # The huge test amount must not change the training median of 2.
    assert model.medians["TransactionAmt"] == 2.0
    assert transformed[3, amount_index] > transformed[0, amount_index]
    frequency_index = model.feature_names.index("freq__ProductCD")
    assert transformed[3, frequency_index] == 0.0


def test_profile_reports_imbalance_and_duplicates() -> None:
    frame = pd.DataFrame(
        {
            "TransactionID": [1, 1, 2],
            "isFraud": [0, 0, 1],
            "TransactionAmt": [5.0, None, 9.0],
        }
    )
    profile = profile_frame(frame, label_column="isFraud", id_column="TransactionID")
    assert profile["rows"] == 3
    assert profile["duplicate_ids"] == 1
    assert profile["positive_rate"] == 1 / 3
    assert profile["missing_fraction"]["TransactionAmt"] > 0
