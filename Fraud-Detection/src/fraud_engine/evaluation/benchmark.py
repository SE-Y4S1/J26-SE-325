"""Train the baselines and the fused model, then write only executed metrics.

A result is marked benchmark_valid only for ieee-cis, elliptic, or dgraphfin, and
only when the loader recorded source files and checksums. Synthetic runs cannot
set that flag.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_engine.data.graphs import causal_edges
from fraud_engine.data.preprocess import (
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    TabularPreprocessor,
    assign_time_splits,
    splits_are_time_ordered,
)
from fraud_engine.evaluation.adversarial import camouflage, evaluate_feature_attack, evaluate_structuring, slow_drift
from fraud_engine.evaluation.metrics import (
    best_f1_threshold,
    classification_metrics,
    gateway_outcome,
    slice_block_rates,
    time_scorer,
)
from fraud_engine.evaluation.plots import save_metric_chart
from fraud_engine.gateway.policy import decide_many, threshold_document, thresholds_from_legitimate_scores
from fraud_engine.models.baselines import fit_forest, fit_logistic, fit_svm, model_parameters, positive_scores
from fraud_engine.models.bundle import save_bundle
from fraud_engine.models.fusion import ScoreFusion
from fraud_engine.models.gnn import (
    GraphAttentionNet,
    GraphConvNet,
    MeanAggregatorNet,
    predict_proba,
    train_classifier,
)
from fraud_engine.models.lstm_ae import (
    LSTMAutoencoder,
    build_sequences,
    errors_to_scores,
    reconstruction_errors,
    train_autoencoder,
)
from fraud_engine.seed import set_seed

REAL_DATASETS = {"ieee-cis", "elliptic", "dgraphfin"}
MODEL_ORDER = [
    "logistic_regression",
    "svm",
    "random_forest",
    "static_gcn",
    "static_gat",
    "lstm_only",
    "gnn_only",
    "fused_lstm_gnn",
]


def refuse_unearned_benchmark_flag(settings: dict[str, Any]) -> None:
    name = str(settings.get("dataset_name"))
    if settings.get("benchmark_valid") and name not in REAL_DATASETS:
        raise ValueError(
            f"Refusing to mark dataset '{name}' as a research benchmark. "
            "Only executed runs on ieee-cis, elliptic, or dgraphfin can set benchmark_valid."
        )
    if name.startswith("synthetic"):
        settings["benchmark_valid"] = False


def run_transaction_experiment(
    frame: pd.DataFrame,
    settings: dict[str, Any],
    output_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score an IEEE-shaped transaction table."""
    refuse_unearned_benchmark_flag(settings)
    if settings.get("benchmark_valid"):
        if not metadata or not metadata.get("files"):
            raise ValueError("A real benchmark requires source-file names, sizes, and checksums.")
    set_seed(int(settings["seed"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = frame["isFraud"].to_numpy(dtype=int)
    times = pd.to_numeric(frame["TransactionDT"], errors="coerce").to_numpy(dtype=float)
    split = assign_time_splits(times, settings["split"]["train"], settings["split"]["val"])
    ordered = splits_are_time_ordered(times, split)
    preprocessor = TabularPreprocessor().fit(frame, split == SPLIT_TRAIN)
    features = preprocessor.transform(frame)
    train_index = np.flatnonzero(split == SPLIT_TRAIN)
    val_index = _time_order(np.flatnonzero(split == SPLIT_VAL), times)
    test_index = np.flatnonzero(split == SPLIT_TEST)
    calibration_index, threshold_index = _half_split(val_index)
    entity = frame["card1"].to_numpy() if "card1" in frame.columns else np.arange(len(frame))
    groups = [entity]
    if "DeviceInfo" in frame.columns:
        groups.append(frame["DeviceInfo"].to_numpy())
    edges = causal_edges(groups, times, int(settings["models"]["max_neighbors"]))
    train_edges = _edges_where(edges, split == SPLIT_TRAIN, split == SPLIT_TRAIN)
    val_edges = _edges_where(edges, split <= SPLIT_VAL, split <= SPLIT_VAL)
    eval_edges = _edges_respecting_split(edges, split)

    rows: list[dict[str, Any]] = []
    score_bank: dict[str, np.ndarray] = {}
    fitted: dict[str, Any] = {}

    def _add_tabular(name: str, model: Any) -> None:
        fitted[name] = model
        scores = positive_scores(model, features)
        score_bank[name] = scores
        rows.append(
            _row_from_scores(
                name,
                model_parameters(model),
                labels,
                scores,
                val_index,
                threshold_index,
                test_index,
                settings,
                "single-transaction tabular features; scaler and class weights fit on the training split only",
            )
        )

    try:
        _add_tabular("logistic_regression", fit_logistic(features[train_index], labels[train_index], settings["seed"]))
    except Exception as exc:  # noqa: BLE001 - one model failure should not hide the others
        rows.append(_failed("logistic_regression", exc))
    try:
        _add_tabular(
            "svm",
            fit_svm(
                features[train_index],
                labels[train_index],
                settings["seed"],
                int(settings["models"]["svm_max_iter"]),
            ),
        )
    except Exception as exc:  # noqa: BLE001
        rows.append(_failed("svm", exc))
    try:
        _add_tabular(
            "random_forest",
            fit_forest(
                features[train_index],
                labels[train_index],
                settings["seed"],
                int(settings["models"]["rf_trees"]),
                int(settings["models"]["rf_depth"]),
            ),
        )
    except Exception as exc:  # noqa: BLE001
        rows.append(_failed("random_forest", exc))

    neural_notes = (
        "Directed edges go from earlier transactions to later ones that share card1 or DeviceInfo. "
        "Training edges stay inside the training split."
    )
    gnn_kwargs = {"n_features": features.shape[1], "hidden": int(settings["models"]["hidden"])}
    for name, network in (
        ("static_gcn", GraphConvNet(**gnn_kwargs)),
        ("static_gat", GraphAttentionNet(**gnn_kwargs)),
        ("gnn_only", MeanAggregatorNet(**gnn_kwargs)),
    ):
        try:
            history = train_classifier(
                network,
                features,
                train_edges,
                labels,
                train_index,
                threshold_index,
                int(settings["models"]["epochs"]),
                float(settings["models"]["lr"]),
                int(settings["models"]["patience"]),
                int(settings["seed"]),
                float(settings["models"]["max_pos_weight"]),
                val_edge_index=val_edges,
            )
            scores = predict_proba(network, features, eval_edges)
            score_bank[name] = scores
            fitted[name] = network
            row = _row_from_scores(
                name,
                {"hidden": gnn_kwargs["hidden"], "epochs_ran": len(history), "learning_rate": settings["models"]["lr"]},
                labels,
                scores,
                val_index,
                threshold_index,
                test_index,
                settings,
                neural_notes,
            )
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            rows.append(_failed(name, exc))

    lstm_norm = {"low": 0.0, "high": 1.0}
    try:
        sequences, masks, owners = build_sequences(
            features, entity, times, int(settings["models"]["seq_len"])
        )
        legitimate = (split[owners] == SPLIT_TRAIN) & (labels[owners] == 0)
        if int(legitimate.sum()) < 8:
            raise ValueError("Fewer than 8 legitimate training sequences.")
        lstm = LSTMAutoencoder(features.shape[1], int(settings["models"]["hidden"]))
        history = train_autoencoder(
            lstm,
            sequences[legitimate],
            masks[legitimate],
            int(settings["models"]["epochs"]),
            float(settings["models"]["lr"]),
            int(settings["models"]["batch_size"]),
            int(settings["models"]["patience"]),
            int(settings["seed"]),
        )
        errors = reconstruction_errors(lstm, sequences, masks, int(settings["models"]["batch_size"]))
        sequence_scores, low, high = errors_to_scores(errors, errors[legitimate])
        lstm_norm = {"low": low, "high": high}
        row_scores = np.full(len(frame), np.nan, dtype=np.float64)
        row_scores[owners] = sequence_scores
        fill = float(np.nanmedian(row_scores[train_index]))
        row_scores = np.where(np.isnan(row_scores), fill, row_scores)
        score_bank["lstm_only"] = row_scores
        fitted["lstm_only"] = lstm
        rows.append(
            _row_from_scores(
                "lstm_only",
                {
                    "hidden": int(settings["models"]["hidden"]),
                    "sequence_length": int(settings["models"]["seq_len"]),
                    "epochs_ran": len(history),
                    "trained_on": "legitimate training sequences only",
                },
                labels,
                row_scores,
                val_index,
                threshold_index,
                test_index,
                settings,
                "Reconstruction error is scaled with the 5th and 95th percentiles of legitimate training errors.",
            )
        )
    except Exception as exc:  # noqa: BLE001
        rows.append(_failed("lstm_only", exc))

    fusion = ScoreFusion(seed=int(settings["seed"]))
    thresholds = None
    try:
        if "lstm_only" not in score_bank or "gnn_only" not in score_bank:
            raise ValueError("Fusion needs both the LSTM stream and the GNN stream.")
        fusion.fit(
            score_bank["lstm_only"][calibration_index],
            score_bank["gnn_only"][calibration_index],
            labels[calibration_index],
        )
        fused_scores = fusion.score(score_bank["lstm_only"], score_bank["gnn_only"])
        score_bank["fused_lstm_gnn"] = fused_scores
        thresholds = thresholds_from_legitimate_scores(
            fused_scores[threshold_index],
            labels[threshold_index],
            float(settings["gateway"]["step_up_fpr"]),
            float(settings["gateway"]["block_fpr"]),
        )
        row = _row_from_scores(
            "fused_lstm_gnn",
            {"fusion": fusion.to_dict(), "threshold_rule": threshold_document(thresholds)},
            labels,
            fused_scores,
            val_index,
            threshold_index,
            test_index,
            settings,
            "The LSTM/GNN mixture weight is chosen on the earlier validation slice. A stream that ranks backwards there is reversed. Gateway cut-points use the later validation slice.",
        )
        rows.append(row)
    except Exception as exc:  # noqa: BLE001
        rows.append(_failed("fused_lstm_gnn", exc))

    attacks = _attacks(
        features,
        labels,
        train_index,
        threshold_index,
        test_index,
        score_bank.get("fused_lstm_gnn"),
        score_bank.get("logistic_regression"),
        fitted.get("logistic_regression"),
        preprocessor.feature_names,
        settings,
    )
    lift = _device_lift(features, labels, train_index, test_index, preprocessor, settings)
    slices = {}
    if thresholds and "fused_lstm_gnn" in score_bank and "DeviceType" in frame.columns:
        decisions = decide_many(score_bank["fused_lstm_gnn"][test_index], thresholds)
        slices = slice_block_rates(labels[test_index], decisions, frame["DeviceType"].to_numpy()[test_index])

    bundle_path = None
    if {"lstm_only", "gnn_only", "logistic_regression"} <= fitted.keys() and thresholds is not None:
        bundle_path = output_dir / "fraud_bundle.joblib"
        save_bundle(
            bundle_path,
            {
                "schema_version": 1,
                "dataset_name": settings["dataset_name"],
                "benchmark_valid": bool(settings.get("benchmark_valid")),
                "seed": int(settings["seed"]),
                "preprocessor": preprocessor.to_dict(),
                "thresholds": thresholds,
                "lstm_state": fitted["lstm_only"].state_dict(),
                "lstm_kwargs": {"n_features": features.shape[1], "hidden": int(settings["models"]["hidden"])},
                "gnn_state": fitted["gnn_only"].state_dict(),
                "gnn_kwargs": gnn_kwargs,
                "lstm_norm": lstm_norm,
                "seq_len": int(settings["models"]["seq_len"]),
                "fusion_alpha": fusion.alpha,
                "fusion_flip_first": fusion.flip_first,
                "fusion_flip_second": fusion.flip_second,
                "fusion_fallback": fusion.fallback,
                "fusion_weights": fusion.weights,
                "logistic_model": fitted["logistic_regression"],
                "model_version": f"{settings['dataset_name']}-seed{settings['seed']}",
            },
        )

    scoring_latency: dict[str, Any] = {}
    if "logistic_regression" in fitted and len(test_index):
        scoring_latency["logistic_regression"] = time_scorer(
            lambda matrix: positive_scores(fitted["logistic_regression"], matrix),
            features[test_index],
        )
    if "gnn_only" in fitted:
        started = time.perf_counter()
        predict_proba(fitted["gnn_only"], features, eval_edges)
        elapsed = max(time.perf_counter() - started, 1e-9)
        scoring_latency["gnn_only_full_graph"] = {
            "batch_throughput_rows_per_s": float(len(features) / elapsed),
            "batch_seconds": float(elapsed),
            "rows": int(len(features)),
            "notes": [
                "One forward pass over the whole causal graph. This is not the time to score one transaction on its own."
            ],
        }

    summary = {
        "dataset": settings["dataset_name"],
        "benchmark_valid": bool(settings.get("benchmark_valid")),
        "results_are_research_benchmark": bool(settings.get("benchmark_valid")),
        "split_strategy": "time-ordered TransactionDT with no shared timestamp across splits" if ordered else "index fallback because timestamps were not separable",
        "time_splits_are_ordered": ordered,
        "rows": int(len(frame)),
        "features": preprocessor.feature_names,
        "class_balance": {
            "train_fraud_rate": _rate(labels[train_index]),
            "validation_fraud_rate": _rate(labels[val_index]),
            "test_fraud_rate": _rate(labels[test_index]),
        },
        "imbalance_method": "sklearn class_weight on training rows; neural pos_weight clipped by max_pos_weight; LSTM autoencoder trained on legitimate training sequences. The test split is not resampled.",
        "gateway_thresholds": thresholds,
        "device_feature_lift": lift,
        "slice_block_rates": slices,
        "adversarial": attacks,
        "scoring_latency": scoring_latency,
        "models": rows,
        "source_metadata": metadata,
        "artifact": str(bundle_path) if bundle_path else None,
        "notes": [
            "Precision, recall, F1, and the confusion matrix use a threshold chosen by validation F1.",
            "ALLOW, STEP_UP, and BLOCK use a separate false-positive budget on a later validation slice.",
            "Simulated biometric columns are not part of this run unless they were already present in the frame.",
        ],
    }
    if not settings.get("benchmark_valid"):
        summary["notes"].append(
            "This run is not a research benchmark. Do not cite these scores as IEEE-CIS, DGraphFin, or Elliptic results."
        )
    _write_outputs(summary, output_dir, settings)
    return summary


def run_graph_experiment(
    prepared: dict[str, Any],
    settings: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Train tabular and graph baselines on Elliptic or DGraphFin."""
    dataset_name = prepared["dataset"]
    local_settings = dict(settings)
    local_settings["dataset_name"] = dataset_name
    local_settings["benchmark_valid"] = dataset_name in REAL_DATASETS and bool(prepared.get("files"))
    refuse_unearned_benchmark_flag(local_settings)
    set_seed(int(settings["seed"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    features = prepared["x"].astype(np.float32)
    labels = prepared["y"].astype(int)
    split = prepared["split"].astype(int)
    edges = prepared["edge_index"]
    labelled = labels >= 0
    train_nodes = (split == SPLIT_TRAIN) & labelled
    val_nodes = (split == SPLIT_VAL) & labelled
    test_nodes = (split == SPLIT_TEST) & labelled
    mean = features[train_nodes].mean(axis=0)
    scale = features[train_nodes].std(axis=0)
    scale[scale < 1e-6] = 1.0
    features = np.clip((features - mean) / scale, -8.0, 8.0).astype(np.float32)
    train_index = np.flatnonzero(train_nodes)
    val_index = np.flatnonzero(val_nodes)
    test_index = np.flatnonzero(test_nodes)
    calibration_index, threshold_index = _half_split(val_index)
    train_edges = _edges_where(edges, split == SPLIT_TRAIN, split == SPLIT_TRAIN)
    val_edges = _edges_where(edges, split <= SPLIT_VAL, split <= SPLIT_VAL)
    eval_edges = _edges_respecting_split(edges, split)
    rows: list[dict[str, Any]] = []
    hidden = int(settings["models"]["hidden"])

    def _tabular(name: str, model: Any, note: str) -> None:
        scores = positive_scores(model, features)
        rows.append(
            _row_from_scores(
                name,
                model_parameters(model),
                labels,
                scores,
                val_index,
                threshold_index,
                test_index,
                local_settings,
                note,
            )
        )

    note = f"{prepared.get('subset')}. Node features are scaled with training-node statistics."
    try:
        _tabular("logistic_regression", fit_logistic(features[train_index], labels[train_index], settings["seed"]), note)
    except Exception as exc:  # noqa: BLE001
        rows.append(_failed("logistic_regression", exc))
    try:
        _tabular(
            "svm",
            fit_svm(features[train_index], labels[train_index], settings["seed"], int(settings["models"]["svm_max_iter"])),
            note,
        )
    except Exception as exc:  # noqa: BLE001
        rows.append(_failed("svm", exc))
    try:
        _tabular(
            "random_forest",
            fit_forest(
                features[train_index],
                labels[train_index],
                settings["seed"],
                int(settings["models"]["rf_trees"]),
                int(settings["models"]["rf_depth"]),
            ),
            note,
        )
    except Exception as exc:  # noqa: BLE001
        rows.append(_failed("random_forest", exc))

    graph_scores = {}
    for name, network in (
        ("static_gcn", GraphConvNet(features.shape[1], hidden)),
        ("static_gat", GraphAttentionNet(features.shape[1], hidden)),
        ("gnn_only", MeanAggregatorNet(features.shape[1], hidden)),
    ):
        try:
            history = train_classifier(
                network,
                features,
                train_edges,
                np.clip(labels, 0, 1),
                train_index,
                threshold_index,
                int(settings["models"]["epochs"]),
                float(settings["models"]["lr"]),
                int(settings["models"]["patience"]),
                int(settings["seed"]),
                float(settings["models"]["max_pos_weight"]),
                val_edge_index=val_edges,
            )
            scores = predict_proba(network, features, eval_edges)
            graph_scores[name] = scores
            rows.append(
                _row_from_scores(
                    name,
                    {"hidden": hidden, "epochs_ran": len(history)},
                    labels,
                    scores,
                    val_index,
                    threshold_index,
                    test_index,
                    local_settings,
                    "Training edges have both ends in the training split so test node features cannot update the weights.",
                )
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(_failed(name, exc))
    rows.append(
        _skipped(
            "lstm_only",
            "This dataset is a node graph, not a per-account transaction sequence, so the LSTM autoencoder is not applicable.",
        )
    )
    rows.append(
        _skipped(
            "fused_lstm_gnn",
            "Fusion needs an LSTM sequence score. It is reported on the IEEE-CIS transaction stream.",
        )
    )
    summary = {
        "dataset": dataset_name,
        "benchmark_valid": bool(local_settings["benchmark_valid"]),
        "results_are_research_benchmark": bool(local_settings["benchmark_valid"]),
        "split_strategy": prepared.get("subset"),
        "rows": int(len(labels)),
        "models": rows,
        "source_metadata": {
            "source": prepared.get("source"),
            "license": prepared.get("license"),
            "files": prepared.get("files"),
            "notes": prepared.get("notes"),
        },
        "graph_models_trained": list(graph_scores),
        "notes": ["Metrics ignore nodes whose label is unknown (-1)."],
    }
    _write_outputs(summary, output_dir, local_settings)
    return summary


def _attacks(
    features: np.ndarray,
    labels: np.ndarray,
    train_index: np.ndarray,
    threshold_index: np.ndarray,
    test_index: np.ndarray,
    fused_scores: np.ndarray | None,
    logistic_scores: np.ndarray | None,
    logistic_model: Any,
    feature_names: list[str],
    settings: dict[str, Any],
) -> dict[str, Any]:
    if logistic_model is None or logistic_scores is None:
        return {"notes": ["Adversarial rescoring needs the trained logistic regression baseline."]}
    try:
        thresholds = thresholds_from_legitimate_scores(
            logistic_scores[threshold_index],
            labels[threshold_index],
            float(settings["gateway"]["step_up_fpr"]),
            float(settings["gateway"]["block_fpr"]),
        )
    except ValueError as exc:
        return {"notes": [f"Adversarial gateway thresholds were not selected: {exc}"]}
    test_features = features[test_index]
    test_labels = labels[test_index]
    legitimate = labels[train_index] == 0
    legitimate_mean = features[train_index][legitimate].mean(axis=0)
    from fraud_engine.evaluation.metrics import safe_auroc

    clean_auroc, _note = safe_auroc(test_labels, logistic_scores[test_index])
    score_fn = lambda matrix: positive_scores(logistic_model, matrix)  # noqa: E731
    alpha = float(settings["adversarial"]["camouflage_alpha"])
    magnitude = float(settings["adversarial"]["drift_magnitude"])
    parts = int(settings["adversarial"]["structuring_parts"])
    camouflaged = camouflage(test_features, test_labels, legitimate_mean, alpha)
    drifted = slow_drift(test_features, magnitude)
    amount_index = feature_names.index("TransactionAmt") if "TransactionAmt" in feature_names else -1
    fused_note = None
    if fused_scores is not None:
        from fraud_engine.evaluation.metrics import safe_auroc

        fused_auroc, _note = safe_auroc(test_labels, fused_scores[test_index])
        fused_note = fused_auroc
    return {
        "scored_model": "logistic_regression",
        "fused_clean_auroc": fused_note,
        "why_tabular_scorer": (
            "Camouflage, drift, and structuring are applied to feature rows and rescored with the "
            "trained logistic regression model. Rebuilding the full graph for every altered row is "
            "recorded as a limitation; the fused clean AUROC is included for comparison."
        ),
        "camouflage": evaluate_feature_attack(
            "camouflage", test_labels, camouflaged, score_fn, thresholds, clean_auroc
        ),
        "slow_drift": evaluate_feature_attack(
            "slow_drift", test_labels, drifted, score_fn, thresholds, clean_auroc
        ),
        "structuring": evaluate_structuring(
            test_features[:, amount_index] if amount_index >= 0 else np.zeros(len(test_features)),
            test_labels,
            test_features,
            amount_index,
            parts,
            score_fn,
            thresholds,
        ),
    }


def _device_lift(
    features: np.ndarray,
    labels: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
    preprocessor: TabularPreprocessor,
    settings: dict[str, Any],
) -> dict[str, Any]:
    from fraud_engine.evaluation.metrics import safe_auroc

    names = preprocessor.feature_names
    device_indexes = [index for index, name in enumerate(names) if name in set(preprocessor.device_features)]
    if not device_indexes:
        return {"auroc_with_device_features": None, "auroc_without_device_features": None, "lift": None, "notes": ["No device or identity columns were available."]}
    keep = [index for index in range(features.shape[1]) if index not in set(device_indexes)]
    try:
        with_model = fit_logistic(features[train_index], labels[train_index], settings["seed"])
        without_model = fit_logistic(features[train_index][:, keep], labels[train_index], settings["seed"])
    except ValueError as exc:
        return {"lift": None, "notes": [str(exc)]}
    with_scores = positive_scores(with_model, features[test_index])
    without_scores = positive_scores(without_model, features[test_index][:, keep])
    with_auroc, with_note = safe_auroc(labels[test_index], with_scores)
    without_auroc, without_note = safe_auroc(labels[test_index], without_scores)
    lift = None if with_auroc is None or without_auroc is None else float(with_auroc - without_auroc)
    notes = [note for note in (with_note, without_note) if note]
    notes.append(
        "Lift is the test AUROC of logistic regression with device/identity columns minus the same model without those columns. "
        "Columns are " + ", ".join(names[index] for index in device_indexes) + "."
    )
    if preprocessor.simulated_features:
        notes.append("Some of these columns are simulated and must not be described as collected biometrics.")
    else:
        notes.append("No simulated biometric column was included in this lift.")
    return {
        "auroc_with_device_features": with_auroc,
        "auroc_without_device_features": without_auroc,
        "lift": lift,
        "device_columns": [names[index] for index in device_indexes],
        "notes": notes,
    }


def _row_from_scores(
    name: str,
    parameters: dict[str, Any],
    labels: np.ndarray,
    scores: np.ndarray,
    val_index: np.ndarray,
    threshold_index: np.ndarray,
    test_index: np.ndarray,
    settings: dict[str, Any],
    feature_note: str,
) -> dict[str, Any]:
    usable_val = val_index[labels[val_index] >= 0] if np.any(labels < 0) else val_index
    usable_threshold = threshold_index[labels[threshold_index] >= 0] if np.any(labels < 0) else threshold_index
    usable_test = test_index[labels[test_index] >= 0] if np.any(labels < 0) else test_index
    threshold, threshold_note = best_f1_threshold(labels[usable_val], scores[usable_val])
    placeholder_threshold = False
    if threshold is None:
        threshold = 0.5
        placeholder_threshold = True
    metrics = classification_metrics(labels[usable_test], scores[usable_test], threshold)
    if threshold_note:
        metrics["notes"].append(threshold_note)
    if placeholder_threshold:
        metrics["notes"].append(
            "Validation could not supply an F1 threshold. The placeholder 0.5 was not fit on test labels."
        )
    gateway = None
    try:
        cut_points = thresholds_from_legitimate_scores(
            scores[usable_threshold],
            labels[usable_threshold],
            float(settings["gateway"]["step_up_fpr"]),
            float(settings["gateway"]["block_fpr"]),
        )
        decisions = decide_many(scores[usable_test], cut_points)
        gateway = gateway_outcome(labels[usable_test], decisions)
        gateway["thresholds"] = cut_points
    except ValueError as exc:
        gateway = {"notes": [str(exc)]}
    timing = {
        "notes": [
            "This table records ranking quality. Single-request scoring latency is measured on the API service."
        ]
    }
    return {
        "model": name,
        "dataset": settings["dataset_name"],
        "benchmark_valid": bool(settings.get("benchmark_valid")),
        "parameters": parameters,
        "features": feature_note,
        "split_strategy": "see experiment summary",
        **{key: metrics[key] for key in ("auroc", "average_precision", "precision", "recall", "f1", "confusion_matrix", "threshold")},
        "gateway": gateway,
        "timing": timing,
        "notes": metrics["notes"],
        "error": None,
    }


def _failed(name: str, exc: Exception) -> dict[str, Any]:
    return {
        "model": name,
        "auroc": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "error": f"{type(exc).__name__}: {exc}",
        "notes": ["This model did not finish. The error is the executed result, not an imputed score."],
        "benchmark_valid": False,
    }


def _skipped(name: str, reason: str) -> dict[str, Any]:
    return {
        "model": name,
        "auroc": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "error": None,
        "notes": [reason],
        "skipped": True,
        "benchmark_valid": False,
    }


def _write_outputs(summary: dict[str, Any], output_dir: Path, settings: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    table = pd.DataFrame(summary["models"])
    csv_path = output_dir / "metrics.csv"
    fieldnames = ["model", "dataset", "benchmark_valid", "auroc", "average_precision", "precision", "recall", "f1", "error"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in summary["models"]:
            writer.writerow({key: row.get(key) for key in fieldnames})
    title = (
        f"{settings['dataset_name']} test AUROC"
        if settings.get("benchmark_valid")
        else f"{settings['dataset_name']} pipeline check — not a research benchmark"
    )
    save_metric_chart(table, "auroc", output_dir / "auroc.png", title)
    lines = [
        f"# Comparison — {settings['dataset_name']}",
        "",
        f"Research benchmark: **{bool(settings.get('benchmark_valid'))}**.",
        "",
        "| Model | AUROC | Precision | Recall | F1 | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["models"]:
        note = "; ".join(row.get("notes") or [])
        if row.get("error"):
            note = row["error"]
        lines.append(
            "| {model} | {auroc} | {precision} | {recall} | {f1} | {note} |".format(
                model=row.get("model"),
                auroc=_fmt(row.get("auroc")),
                precision=_fmt(row.get("precision")),
                recall=_fmt(row.get("recall")),
                f1=_fmt(row.get("f1")),
                note=note.replace("|", "/"),
            )
        )
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _edges_where(edge_index: np.ndarray, source_ok: np.ndarray, target_ok: np.ndarray) -> np.ndarray:
    if edge_index.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    source, target = edge_index
    mask = source_ok[source] & target_ok[target]
    return edge_index[:, mask]


def _edges_respecting_split(edge_index: np.ndarray, split: np.ndarray) -> np.ndarray:
    if edge_index.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    source, target = edge_index
    mask = split[source] <= split[target]
    return edge_index[:, mask]


def _half_split(indexes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(indexes) < 2:
        return indexes, indexes
    middle = max(1, len(indexes) // 2)
    return indexes[:middle], indexes[middle:]


def _time_order(indexes: np.ndarray, times: np.ndarray) -> np.ndarray:
    return indexes[np.argsort(times[indexes], kind="mergesort")]


def _rate(labels: np.ndarray) -> float | None:
    usable = labels[labels >= 0]
    if len(usable) == 0:
        return None
    return float((usable == 1).mean())


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialise {type(value)}")
