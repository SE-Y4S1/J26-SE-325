"""Executed synthetic experiment. The output must not claim to be IEEE-CIS."""

from __future__ import annotations

import json
from pathlib import Path

from fraud_engine.config import load_settings
from fraud_engine.data.synthetic import make_synthetic_transactions
from fraud_engine.evaluation.benchmark import run_transaction_experiment


def test_synthetic_run_records_executed_metrics(tmp_path: Path) -> None:
    settings = load_settings("synthetic")
    settings["max_rows"] = 180
    settings["models"]["epochs"] = 2
    settings["models"]["hidden"] = 8
    settings["log_dir"] = str(tmp_path)
    frame = make_synthetic_transactions(180, seed=int(settings["seed"]))
    summary = run_transaction_experiment(frame, settings, tmp_path / "run", metadata={"origin": "synthetic-fixture"})
    assert summary["results_are_research_benchmark"] is False
    assert summary["benchmark_valid"] is False
    written = json.loads((tmp_path / "run" / "metrics.json").read_text(encoding="utf-8"))
    assert written["results_are_research_benchmark"] is False
    names = [row["model"] for row in summary["models"]]
    for required in (
        "logistic_regression",
        "svm",
        "random_forest",
        "static_gcn",
        "static_gat",
        "lstm_only",
        "gnn_only",
        "fused_lstm_gnn",
    ):
        assert required in names
    for row in summary["models"]:
        if row.get("error"):
            continue
        if row.get("auroc") is not None:
            assert 0.0 <= row["auroc"] <= 1.0
    assert "fraud_blocked" in summary["models"][-1].get("gateway", {}) or summary["models"][-1].get("error")
    assert (tmp_path / "run" / "metrics.csv").is_file()
    assert (tmp_path / "run" / "comparison.md").is_file()
