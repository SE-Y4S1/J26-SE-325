"""Train baselines and the fused model.

Examples (from the Fraud-Detection folder):
  py scripts/train_benchmark.py --profile synthetic
  py scripts/train_benchmark.py --profile dev --dataset ieee
  py scripts/train_benchmark.py --profile wider --dataset ieee
  py scripts/train_benchmark.py --profile dev --dataset elliptic
  py scripts/train_benchmark.py --profile dev --dataset dgraphfin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_engine.config import load_settings
from fraud_engine.data.dgraphfin import load_dgraphfin
from fraud_engine.data.elliptic import load_elliptic
from fraud_engine.data.ieee import load_ieee_frame
from fraud_engine.data.synthetic import make_synthetic_transactions
from fraud_engine.evaluation.benchmark import run_graph_experiment, run_transaction_experiment
from fraud_engine.paths import RESULT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fraud-detection experiment.")
    parser.add_argument("--profile", default="synthetic", choices=["synthetic", "dev", "wider", "full", "default"])
    parser.add_argument("--dataset", default=None, choices=["ieee", "elliptic", "dgraphfin"])
    args = parser.parse_args()
    settings = load_settings(args.profile)
    if args.profile == "synthetic":
        dataset = "synthetic"
    else:
        dataset = args.dataset or "ieee"
    try:
        if dataset == "synthetic":
            frame = make_synthetic_transactions(n=int(settings.get("max_rows") or 360), seed=int(settings["seed"]))
            output = RESULT_DIR / "synthetic-fixture"
            summary = run_transaction_experiment(frame, settings, output, metadata={"origin": "synthetic-fixture"})
        elif dataset == "ieee":
            frame, metadata = load_ieee_frame(max_rows=settings.get("max_rows"))
            output = _ieee_output(settings.get("max_rows"))
            summary = run_transaction_experiment(frame, settings, output, metadata=metadata)
        elif dataset == "elliptic":
            prepared = load_elliptic(max_nodes=settings["graph_limits"]["elliptic_max_nodes"])
            output = RESULT_DIR / "elliptic"
            summary = run_graph_experiment(prepared, settings, output)
        else:
            prepared = load_dgraphfin(max_nodes=settings["graph_limits"]["dgraph_max_nodes"])
            output = RESULT_DIR / "dgraphfin"
            summary = run_graph_experiment(prepared, settings, output)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(exc)
        print("No benchmark file was written for this failed run.")
        sys.exit(2)
    print(f"Wrote results to {output}")
    print(f"results_are_research_benchmark={summary['results_are_research_benchmark']}")
    for row in summary["models"]:
        print(f"{row.get('model')}: auroc={row.get('auroc')} error={row.get('error')}")


def _ieee_output(max_rows: object) -> Path:
    """Keep the finished 20,000-row result folder and write larger runs beside it."""
    if max_rows in (None, ""):
        return RESULT_DIR / "ieee-cis-full"
    rows = int(max_rows)  # type: ignore[arg-type]
    if rows == 20000:
        return RESULT_DIR / "ieee-cis"
    return RESULT_DIR / f"ieee-cis-{rows}"


if __name__ == "__main__":
    main()
