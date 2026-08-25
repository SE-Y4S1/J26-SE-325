"""One-command pipeline runner for Component 1.

Chains the phases in dependency order so a supervisor or teammate can reproduce every result
without knowing which script to run in which sequence:

    uv run python run_pipeline.py                 # everything runnable offline
    uv run python run_pipeline.py --stages 3 4    # just those stages
    uv run python run_pipeline.py --list

Each stage is skipped if its output already exists, so re-running is cheap and a failure
part-way through can be resumed rather than restarted.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It never downloads a foundation checkpoint (~821MB each) and never fine-tunes. Those belong
in Colab -- see experiments/colab_finetune.ipynb -- for two independent reasons documented in
the README: no CUDA on the dev machine, and a ~24 KB/s link. Stages that would require them
report "needs Colab" rather than starting an hours-long download.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("pipeline")

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "artifacts" / "results"
CONFIGS = ROOT / "configs"
TRAJECTORIES = ROOT / "artifacts" / "trajectories"


@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    output: Path            # existence means "already done"
    run: Callable[[], None]


# --------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------

def stage_resolve_universe() -> None:
    """Phase 1: derive each symbol's history window and horizons from its own criteria."""
    from datetime import date

    from data.window_selector import resolve_universe

    resolved = resolve_universe(
        CONFIGS / "universe.yaml", CONFIGS / "resolved_universe.yaml",
        as_of=date.today(), typical_position_value=50_000.0,
    )
    logger.info("resolved %d symbols", len(resolved))


def stage_rq1_baseline() -> None:
    """Phase 3 + 7: walk-forward backtest of the baseline forecaster on real data."""
    import subprocess

    subprocess.run([sys.executable, "-u", str(ROOT / "scripts_rq1.py")], check=True, cwd=ROOT)


def stage_rq2_allocation() -> None:
    """Phase 5a: MOEA/D versus Markowitz across all three Pareto selection rules."""
    from experiments.run_rq_analysis import rq2_allocation_comparison

    RESULTS.mkdir(parents=True, exist_ok=True)
    rq2_allocation_comparison().to_csv(RESULTS / "rq2_allocation.csv", index=False)


def stage_rq3_withdrawal() -> None:
    """Phase 5b: fuzzy GA versus the naive liquidation baselines."""
    from experiments.run_rq_analysis import rq3_withdrawal_vs_naive

    RESULTS.mkdir(parents=True, exist_ok=True)
    rq3_withdrawal_vs_naive().to_csv(RESULTS / "rq3_withdrawal.csv", index=False)


def stage_rq4_stress() -> None:
    """Phase 5b + 7: degradation across the stress-severity ladder."""
    from experiments.run_rq_analysis import rq4_stress_degradation

    RESULTS.mkdir(parents=True, exist_ok=True)
    _, summary = rq4_stress_degradation()
    summary.to_csv(RESULTS / "rq4_summary.csv", index=False)


def stage_trajectories() -> None:
    """Phase 5c: the grounded SFT dataset that Colab consumes."""
    from agent.trajectory_generation import Driver, generate_dataset

    path = generate_dataset(n_scenarios=800, driver=Driver.SCRIPTED, seed=42)
    logger.info("wrote %s", path)


STAGES: tuple[Stage, ...] = (
    Stage("1", "resolve per-symbol windows", CONFIGS / "resolved_universe.yaml", stage_resolve_universe),
    Stage("2", "RQ1 baseline forecast", RESULTS / "rq1_baseline.csv", stage_rq1_baseline),
    Stage("3", "RQ2 allocation", RESULTS / "rq2_allocation.csv", stage_rq2_allocation),
    Stage("4", "RQ3 withdrawal", RESULTS / "rq3_withdrawal.csv", stage_rq3_withdrawal),
    Stage("5", "RQ4 stress degradation", RESULTS / "rq4_summary.csv", stage_rq4_stress),
    Stage("6", "SFT trajectories", TRAJECTORIES / "withdrawal_sft_scripted_800.jsonl", stage_trajectories),
)


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stages", nargs="*", choices=[s.key for s in STAGES],
                        help="run only these stages (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="re-run even when the output already exists")
    parser.add_argument("--list", action="store_true", help="show stages and their status")
    args = parser.parse_args()

    if args.list:
        print(f"{'#':3} {'stage':32} {'status':12} output")
        for stage in STAGES:
            status = "done" if stage.output.exists() else "pending"
            print(f"{stage.key:3} {stage.title:32} {status:12} {stage.output.relative_to(ROOT)}")
        return 0

    selected = [s for s in STAGES if not args.stages or s.key in args.stages]
    failures: list[str] = []

    for stage in selected:
        if stage.output.exists() and not args.force:
            logger.info("[%s] %s -- already done (%s); use --force to redo",
                        stage.key, stage.title, stage.output.name)
            continue

        logger.info("[%s] %s -- running", stage.key, stage.title)
        started = time.time()
        try:
            stage.run()
            logger.info("[%s] done in %.0fs", stage.key, time.time() - started)
        except Exception as exc:  # noqa: BLE001 - one stage must not abort the rest
            logger.error("[%s] FAILED: %s: %s", stage.key, type(exc).__name__, exc)
            failures.append(stage.key)

    print()
    print("=== pipeline summary ===")
    for stage in STAGES:
        mark = "ok " if stage.output.exists() else "-- "
        print(f"  {mark} [{stage.key}] {stage.title}")

    if failures:
        # Named explicitly: a partially-complete run that looks successful is how stale
        # results end up in a dissertation.
        print(f"\nFAILED stages: {failures}")
        return 1

    print("\nNext: fine-tuning and RQ1's foundation/hybrid rows need a GPU and a fast link.")
    print("See experiments/colab_finetune.ipynb.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
