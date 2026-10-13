"""Colab Part C: fine-tune Laya on the liquidation questions and export a head-only update.

    python -m experiments.train_laya --out /content/drive/MyDrive/j26_laya \
        --bars /content/drive/MyDrive/j26_hybrid/bars.parquet \
        --backfill /content/drive/MyDrive/j26_se_325_backfill/fundamentals_backfill.parquet

STAGES (each resumable; outputs on Drive)
  1. stage1_{train,test}.jsonl   point-in-time technical/fundamental rows, outcome labels
  2. stage1/                     fine-tune #1 (encoder frozen) on Stage 1 rows
  3. decision_{train,test}.jsonl withdrawal scenarios; each holding's Stage 1 summary is the
                                 STAGE-1 MODEL'S OWN ANSWERS at that date, exactly what the
                                 service feeds Stage 2 at inference (no train/serve mismatch)
  4. final/                      fine-tune #2 from stage1/ on decision + portfolio + Stage 1
                                 rows (multi-task: one model, Stage 1 is not forgotten)
  5. evaluation.csv              test split (dates after the cutoff): zero-shot base vs
                                 fine-tuned, per question -- the RQ5 numbers
  6. laya_head.zip               head + calibration only (encoder frozen), a few MB
Install locally: python -m decision.checkpoint --install laya_head.zip
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BASE = "convaiinnovations/laya"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _benchmarks(bars: pd.DataFrame) -> dict[str, pd.Series]:
    out = {}
    for key, symbol in (("equity", "SPY"), ("etf", "SPY"), ("crypto", "BTCUSDT")):
        frame = bars[bars["symbol"] == symbol]
        if not frame.empty:
            out[key] = pd.Series(frame["close"].to_numpy(), index=pd.to_datetime(frame["timestamp"]))
    try:
        from data.exchanges import cse

        out["cse_equity"] = cse.aspi_daily()
    except Exception as exc:  # noqa: BLE001
        logger.warning("ASPI unavailable; CSE trend labels use absolute returns: %s", exc)
    return out


def build_stage1(out: Path, bars: pd.DataFrame, loader, sample_every: int) -> None:  # noqa: ANN001
    from decision.labels import split_by_date, stage1_rows, write_jsonl

    if (out / "stage1_train.jsonl").exists():
        return
    benchmarks = _benchmarks(bars)
    rows = []
    for symbol, frame in bars.groupby("symbol"):
        asset_class = str(frame["asset_class"].iloc[0])
        rows += stage1_rows(str(symbol), frame, asset_class=asset_class, benchmark=benchmarks.get(asset_class),
                            records=loader(str(symbol)), sample_every=sample_every)
    train, test, cutoff = split_by_date(rows)
    write_jsonl(train, out / "stage1_train.jsonl")
    write_jsonl(test, out / "stage1_test.jsonl")
    (out / "split.json").write_text(json.dumps({"cutoff": cutoff, "stage1_train": len(train), "stage1_test": len(test)}))
    logger.info("stage 1 rows: %d train / %d test (cutoff %s)", len(train), len(test), cutoff)


def finetune(data: Path, model_dir: str, output: Path, eval_data: Path, epochs: int) -> None:
    from laya.train import TrainConfig, finetune as laya_finetune

    if (output / "model.safetensors").exists():
        return
    config = TrainConfig(epochs=epochs, freeze_encoder=True, eval_data=str(eval_data),
                         shuffle_options=("choice",), micro_batch=16, grad_accum=4)
    summary = laya_finetune(str(data), model_dir, str(output), config=config)
    (output / "train_summary.json").write_text(json.dumps(summary, indent=1, default=str))


def build_decision(out: Path, bars: pd.DataFrame, loader, n_scenarios: int) -> None:  # noqa: ANN001
    import laya

    from decision.engine import LayaEngine, stage1_states
    from decision.labels import _record_at, scenario_rows, split_by_date, write_jsonl
    from decision import questions as Q
    from decision.states import summarise_stage1
    from features.technical import technical_snapshot

    if (out / "decision_train.jsonl").exists():
        return
    engine = LayaEngine(laya.load(str(out / "stage1")), "stage1")
    panel = {str(s): f.sort_values("timestamp").reset_index(drop=True) for s, f in bars.groupby("symbol")}
    classes = {s: str(f["asset_class"].iloc[0]) for s, f in panel.items()}
    benchmarks = _benchmarks(bars)
    cache: dict[tuple[str, str], dict] = {}

    def summary(symbol: str, when: pd.Timestamp) -> dict:
        key = (symbol, str(pd.Timestamp(when).date()))
        if key not in cache:
            frame = panel[symbol]
            upto = frame[pd.to_datetime(frame["timestamp"]) <= when]
            bench = benchmarks.get(classes[symbol])
            record = _record_at(loader(symbol), pd.Timestamp(when))
            metrics = {"technical": technical_snapshot(upto, bench[bench.index <= when] if bench is not None else None),
                       "fundamentals": record}
            answers = {}
            for group, state in stage1_states(metrics, classes[symbol]).items():
                answers.update(engine.ask([state], Q.STAGE1[group])[0])
            cache[key] = summarise_stage1(answers)
        return cache[key]

    rows = scenario_rows(panel, n_scenarios=n_scenarios, rng=random.Random(7), stage1_summary=summary,
                         asset_classes=classes)
    train, test, cutoff = split_by_date(rows)
    write_jsonl(train + _read_jsonl(out / "stage1_train.jsonl"), out / "decision_train.jsonl")
    write_jsonl(test, out / "decision_test.jsonl")
    logger.info("decision rows: %d train / %d test (cutoff %s)", len(train), len(test), cutoff)


def evaluate(out: Path) -> pd.DataFrame:
    import laya

    from decision.labels import evaluate as score

    path = out / "evaluation.csv"
    if path.exists():
        return pd.read_csv(path)
    test = _read_jsonl(out / "stage1_test.jsonl") + _read_jsonl(out / "decision_test.jsonl")
    frames = []
    for name, model in (("zero_shot", BASE), ("fine_tuned", str(out / "final"))):
        agent = laya.load(model)
        answered = []
        for row in test:
            answered.append((row, agent.predict_batch([row["state"]], row["questions"])[0]["answers"]))
        frame = score(answered)
        frame.insert(0, "model", name)
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(path, index=False)
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("artifacts/laya_training"))
    parser.add_argument("--bars", type=Path, required=True, help="bars.parquet from Part A")
    parser.add_argument("--backfill", type=Path, default=None)
    parser.add_argument("--scenarios", type=int, default=20_000)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    from features.enrichment import backfill_loader

    args.out.mkdir(parents=True, exist_ok=True)
    bars = pd.read_parquet(args.bars)
    loader = backfill_loader(args.backfill) if args.backfill and args.backfill.exists() else (lambda s: [])

    build_stage1(args.out, bars, loader, args.sample_every)
    finetune(args.out / "stage1_train.jsonl", BASE, args.out / "stage1", args.out / "stage1_test.jsonl", args.epochs)
    build_decision(args.out, bars, loader, args.scenarios)
    finetune(args.out / "decision_train.jsonl", str(args.out / "stage1"), args.out / "final",
             args.out / "decision_test.jsonl", args.epochs)
    table = evaluate(args.out)
    print(table.to_string(index=False))

    from decision.checkpoint import export_head

    split = json.loads((args.out / "split.json").read_text())
    tuned = table[table["model"] == "fine_tuned"]
    metrics = {f"accuracy_{r.question}": r.accuracy for r in tuned.itertuples()}
    metrics.update({f"brier_{r.question}": r.brier for r in tuned.itertuples()})
    dates = pd.to_datetime(bars["timestamp"])
    zip_path = export_head(args.out / "final", args.out / "laya_head.zip", metrics=metrics,
                           train_window={"start": str(dates.min().date()), "end": split["cutoff"]})
    print("export:", zip_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
