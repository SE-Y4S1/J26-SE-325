"""The experiment, in two stages.

    uv run python -m assurance.run tune     # fit models, compute signals, tune every router on
                                            # the tuning split, export the tuned P002 policy
    (commit the exported policy file)
    uv run python -m assurance.run final    # evaluate on the held-out test split, once

`tune` never looks at test labels. `final` refuses to run until the exported policy is
committed, evaluates exactly that policy, and records the run (test-set fingerprint, git
commit, policy hash) in results/ieee-cis/RUNLOG.json; a second final run for the same policy
needs --rerun-reason, which is logged.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config, report
from .data import load_ieee_cis, test_index_digest, time_split
from .engine import policy_hash_v2
from .evaluate import aurc, bootstrap, deferral_curve, point_metrics
from .features import Preprocessor
from .models import CalibratedEnsemble
from .routers import Fitted, _assurance_routes, assurance_score, fit_all
from .signals import ContextModel, compute, first_seen_flags

CACHE_NAME = "signals-cache.npz"
# The fitted preprocessor, calibrated models and context model, so assurance.replay can score
# real transactions without refitting. Git-ignored with the rest of cache/ (trained on Kaggle data).
MODELS_NAME = "models.joblib"
EXPORTED_POLICY = config.POLICY_DIR / "P002-assurance-v1.1.json"
BASELINES = ["risk_band", "confidence_only", "p001_retuned", "random_review", "learned_deferral"]


# --- building signals ------------------------------------------------------------------------

def build(data_dir: Path | None, max_rows: int | None, fast: bool) -> tuple[dict, dict, dict]:
    frame = load_ieee_cis(data_dir, max_rows)
    splits = time_split(frame)
    parts = {name: splits.part(name) for name in splits.index}
    pre = Preprocessor().fit(parts["train"])
    x = {name: pre.transform(part) for name, part in parts.items()}
    y = {name: part["isFraud"].to_numpy(dtype=int) for name, part in parts.items()}

    ensemble = CalibratedEnsemble.fit(x["train"], y["train"], x["calibration"], y["calibration"], fast=fast)
    context = ContextModel().fit(parts["train"], x["train"])
    new_card = first_seen_flags(splits.frame)

    signals = {}
    for name in ["calibration", "tuning", "test"]:
        members = ensemble.member_probabilities(x[name])
        signals[name] = compute(parts[name], x[name], members, new_card[splits.index[name]], context)

    cal = signals["calibration"]
    meta = {
        "rows": {k: int(len(v)) for k, v in splits.index.items()},
        "fraud_rate": {k: float(parts[k]["isFraud"].mean()) for k in splits.index},
        "time_ranges": {k: [int(parts[k]["TransactionDT"].min()), int(parts[k]["TransactionDT"].max())] for k in splits.index},
        "features": len(pre.columns),
        "decision_threshold": config.DECISION_THRESHOLD,
        "test_index_digest": test_index_digest(splits),
        "calibration_split_model_quality": {
            name: {"auroc": float(roc_auc_score(cal["y"], cal[f"member_{name}"])),
                   "average_precision": float(average_precision_score(cal["y"], cal[f"member_{name}"]))}
            for name in ensemble.models
        },
        "fast": fast,
        "max_rows": max_rows,
    }
    models = {"preprocessor": pre, "ensemble": ensemble, "context": context,
              "threshold": config.DECISION_THRESHOLD, "trained_rows": meta["rows"]["train"]}
    return signals, meta, models


def save_cache(path: Path, signals: dict, meta: dict) -> None:
    arrays = {f"{split}__{k}": v for split, s in signals.items() for k, v in s.items()}
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)


def load_cache(path: Path) -> tuple[dict, dict]:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    signals: dict = {}
    for key in data.files:
        if key == "meta":
            continue
        split, name = key.split("__", 1)
        signals.setdefault(split, {})[name] = data[key]
    return signals, meta


# --- policy export ---------------------------------------------------------------------------

def export_policy(params: dict, meta: dict, budget: float, source: Path, target: Path) -> dict:
    """The tuned P002 as a new version (v1.1): v1.0 stays, so records it routed still resolve."""
    base = json.loads(source.read_text(encoding="utf-8"))
    doc = json.loads(json.dumps(base["document"]))
    max_d = round(params["maxDisagreement"], 4)
    doc.update({
        "version": "1.1",
        "status": "tuned",
        "decision": {"rejectThreshold": config.DECISION_THRESHOLD},
        "weights": {k: round(v, 4) for k, v in params["weights"].items()},
        "routing": {"autoMin": round(params["autoMin"], 4), "verifyMin": round(params["verifyMin"], 4),
                    "degradedCap": doc["routing"]["degradedCap"]},
        "missing": {"disagreement": max_d},
        "tunedOn": {
            "dataset": "IEEE-CIS Fraud Detection (Kaggle 2019), labelled train_transaction + train_identity",
            "split": "time-ordered; tuned on the policy-tuning split only",
            "rows": meta["rows"],
            "humanReviewBudget": budget,
            "verifyBudget": config.VERIFY_BUDGET,
            "objective": "minimise incorrect automatic decisions (VERIFY counted as automatic)",
            "offlineContext": "mean of amount percentile, IsolationForest novelty, first-seen card, missing identity",
        },
    })
    doc["vetoes"] = {**doc["vetoes"], "maxDisagreement": max_d}
    w = doc["weights"]
    drift = round(1 - w["confidence"] - w["disagreement"] - w["context"], 4)
    w["context"] = round(w["context"] + drift, 4)      # keep the sum exactly 1 after rounding
    policy = {
        "meta": {**base["meta"], "description": (
            "AI Decision Assurance Policy, tuned on IEEE-CIS (see research/). Weights, thresholds and the "
            "disagreement veto come from the policy-tuning split; the online context sub-signals are unchanged "
            "from v1.0 because the offline context signals are not available online.")},
        "document": doc,
    }
    target.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    return policy


def assurance_from_policy(doc: dict) -> Fitted:
    """The exported policy as a router -- what `final` evaluates, so evaluated = enforced."""
    w = (doc["weights"]["confidence"], doc["weights"]["disagreement"], doc["weights"]["context"])
    max_d, vmin, amin = doc["vetoes"]["maxDisagreement"], doc["routing"]["verifyMin"], doc["routing"]["autoMin"]
    return Fitted("assurance", lambda s: _assurance_routes(s, w, max_d, vmin, amin),
                  {"policyHash": policy_hash_v2(doc)},
                  rank=lambda s: 1 - assurance_score(s, w) + (s["u_disagreement"] > max_d))


# --- stages ----------------------------------------------------------------------------------

def stage_tune(args) -> None:
    results = Path(args.results_dir)
    cache = results / "cache" / CACHE_NAME
    cache.parent.mkdir(parents=True, exist_ok=True)
    signals, meta, models = build(args.data, args.max_rows, args.fast)
    save_cache(cache, signals, meta)
    joblib.dump(models, cache.parent / MODELS_NAME, compress=3)

    tune = signals["tuning"]
    by_budget = {}
    exported_params = None
    for budget in config.BUDGETS:
        fitted = fit_all(tune, budget)
        by_budget[str(budget)] = {name: {"params": f.params, "tuning_metrics": point_metrics(f.route(tune), tune)}
                                  for name, f in fitted.items()}
        if abs(budget - config.POLICY_BUDGET) < 1e-12:
            exported_params = fitted["assurance"].params

    payload = {"stage": "tune", "generated_at": _now(), "meta": meta, "note": "Tuning-split metrics only. No test labels were used.",
               "budgets": by_budget}
    (results / "tuning.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    if not args.no_export:
        policy = export_policy(exported_params, meta, config.POLICY_BUDGET, config.POLICY_FILE, Path(args.policy_out))
        print(f"exported {Path(args.policy_out).name}: {policy['document']['weights']} veto>{policy['document']['vetoes']['maxDisagreement']}"
              f" verifyMin={policy['document']['routing']['verifyMin']} autoMin={policy['document']['routing']['autoMin']}")
    print(f"tuning done: {meta['rows']} rows; cache {cache}")


def stage_final(args) -> None:
    results = Path(args.results_dir)
    signals, meta = load_cache(results / "cache" / CACHE_NAME)
    policy_file = Path(args.policy_out)
    policy = json.loads(policy_file.read_text(encoding="utf-8"))
    doc = policy["document"]
    phash = policy_hash_v2(doc)

    runlog_file = results / "RUNLOG.json"
    runlog = json.loads(runlog_file.read_text(encoding="utf-8")) if runlog_file.exists() else []
    commit = None
    if not args.allow_uncommitted:
        commit = _require_committed(policy_file)
        if any(r.get("stage") == "final" and r.get("policyHash") == phash for r in runlog) and not args.rerun_reason:
            raise SystemExit("A final evaluation of this policy already exists. Pass --rerun-reason \"why\" to repeat it (it is logged).")

    tune, test = signals["tuning"], signals["test"]
    by_budget = {}
    curves = {}
    for budget in config.BUDGETS:
        fitted = fit_all(tune, budget)
        if abs(budget - config.POLICY_BUDGET) < 1e-12:
            fitted["assurance"] = assurance_from_policy(doc)
        routes = {name: f.route(test) for name, f in fitted.items()}
        metrics = {name: point_metrics(r, test) for name, r in routes.items()}
        boot = bootstrap(routes, test, reference="assurance", comparators=BASELINES + ["p001_deployed", "auto_all"])
        half = len(test["y"]) // 2
        first = {k: v[:half] for k, v in test.items()}
        second = {k: v[half:] for k, v in test.items()}
        drift = {name: {"first_half": point_metrics(f.route(first), first)["incorrect_automatic_rate"],
                        "second_half": point_metrics(f.route(second), second)["incorrect_automatic_rate"]}
                 for name, f in fitted.items()}
        by_budget[str(budget)] = {
            "methods": {name: {**metrics[name], "aurc": aurc(fitted[name], test), "params": fitted[name].params,
                               "ci95": boot["intervals"][name], "drift": drift[name]} for name in fitted},
            "paired_vs_assurance": boot["paired"],
        }
        if abs(budget - config.POLICY_BUDGET) < 1e-12:
            curves = {name: deferral_curve(f, test) for name, f in fitted.items() if f.rank is not None}

    payload = {
        "stage": "final",
        "status": "FINAL" if not args.allow_uncommitted else "NOT_FINAL_UNCOMMITTED_POLICY",
        "generated_at": _now(),
        "policy": {"file": policy_file.name, "id": doc["id"], "version": doc["version"], "hash": phash},
        "meta": meta,
        "budgets_compared": config.BUDGETS,
        "policy_budget": config.POLICY_BUDGET,
        "assumptions": {
            "verify_effectiveness": "not measured; primary metric uses 0 (VERIFY counted as automatic), sweep reported",
            "analyst_accuracy": "not measured; sweep reported",
        },
        "results": by_budget,
        "curves_at_policy_budget": curves,
    }
    results.mkdir(parents=True, exist_ok=True)
    (results / "metrics.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    report.write_all(payload, results)

    runlog.append({"stage": "final", "at": payload["generated_at"], "policyHash": phash, "commit": commit,
                   "testIndexDigest": meta["test_index_digest"], "status": payload["status"],
                   **({"rerunReason": args.rerun_reason} if args.rerun_reason else {})})
    runlog_file.write_text(json.dumps(runlog, indent=1) + "\n", encoding="utf-8")
    print(f"final evaluation written to {results} ({payload['status']})")


def _require_committed(policy_file: Path) -> str:
    def git(*a):
        return subprocess.run(["git", *a], cwd=config.REPO_DIR, capture_output=True, text=True)
    resolved = policy_file.resolve()
    if not resolved.is_relative_to(config.REPO_DIR.resolve()):
        raise SystemExit(f"Commit {policy_file} first: it is outside the repository, so it cannot be a committed policy.")
    rel = resolved.relative_to(config.REPO_DIR.resolve()).as_posix()
    if git("ls-files", "--error-unmatch", rel).returncode != 0 or git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise SystemExit(f"Commit {rel} first: the final evaluation must be of a committed policy.")
    return git("rev-parse", "HEAD").stdout.strip()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["tune", "final"])
    parser.add_argument("--data", type=Path, default=None, help="IEEE-CIS directory (default: IEEE_CIS_DIR or Component 2's data/raw/ieee-cis)")
    parser.add_argument("--max-rows", type=int, default=None, help="Read only the first N transactions (development runs)")
    parser.add_argument("--fast", action="store_true", help="Smaller models (development runs)")
    parser.add_argument("--results-dir", default=str(config.RESULTS_DIR))
    parser.add_argument("--policy-out", default=str(EXPORTED_POLICY))
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--allow-uncommitted", action="store_true", help="Smoke tests only: results are marked NOT_FINAL")
    parser.add_argument("--rerun-reason", default=None)
    args = parser.parse_args(argv)
    (stage_tune if args.stage == "tune" else stage_final)(args)


if __name__ == "__main__":
    main()
