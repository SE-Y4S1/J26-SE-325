"""Replay real IEEE-CIS transactions through the live system.

    uv run python -m assurance.replay --test-sample 20            # held-out test transactions
    uv run python -m assurance.replay --csv rows.csv [--identity identity.csv]   # any IEEE-format file
    options: --service http://localhost:8002   --token <jwt>   --dry-run   --tag <suffix>

Each transaction is scored by the three models `run tune` trained and calibrated, then sent to
Component 3's POST /api/audit/evaluate with those real model scores as assurance signals. The
service routes it under the policy in force and anchors it on the ledger it is configured for, so
it appears in the audit console and verifies like any other decision.

What is sent, and what is not:
  - riskScore is the calibrated ensemble probability x 100; modelScores are the three members.
  - rules is [] because a replayed transaction has not passed through Component 2's gateway, so
    no gateway rule fired. transactionType is PAYMENT (IEEE-CIS is card-not-present e-commerce).
  - Labels (isFraud) never leave this process. They are printed next to the route, when the input
    has them, so the outcome can be checked; the service decides without them.
  - The Kaggle data stays local: only the scores, the amount and a TransactionID-derived id are sent.

Rows that cannot be scored (no TransactionID, a missing, non-numeric or non-positive amount) are
reported and skipped; columns absent from a CSV are treated as missing values, exactly as the
models handle missing values in training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .data import load_ieee_cis, time_split, transaction_columns
from .engine import margin_uncertainty
from .run import MODELS_NAME

TRANSACTION_TYPE = "PAYMENT"
TIMEOUT_SECONDS = 30


# --- inputs ----------------------------------------------------------------------------------

def load_models(results_dir: Path) -> tuple[dict, str]:
    path = results_dir / "cache" / MODELS_NAME
    if not path.exists():
        raise SystemExit(f"{path} not found. Train the models first: python -m assurance.run tune")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return joblib.load(path), digest


def test_sample(data_dir: Path | None, n: int, frauds: int, seed: int) -> pd.DataFrame:
    """n transactions from the held-out test split, `frauds` of them known frauds (so a short
    demonstration shows both kinds); the rest drawn at random. Deterministic for a given seed."""
    test = time_split(load_ieee_cis(data_dir)).part("test")
    rng = np.random.default_rng(seed)
    fraud_rows = test.index[test["isFraud"] == 1].to_numpy()
    legit_rows = test.index[test["isFraud"] == 0].to_numpy()
    frauds = min(frauds, n, len(fraud_rows))
    chosen = np.concatenate([rng.choice(fraud_rows, frauds, replace=False),
                             rng.choice(legit_rows, n - frauds, replace=False)])
    return test.loc[np.sort(chosen)]


def _kaggle_names(frame: pd.DataFrame) -> pd.DataFrame:
    # Kaggle's test files spell identity columns "id-01"; the training files use "id_01".
    return frame.rename(columns=lambda c: c.replace("id-", "id_") if c.startswith("id-") else c)


def read_csv(csv: Path, identity: Path | None) -> tuple[pd.DataFrame, list[str]]:
    """An IEEE-format file, with any missing model columns added as missing values."""
    frame = _kaggle_names(pd.read_csv(csv, low_memory=False))
    if "TransactionID" not in frame.columns or "TransactionAmt" not in frame.columns:
        raise SystemExit(f"{csv} must have TransactionID and TransactionAmt columns (IEEE-CIS format).")
    if identity is not None:
        ident = _kaggle_names(pd.read_csv(identity, low_memory=False))
        ident = ident[[c for c in ident.columns if c in {"TransactionID", *config.IDENTITY_NUMERIC, *config.IDENTITY_CATEGORICAL}]]
        ident["has_identity"] = 1
        frame = frame.merge(ident, on="TransactionID", how="left", suffixes=("", "_identity"))
    if "has_identity" not in frame.columns:
        identity_seen = [c for c in config.IDENTITY_NUMERIC + config.IDENTITY_CATEGORICAL if c in frame.columns]
        frame["has_identity"] = frame[identity_seen].notna().any(axis=1).astype(int) if identity_seen else 0
    frame["has_identity"] = frame["has_identity"].fillna(0).astype(int)
    needed = [c for c in transaction_columns() if c != "isFraud"] + config.IDENTITY_NUMERIC + config.IDENTITY_CATEGORICAL
    missing = [c for c in needed if c not in frame.columns]
    for column in missing:
        frame[column] = np.nan
    return frame, missing


def split_scoreable(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Rows the service can accept, and (id, reason) for every row that cannot be sent."""
    ids = pd.to_numeric(frame["TransactionID"], errors="coerce")
    amounts = pd.to_numeric(frame["TransactionAmt"], errors="coerce")
    bad_id = ids.isna() | (ids != ids.round()) | (ids < 0)
    bad_amount = amounts.isna() | ~np.isfinite(amounts) | (amounts <= 0)
    skipped = []
    for index in frame.index[bad_id | bad_amount]:
        label = str(frame.at[index, "TransactionID"])
        skipped.append((label, "no usable TransactionID" if bad_id[index] else f"amount {frame.at[index, 'TransactionAmt']!r} is not a positive number"))
    keep = frame.loc[~(bad_id | bad_amount)].copy()
    keep["TransactionID"] = ids[keep.index].astype(np.int64)
    keep["TransactionAmt"] = amounts[keep.index].astype(float)
    return keep, skipped


# --- scoring and sending ---------------------------------------------------------------------

def score(models: dict, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    x = models["preprocessor"].transform(frame)
    members = models["ensemble"].member_probabilities(x)
    return np.vstack(list(members.values())).mean(axis=0), members


def payload(row: pd.Series, p: float, members: dict[str, float], threshold: float, model_version: str, tag: str) -> dict:
    return {
        "transactionId": f"IEEE-{int(row['TransactionID'])}-{tag}",
        "riskScore": round(float(np.clip(p, 0, 1)) * 100, 4),
        # Not used by the assurance engine (it derives confidence from the score); required by the API.
        "confidence": round(100 * (1 - float(margin_uncertainty(p, threshold))), 2),
        "amount": round(float(row["TransactionAmt"]), 2),
        "transactionType": TRANSACTION_TYPE,
        "modelVersion": model_version,
        "signals": {
            "modelScores": {name: round(float(v), 6) for name, v in members.items()},
            "rules": [],
            "benchmarkModel": True,
            "sourceModelVersion": model_version,
        },
    }


def post_json(url: str, body: dict, token: str | None) -> tuple[int, dict]:
    """POST and return (status, body). Status 0 means the service could not be reached."""
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as err:
        try:
            return err.code, json.loads(err.read() or b"{}")
        except ValueError:
            return err.code, {"message": f"HTTP {err.code}"}
    except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
        return 0, {"message": str(getattr(err, "reason", err))}


def outcome(route: str | None, ai_decision: str | None, label: int | None) -> str:
    if route is None:
        return "-"
    if route != "AUTO":
        return "deferred"
    if label is None:
        return "automatic"
    correct = (ai_decision == "REJECT") == (label == 1)
    return "correct automatic" if correct else "WRONG automatic"


# --- run -------------------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--test-sample", type=int, metavar="N", help="N transactions from the held-out test split")
    source.add_argument("--csv", type=Path, help="An IEEE-format transaction CSV (e.g. Kaggle's test_transaction.csv, or rows you pick)")
    parser.add_argument("--identity", type=Path, default=None, help="The matching identity CSV (optional)")
    parser.add_argument("--limit", type=int, default=50, help="Send at most this many rows from --csv (default 50)")
    parser.add_argument("--frauds", type=int, default=None, help="Known frauds in --test-sample (default: a quarter)")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--data", type=Path, default=None, help="IEEE-CIS directory (for --test-sample)")
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--service", default=os.getenv("AUDIT_SERVICE_URL", "http://localhost:8002"))
    parser.add_argument("--token", default=os.getenv("AUDIT_TOKEN"), help="Bearer token when the service enforces auth")
    parser.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%m%d%H%M%S"),
                        help="Suffix that keeps transaction ids unique across replays")
    parser.add_argument("--dry-run", action="store_true", help="Score and print, send nothing")
    args = parser.parse_args(argv)

    models, digest = load_models(Path(args.results_dir))
    model_version = f"ieee-cis-ensemble:{digest}"
    threshold = models["threshold"]

    missing: list[str] = []
    if args.test_sample is not None:
        frauds = args.frauds if args.frauds is not None else max(1, args.test_sample // 4)
        frame = test_sample(args.data, args.test_sample, frauds, args.seed)
        print(f"Source: {len(frame)} transactions from the held-out IEEE-CIS test split ({int(frame['isFraud'].sum())} known frauds).")
    else:
        frame, missing = read_csv(args.csv, args.identity)
        frame = frame.head(args.limit)
        print(f"Source: {args.csv.name}, first {len(frame)} rows.")
        if missing:
            print(f"  {len(missing)} model column(s) absent from the file are treated as missing values: {', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}")

    frame, skipped = split_scoreable(frame)
    for tx, reason in skipped:
        print(f"  skipped {tx}: {reason}")
    if frame.empty:
        print("Nothing to send.")
        return 1

    p, members = score(models, frame)
    labels = frame["isFraud"].to_numpy() if "isFraud" in frame.columns and frame["isFraud"].notna().all() else None
    url = f"{args.service.rstrip('/')}/api/audit/evaluate"
    print(f"Models: {model_version} (logistic, forest, boosting; trained on {models['trained_rows']:,} rows). "
          f"{'Dry run: nothing is sent.' if args.dry_run else 'Sending to ' + url}\n")

    header = f"{'transaction':<26}{'amount':>10}{'risk':>7}{'  logit  forest  boost':<23}{'AI':>7}  {'route':<13}{'score':>6}  {'block':>6}  {'label':<6}  outcome"
    print(header)
    print("-" * len(header))
    log, counts = [], {}
    for i, (_, row) in enumerate(frame.iterrows()):
        m = {name: float(v[i]) for name, v in members.items()}
        body = payload(row, float(p[i]), m, threshold, model_version, args.tag)
        label = int(labels[i]) if labels is not None else None
        status, response = (200, {}) if args.dry_run else post_json(url, body, args.token)
        if status == 0:
            print(f"\nThe audit service at {args.service} is not reachable ({response.get('message')}). Start it and run again.")
            return 2
        assurance = response.get("assurance") or {}
        route, ai = assurance.get("route"), assurance.get("aiDecision")
        block = (response.get("blockchain") or {}).get("blockNumber")
        result = outcome(route, ai, label)
        if status != 200 and not args.dry_run:
            result = f"HTTP {status}: {response.get('message', '')}"[:60]
        counts[route or "not routed"] = counts.get(route or "not routed", 0) + 1
        scores = "  ".join(f"{m[k]:.3f}" for k in ("logistic", "forest", "boosting") if k in m)
        score_txt = f"{assurance['assuranceScore']:.2f}" if assurance.get("assuranceScore") is not None else "-"
        print(f"{body['transactionId']:<26}{body['amount']:>10,.2f}{body['riskScore']:>7.1f}  {scores:<21}{(ai or '-'):>7}  "
              f"{(route or '-'):<13}{score_txt:>6}  {str(block or '-'):>6}  {('fraud' if label == 1 else 'legit' if label == 0 else '?'):<6}  {result}")
        log.append({"request": body, "status": status, "route": route, "aiDecision": ai, "block": block,
                    "label": label, "outcome": result})

    print("\nRoutes: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    if labels is not None and not args.dry_run:
        wrong = sum(1 for r in log if r["outcome"] == "WRONG automatic")
        caught = sum(1 for r in log if r["label"] == 1 and r["route"] in ("VERIFY", "HUMAN_REVIEW"))
        auto_frauds_rejected = sum(1 for r in log if r["label"] == 1 and r["outcome"] == "correct automatic")
        print(f"Known frauds: {int(labels.sum())} -> {auto_frauds_rejected} rejected automatically, {caught} held for verification or review. "
              f"Incorrect automatic decisions: {wrong} of {len(log)}.")
        print("A sample this small illustrates the process; the measured result is the held-out evaluation (Audit -> Evaluation).")
    if not args.dry_run:
        print("Open http://localhost:3000/audit/decisions to see them; each can be verified against the chain.")

    out = Path(args.results_dir) / "cache" / "replays"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"replay-{args.tag}.json").write_text(json.dumps(log, indent=1, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
