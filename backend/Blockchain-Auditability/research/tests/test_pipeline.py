"""Tests of the evaluation pipeline's code (on a synthetic fixture; never reported as results)."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from assurance import config, run
from assurance.data import load_ieee_cis, time_split
from assurance.engine import assess, canonical_json, js_round6, margin_uncertainty, policy_hash_v2
from assurance.evaluate import bootstrap, point_metrics
from assurance.features import Preprocessor
from assurance.routers import AUTO, HUMAN, VERIFY, fit_all, fit_assurance, fit_p001_deployed, incorrect_automatic
from assurance.signals import first_seen_flags
from tests.fixture import make_ieee_like

P002_V1 = json.loads((config.POLICY_DIR / "P002-assurance-v1.json").read_text(encoding="utf-8"))["document"]


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory) -> Path:
    return make_ieee_like(tmp_path_factory.mktemp("ieee"))


# --- engine port -----------------------------------------------------------------------------

def test_policy_hash_matches_the_javascript_registry():
    # The value the service computed and registered on-chain for P002 v1.0.
    assert policy_hash_v2(P002_V1) == "0xa2a3d4725c434cbbd0cd9ec5567a9b06b78a2687a02002b26ea249640c087e5c"


@pytest.mark.parametrize("value,expected", [(1.0, "1"), (0.35, "0.35"), (1e-7, "1e-7"), (100000, "100000"), (0.1 + 0.2, "0.30000000000000004")])
def test_canonical_json_formats_numbers_like_javascript(value, expected):
    assert canonical_json(value) == expected


def test_canonical_json_sorts_keys_at_every_depth():
    assert canonical_json({"b": {"y": 1, "x": [3, 1]}, "a": True}) == '{"a":true,"b":{"x":[3,1],"y":1}}'


def test_js_rounding_rounds_halves_up():
    assert js_round6(0.0000005) == 0.000001
    assert js_round6(0.1234565) == pytest.approx(0.123457)


def test_engine_port_reproduces_known_routes():
    validated = {"benchmarkModel": True, "sourceModelVersion": "fraud-bundle-2026-10"}
    clean = assess({"riskScore": 10, "confidence": 90, "amount": 500, "transactionType": "PAYMENT",
                    "signals": {"modelScores": {"lstm": 0.1, "gnn": 0.12}, "rules": [], **validated}}, P002_V1)
    assert clean["route"] == "AUTO" and clean["signalQuality"] == "FULL"
    disagree = assess({"riskScore": 35, "confidence": 90, "amount": 2500, "transactionType": "PORTFOLIO_WITHDRAWAL",
                       "signals": {"modelScores": {"lstm": 0.1, "gnn": 0.75}, "rules": [], **validated}}, P002_V1)
    assert disagree["route"] == "HUMAN_REVIEW" and disagree["vetoes"] == ["MODEL_DISAGREEMENT"]
    # The live demonstration's value: assurance score 0.44942 for this input.
    assert disagree["assuranceScore"] == pytest.approx(0.44942, abs=1e-6)
    heuristic = assess({"riskScore": 0.998, "confidence": 50, "amount": 9980, "transactionType": "PORTFOLIO_WITHDRAWAL",
                        "signals": {"modelScores": {}, "rules": ["score_below_step_up_threshold"], "benchmarkModel": False,
                                    "sourceModelVersion": "heuristic-untrained"}}, P002_V1)
    assert heuristic["route"] == "VERIFY" and heuristic["degradedSignals"] == ["disagreement"]


def test_margin_uncertainty_matches_the_engine_formula():
    t = 0.0909
    assert margin_uncertainty(t, t) == pytest.approx(1.0)
    assert margin_uncertainty(1.0, t) == pytest.approx(0.0)
    assert margin_uncertainty(0.0, t) == pytest.approx(0.0)


# --- data and leakage --------------------------------------------------------------------------

def test_time_split_is_ordered_disjoint_and_shares_no_timestamp(fixture_dir):
    splits = time_split(load_ieee_cis(fixture_dir))
    names = list(config.SPLITS)
    seen = set()
    for a, b in zip(names, names[1:]):
        ta, tb = splits.part(a)["TransactionDT"], splits.part(b)["TransactionDT"]
        assert ta.max() < tb.min(), f"{a} and {b} overlap in time"
    for name in names:
        ids = set(splits.part(name)["TransactionID"])
        assert not ids & seen
        seen |= ids
    assert len(seen) == len(splits.frame)


def test_preprocessing_learns_only_from_the_training_split(fixture_dir):
    splits = time_split(load_ieee_cis(fixture_dir))
    train, test = splits.part("train"), splits.part("test").copy()
    pre = Preprocessor().fit(train)
    assert pre.medians["C1"] == pytest.approx(pd.to_numeric(train["C1"]).median())
    test["C1"] = np.nan
    test["ProductCD"] = "NEVER_SEEN"
    x = pd.DataFrame(pre.transform(test), columns=pre.columns)
    assert (x["C1"] == pre.medians["C1"]).all()        # imputed with the train median
    assert (x["freq__ProductCD"] == 0).all()           # unseen category: no value from later data


def test_first_seen_card_looks_only_backwards():
    frame = pd.DataFrame({"card1": [5, 6, 5, None, 6, 7]})
    assert first_seen_flags(frame).tolist() == [1, 1, 0, 1, 0, 1]
    # Appending future rows never changes an earlier flag.
    longer = pd.concat([frame, pd.DataFrame({"card1": [8, 5]})], ignore_index=True)
    assert first_seen_flags(longer)[:6].tolist() == first_seen_flags(frame).tolist()


# --- routers and metrics -----------------------------------------------------------------------

def synthetic_signals(n=4000, seed=3):
    rng = np.random.default_rng(seed)
    p = rng.beta(0.6, 6, size=n)
    y = (rng.random(n) < p).astype(int)
    d = np.abs(rng.normal(0, 0.12, size=n))
    return {"y": y, "p": p, "decision": (p >= 0.0909).astype(int), "u_confidence": np.asarray(margin_uncertainty(p, 0.0909)),
            "u_disagreement": np.clip(d, 0, 1), "u_context": rng.random(n), "amount_value": rng.random(n) * 100,
            "ctx_amount": rng.random(n), "ctx_novelty": rng.random(n), "ctx_new_card": (rng.random(n) < 0.1).astype(float),
            "ctx_no_identity": (rng.random(n) < 0.6).astype(float)}


@pytest.mark.parametrize("budget", config.BUDGETS)
def test_every_budgeted_router_respects_the_budget_on_the_tuning_split(budget):
    s = synthetic_signals()
    for name, fitted in fit_all(s, budget).items():
        if name in ("auto_all", "p001_deployed"):
            continue
        share = np.mean(fitted.route(s) == HUMAN)
        assert share <= budget + 0.01, f"{name} sends {share:.3f} to review at budget {budget}"


def test_two_tier_baselines_never_use_verify_and_assurance_does():
    s = synthetic_signals()
    fitted = fit_all(s, 0.1)
    for name in ("risk_band", "confidence_only", "p001_retuned", "random_review", "learned_deferral"):
        assert not np.any(fitted[name].route(s) == VERIFY)
    assert np.any(fitted["assurance"].route(s) == VERIFY)


def test_ablations_really_drop_their_signal():
    s = synthetic_signals()
    for signal in ("confidence", "disagreement", "context"):
        assert fit_assurance(s, 0.1, fixed_zero=signal).params["weights"][signal] == 0


def test_p001_deployed_is_the_legacy_rule():
    s = {"p": np.array([0.9, 0.9, 0.5]), "u_confidence": np.array([0.5, 0.1, 0.9])}
    assert fit_p001_deployed(s, 0.1).route(s).tolist() == [HUMAN, AUTO, AUTO]


def test_metrics_on_a_hand_checked_example():
    s = {"y": np.array([1, 0, 1, 0]), "decision": np.array([0, 0, 1, 1]), "amount_value": np.array([10.0, 1, 1, 1])}
    routes = np.array([AUTO, AUTO, HUMAN, VERIFY])
    m = point_metrics(routes, s)
    # errors: row 0 (missed fraud, automatic) and row 3 (false alarm, VERIFY -> counted automatic)
    assert m["incorrect_automatic_rate"] == 0.5
    assert m["human_review_rate"] == 0.25 and m["verify_rate"] == 0.25
    assert m["fraud_auto_approved"] == 1 and m["fraud_auto_approved_amount"] == 10.0
    assert m["verify_effectiveness_sweep"]["0.5"] == pytest.approx(0.25 + 0.5 * 0.25)
    assert incorrect_automatic(routes, s) == 0.5


def test_bootstrap_is_reproducible_and_brackets_the_estimate():
    s = synthetic_signals()
    fitted = fit_all(s, 0.1)
    routes = {k: f.route(s) for k, f in fitted.items()}
    a = bootstrap(routes, s, "assurance", ["risk_band"], b=200)
    b = bootstrap(routes, s, "assurance", ["risk_band"], b=200)
    assert a == b
    lo, hi = a["intervals"]["assurance"]["incorrect_automatic_rate"]
    assert lo <= point_metrics(routes["assurance"], s)["incorrect_automatic_rate"] <= hi


# --- the two stages end to end -----------------------------------------------------------------

def test_tune_then_final_on_the_fixture(fixture_dir, tmp_path):
    results, policy_out = tmp_path / "results", tmp_path / "P002-assurance-v1.1.json"
    common = ["--data", str(fixture_dir), "--results-dir", str(results), "--policy-out", str(policy_out), "--fast"]
    run.main(["tune", *common])
    tuning = json.loads((results / "tuning.json").read_text())
    assert "test" not in json.dumps(tuning["budgets"])   # tuning output carries no test metrics
    doc = json.loads(policy_out.read_text())["document"]
    assert doc["version"] == "1.1" and doc["status"] == "tuned"
    assert math.isclose(sum(doc["weights"].values()), 1.0, abs_tol=1e-9)
    assert doc["missing"]["disagreement"] >= doc["vetoes"]["maxDisagreement"]     # registry rule
    assert doc["routing"]["verifyMin"] <= doc["routing"]["autoMin"]

    with pytest.raises(SystemExit, match="Commit"):
        run.main(["final", *common])                     # an uncommitted policy is refused

    run.main(["final", *common, "--allow-uncommitted"])
    metrics = json.loads((results / "metrics.json").read_text())
    assert metrics["status"] == "NOT_FINAL_UNCOMMITTED_POLICY"
    assert metrics["policy"]["hash"] == policy_hash_v2(doc)
    for name in ("REPORT.md", "budgets.csv", "figures/deferral_curves.png", "RUNLOG.json"):
        assert (results / name).exists()
    assert not (config.RESULTS_DIR / "metrics.json").exists() or config.RESULTS_DIR != results
