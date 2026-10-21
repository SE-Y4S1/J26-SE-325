"""Tests of assurance.replay on the synthetic fixture (code checks only; never results)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from assurance import replay, run
from tests.fixture import make_ieee_like


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> tuple[Path, Path]:
    data = make_ieee_like(tmp_path_factory.mktemp("ieee"))
    results = tmp_path_factory.mktemp("results")
    run.main(["tune", "--data", str(data), "--results-dir", str(results), "--fast", "--no-export"])
    return data, results


def replay_log(results: Path, tag: str) -> list[dict]:
    return json.loads((results / "cache" / "replays" / f"replay-{tag}.json").read_text(encoding="utf-8"))


def test_tune_saves_the_models_replay_needs(trained):
    _, results = trained
    assert (results / "cache" / run.MODELS_NAME).exists()


def test_missing_models_are_refused_with_a_clear_message(tmp_path):
    with pytest.raises(SystemExit, match="assurance.run tune"):
        replay.main(["--test-sample", "3", "--results-dir", str(tmp_path), "--dry-run"])


def test_test_sample_sends_real_model_scores_and_never_the_label(trained):
    data, results = trained
    assert replay.main(["--test-sample", "8", "--frauds", "2", "--data", str(data), "--results-dir", str(results),
                        "--dry-run", "--tag", "t1"]) == 0
    log = replay_log(results, "t1")
    assert len(log) == 8 and sum(r["label"] for r in log) == 2
    for entry in log:
        body = entry["request"]
        assert set(body["signals"]["modelScores"]) == {"logistic", "forest", "boosting"}
        assert all(0 <= v <= 1 for v in body["signals"]["modelScores"].values())
        assert body["signals"]["rules"] == [] and body["signals"]["benchmarkModel"] is True
        assert body["transactionType"] == "PAYMENT" and body["amount"] > 0 and 0 <= body["riskScore"] <= 100
        assert body["transactionId"].startswith("IEEE-") and body["transactionId"].endswith("-t1")
        assert "isFraud" not in json.dumps(body) and "label" not in body


def test_the_service_response_is_reported_against_the_label(trained, monkeypatch):
    data, results = trained
    sent = []

    def fake_post(url, body, token):
        sent.append((url, body))
        # Route everything AUTO and REJECT, so legitimate rows are wrong automatic decisions.
        return 200, {"assurance": {"route": "AUTO", "aiDecision": "REJECT", "assuranceScore": 0.9}, "blockchain": {"blockNumber": 42}}

    monkeypatch.setattr(replay, "post_json", fake_post)
    assert replay.main(["--test-sample", "4", "--frauds", "1", "--data", str(data), "--results-dir", str(results),
                        "--service", "http://audit:8002", "--tag", "t2"]) == 0
    assert all(url == "http://audit:8002/api/audit/evaluate" for url, _ in sent)
    outcomes = [r["outcome"] for r in replay_log(results, "t2")]
    assert outcomes.count("correct automatic") == 1 and outcomes.count("WRONG automatic") == 3


def test_an_unreachable_service_stops_the_replay(trained, monkeypatch):
    data, results = trained
    monkeypatch.setattr(replay, "post_json", lambda *a: (0, {"message": "connection refused"}))
    assert replay.main(["--test-sample", "2", "--data", str(data), "--results-dir", str(results), "--tag", "t3"]) == 2


def test_a_messy_csv_is_scored_where_possible_and_bad_rows_are_reported(trained, tmp_path, capsys):
    _, results = trained
    csv = tmp_path / "rows.csv"
    pd.DataFrame({
        "TransactionID": [3_100_001, 3_100_002, 3_100_003, None, 3_100_005],
        "TransactionAmt": [49.95, 0, "abc", 10, 1200.0],
        "ProductCD": ["W", "C", "W", "W", "NEVER_SEEN"],
    }).to_csv(csv, index=False)
    identity = tmp_path / "identity.csv"
    pd.DataFrame({"TransactionID": [3_100_005], "id-01": [-5.0], "DeviceType": ["mobile"]}).to_csv(identity, index=False)

    assert replay.main(["--csv", str(csv), "--identity", str(identity), "--results-dir", str(results),
                        "--dry-run", "--tag", "t4"]) == 0
    printed = capsys.readouterr().out
    assert "treated as missing values" in printed
    assert "amount 0 is not a positive number" in printed or "amount '0' is not a positive number" in printed
    assert "no usable TransactionID" in printed
    log = replay_log(results, "t4")
    assert [r["request"]["transactionId"] for r in log] == ["IEEE-3100001-t4", "IEEE-3100005-t4"]
    assert all(r["label"] is None for r in log)
