"""Phase 4b tests: Laya questions, states, outcome labels (validated with laya's own training
code), the engine's application to the plan, the Stage 1 step, and head install.

No Laya weights are loaded: the model is a stub with laya's answer shapes (verified against
laya 0.4.1 agent.py: choice -> choice/probabilities/confidence, noul -> noul/confidence,
score -> score/probabilities/confidence).
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from decision import engine as E
from decision import labels as L
from decision import questions as Q
from decision.states import fundamental_state, news_state, summarise_stage1, technical_state

laya_train = pytest.importorskip("laya.train")


class StubAgent:
    """Answers every question with a fixed label/probability; records what it was asked."""

    def __init__(self, choice: str = "sell_now", confidence: float = 0.9, noul: float = 0.8, score: float = 3.0):
        self.choice, self.confidence, self.noul, self.score = choice, confidence, noul, score
        self.calls: list[tuple[int, tuple[str, ...]]] = []

    def predict_batch(self, states, questions):  # noqa: ANN001
        self.calls.append((len(states), tuple(questions)))
        out = []
        for _ in states:
            answers = {}
            for qid, q in questions.items():
                if q["type"] == "choice":
                    labels = list(q["criteria"])
                    pick = self.choice if self.choice in labels else labels[0]
                    probs = {k: (self.confidence if k == pick else (1 - self.confidence) / (len(labels) - 1)) for k in labels}
                    answers[qid] = {"choice": pick, "probabilities": probs, "confidence": self.confidence}
                elif q["type"] == "noul":
                    answers[qid] = {"noul": self.noul, "confidence": max(self.noul, 1 - self.noul)}
                else:
                    answers[qid] = {"score": self.score, "probabilities": {}, "confidence": self.confidence}
            out.append({"answers": answers})
        return out


@pytest.fixture
def engine():
    agent = StubAgent()
    E.set_engine(E.LayaEngine(agent, "laya_decision-v0.1+test"))
    yield agent
    E.reset()


# --------------------------------------------------------------------------------------
# Questions and states
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("qid", sorted(Q.ALL_QUESTIONS))
def test_every_question_passes_laya_validation(qid: str) -> None:
    from laya.agent import Agent

    Agent._check_question(qid, Q.ALL_QUESTIONS[qid])
    laya_train.to_internal(qid, Q.ALL_QUESTIONS[qid])


def test_states_are_compact_and_skip_missing_groups() -> None:
    tech = technical_state({"trend": "bearish", "rsi": 28.123456, "close": 18.4}, asset_class="cse_equity")
    assert tech == {"asset_class": "cse_equity", "trend": "bearish", "rsi": 28.123}
    assert fundamental_state(None, asset_class="equity") is None
    assert fundamental_state({"usable": False, "ratios": {"roe": 1}}, asset_class="equity") is None
    record = {"usable": True, "ratios": {"roe": 0.1, "f_score": 7, "f_score_max": 9, "template": "general"},
              "flags": {"going_concern_doubt": True, "unaudited": True}}
    state = fundamental_state(record, asset_class="equity")
    assert state["f_score"] == "7/9" and state["red_flags"] == ["going_concern_doubt"]
    tokens = fundamental_state({"kind": "tokenomics", "tokenomics": {"mcap_to_fdv": 0.9}}, asset_class="crypto")
    assert tokens["kind"] == "tokenomics"
    assert news_state({"items_relevant": 0}) is None
    assert len(json.dumps(state)) < 1200          # far inside 512 tokens


def test_summarise_stage1() -> None:
    summary = summarise_stage1({"trend": {"choice": "bearish", "confidence": 0.71},
                                "distress_risk": {"noul": 0.123, "confidence": 0.88},
                                "financial_health": {"score": 1.6, "confidence": 0.5}})
    assert summary == {"trend": "bearish (0.71)", "distress_risk": 0.12, "financial_health": 1.6}


# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------

def test_three_way_soft_edges() -> None:
    assert L.three_way(0.10, 0.03, ("up", "flat", "down"))["up"] > 0.99
    assert L.three_way(0.0, 0.03, ("up", "flat", "down"))["flat"] > 0.9
    edge = L.three_way(0.03, 0.03, ("up", "flat", "down"))
    assert 0.4 < edge["up"] < 0.6


def test_action_costs_follow_the_realised_path() -> None:
    falling = np.linspace(100, 90, 6)
    rising = np.linspace(100, 110, 6)
    kw = {"value": 1e5, "adv": 1e7, "sigma": 0.01, "deadline": 5}
    down = L.action_costs(falling, others_best=0.001, **kw)
    up = L.action_costs(rising, others_best=0.001, **kw)
    assert min(down, key=down.get) == "sell_now"
    assert min(up, key=up.get) in ("exclude", "defer")
    assert L.action_costs(falling, others_best=None, **kw)["exclude"] == float("inf")
    probs = L.softmax_costs(down)
    assert sum(probs.values()) == pytest.approx(1.0) and probs["exclude"] < probs["sell_now"]


def bars(n=200, drift=0.001, symbol="X") -> pd.DataFrame:
    rng = np.random.default_rng(3)
    close = 50 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))
    return pd.DataFrame({"timestamp": pd.bdate_range("2024-01-01", periods=n), "symbol": symbol,
                         "asset_class": "cse_equity", "open": np.nan, "high": close * 1.01,
                         "low": close * 0.99, "close": close, "volume": np.full(n, 5e4)})


def _validate_with_laya(rows: list[dict]) -> None:
    for row in rows:
        for qid, question in row["questions"].items():
            q = laya_train.to_internal(qid, question)
            target = laya_train.target_from_gold(q, row["gold"][qid])
            assert sum(target) == pytest.approx(1.0)


def test_stage1_rows_are_point_in_time_and_laya_compatible() -> None:
    frame = bars()
    record = {"usable": True, "available_from": "2024-03-01T00:00:00+00:00",
              "ratios": {"roe": 0.1, "f_score": 6, "f_score_max": 9, "z_zone": "grey", "earnings_growth_yoy": 0.1}}
    later = {"usable": True, "available_from": "2024-06-01T00:00:00+00:00",
             "ratios": {"roe": 0.05, "f_score": 4, "f_score_max": 9, "earnings_growth_yoy": -0.2}}
    rows = L.stage1_rows("X", frame, asset_class="cse_equity", records=[record, later])
    groups = {r["meta"]["group"] for r in rows}
    assert groups == {"technical", "fundamental"}
    fundamental = [r for r in rows if r["meta"]["group"] == "fundamental"]
    # No fundamental row before the first report was uploaded.
    assert min(r["meta"]["date"] for r in fundamental) >= "2024-03-01"
    # Rows before the later report learn that it deteriorated.
    early = [r for r in fundamental if r["meta"]["date"] < "2024-06-01" and "earnings_deteriorating" in r["gold"]]
    assert early and early[0]["gold"]["earnings_deteriorating"]["probabilities"]["true"] == pytest.approx(0.9)
    _validate_with_laya(rows)


def test_scenario_rows_are_laya_compatible() -> None:
    panel = {f"S{i}": bars(drift=d, symbol=f"S{i}") for i, d in enumerate((-0.004, 0.0, 0.004))}
    rows = L.scenario_rows(panel, n_scenarios=15, rng=random.Random(1),
                           stage1_summary=lambda s, w: {"trend": "neutral (0.50)"},
                           asset_classes={s: "cse_equity" for s in panel})
    assert {r["meta"]["group"] for r in rows} == {"decision", "portfolio"}
    _validate_with_laya(rows)


def test_split_by_date() -> None:
    rows = [{"meta": {"date": f"2024-01-{d:02d}"}} for d in range(1, 11)]
    train, test, cutoff = L.split_by_date(rows)
    assert cutoff == "2024-01-09" and len(train) == 8 and len(test) == 2
    assert max(r["meta"]["date"] for r in train) < min(r["meta"]["date"] for r in test)


def test_evaluate_scores_argmax_and_brier() -> None:
    row = {"gold": {"trend": {"probabilities": {"bullish": 1.0, "neutral": 0.0, "bearish": 0.0}}}}
    answers = {"trend": {"choice": "bullish", "probabilities": {"bullish": 0.8, "neutral": 0.1, "bearish": 0.1},
                         "confidence": 0.8}}
    table = L.evaluate([(row, answers)])
    assert table.loc[0, "accuracy"] == 1.0
    assert table.loc[0, "brier"] == pytest.approx(0.04 + 0.01 + 0.01)


# --------------------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------------------

def holding(value=1e6, adv=1e7) -> dict:
    return {"value": value, "price": 10.0, "adv_usd": adv, "cost_basis": 8.0, "daily_volatility": 0.01}


def analysed() -> dict:
    return {"metrics": {"laya_analysis": {"available": True, "summary": {"trend": "bearish (0.80)"}},
                        "forecast": {"h5": {"horizon": 5, "method": "empirical", "p10": -0.05, "p50": 0.0, "p90": 0.05}}}}


INPUTS = {"urgency": 0.3, "risk_tolerance": 0.5, "liquidity_target": 5e5, "deadline_days": 3}


def test_apply_folds_confident_answers_into_inputs(engine) -> None:
    state = {"A": holding(), "B": holding()}
    new_state, inputs, decisions, meta = E.apply(state, {"A": analysed(), "B": None}, INPUTS)
    assert new_state["A"]["laya_priority_adjustment"] == E.ACTION_PRIORITY_POINTS + E.DOWNSIDE_PRIORITY_POINTS
    assert "laya_priority_adjustment" not in new_state["B"]
    assert decisions["A"]["source"] == "laya" and decisions["B"]["source"] == "fuzzy"
    assert inputs["urgency"] == pytest.approx(3.0 / 4) and meta["requested_urgency"] == 0.3
    assert "laya_priority_adjustment" not in state["A"]          # caller's dict untouched
    # One batched Stage 2 pass for the holdings, one for the portfolio.
    assert engine.calls == [(1, tuple(Q.DECISION)), (1, tuple(Q.PORTFOLIO))]


def test_low_confidence_answers_are_ignored(engine) -> None:
    engine.confidence, engine.noul = 0.4, 0.3
    new_state, inputs, decisions, _ = E.apply({"A": holding()}, {"A": analysed()}, INPUTS)
    assert "laya_priority_adjustment" not in new_state["A"]
    assert inputs["urgency"] == 0.3
    assert "below confidence threshold" in decisions["A"]["applied"][0]


def test_exclude_only_when_others_can_cover(engine) -> None:
    engine.choice = "exclude"
    state = {"A": holding(value=1e6), "B": holding(value=1e6)}
    covered, *_ = E.apply(state, {"A": analysed(), "B": analysed()}, INPUTS)
    assert covered["A"]["max_sell_fraction"] == 0.0
    tight = {**INPUTS, "liquidity_target": 1.5e6}
    uncovered, _, decisions, _ = E.apply(state, {"A": analysed(), "B": analysed()}, tight)
    assert "max_sell_fraction" not in uncovered["A"]
    assert "cannot cover" in decisions["A"]["applied"][0]


def test_without_engine_nothing_changes() -> None:
    E.set_engine(None, "no fine-tuned Laya head installed")
    try:
        state, inputs, decisions, meta = E.apply({"A": holding()}, {"A": analysed()}, INPUTS)
        assert inputs == INPUTS and meta["source"] == "fuzzy"
        assert decisions["A"] == {"source": "fuzzy", "reason": "no fine-tuned Laya head installed"}
    finally:
        E.reset()


def test_excluded_holding_is_not_sold() -> None:
    from optimization.ga_withdrawal import GAConfig, optimize_withdrawal

    holdings = {"KEEP": {**holding(), "max_sell_fraction": 0.0}, "SELL": holding()}
    plan = optimize_withdrawal(holdings, 4e5, deadline_days=3, config=GAConfig(population_size=30, n_generations=20))
    assert {step.symbol for step in plan.steps} == {"SELL"}


def test_stage1_step_reads_this_runs_metrics(engine, tmp_path, monkeypatch) -> None:
    from data import active_set
    from data.instruments import Exchange, Instrument
    from data.schema import AssetClass

    monkeypatch.setattr(active_set, "REGISTRY_PATH", tmp_path / "a.sqlite")
    seen = {}

    def liquidity(instrument):  # noqa: ANN001
        return True, {"daily_volatility": 0.02}

    def technical(instrument):  # noqa: ANN001
        return True, {"trend": "bearish", "rsi": 25.0}

    def laya_step(instrument, metrics):  # noqa: ANN001
        seen.update(metrics)
        return E.analysis_step(instrument, metrics)

    laya_step.needs_metrics = True
    monkeypatch.setattr(active_set, "PIPELINE", [("liquidity", liquidity), ("technical", technical),
                                                 ("laya_analysis", laya_step)])
    jkh = Instrument(symbol="JKH.N0000", name="JKH", exchange=Exchange.CSE, asset_class=AssetClass.CSE_EQUITY,
                     currency="LKR", provider_id="508")
    active_set.track([jkh])
    record = active_set.prepare("JKH.N0000")
    assert seen["technical"]["trend"] == "bearish"
    analysis = record["metrics"]["laya_analysis"]
    assert analysis["available"] and "trend" in analysis["groups"]["technical"]
    assert record["status"] == "ready"


def test_unavailable_laya_does_not_mark_symbol_partial() -> None:
    E.set_engine(None, "base checkpoint not cached")
    try:
        ok, block = E.analysis_step(None, {})
        assert ok and block == {"available": False, "reason": "base checkpoint not cached"}
    finally:
        E.reset()


# --------------------------------------------------------------------------------------
# Head export / install
# --------------------------------------------------------------------------------------

def test_head_export_and_install_round_trip(tmp_path, monkeypatch) -> None:
    import torch
    from safetensors.torch import load_file, save_file

    from decision import checkpoint
    from forecasting import model_registry

    monkeypatch.setattr(model_registry, "REGISTRY_PATH", tmp_path / "registry.sqlite")
    monkeypatch.setattr(E, "CHECKPOINT_ROOT", tmp_path / "laya")

    base = tmp_path / "base"
    (base / "encoder").mkdir(parents=True)
    (base / "tokenizer").mkdir()
    (base / "encoder" / "config.json").write_text("{}")
    save_file({"encoder.w": torch.zeros(2, dtype=torch.float16), "head.w": torch.zeros(2, dtype=torch.float16)},
              str(base / "model.safetensors"))
    (base / "rl_agent_config.json").write_text(json.dumps({"temperature": [1.0, 1.0, 1.0]}))

    tuned = tmp_path / "tuned"
    tuned.mkdir()
    save_file({"encoder.w": torch.zeros(2), "head.w": torch.ones(2)}, str(tuned / "model.safetensors"))
    (tuned / "rl_agent_config.json").write_text(json.dumps({"temperature": [1.3, 0.9, 1.1]}))

    archive = checkpoint.export_head(tuned, tmp_path / "laya_head.zip", metrics={"accuracy_trend": 0.61},
                                     train_window={"start": "2021-01-01", "end": "2026-01-01"})
    version = checkpoint.install_head(archive, base_dir=base)
    assert version.startswith("laya_decision-v0.1")
    record = model_registry.get_record(version)
    merged_dir = tmp_path / "laya" / record.content_hash[:12]
    merged = load_file(str(merged_dir / "model.safetensors"))
    assert merged["head.w"].tolist() == [1.0, 1.0] and merged["encoder.w"].tolist() == [0.0, 0.0]
    assert json.loads((merged_dir / "rl_agent_config.json").read_text())["temperature"] == [1.3, 0.9, 1.1]
    assert record.metrics == {"accuracy_trend": 0.61}


# --------------------------------------------------------------------------------------
# Endpoint
# --------------------------------------------------------------------------------------

def test_withdraw_reports_laya_decisions(engine, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from data import active_set
    from service import events as events_mod
    from service.api import app
    from service.events import NullProducer

    monkeypatch.setattr(events_mod, "_producer_cache", {"producer": NullProducer()})
    monkeypatch.setattr(active_set, "touch", lambda symbols: None)
    monkeypatch.setattr(active_set, "metrics_for", lambda symbol: None)
    monkeypatch.setattr(active_set, "snapshot_for", lambda symbol: {**analysed(), "status": "ready", "age_hours": 1.0}
                        if symbol == "A" else None)
    holdings = [{"symbol": "A", "quantity": 1000, "current_price": 100.0, "avg_daily_volume": 1e7},
                {"symbol": "B", "quantity": 1000, "current_price": 100.0, "avg_daily_volume": 1e7}]
    body = TestClient(app).post("/portfolio/withdraw", json={"holdings": holdings, "target_amount": 50_000,
                                                             "deadline_days": 3}).json()
    assert body["market_context"]["A"]["decision"]["source"] == "laya"
    assert body["market_context"]["B"]["decision"]["source"] == "fuzzy"
    assert body["effective_inputs"]["decision"]["source"] == "laya"
    trace_a = next(t for t in body["fuzzy_rule_trace"] if t["symbol"] == "A")
    assert trace_a["laya_adjustment"] == E.ACTION_PRIORITY_POINTS + E.DOWNSIDE_PRIORITY_POINTS
