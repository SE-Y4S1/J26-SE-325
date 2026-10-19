"""Loading Laya, asking it, and applying its answers to the liquidation plan's inputs.

WHICH MODEL
  1. the active fine-tuned head (model registry `laya_decision`, installed by
     decision/checkpoint.py from Colab Part C), else
  2. the base checkpoint zero-shot, if its weights are already cached, else
  3. none -- every holding is planned by the fuzzy rules alone, and the response says why.
Loading an ~842 MB checkpoint on CPU takes tens of seconds, so it happens on a background
thread; a request that arrives first is planned without Laya rather than made to wait.

WHERE IT RUNS
  Stage 1 (analysis: technical / fundamental / news) depends only on the holding, so it runs
  in the background preparation pipeline and is stored with the other analysis.
  Stage 2 (decision) depends on the request (deadline, target, position size), so it runs on
  the request path: one batched forward pass for all holdings plus one for the portfolio.

HOW ANSWERS ARE USED (confidence >= CONFIDENCE_THRESHOLD, else ignored and recorded)
  action sell_now / defer  -> +/- ACTION_PRIORITY_POINTS on the fuzzy sell priority
  action exclude           -> the holding's sell fraction capped at 0, only if the others can
                              still cover the target with margin
  downside_risk            -> + DOWNSIDE_PRIORITY_POINTS when likely
  urgency / risk_tolerance -> replace the request's values (score level / 4)
The GA still computes every amount.
"""

from __future__ import annotations

import logging
import threading
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from decision import questions as Q
from decision.states import (
    decision_state,
    fundamental_state,
    news_state,
    portfolio_state,
    summarise_stage1,
    technical_state,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "laya_decision"
CHECKPOINT_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "checkpoints" / "laya"
BASE_REPO = "convaiinnovations/laya"
CONFIDENCE_THRESHOLD = 0.6
ACTION_PRIORITY_POINTS = 15.0
DOWNSIDE_PRIORITY_POINTS = 5.0
EXCLUDE_COVER_MARGIN = 1.05


class LayaEngine:
    def __init__(self, agent: Any, version: str) -> None:
        self.agent = agent
        self.version = version

    def ask(self, states: list[dict[str, Any]], questions: dict[str, Any]) -> list[dict[str, dict]]:
        """Answers per state for one question set, in one batched pass."""
        if not states:
            return []
        results = self.agent.predict_batch(states, questions)
        return [r.get("answers", {}) for r in results]


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------

_engine: LayaEngine | None = None
_reason = "not loaded yet"
_lock = threading.Lock()
_loader: threading.Thread | None = None


def _base_cached() -> bool:
    hub = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{BASE_REPO.replace('/', '--')}"
    return hub.exists() and any(f.stat().st_size > 1_000_000 for f in hub.rglob("model.safetensors"))


def _load() -> tuple[LayaEngine | None, str]:
    if find_spec("laya") is None:
        return None, "laya is not installed (uv sync --extra decision)"
    import laya

    try:
        from forecasting.model_registry import get_active_version, get_record

        version = get_active_version(MODEL_NAME)
        if version != "unregistered":
            record = get_record(version)
            directory = CHECKPOINT_ROOT / record.content_hash[:12] if record else None
            if directory and (directory / "model.safetensors").exists():
                return LayaEngine(laya.load(str(directory), device="cpu"), version), "fine-tuned"
            logger.warning("active %s has no checkpoint at %s", version, directory)
        if _base_cached():
            import os

            os.environ.setdefault("HF_HUB_OFFLINE", "1")   # never start an 842 MB download here
            return LayaEngine(laya.load(BASE_REPO, device="cpu"), f"laya-base-{laya.__version__}-zero-shot"), "zero-shot"
    except Exception as exc:  # noqa: BLE001
        logger.warning("laya failed to load: %s", exc)
        return None, f"laya failed to load: {type(exc).__name__}: {exc}"
    return None, ("no fine-tuned Laya head installed and the base checkpoint is not cached "
                  f"(huggingface-cli download {BASE_REPO})")


def _load_in_background() -> None:
    global _engine, _reason
    engine, reason = _load()
    with _lock:
        _engine, _reason = engine, reason
    logger.info("laya engine: %s", reason if engine is None else f"{engine.version} ({reason})")


def get_engine(*, block: bool = False) -> tuple[LayaEngine | None, str]:
    """(engine, reason). Starts loading on first call; non-blocking callers get None + 'loading'
    until it is ready."""
    global _loader, _reason
    with _lock:
        if _engine is not None:
            return _engine, "ready"
        if _loader is None:
            _reason = "loading"
            _loader = threading.Thread(target=_load_in_background, name="laya-loader", daemon=True)
            _loader.start()
        loader = _loader
    if block:
        loader.join()
    with _lock:
        return _engine, _reason


def reset() -> None:
    """Testing only."""
    global _engine, _reason, _loader
    with _lock:
        _engine, _reason, _loader = None, "not loaded yet", None


def set_engine(engine: LayaEngine | None, reason: str = "ready") -> None:
    """Testing / tooling: install an engine directly."""
    global _engine, _reason, _loader
    done = threading.Thread(target=lambda: None)   # a finished "load", so blocking callers return
    done.start()
    done.join()
    with _lock:
        _engine, _reason, _loader = engine, reason, done


# --------------------------------------------------------------------------------------
# Stage 1 (background preparation)
# --------------------------------------------------------------------------------------

def stage1_states(metrics: dict[str, Any], asset_class: str) -> dict[str, dict[str, Any]]:
    """The state for each Stage 1 group that has data."""
    states = {"technical": technical_state(metrics.get("technical"), asset_class=asset_class)}
    fundamental = fundamental_state(metrics.get("fundamentals"), asset_class=asset_class)
    if fundamental:
        states["fundamental"] = fundamental
    news = news_state(metrics.get("sentiment"))
    if news:
        states["news"] = news
    return states


def analyse(engine: LayaEngine, metrics: dict[str, Any], asset_class: str) -> dict[str, Any]:
    groups = {}
    for group, state in stage1_states(metrics, asset_class).items():
        [answers] = engine.ask([state], Q.STAGE1[group])
        groups[group] = answers
    merged = {qid: answer for answers in groups.values() for qid, answer in answers.items()}
    return {"available": True, "model_version": engine.version, "groups": groups,
            "summary": summarise_stage1(merged)}


def analysis_step(instrument, metrics: dict[str, Any]) -> tuple[bool, dict[str, Any] | str]:  # noqa: ANN001
    """Active-set step. Laya being absent is a normal configuration, not a failure: the step
    succeeds with `available: false` and the reason, so the symbol is not marked `partial`."""
    engine, reason = get_engine(block=True)
    if engine is None:
        return True, {"available": False, "reason": reason}
    return True, analyse(engine, metrics, instrument.asset_class.value)


analysis_step.needs_metrics = True  # type: ignore[attr-defined]


# --------------------------------------------------------------------------------------
# Stage 2 (request path) and application to the plan inputs
# --------------------------------------------------------------------------------------

def _position(holding: dict[str, Any], total: float, participation_cap: float) -> dict[str, Any]:
    value = float(holding.get("value", 0.0))
    adv = float(holding.get("adv_usd", 0.0))
    basis = holding.get("cost_basis")
    price = float(holding.get("price", 0.0))
    return {
        "share_of_portfolio": value / total if total else 0.0,
        "days_to_exit": value / (participation_cap * adv) if adv > 0 else 999.0,
        "unrealized_pnl_pct": (price / float(basis) - 1.0) if basis else 0.0,
        "needs_fx": bool(holding.get("fx_cost_pct")),
    }


def _confident(answer: dict[str, Any] | None) -> bool:
    return bool(answer) and float(answer.get("confidence", 0.0)) >= CONFIDENCE_THRESHOLD


def apply(
    state: dict[str, dict[str, Any]],
    snapshots: dict[str, dict[str, Any] | None],
    inputs: dict[str, Any],
    *,
    participation_cap: float = 0.10,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    """Run Stage 2 and fold confident answers into the GA/fuzzy inputs.

    Returns (state, inputs, per-holding decision records, meta). `state`/`inputs` are new
    dicts; the caller's are untouched. Without an engine, or for a holding without Stage 1
    analysis, nothing changes and the record says why.
    """
    engine, reason = get_engine(block=False)
    state = {s: dict(h) for s, h in state.items()}
    inputs = dict(inputs)
    if engine is None:
        return state, inputs, {s: {"source": "fuzzy", "reason": reason} for s in state}, \
            {"source": "fuzzy", "reason": reason}

    total = sum(float(h.get("value", 0.0)) for h in state.values())
    request = {"deadline_days": inputs["deadline_days"], "urgency": inputs["urgency"],
               "target_share": inputs["liquidity_target"] / total if total else 1.0}
    decisions: dict[str, dict[str, Any]] = {}
    asked: list[str] = []
    stage2_states: list[dict[str, Any]] = []
    for symbol, holding in state.items():
        analysis = ((snapshots.get(symbol) or {}).get("metrics") or {}).get("laya_analysis") or {}
        if not analysis.get("available"):
            decisions[symbol] = {"source": "fuzzy", "reason": analysis.get("reason") or "no Stage 1 analysis yet"}
            continue
        forecast = (((snapshots.get(symbol) or {}).get("metrics") or {}).get("forecast") or {})
        band = forecast.get("h5" if inputs["deadline_days"] <= 10 else "h21")
        asked.append(symbol)
        stage2_states.append(decision_state(stage1=analysis.get("summary") or {}, forecast=band,
                                            position=_position(holding, total, participation_cap), request=request))

    for symbol, answers in zip(asked, engine.ask(stage2_states, Q.DECISION), strict=True):
        action, downside = answers.get("action"), answers.get("downside_risk")
        record: dict[str, Any] = {"source": "laya", "model_version": engine.version, "action": action,
                                  "downside_risk": downside, "applied": []}
        holding = state[symbol]
        adjustment = 0.0
        if _confident(action):
            choice = action["choice"]
            if choice == "sell_now":
                adjustment += ACTION_PRIORITY_POINTS
            elif choice == "defer":
                adjustment -= ACTION_PRIORITY_POINTS
            elif choice == "exclude":
                others = total - float(holding.get("value", 0.0))
                if others >= inputs["liquidity_target"] * EXCLUDE_COVER_MARGIN:
                    holding["max_sell_fraction"] = 0.0
                else:
                    record["applied"].append("exclude ignored: other holdings cannot cover the target")
            if choice != "spread" and not record["applied"]:
                record["applied"].append(f"action {choice}")
        else:
            record["applied"].append("action below confidence threshold; fuzzy rules decide")
        if downside and float(downside.get("noul", 0.0)) >= CONFIDENCE_THRESHOLD:
            adjustment += DOWNSIDE_PRIORITY_POINTS
            record["applied"].append("downside risk likely")
        if adjustment:
            holding["laya_priority_adjustment"] = adjustment
        decisions[symbol] = record

    meta: dict[str, Any] = {"source": "laya" if asked else "fuzzy", "model_version": engine.version}
    if asked:
        summary = [{"symbol": s, **_position(state[s], total, participation_cap)} for s in asked]
        [portfolio] = engine.ask([portfolio_state(holdings=summary, request=request)], Q.PORTFOLIO)
        for key in ("urgency", "risk_tolerance"):
            answer = portfolio.get(key)
            if _confident(answer):
                levels = len(Q.PORTFOLIO[key]["criteria"]) - 1
                meta[f"requested_{key}"] = inputs[key]
                inputs[key] = round(float(answer["score"]) / levels, 4)
        meta["portfolio_answers"] = portfolio
    else:
        meta["reason"] = "no holding has Stage 1 analysis yet"
    return state, inputs, decisions, meta
