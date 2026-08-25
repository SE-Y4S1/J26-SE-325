"""Generate the supervised fine-tuning dataset for Colab.

Sweeps synthetic withdrawal scenarios (urgency x market condition x portfolio composition,
reusing optimization/stress_scenarios.py for variety) and writes one JSON object per
scenario containing the full message/tool-call sequence, ready for SFT.

TWO DRIVERS
-----------
  ollama   -- the reference agent. Realistic reasoning traces, but slow and non-deterministic.
  scripted -- a deterministic rule-based policy. Fast, free, reproducible, and grounded by
              construction.

The scripted policy is the DEFAULT for bulk generation. It is not a shortcut: because it
always routes through run_fuzzy_ga_withdrawal, every trajectory it emits satisfies the
grounding constraint automatically, so the fine-tuned model learns the tool-calling
discipline rather than imitating a small model's occasional shortcuts. The Ollama driver is
used for a smaller, more varied slice and as a sanity check that the tool interface behaves
under genuine model-driven reasoning.

Both drivers pass through the same enforce_grounding() check before a trajectory is written.
Rejected trajectories are logged with a reason, not silently dropped -- a rising rejection
rate is the signal that something upstream broke.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from agent.reference_agent import AgentStep, AgentTranscript, build_system_prompt, enforce_grounding
from agent.tool_schema import GROUNDING_TOOL_NAME
from agent.tools import dispatch

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "trajectories"

# Symbols used to build synthetic portfolios, spanning the liquidity spectrum. THIN and
# MICRO exist so the dataset contains cases where naive liquidation is clearly wrong.
_SYMBOL_POOL = [
    ("AAPL", 2.5e9, 0.014), ("MSFT", 2.0e9, 0.013), ("SPY", 4.0e10, 0.009),
    ("QQQ", 2.0e10, 0.012), ("GLD", 1.5e9, 0.008), ("TLT", 1.0e9, 0.010),
    ("XLE", 8.0e8, 0.015), ("IWM", 3.0e9, 0.013),
    ("THIN", 8.0e5, 0.030), ("MICRO", 1.2e5, 0.045),
]


class Driver(StrEnum):
    SCRIPTED = "scripted"
    OLLAMA = "ollama"


@dataclass(frozen=True)
class ScenarioSpec:
    """One synthetic withdrawal situation."""

    scenario_id: str
    portfolio: dict[str, dict[str, float]]
    target_amount: float
    urgency: float
    deadline_days: int
    stress_scenario: str
    market_regime: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "portfolio": self.portfolio,
            "target_amount": self.target_amount,
            "urgency": self.urgency,
            "deadline_days": self.deadline_days,
            "market_regime": self.market_regime,
        }


def generate_scenarios(
    *,
    n_scenarios: int = 500,
    seed: int = 42,
    include_stress: bool = True,
) -> list[ScenarioSpec]:
    """Build a varied scenario grid.

    Deliberately over-samples hard cases -- high urgency, illiquid holdings, compound stress
    -- because those are where an ungrounded model is most tempted to invent a number, and
    therefore where the training signal matters most.
    """
    from optimization.stress_scenarios import ScenarioType, apply_to_holdings, make_scenario

    rng = random.Random(seed)
    scenarios: list[ScenarioSpec] = []

    for i in range(n_scenarios):
        n_holdings = rng.randint(3, 7)
        chosen = rng.sample(_SYMBOL_POOL, n_holdings)

        # Guarantee an illiquid name in ~40% of portfolios.
        if rng.random() < 0.4 and not any(s in {"THIN", "MICRO"} for s, _, _ in chosen):
            chosen[-1] = rng.choice([p for p in _SYMBOL_POOL if p[0] in {"THIN", "MICRO"}])

        portfolio: dict[str, dict[str, float]] = {}
        for symbol, adv, vol in chosen:
            value = rng.uniform(20_000, 500_000)
            portfolio[symbol] = {
                "value": round(value, 2),
                "price": round(rng.uniform(15, 400), 2),
                "adv_usd": adv,
                "daily_volatility": vol,
                "volatility_pct": round(min(1.0, vol / 0.03), 3),
            }

        total = sum(h["value"] for h in portfolio.values())
        # Over-sample large withdrawals: a 10% withdrawal is easy for every method, so it
        # carries almost no training signal.
        target = total * rng.choice([0.1, 0.25, 0.4, 0.5, 0.6, 0.75])

        urgency = rng.choice([0.1, 0.3, 0.5, 0.7, 0.9, 0.95])
        deadline = 1 if urgency > 0.7 else rng.choice([1, 2, 3, 5, 10])

        stress_name, regime = "baseline", "normal"
        if include_stress and rng.random() < 0.45:
            scenario_type = rng.choice([
                ScenarioType.ADV_COLLAPSE, ScenarioType.VOLATILITY_SPIKE, ScenarioType.COMPOUND
            ])
            stress = make_scenario(scenario_type, rng.choice([0.33, 0.66, 1.0]))
            portfolio = apply_to_holdings(portfolio, stress)
            stress_name, regime = stress.name, stress.scenario_type.value

        scenarios.append(
            ScenarioSpec(
                scenario_id=f"scn_{i:05d}",
                portfolio=portfolio,
                target_amount=round(target, 2),
                urgency=urgency,
                deadline_days=deadline,
                stress_scenario=stress_name,
                market_regime=regime,
            )
        )

    return scenarios


def scripted_policy(scenario: ScenarioSpec) -> AgentTranscript:
    """Deterministic tool-calling policy: gather context, then call the optimizer.

    Grounded by construction -- it copies the tool output verbatim into the decision and
    never composes a number itself. The reasoning trace records WHY the urgency and risk
    parameters were chosen, which is the behaviour we want the fine-tuned model to learn.
    """
    steps: list[AgentStep] = []
    tool_calls: list[str] = []
    tool_outputs: dict[str, Any] = {}
    reasoning: list[str] = []

    # 1. Assess the portfolio's liquidity profile without calling the network. The scripted
    #    driver reasons from the scenario state directly; the Ollama driver uses the tools.
    illiquid = [
        symbol for symbol, holding in scenario.portfolio.items()
        if holding["value"] / max(holding["adv_usd"], 1.0) > 0.05
    ]
    if illiquid:
        reasoning.append(
            f"{', '.join(illiquid)} exceed 5% of ADV; these need lead time and should be "
            "started early rather than left to the deadline."
        )

    high_vol = [s for s, h in scenario.portfolio.items() if h.get("volatility_pct", 0) > 0.7]
    if high_vol:
        reasoning.append(f"{', '.join(high_vol)} are in a turbulent volatility regime.")

    # 2. Choose optimizer parameters -- this is where judgement lives.
    urgency = scenario.urgency
    if scenario.deadline_days == 1 and urgency < 0.6:
        urgency = min(1.0, urgency + 0.25)
        reasoning.append("Same-day deadline; raising effective urgency above the stated level.")

    risk_tolerance = 0.3 if high_vol else 0.5
    reasoning.append(
        f"Calling {GROUNDING_TOOL_NAME} with urgency={urgency:.2f}, "
        f"risk_tolerance={risk_tolerance:.2f}, deadline={scenario.deadline_days}d."
    )

    # 3. The grounded tool call. Everything numeric comes from here.
    arguments = {
        "urgency": urgency,
        "risk_tolerance": risk_tolerance,
        "liquidity_target": scenario.target_amount,
        "portfolio_state": scenario.portfolio,
        "deadline_days": scenario.deadline_days,
    }
    observation = dispatch(GROUNDING_TOOL_NAME, arguments)
    tool_calls.append(GROUNDING_TOOL_NAME)
    if "error" not in observation:
        tool_outputs[GROUNDING_TOOL_NAME] = observation

    steps.append(AgentStep(0, " ".join(reasoning), GROUNDING_TOOL_NAME, arguments, observation))

    if "error" in observation:
        decision = None
        reasoning.append(f"Optimizer failed: {observation['error']}")
    else:
        if not observation.get("feasible", False):
            reasoning.append(
                f"Plan is infeasible: raises {observation['raised_amount']:,.0f} of "
                f"{scenario.target_amount:,.0f}. Reporting the shortfall rather than "
                "overstating what can be raised."
            )
        # Copied verbatim -- the defining property of a grounded trajectory.
        decision = {
            "assets_to_sell": observation["assets_to_sell"],
            "raised_amount": observation["raised_amount"],
            "expected_slippage_pct": observation["expected_slippage_pct"],
            "feasible": observation["feasible"],
            "reasoning_trace": reasoning,
        }

    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": json.dumps(scenario.as_dict(), default=str)},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": GROUNDING_TOOL_NAME,
                        "arguments": json.dumps(arguments, default=str),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": json.dumps(observation, default=str)},
        {"role": "assistant", "content": json.dumps(decision, default=str) if decision else ""},
    ]

    transcript = AgentTranscript(
        scenario_id=scenario.scenario_id,
        messages=messages, steps=steps, decision=decision,
        grounded=False, tool_calls_made=tool_calls, tool_outputs=tool_outputs,
    )
    transcript.grounded, transcript.grounding_error = enforce_grounding(transcript)
    return transcript


def to_sft_record(transcript: AgentTranscript, scenario: ScenarioSpec) -> dict[str, Any]:
    """Convert a transcript to the SFT JSON shape (messages + tool_calls + tool responses)."""
    return {
        "scenario_id": transcript.scenario_id,
        "messages": transcript.messages,
        "metadata": {
            "market_regime": scenario.market_regime,
            "stress_scenario": scenario.stress_scenario,
            "urgency": scenario.urgency,
            "deadline_days": scenario.deadline_days,
            "target_amount": scenario.target_amount,
            "n_holdings": len(scenario.portfolio),
            "tool_calls": transcript.tool_calls_made,
            "grounded": transcript.grounded,
        },
    }


def generate_dataset(
    *,
    n_scenarios: int = 500,
    driver: Driver = Driver.SCRIPTED,
    output_path: Path | None = None,
    seed: int = 42,
    include_stress: bool = True,
) -> Path:
    """Generate, validate and write the dataset as JSONL. Returns the file path."""
    driver = Driver(driver)
    scenarios = generate_scenarios(n_scenarios=n_scenarios, seed=seed, include_stress=include_stress)

    output_path = output_path or (OUTPUT_DIR / f"withdrawal_sft_{driver.value}_{n_scenarios}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if driver is Driver.OLLAMA:
        from agent.reference_agent import check_ollama_available, run_episode

        if not check_ollama_available():
            raise RuntimeError(
                "Ollama is not reachable or gemma4-e4b is not pulled. Start Ollama, or use "
                "driver=Driver.SCRIPTED."
            )

    accepted = rejected = 0
    rejection_reasons: dict[str, int] = {}

    with output_path.open("w", encoding="utf-8") as handle:
        for scenario in scenarios:
            if driver is Driver.SCRIPTED:
                transcript = scripted_policy(scenario)
            else:
                from agent.reference_agent import run_episode

                transcript = run_episode(scenario.as_dict())

            if not transcript.grounded:
                rejected += 1
                reason = transcript.grounding_error or "unknown"
                rejection_reasons[reason[:80]] = rejection_reasons.get(reason[:80], 0) + 1
                logger.debug("rejected %s: %s", scenario.scenario_id, reason)
                continue

            handle.write(json.dumps(to_sft_record(transcript, scenario), default=str) + "\n")
            accepted += 1

    total = accepted + rejected
    logger.info(
        "wrote %d/%d trajectories to %s (%.1f%% rejected)",
        accepted, total, output_path, 100 * rejected / total if total else 0.0,
    )
    if rejection_reasons:
        # Surfaced rather than swallowed: a rising rejection rate means something upstream
        # broke, and a silently short dataset is the hardest kind of bug to notice.
        for reason, count in sorted(rejection_reasons.items(), key=lambda kv: -kv[1])[:5]:
            logger.warning("  %d rejected: %s", count, reason)

    return output_path
