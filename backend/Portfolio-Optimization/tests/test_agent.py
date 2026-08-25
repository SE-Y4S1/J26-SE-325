"""Phase 5c tests: tool schemas, dispatch, and THE GROUNDING CONSTRAINT.

The grounding tests are the most important in this repo. The build brief's rule is that the
agent must never output a sell amount that did not come from run_fuzzy_ga_withdrawal, and
the TAF's Legal Impact section is the reason (regulators require auditable derivations for
automated financial decisions). These tests are what turn that rule from a comment into an
enforced invariant.

Note the test is stronger than "the tool was called": it verifies the decision's NUMBERS
match the tool's output. A model that calls the tool, ignores it, and states its own figures
would pass the weaker check and still be exactly the failure we are guarding against.
"""

from __future__ import annotations

import json

import pytest

from agent.reference_agent import (
    AgentTranscript,
    build_system_prompt,
    check_ollama_available,
    enforce_grounding,
)
from agent.tool_schema import (
    GROUNDING_TOOL_NAME,
    all_tool_schemas,
    tool_schema,
    validate_tool_arguments,
)
from agent.tools import TOOL_REGISTRY, dispatch
from agent.trajectory_generation import (
    Driver,
    generate_dataset,
    generate_scenarios,
    scripted_policy,
)

PORTFOLIO = {
    "LIQ": {"value": 400_000, "price": 200.0, "adv_usd": 5.0e8, "daily_volatility": 0.012, "volatility_pct": 0.4},
    "THIN": {"value": 300_000, "price": 15.0, "adv_usd": 8.0e5, "daily_volatility": 0.030, "volatility_pct": 0.8},
}


# --------------------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------------------

def test_all_schemas_are_valid_openai_function_shape() -> None:
    """The format Llama tool-calling fine-tunes expect; a shape error here costs a Colab run."""
    for schema in all_tool_schemas():
        assert schema["type"] == "function"
        function = schema["function"]
        assert isinstance(function["name"], str) and function["name"]
        assert isinstance(function["description"], str) and function["description"]

        parameters = function["parameters"]
        assert parameters["type"] == "object"
        assert isinstance(parameters["properties"], dict)
        for required in parameters.get("required", []):
            assert required in parameters["properties"], (
                f"{function['name']}: '{required}' is required but not defined"
            )


def test_every_schema_has_a_matching_implementation() -> None:
    """A schema the model can call but that has no implementation is a runtime crash."""
    schema_names = {s["function"]["name"] for s in all_tool_schemas()}
    assert schema_names == set(TOOL_REGISTRY)


def test_grounding_tool_description_forbids_inventing_numbers() -> None:
    """The description is the model's only instruction about the division of labour, so the
    prohibition must actually be in it."""
    description = tool_schema(GROUNDING_TOOL_NAME)["function"]["description"].lower()
    assert "only way" in description or "only" in description
    assert "never" in description
    assert any(word in description for word in ("invent", "estimate", "adjust"))


def test_system_prompt_states_the_hard_rule() -> None:
    prompt = build_system_prompt().lower()
    assert GROUNDING_TOOL_NAME.lower() in prompt
    assert "never" in prompt
    # It must also tell the model NOT to write end-user prose (Component 4's job).
    assert "prose" in prompt or "end users" in prompt


def test_schemas_are_json_serializable() -> None:
    """They are sent over the wire to Ollama, so anything non-serializable breaks the loop."""
    assert json.loads(json.dumps(all_tool_schemas())) == all_tool_schemas()


# --------------------------------------------------------------------------------------
# Argument validation
# --------------------------------------------------------------------------------------

def test_validate_accepts_a_well_formed_call() -> None:
    validated = validate_tool_arguments(
        GROUNDING_TOOL_NAME,
        {"urgency": 0.8, "risk_tolerance": 0.5, "liquidity_target": 100_000,
         "portfolio_state": PORTFOLIO, "deadline_days": 3},
    )
    assert validated["urgency"] == 0.8
    assert validated["deadline_days"] == 3


def test_validate_coerces_quoted_numbers() -> None:
    """Small models routinely quote their numbers; rejecting that would waste a whole turn."""
    validated = validate_tool_arguments(
        GROUNDING_TOOL_NAME,
        {"urgency": "0.8", "risk_tolerance": "0.5", "liquidity_target": "100000",
         "portfolio_state": PORTFOLIO},
    )
    assert validated["urgency"] == pytest.approx(0.8)
    assert validated["liquidity_target"] == pytest.approx(100_000.0)


def test_validate_reports_missing_arguments_by_name() -> None:
    with pytest.raises(ValueError, match="missing required argument"):
        validate_tool_arguments(GROUNDING_TOOL_NAME, {"urgency": 0.5})


def test_validate_enforces_ranges() -> None:
    with pytest.raises(ValueError, match="must be <= 1"):
        validate_tool_arguments(
            GROUNDING_TOOL_NAME,
            {"urgency": 5.0, "risk_tolerance": 0.5, "liquidity_target": 1000,
             "portfolio_state": PORTFOLIO},
        )


def test_validate_enforces_enums() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        validate_tool_arguments("get_forecast", {"symbol": "AAPL", "horizon": 7})


def test_validate_rejects_unknown_arguments() -> None:
    with pytest.raises(ValueError, match="unexpected argument"):
        validate_tool_arguments("get_technical_signals", {"symbol": "AAPL", "extra": 1})


def test_dispatch_returns_errors_as_observations_not_exceptions() -> None:
    """A crashed run yields no trajectory; a returned error lets the model self-correct."""
    result = dispatch("get_technical_signals", {"wrong_arg": "AAPL"})
    assert "error" in result
    assert "missing required argument" in result["error"]


def test_dispatch_reports_unknown_tools_helpfully() -> None:
    result = dispatch("not_a_real_tool", {})
    assert "error" in result
    assert "unknown tool" in result["error"]
    assert GROUNDING_TOOL_NAME in result["error"]      # lists what IS available


def test_dispatch_runs_the_grounded_tool_end_to_end() -> None:
    result = dispatch(
        GROUNDING_TOOL_NAME,
        {"urgency": 0.6, "risk_tolerance": 0.5, "liquidity_target": 150_000,
         "portfolio_state": PORTFOLIO, "deadline_days": 3},
    )
    assert "error" not in result, result.get("error")
    for field in ("assets_to_sell", "raised_amount", "expected_slippage_pct",
                  "residual_portfolio_weights", "fuzzy_rule_trace", "feasible"):
        assert field in result, f"grounded tool output missing {field}"
    assert result["fuzzy_rule_trace"], "fuzzy layer did not run"
    assert json.loads(json.dumps(result))              # must survive the wire


# --------------------------------------------------------------------------------------
# THE GROUNDING CONSTRAINT
# --------------------------------------------------------------------------------------

def _transcript(*, decision, tool_calls, tool_output) -> AgentTranscript:
    return AgentTranscript(
        scenario_id="t", messages=[], steps=[], decision=decision,
        grounded=False, tool_calls_made=tool_calls,
        tool_outputs={GROUNDING_TOOL_NAME: tool_output} if tool_output else {},
    )


TOOL_OUTPUT = {
    "assets_to_sell": [
        {"symbol": "LIQ", "sell_fraction": 0.25, "quantity": 500.0},
        {"symbol": "THIN", "sell_fraction": 0.10, "quantity": 2000.0},
    ],
    "raised_amount": 130_000.0,
    "expected_slippage_pct": 0.0021,
    "feasible": True,
}


def test_grounding_rejects_a_transcript_that_never_called_the_tool() -> None:
    """The headline check: reaching a decision without the optimizer is a BUG."""
    grounded, error = enforce_grounding(
        _transcript(
            decision={"assets_to_sell": [{"symbol": "LIQ", "sell_fraction": 0.4}],
                      "raised_amount": 200_000},
            tool_calls=["get_technical_signals", "get_sentiment"],
            tool_output=None,
        )
    )
    assert grounded is False
    assert "never called" in error


def test_grounding_rejects_altered_numbers_even_when_the_tool_was_called() -> None:
    """The check that actually matters. Calling the tool then ignoring its answer is exactly
    the failure mode the constraint exists to prevent."""
    tampered = json.loads(json.dumps(TOOL_OUTPUT))
    tampered["raised_amount"] = 175_000.0          # model "rounded up"

    grounded, error = enforce_grounding(
        _transcript(decision=tampered, tool_calls=[GROUNDING_TOOL_NAME], tool_output=TOOL_OUTPUT)
    )
    assert grounded is False
    assert "raised_amount was altered" in error


def test_grounding_rejects_an_altered_sell_fraction() -> None:
    tampered = json.loads(json.dumps(TOOL_OUTPUT))
    tampered["assets_to_sell"][0]["sell_fraction"] = 0.40    # was 0.25

    grounded, error = enforce_grounding(
        _transcript(decision=tampered, tool_calls=[GROUNDING_TOOL_NAME], tool_output=TOOL_OUTPUT)
    )
    assert grounded is False
    assert "sell_fraction was altered" in error


def test_grounding_rejects_a_hallucinated_symbol() -> None:
    tampered = json.loads(json.dumps(TOOL_OUTPUT))
    tampered["assets_to_sell"].append({"symbol": "NVDA", "sell_fraction": 0.5, "quantity": 10.0})

    grounded, error = enforce_grounding(
        _transcript(decision=tampered, tool_calls=[GROUNDING_TOOL_NAME], tool_output=TOOL_OUTPUT)
    )
    assert grounded is False
    assert "NVDA" in error


def test_grounding_accepts_a_faithful_copy() -> None:
    grounded, error = enforce_grounding(
        _transcript(
            decision=json.loads(json.dumps(TOOL_OUTPUT)),
            tool_calls=[GROUNDING_TOOL_NAME], tool_output=TOOL_OUTPUT,
        )
    )
    assert grounded is True, error


def test_grounding_tolerates_json_roundtrip_rounding() -> None:
    """Must not reject a faithful copy over float noise -- that would make it unusable."""
    rounded = json.loads(json.dumps(TOOL_OUTPUT))
    rounded["expected_slippage_pct"] = 0.00210000001

    grounded, _ = enforce_grounding(
        _transcript(decision=rounded, tool_calls=[GROUNDING_TOOL_NAME], tool_output=TOOL_OUTPUT)
    )
    assert grounded is True


def test_grounding_rejects_a_missing_decision() -> None:
    grounded, error = enforce_grounding(
        _transcript(decision=None, tool_calls=[GROUNDING_TOOL_NAME], tool_output=TOOL_OUTPUT)
    )
    assert grounded is False
    assert "no structured decision" in error


# --------------------------------------------------------------------------------------
# Trajectory generation
# --------------------------------------------------------------------------------------

def test_scenarios_cover_urgency_stress_and_illiquidity() -> None:
    scenarios = generate_scenarios(n_scenarios=60, seed=1)
    assert len(scenarios) == 60
    assert len({s.urgency for s in scenarios}) >= 3
    assert any(s.stress_scenario != "baseline" for s in scenarios), "no stressed scenarios"
    assert any(
        any(sym in {"THIN", "MICRO"} for sym in s.portfolio) for s in scenarios
    ), "no illiquid holdings -- the hard cases are missing"


def test_scenarios_are_reproducible_under_a_seed() -> None:
    first = generate_scenarios(n_scenarios=10, seed=7)
    second = generate_scenarios(n_scenarios=10, seed=7)
    assert [s.scenario_id for s in first] == [s.scenario_id for s in second]
    assert [s.target_amount for s in first] == [s.target_amount for s in second]


def test_scripted_policy_always_calls_the_grounded_tool() -> None:
    """Grounded by construction -- this is why it is the default bulk driver."""
    for scenario in generate_scenarios(n_scenarios=8, seed=3):
        transcript = scripted_policy(scenario)
        assert GROUNDING_TOOL_NAME in transcript.tool_calls_made
        assert transcript.grounded, transcript.grounding_error


def test_scripted_policy_produces_no_end_user_prose() -> None:
    """Component 4 owns explanation; duplicating it here is scope overlap."""
    transcript = scripted_policy(generate_scenarios(n_scenarios=1, seed=5)[0])
    assert transcript.decision is not None
    assert set(transcript.decision) <= {
        "assets_to_sell", "raised_amount", "expected_slippage_pct", "feasible", "reasoning_trace"
    }
    # reasoning_trace is an INTERNAL trace, not user-facing text.
    assert "reasoning_trace" in transcript.decision
    assert isinstance(transcript.decision["reasoning_trace"], list)


def test_scripted_policy_reports_infeasibility_honestly() -> None:
    """A plan that cannot raise the cash must say so rather than overstating."""
    from agent.trajectory_generation import ScenarioSpec

    scenario = ScenarioSpec(
        scenario_id="impossible",
        portfolio={"THIN": PORTFOLIO["THIN"]},
        target_amount=10_000_000,      # far beyond the position, let alone the daily cap
        urgency=0.9, deadline_days=1,
        stress_scenario="baseline", market_regime="normal",
    )
    transcript = scripted_policy(scenario)
    assert transcript.decision is not None
    assert transcript.decision["feasible"] is False
    assert transcript.decision["raised_amount"] < scenario.target_amount


def test_generate_dataset_writes_only_grounded_trajectories(tmp_path) -> None:
    """THE automated check the brief asks for: every accepted trajectory used the tool."""
    output = tmp_path / "sft.jsonl"
    path = generate_dataset(n_scenarios=12, driver=Driver.SCRIPTED, output_path=output, seed=11)

    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "dataset is empty"

    for line in lines:
        record = json.loads(line)
        assert record["metadata"]["grounded"] is True
        assert GROUNDING_TOOL_NAME in record["metadata"]["tool_calls"], (
            f"{record['scenario_id']} reached a decision without calling the optimizer"
        )
        # SFT shape: a tool call followed by a tool response.
        roles = [m["role"] for m in record["messages"]]
        assert roles[0] == "system" and roles[1] == "user"
        assert "tool" in roles


def test_generate_dataset_records_scenario_metadata(tmp_path) -> None:
    """Phase 7 slices RQ4 results by regime, so the metadata must survive into the dataset."""
    path = generate_dataset(
        n_scenarios=10, driver=Driver.SCRIPTED, output_path=tmp_path / "d.jsonl", seed=2
    )
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for record in records:
        for key in ("market_regime", "stress_scenario", "urgency", "deadline_days"):
            assert key in record["metadata"]


@pytest.mark.integration
@pytest.mark.slow
def test_reference_agent_against_live_ollama() -> None:
    """End-to-end against the real model. Skipped unless Ollama is running with gemma4-e4b.

    This is the check that the tool INTERFACE works under genuine model-driven reasoning --
    the reason the reference agent exists at all.

    It caught a real behaviour: without the NUDGE_AFTER_STEPS guard, gemma4-e4b cycles the
    three context tools indefinitely and never calls the optimizer, exhausting its iteration
    budget without producing a plan. That finding is why the scripted policy -- not this
    agent -- is the default driver for bulk trajectory generation.

    Trimmed to two holdings: each context tool hits yfinance and GDELT live, so a full
    seven-holding portfolio makes this a multi-minute test dominated by network latency.
    """
    if not check_ollama_available():
        pytest.skip("Ollama not running or gemma4-e4b not pulled")

    from dataclasses import replace

    from agent.reference_agent import run_episode

    scenario = generate_scenarios(n_scenarios=1, seed=99)[0]
    scenario = replace(scenario, portfolio=dict(list(scenario.portfolio.items())[:2]))

    # Override the default episode budget deliberately. EPISODE_TIMEOUT_SECONDS (300s) is a
    # SAFETY bound for anything user-facing; this test's purpose is to validate the tool
    # interface, not to enforce latency. Measured on this machine: gemma4-e4b runs ~150s per
    # turn on CPU (`ollama ps` reports 100% CPU, no GPU offload), and an episode needs
    # roughly four turns -- gather context, call the optimizer, report. 300s abandons it
    # mid-episode and tells us nothing about whether the interface works.
    transcript = run_episode(scenario.as_dict(), timeout_seconds=1200.0)

    assert transcript.tool_calls_made, "model made no tool calls at all"
    assert GROUNDING_TOOL_NAME in transcript.tool_calls_made, (
        f"model reached a decision without the optimizer; called: {transcript.tool_calls_made}"
    )
    # The stronger claim: the interface works well enough to produce a VALID trajectory.
    assert transcript.grounded, transcript.grounding_error


def test_grounding_accepts_a_multi_day_sale_of_one_symbol() -> None:
    """Regression guard. A holding is legitimately sold across several days whenever the ADV
    participation cap bites -- exactly the illiquid case this component exists to handle.

    The validator used to key assets_to_sell by SYMBOL alone, collapsing those rows to the
    last one, so day 0's fraction was compared against day 2's and a perfectly faithful plan
    was reported as tampered with. Observed rejecting real trajectories during SFT dataset
    generation.
    """
    multi_day = {
        "assets_to_sell": [
            {"symbol": "SPY", "sell_fraction": 0.40, "quantity": 100.0, "execution_day": 0},
            {"symbol": "THIN", "sell_fraction": 0.17106, "quantity": 500.0, "execution_day": 0},
            {"symbol": "THIN", "sell_fraction": 0.17106, "quantity": 500.0, "execution_day": 1},
            {"symbol": "THIN", "sell_fraction": 0.122843, "quantity": 360.0, "execution_day": 2},
        ],
        "raised_amount": 250_000.0,
        "expected_slippage_pct": 0.0031,
        "feasible": True,
    }

    grounded, error = enforce_grounding(
        _transcript(decision=json.loads(json.dumps(multi_day)),
                    tool_calls=[GROUNDING_TOOL_NAME], tool_output=multi_day)
    )
    assert grounded is True, error


def test_grounding_still_catches_tampering_within_a_multi_day_sale() -> None:
    """The fix must not weaken the check: altering ONE day's fraction is still caught."""
    tool_output = {
        "assets_to_sell": [
            {"symbol": "THIN", "sell_fraction": 0.17106, "quantity": 500.0, "execution_day": 0},
            {"symbol": "THIN", "sell_fraction": 0.122843, "quantity": 360.0, "execution_day": 1},
        ],
        "raised_amount": 100_000.0, "expected_slippage_pct": 0.002, "feasible": True,
    }
    tampered = json.loads(json.dumps(tool_output))
    tampered["assets_to_sell"][1]["sell_fraction"] = 0.99

    grounded, error = enforce_grounding(
        _transcript(decision=tampered, tool_calls=[GROUNDING_TOOL_NAME], tool_output=tool_output)
    )
    assert grounded is False
    assert "day 1" in error or "on day 1" in error


def test_grounding_rejects_a_dropped_sale_step() -> None:
    """Omitting a day's sale understates what the user must actually liquidate."""
    tool_output = {
        "assets_to_sell": [
            {"symbol": "THIN", "sell_fraction": 0.5, "quantity": 100.0, "execution_day": 0},
            {"symbol": "THIN", "sell_fraction": 0.5, "quantity": 100.0, "execution_day": 1},
        ],
        "raised_amount": 100_000.0, "expected_slippage_pct": 0.002, "feasible": True,
    }
    truncated = json.loads(json.dumps(tool_output))
    truncated["assets_to_sell"] = truncated["assets_to_sell"][:1]

    grounded, error = enforce_grounding(
        _transcript(decision=truncated, tool_calls=[GROUNDING_TOOL_NAME], tool_output=tool_output)
    )
    assert grounded is False
    assert "sale step" in error


def test_episode_respects_a_wall_clock_budget() -> None:
    """An iteration cap alone does not bound runtime: each turn is a forward pass over a
    context that grows with every tool result, and the local model runs CPU-only. A single
    episode was observed exceeding 50 minutes before this guard existed.

    A timed-out episode must come back UNGROUNDED so the caller discards it -- a truncated
    transcript is not a valid training example.
    """
    from unittest.mock import MagicMock, patch

    from agent.reference_agent import run_episode

    def slow_client(**_):
        client = MagicMock()

        def slow_create(**__):
            import time as _t
            _t.sleep(0.05)
            message = MagicMock()
            message.tool_calls = None
            message.content = "still thinking..."      # never returns a decision
            message.model_dump.return_value = {"role": "assistant", "content": "..."}
            return MagicMock(choices=[MagicMock(message=message)])

        client.chat.completions.create.side_effect = slow_create
        return client

    scenario = {"scenario_id": "slow", "target_amount": 1000.0, "portfolio": {}, "urgency": 0.5}

    with patch("openai.OpenAI", side_effect=slow_client):
        transcript = run_episode(scenario, timeout_seconds=0.2, max_iterations=10_000)

    assert transcript.grounded is False
    assert "timed out" in (transcript.grounding_error or "")
    # It must actually stop early rather than burning the whole iteration budget.
    assert len(transcript.steps) < 100


def test_client_is_constructed_with_a_request_timeout() -> None:
    """The episode deadline is only checked BETWEEN turns, so it cannot interrupt a single
    blocking call. Without a per-request ceiling an episode overruns its budget while looking
    idle -- observed on the CPU-only local model. Bounding the request is what makes the
    episode budget enforceable at all.
    """
    from unittest.mock import MagicMock, patch

    from agent.reference_agent import REQUEST_TIMEOUT_SECONDS, run_episode

    captured: dict = {}

    def capture_client(**kwargs):
        captured.update(kwargs)
        client = MagicMock()
        message = MagicMock()
        message.tool_calls = None
        message.content = '{"assets_to_sell": [], "raised_amount": 0}'
        message.model_dump.return_value = {"role": "assistant", "content": "{}"}
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=message)]
        )
        return client

    scenario = {"scenario_id": "t", "target_amount": 1000.0, "portfolio": {}, "urgency": 0.5}
    with patch("openai.OpenAI", side_effect=capture_client):
        run_episode(scenario, timeout_seconds=30.0)

    assert captured.get("timeout") == REQUEST_TIMEOUT_SECONDS
    # A retry storm would silently multiply the wall-clock cost of every turn.
    assert captured.get("max_retries") == 1


def test_request_ceiling_exceeds_the_default_nudge_budget() -> None:
    """Guard against re-introducing a self-defeating timeout.

    REQUEST_TIMEOUT_SECONDS was briefly set to 120s right after measuring ~150s per turn on
    this hardware, so no turn could ever complete and every episode failed with a timeout
    that looked like a model fault. A per-request ceiling below a realistic turn is worse than
    no ceiling: it converts a slow system into a broken one.
    """
    from agent.reference_agent import EPISODE_TIMEOUT_SECONDS, REQUEST_TIMEOUT_SECONDS

    # A single turn must be able to finish inside the request ceiling with margin, and the
    # episode must allow at least a couple of turns.
    assert REQUEST_TIMEOUT_SECONDS >= 300.0
    assert EPISODE_TIMEOUT_SECONDS >= REQUEST_TIMEOUT_SECONDS
