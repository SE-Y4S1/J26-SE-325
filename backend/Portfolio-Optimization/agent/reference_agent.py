"""Reference ReAct agent -- a STAND-IN, not part of the production pipeline.

=============================================================================
THIS IS SCAFFOLDING. It exists to prove the tool interface works end-to-end
before the fine-tuned Llama comes back from Colab. Nothing outside agent/
should import it, and the FastAPI service must never call it. When the
fine-tuned model is ready it replaces the driver here; the tools, schemas and
grounding validator stay exactly as they are.
=============================================================================

Driven by local Ollama (`gemma4-e4b`) through its OpenAI-compatible endpoint at
http://localhost:11434/v1. Chosen because it is already installed, free to run at
trajectory-generation scale, reproducible offline, and -- verified via `ollama show` -- it
advertises native `tools` support, so it exercises the same function-calling path the
fine-tuned model will use. A paid API would have made the SFT dataset cost money to
regenerate.

THE GROUNDING VALIDATOR
-----------------------
enforce_grounding() is the whole point of this module. A transcript is only valid if:

  1. run_fuzzy_ga_withdrawal was called at least once, AND
  2. every numeric field in the final decision matches that call's output exactly.

Check (2) is what makes this real. Requiring only (1) would let the model call the tool,
ignore the answer, and state its own numbers -- which is precisely the failure mode the
constraint exists to prevent. A transcript that fails either check is a BUG, not a
low-quality trajectory: training on it would teach the fine-tuned model to fabricate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from agent.tool_schema import GROUNDING_TOOL_NAME, all_tool_schemas
from agent.tools import dispatch

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "gemma4-e4b"
OLLAMA_API_KEY = "ollama"       # placeholder; Ollama ignores it but the client requires one

MAX_TOOL_ITERATIONS = 12

# After this many turns without calling the optimizer, inject an explicit instruction to do
# so. Observed behaviour with gemma4-e4b: it cycles get_technical_signals -> get_sentiment ->
# get_forecast indefinitely, gathering context it never acts on, and burns the whole budget
# without ever producing a plan. Small models are prone to this when several tools look
# equally applicable. The nudge is scaffolding for the REFERENCE model only -- the
# Colab-tuned model learns the call from the trajectories instead.
NUDGE_AFTER_STEPS = 3

# Wall-clock budget for one episode. An iteration cap alone does not bound runtime: each turn
# is a full forward pass over a context that grows with every tool result, and `ollama ps`
# confirms gemma4-e4b runs 100% on CPU here with no GPU offload. A single episode was
# observed exceeding 50 minutes. A loop with no time bound is a hazard in any path that could
# ever face a user, not merely an inconvenience in tests.
EPISODE_TIMEOUT_SECONDS = 300.0

# Per-request ceiling. The episode deadline is only checked BETWEEN turns, so it cannot
# interrupt a single blocking call -- and on a CPU-only 7.5B model with a growing context one
# call can run for many minutes, which is how an episode overruns its budget while looking
# idle. Bounding the request itself is what makes the episode budget actually enforceable.
#
# MUST exceed a realistic single turn, or no turn ever completes and every episode fails with
# a timeout that looks like a model problem. Measured here: gemma4-e4b answers in ~150s per
# turn on CPU, so an earlier 120s value guaranteed failure. 300s leaves headroom for the later
# turns, whose context has grown with each tool result and which are therefore slower.
REQUEST_TIMEOUT_SECONDS = 300.0

# Tolerance for matching a decision figure against the tool output. Tight enough to catch a
# model that re-derived a number, loose enough to allow JSON round-trip rounding.
GROUNDING_TOLERANCE = 1e-4


@dataclass
class AgentStep:
    """One turn of the loop."""

    step_index: int
    thought: str | None
    tool_name: str | None
    tool_arguments: dict[str, Any] | None
    observation: dict[str, Any] | None


@dataclass
class AgentTranscript:
    """A full episode: the SFT training record and the audit trail.

    `decision` is structured only. No user-facing prose: natural-language explanation is
    Component 4's responsibility, and duplicating it here would create scope overlap that is
    hard to defend to a supervisor.
    """

    scenario_id: str
    messages: list[dict[str, Any]]
    steps: list[AgentStep]
    decision: dict[str, Any] | None
    grounded: bool
    grounding_error: str | None = None
    tool_calls_made: list[str] = field(default_factory=list)
    tool_outputs: dict[str, Any] = field(default_factory=dict)


def build_system_prompt() -> str:
    """The agent's operating instructions.

    States the division of labour explicitly: reason freely over signals, decide how to
    invoke the optimizer, never emit a sell amount that did not come from it.
    """
    return (
        "You are a liquidation strategy agent for a portfolio platform. Given a withdrawal "
        "request, you decide HOW to liquidate by choosing the parameters for the "
        "optimizer -- you do not compute the trade sizes yourself.\n"
        "\n"
        "Your process:\n"
        "1. Gather context with get_technical_signals, get_sentiment and get_forecast for "
        "the relevant holdings. Reason about momentum, volatility and downside risk.\n"
        f"2. Call {GROUNDING_TOOL_NAME} with the urgency and risk_tolerance your analysis "
        "supports. This is where your judgement matters.\n"
        "3. Report the plan the tool returned.\n"
        "\n"
        "HARD RULE: every sell amount, sell fraction, quantity and slippage figure in your "
        f"final answer must be copied EXACTLY from the {GROUNDING_TOOL_NAME} result. Never "
        "estimate, round, adjust, average or invent any of these numbers. If you have not "
        "called that tool, you cannot answer.\n"
        "\n"
        "Respond with a final JSON object only -- no prose explanation for end users, which "
        "is another component's responsibility. Use this shape:\n"
        '{"assets_to_sell": [...], "raised_amount": N, "expected_slippage_pct": N, '
        '"reasoning_trace": ["short internal notes"]}'
    )


def check_ollama_available(base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL) -> bool:
    """Whether Ollama is up and the model is pulled. Lets tests skip cleanly instead of
    failing when the daemon is not running."""
    import requests

    try:
        response = requests.get(base_url.replace("/v1", "/api/tags"), timeout=3)
        response.raise_for_status()
        names = {m.get("name", "").split(":")[0] for m in response.json().get("models", [])}
        return model.split(":")[0] in names
    except Exception:  # noqa: BLE001 - any failure means "not usable"
        return False


def run_episode(
    scenario: dict[str, Any],
    *,
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    max_iterations: int = MAX_TOOL_ITERATIONS,
    timeout_seconds: float = EPISODE_TIMEOUT_SECONDS,
) -> AgentTranscript:
    """Run one withdrawal scenario to a structured decision.

    Bounded by BOTH `max_iterations` and `timeout_seconds`; whichever binds first ends the
    episode. Exhausting either yields an ungrounded transcript, which the caller discards --
    a truncated episode is not a valid training example.
    """
    import time

    from openai import OpenAI

    deadline = time.monotonic() + timeout_seconds

    client = OpenAI(base_url=base_url, api_key=OLLAMA_API_KEY, timeout=REQUEST_TIMEOUT_SECONDS,
                    max_retries=1)      # a retry storm would blow the episode budget

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": _format_request(scenario)},
    ]
    steps: list[AgentStep] = []
    tool_calls_made: list[str] = []
    tool_outputs: dict[str, Any] = {}
    decision: dict[str, Any] | None = None
    retried_for_json = False

    for index in range(max_iterations):
        if time.monotonic() > deadline:
            logger.warning(
                "episode %s exceeded %.0fs after %d step(s); abandoning",
                scenario.get("scenario_id", "unknown"), timeout_seconds, index,
            )
            transcript = AgentTranscript(
                scenario_id=scenario.get("scenario_id", "unknown"),
                messages=messages, steps=steps, decision=None, grounded=False,
                grounding_error=f"episode timed out after {timeout_seconds:.0f}s",
                tool_calls_made=tool_calls_made, tool_outputs=tool_outputs,
            )
            return transcript

        try:
            # Never let one request outlive what remains of the episode budget.
            remaining = max(1.0, deadline - time.monotonic())
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=all_tool_schemas(),
                temperature=0.3,
                timeout=min(REQUEST_TIMEOUT_SECONDS, remaining),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Ollama call failed on step %d: %s", index, exc)
            return AgentTranscript(
                scenario_id=scenario.get("scenario_id", "unknown"),
                messages=messages, steps=steps, decision=None,
                grounded=False, grounding_error=f"model call failed: {exc}",
                tool_calls_made=tool_calls_made, tool_outputs=tool_outputs,
            )

        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            decision = _parse_decision(message.content)
            steps.append(AgentStep(index, message.content, None, None, None))

            # A final answer with no optimizer call is ungrounded by definition. Give the
            # model one chance to correct rather than discarding the episode.
            if GROUNDING_TOOL_NAME not in tool_calls_made and index < max_iterations - 1:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"You have not called {GROUNDING_TOOL_NAME}, so you have no "
                            "authorised numbers. Call it now with the portfolio from the "
                            "request and the urgency your analysis supports."
                        ),
                    }
                )
                decision = None
                continue

            # Tool WAS called but the reply was prose, not JSON. Observed with gemma4-e4b:
            # it narrates the plan correctly and never emits the object. Re-ask once, handing
            # back the exact tool output so copying it is the path of least resistance --
            # which is also the behaviour the grounding rule requires.
            if decision is None and index < max_iterations - 1:
                tool_result = tool_outputs.get(GROUNDING_TOOL_NAME)
                if tool_result is not None and not retried_for_json:
                    retried_for_json = True
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Reply with ONLY a JSON object and no other text. Copy these "
                                "values exactly:\n"
                                + json.dumps(
                                    {
                                        "assets_to_sell": tool_result.get("assets_to_sell"),
                                        "raised_amount": tool_result.get("raised_amount"),
                                        "expected_slippage_pct": tool_result.get("expected_slippage_pct"),
                                        "feasible": tool_result.get("feasible"),
                                    },
                                    default=str,
                                )
                                + '\nAdd a "reasoning_trace" array of short internal notes.'
                            ),
                        }
                    )
                    continue
            break

        # Break the info-gathering loop before it consumes the whole budget.
        if (
            index >= NUDGE_AFTER_STEPS
            and GROUNDING_TOOL_NAME not in tool_calls_made
            and not any(c.function.name == GROUNDING_TOOL_NAME for c in message.tool_calls)
        ):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Stop gathering context. You have enough. Call {GROUNDING_TOOL_NAME} "
                        "now -- it is the only tool that can produce the sell amounts, and "
                        "you cannot answer without it."
                    ),
                }
            )

        for call in message.tool_calls:
            name = call.function.name
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                arguments, observation = {}, {"error": f"arguments were not valid JSON: {exc}"}
            else:
                observation = dispatch(name, arguments)

            tool_calls_made.append(name)
            if "error" not in observation:
                tool_outputs[name] = observation

            steps.append(AgentStep(index, message.content, name, arguments, observation))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(observation, default=str),
                }
            )

    transcript = AgentTranscript(
        scenario_id=scenario.get("scenario_id", "unknown"),
        messages=messages, steps=steps, decision=decision,
        grounded=False, tool_calls_made=tool_calls_made, tool_outputs=tool_outputs,
    )
    transcript.grounded, transcript.grounding_error = enforce_grounding(transcript)
    return transcript


def enforce_grounding(transcript: AgentTranscript) -> tuple[bool, str | None]:
    """Verify the decision's numbers came from run_fuzzy_ga_withdrawal.

    Returns (is_grounded, error). Compares assets_to_sell, sell fractions, raised_amount and
    expected_slippage_pct against the tool's recorded output within float tolerance.
    """
    if GROUNDING_TOOL_NAME not in transcript.tool_calls_made:
        return False, (
            f"{GROUNDING_TOOL_NAME} was never called, so any figures in the decision were "
            "fabricated by the model"
        )

    if transcript.decision is None:
        return False, "no structured decision was produced"

    tool_output = transcript.tool_outputs.get(GROUNDING_TOOL_NAME)
    if tool_output is None:
        return False, f"{GROUNDING_TOOL_NAME} was called but returned an error, so no plan exists"

    decision = transcript.decision

    for field_name in ("raised_amount", "expected_slippage_pct"):
        if field_name not in decision:
            continue
        claimed = _as_float(decision[field_name])
        actual = _as_float(tool_output.get(field_name))
        if claimed is None or actual is None:
            return False, f"{field_name} is not numeric in the decision or the tool output"
        if abs(claimed - actual) > max(GROUNDING_TOLERANCE, abs(actual) * 1e-4):
            return False, (
                f"{field_name} was altered: decision says {claimed}, "
                f"{GROUNDING_TOOL_NAME} returned {actual}"
            )

    claimed_sales = decision.get("assets_to_sell")
    if claimed_sales is not None:
        # Keyed on (symbol, execution_day), NOT symbol alone. A single holding is legitimately
        # sold across several days whenever the ADV participation cap bites -- which is
        # precisely the illiquid case this component exists to handle. Keying by symbol
        # collapses those rows to the last one, so day 0's fraction gets compared against
        # day 2's and a perfectly faithful plan is reported as tampered with.
        actual_rows: dict[tuple[str, object], dict[str, Any]] = {}
        for row in tool_output.get("assets_to_sell", []):
            actual_rows[(row.get("symbol"), row.get("execution_day"))] = row

        actual_symbols = {symbol for symbol, _ in actual_rows}

        for row in claimed_sales:
            symbol = row.get("symbol")
            if symbol not in actual_symbols:
                return False, (
                    f"decision sells {symbol!r}, which is not in the "
                    f"{GROUNDING_TOOL_NAME} plan"
                )

            key = (symbol, row.get("execution_day"))
            actual_row = actual_rows.get(key)
            if actual_row is None:
                return False, (
                    f"decision sells {symbol!r} on day {row.get('execution_day')}, which is "
                    f"not a step in the {GROUNDING_TOOL_NAME} plan"
                )

            for numeric_field in ("sell_fraction", "quantity"):
                if numeric_field not in row:
                    continue
                claimed = _as_float(row[numeric_field])
                actual = _as_float(actual_row.get(numeric_field))
                if claimed is None or actual is None:
                    continue
                if abs(claimed - actual) > max(GROUNDING_TOLERANCE, abs(actual) * 1e-4):
                    # Name the day only when the plan actually has one; "(day None)" is noise
                    # for a single-step plan and this message is read by humans debugging a
                    # rejected trajectory.
                    day = row.get("execution_day")
                    where = f" on day {day}" if day is not None else ""
                    return False, (
                        f"{symbol}.{numeric_field}{where} was altered: "
                        f"decision says {claimed}, tool returned {actual}"
                    )

        # A plan cannot be faithful if steps were dropped: omitting a day's sale understates
        # what the user must actually liquidate.
        if len(claimed_sales) != len(tool_output.get("assets_to_sell", [])):
            return False, (
                f"decision lists {len(claimed_sales)} sale step(s) but "
                f"{GROUNDING_TOOL_NAME} returned {len(tool_output.get('assets_to_sell', []))}"
            )

    return True, None


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _format_request(scenario: dict[str, Any]) -> str:
    """Render a scenario as the user turn."""
    return (
        f"Withdrawal request: raise {scenario['target_amount']:,.2f} "
        f"within {scenario.get('deadline_days', 1)} trading day(s).\n"
        f"Stated urgency: {scenario.get('urgency', 'unspecified')}\n"
        f"Market context: {scenario.get('market_regime', 'normal')}\n"
        f"Current portfolio:\n{json.dumps(scenario['portfolio'], indent=2, default=str)}"
    )


def _parse_decision(content: str | None) -> dict[str, Any] | None:
    """Extract the final JSON object from a model response.

    Small models routinely wrap JSON in prose or a markdown fence, so we locate the outermost
    braces rather than requiring a clean response.
    """
    if not content:
        return None
    text = content.strip()

    if "```" in text:
        blocks = text.split("```")
        for block in blocks:
            candidate = block.removeprefix("json").strip()
            if candidate.startswith("{"):
                text = candidate
                break

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        logger.debug("could not parse decision JSON from: %s", text[:200])
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
