"""OpenAI-style function-calling schemas for every agent tool.

Written in OpenAI function-calling shape (name / description / parameters-as-JSON-Schema)
because that is the format Llama tool-calling fine-tunes expect, and the format Ollama's
OpenAI-compatible endpoint speaks. Building the tools in this shape now means the Colab
fine-tuning run and the eventual swap from the reference model to the fine-tuned one need no
rewrite on this side.

THE GROUNDING CONSTRAINT
------------------------
The tool descriptions here are load-bearing, not documentation. They are the SLM's only
instruction about the division of labour:

  * The model MAY reason over technical signals, sentiment and forecasts, and it decides
    HOW to invoke the optimizer -- what urgency and risk tolerance to pass.
  * The model MAY NOT state a sell amount or percentage of its own. Every number in a final
    decision comes from run_fuzzy_ga_withdrawal.

This is a regulatory requirement, not a style preference. The TAF's Legal Impact section
states that "financial regulators increasingly mandate explainability, auditability, and
bias mitigation" and that platforms without auditable trails "face compliance exposure". A
language model emitting "sell 40% of AAPL" from raw signals has no deterministic, auditable
derivation behind it; the fuzzy-GA output does, complete with a rule trace.

agent/reference_agent.py enforces this in code -- the descriptions set the expectation, the
validator makes it true.
"""

from __future__ import annotations

from typing import Any

# The tool whose output is the ONLY legitimate source of withdrawal numbers. Referenced by
# the reference agent's validator and by the Phase 5c grounding test.
GROUNDING_TOOL_NAME = "run_fuzzy_ga_withdrawal"


TOOL_GET_TECHNICAL_SIGNALS: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_technical_signals",
        "description": (
            "Get the latest technical indicators for one symbol: MACD, RSI, MFI and ATR. "
            "Use this to assess momentum and volatility before deciding how urgently a "
            "position should be liquidated. Note MFI is null for forex pairs, which report "
            "no volume."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker, e.g. 'AAPL', 'SPY' or 'EURUSD=X'.",
                }
            },
            "required": ["symbol"],
            "additionalProperties": False,
        },
    },
}

TOOL_GET_SENTIMENT: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_sentiment",
        "description": (
            "Get FinBERT news-sentiment features for one symbol: mean_sentiment in [-1, 1], "
            "sentiment_volume (headline count, a proxy for attention) and sentiment_momentum "
            "(the shift versus the trailing 5-day mean). Strongly negative sentiment with "
            "rising volume may argue for exiting a position sooner."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker to score."}
            },
            "required": ["symbol"],
            "additionalProperties": False,
        },
    },
}

TOOL_GET_FORECAST: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_forecast",
        "description": (
            "Get a quantile return forecast (p10, p50, p90) for one symbol at a given "
            "horizon, from the hybrid foundation-model + residual-head forecaster. A wide "
            "p10-p90 band means high uncertainty. Use p10 to reason about downside risk if "
            "liquidation is delayed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker to forecast."},
                "horizon": {
                    "type": "integer",
                    "description": "Forecast horizon in trading days.",
                    "enum": [1, 5, 21],
                },
            },
            "required": ["symbol", "horizon"],
            "additionalProperties": False,
        },
    },
}

TOOL_RUN_FUZZY_GA_WITHDRAWAL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": GROUNDING_TOOL_NAME,
        "description": (
            "Compute the optimal liquidation plan using the fuzzy inference system and "
            "genetic algorithm. This is the ONLY way to produce sell amounts. You must call "
            "this tool and report its numbers exactly as returned -- never estimate, adjust "
            "or invent a sell amount or percentage yourself. Your judgement goes into the "
            "arguments you choose (especially urgency and risk_tolerance), not into the "
            "output figures. Returns assets_to_sell with per-asset fractions and quantities, "
            "raised_amount, expected_slippage_pct and a fuzzy_rule_trace."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "urgency": {
                    "type": "number",
                    "description": (
                        "How urgently the user needs the cash, 0.0 (no rush, optimize for "
                        "cost) to 1.0 (immediate). Set this from the user's stated deadline "
                        "and from market conditions you observed via the other tools."
                    ),
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "risk_tolerance": {
                    "type": "number",
                    "description": "User's tolerance for execution risk, 0.0 to 1.0.",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "liquidity_target": {
                    "type": "number",
                    "description": "Cash amount to raise, in account currency.",
                    "exclusiveMinimum": 0,
                },
                "portfolio_state": {
                    "type": "object",
                    "description": (
                        "Current holdings keyed by symbol. Each entry has value, price, "
                        "adv_usd, daily_volatility and volatility_pct."
                    ),
                    "additionalProperties": {"type": "object"},
                },
                "deadline_days": {
                    "type": "integer",
                    "description": "Trading days available to raise the cash.",
                    "minimum": 1,
                    "maximum": 30,
                },
            },
            "required": ["urgency", "risk_tolerance", "liquidity_target", "portfolio_state"],
            "additionalProperties": False,
        },
    },
}

TOOL_RUN_MOEAD_REBALANCE: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "run_moead_rebalance",
        "description": (
            "Compute a long-term target allocation using the MOEA/D multi-objective "
            "optimizer over expected return, CVaR and liquidity cost. Use this for "
            "rebalancing questions, NOT for raising cash -- withdrawals must go through "
            "run_fuzzy_ga_withdrawal. As with that tool, report the returned weights "
            "exactly; do not adjust them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "portfolio_state": {
                    "type": "object",
                    "description": "Current holdings keyed by symbol.",
                    "additionalProperties": {"type": "object"},
                },
                "risk_preference": {
                    "type": "number",
                    "description": "0.0 = minimize risk, 1.0 = maximize return.",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "selection_rule": {
                    "type": "string",
                    "description": "Which Pareto point to return.",
                    "enum": ["knee", "max_sharpe", "scalarized"],
                },
            },
            "required": ["portfolio_state"],
            "additionalProperties": False,
        },
    },
}


_ALL_SCHEMAS = [
    TOOL_GET_TECHNICAL_SIGNALS,
    TOOL_GET_SENTIMENT,
    TOOL_GET_FORECAST,
    TOOL_RUN_FUZZY_GA_WITHDRAWAL,
    TOOL_RUN_MOEAD_REBALANCE,
]


def all_tool_schemas() -> list[dict[str, Any]]:
    """Every schema, in the `tools=[...]` shape the chat-completions API expects."""
    return list(_ALL_SCHEMAS)


def tool_schema(name: str) -> dict[str, Any]:
    """One schema by name."""
    for schema in _ALL_SCHEMAS:
        if schema["function"]["name"] == name:
            return schema
    raise KeyError(f"unknown tool: {name!r}")


def validate_tool_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate and coerce model-supplied arguments against a tool's JSON Schema.

    Small models hallucinate argument shapes, so this must reject bad calls with a message
    the agent loop can feed back as an observation rather than crashing the run. Validation
    is done by hand against the subset of JSON Schema used above, rather than pulling in a
    full validator, because the failure MESSAGE matters -- it is fed back to the model, so it
    has to say what was wrong in terms the model can act on.
    """
    spec = tool_schema(name)["function"]["parameters"]
    properties: dict[str, Any] = spec.get("properties", {})
    required: list[str] = spec.get("required", [])

    missing = [key for key in required if key not in arguments]
    if missing:
        raise ValueError(f"{name}: missing required argument(s): {', '.join(missing)}")

    unexpected = [key for key in arguments if key not in properties]
    if unexpected and spec.get("additionalProperties") is False:
        raise ValueError(
            f"{name}: unexpected argument(s): {', '.join(unexpected)}. "
            f"Valid arguments are: {', '.join(properties)}"
        )

    coerced: dict[str, Any] = {}
    for key, value in arguments.items():
        if key not in properties:
            continue
        rule = properties[key]
        expected = rule.get("type")

        if expected == "number":
            value = _coerce_number(name, key, value)
            _check_range(name, key, value, rule)
        elif expected == "integer":
            value = _coerce_integer(name, key, value)
            _check_range(name, key, value, rule)
        elif expected == "string" and not isinstance(value, str):
            raise ValueError(f"{name}.{key}: expected a string, got {type(value).__name__}")
        elif expected == "object" and not isinstance(value, dict):
            raise ValueError(f"{name}.{key}: expected an object, got {type(value).__name__}")

        if "enum" in rule and value not in rule["enum"]:
            raise ValueError(f"{name}.{key}: must be one of {rule['enum']}, got {value!r}")

        coerced[key] = value

    return coerced


def _coerce_number(tool: str, key: str, value: Any) -> float:
    """Accept a numeric string -- small models frequently quote their numbers."""
    if isinstance(value, bool):
        raise ValueError(f"{tool}.{key}: expected a number, got a boolean")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{tool}.{key}: expected a number, got {value!r}") from exc


def _coerce_integer(tool: str, key: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{tool}.{key}: expected an integer, got a boolean")
    try:
        as_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{tool}.{key}: expected an integer, got {value!r}") from exc
    if as_float != int(as_float):
        raise ValueError(f"{tool}.{key}: expected a whole number, got {value!r}")
    return int(as_float)


def _check_range(tool: str, key: str, value: float, rule: dict[str, Any]) -> None:
    if "minimum" in rule and value < rule["minimum"]:
        raise ValueError(f"{tool}.{key}: must be >= {rule['minimum']}, got {value}")
    if "maximum" in rule and value > rule["maximum"]:
        raise ValueError(f"{tool}.{key}: must be <= {rule['maximum']}, got {value}")
    if "exclusiveMinimum" in rule and value <= rule["exclusiveMinimum"]:
        raise ValueError(f"{tool}.{key}: must be > {rule['exclusiveMinimum']}, got {value}")
