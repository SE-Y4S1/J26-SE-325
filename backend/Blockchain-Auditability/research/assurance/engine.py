"""Python port of src/services/assuranceEngine.js, so the policy evaluated here is the policy
the service enforces.

`assess` mirrors the JavaScript engine operation for operation on the service's inputs; the
golden vectors written by `golden.py` are checked against the JavaScript engine by
tests/goldenVectors.test.js, so any drift between the two fails CI.

`route_from_components` is the shared core -- score, thresholds, vetoes, degraded cap -- used
directly by the offline evaluation, where the three uncertainties come from IEEE-CIS signals.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

AUTOMATION_LEVEL = {"HUMAN_REVIEW": 0, "VERIFY": 1, "AUTO": 2}


def js_round6(x: float) -> float:
    """Math.round(x * 1e6) / 1e6 -- JavaScript rounds halves towards +infinity."""
    return math.floor(x * 1e6 + 0.5) / 1e6


def clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))


def margin_uncertainty(p: np.ndarray | float, t: float) -> np.ndarray | float:
    """1 - (distance of p from the decision threshold t, scaled to [0, 1])."""
    p = np.clip(p, 0.0, 1.0)
    margin = np.where(p >= t, (p - t) / (1 - t), (t - p) / t)
    return np.clip(1 - margin, 0.0, 1.0)


def _model_validation(signals: dict, patterns: list[str]) -> str:
    version = signals.get("sourceModelVersion")
    version = version.lower() if isinstance(version, str) else None
    if signals.get("benchmarkModel") is False:
        return "unvalidated"
    if version and any(p.lower() in version for p in patterns):
        return "unvalidated"
    if signals.get("benchmarkModel") is True:
        return "validated"
    return "unknown"


def route_from_components(score: float, vetoed: bool, degraded: bool, routing: dict) -> str:
    route = "AUTO" if score >= routing["autoMin"] else "VERIFY" if score >= routing["verifyMin"] else "HUMAN_REVIEW"
    cap = routing["degradedCap"]
    if degraded and AUTOMATION_LEVEL[route] > AUTOMATION_LEVEL[cap]:
        route = cap
    if vetoed:
        route = "HUMAN_REVIEW"
    return route


def assess(inp: dict, doc: dict) -> dict:
    """The JavaScript assessAssurance(input, policy) on one request."""
    signals = inp.get("signals") or {}
    degraded: list[str] = []
    unreliable: list[str] = []
    vetoes: list[str] = []

    t = doc["decision"]["rejectThreshold"]
    p = clamp01(inp["riskScore"] / 100)
    ai_decision = "REJECT" if p >= t else "APPROVE"

    margin = (p - t) / (1 - t) if p >= t else (t - p) / t
    u_c = clamp01(1 - margin)
    validation = _model_validation(signals, doc["unreliable"].get("modelVersionPatterns", []))
    if validation != "validated":
        u_c = max(u_c, doc["unreliable"]["confidenceFloor"])
        unreliable.append("confidence")

    scores = list((signals.get("modelScores") or {}).values())
    if len(scores) >= 2:
        d = clamp01(max(scores) - min(scores))
        if d > doc["vetoes"]["maxDisagreement"]:
            vetoes.append("MODEL_DISAGREEMENT")
    else:
        d = doc["missing"]["disagreement"]
        degraded.append("disagreement")

    ctx = doc["context"]
    low, high = ctx["amountScale"]["low"], ctx["amountScale"]["high"]
    amount_risk = clamp01(math.log(max(inp["amount"], low) / low) / math.log(high / low))

    if isinstance(signals.get("rules"), list):
        rule_risk = 0.0
        for rule in signals["rules"]:
            rule_risk = max(rule_risk, ctx["ruleSeverity"].get(rule, ctx["unknownRuleSeverity"]))
    else:
        rule_risk = 1.0
        degraded.append("context.upstreamRules")

    if validation == "validated":
        validation_risk = 0.0
    elif validation == "unvalidated":
        validation_risk = 1.0
    else:
        validation_risk = 1.0
        degraded.append("context.modelValidation")

    tx_type = str(inp.get("transactionType") or "").upper()
    type_risk = ctx["transactionTypeRisk"].get(tx_type, ctx["defaultTransactionTypeRisk"])

    cw = ctx["weights"]
    u_r = clamp01(cw["amount"] * amount_risk + cw["upstreamRules"] * rule_risk
                  + cw["modelValidation"] * validation_risk + cw["transactionType"] * type_risk)

    if inp["amount"] >= doc["vetoes"]["reviewAmount"]:
        vetoes.append("AMOUNT_REVIEW")

    w = doc["weights"]
    score = clamp01(1 - (w["confidence"] * u_c + w["disagreement"] * d + w["context"] * u_r))
    quality = "DEGRADED" if degraded or unreliable else "FULL"
    route = route_from_components(score, bool(vetoes), quality == "DEGRADED", doc["routing"])

    return {
        "aiDecision": ai_decision,
        "route": route,
        "assuranceScore": js_round6(score),
        "components": {"confidence": js_round6(u_c), "disagreement": js_round6(d), "context": js_round6(u_r)},
        "vetoes": vetoes,
        "signalQuality": quality,
        "degradedSignals": degraded,
        "unreliableSignals": unreliable,
    }


# --- policy hashing (scheme v2), matching src/services/canonicalJson.js -----------------------

def _js_number(x: float | int) -> str:
    if isinstance(x, bool):
        raise TypeError("booleans are not numbers")
    if isinstance(x, int):
        return str(x)
    if not math.isfinite(x):
        raise TypeError("canonical JSON rejects non-finite numbers")
    if x.is_integer() and abs(x) < 1e21:
        return str(int(x))
    text = repr(x)
    if "e" in text:  # Python 1e-07 -> JavaScript 1e-7
        mantissa, exponent = text.split("e")
        sign = "-" if exponent.startswith("-") else "+"
        text = f"{mantissa}e{sign}{exponent.lstrip('+-').lstrip('0') or '0'}"
    return text


def canonical_json(value: Any) -> str:
    import json
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _js_number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonical_json(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{json.dumps(k, ensure_ascii=False)}:{canonical_json(value[k])}" for k in sorted(value)) + "}"
    raise TypeError(f"canonical JSON: unsupported {type(value).__name__}")


def policy_hash_v2(doc: dict) -> str:
    return "0x" + hashlib.sha256(canonical_json(doc).encode("utf-8")).hexdigest()
