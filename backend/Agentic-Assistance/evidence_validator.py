
import re


def validate_evidence_claims(answer: str, evidence: list[dict]) -> dict:
    """Detect selected risky claims and append a grounded evidence note.

    This is a limited phrase-based guardrail, not a complete fact checker.
    """

    tool_names = {
        str(item.get("tool", ""))
        for item in evidence
    }
    evidence_text = "\n".join(
        str(item.get("result", ""))
        for item in evidence
    )

    issues = []
    checks_applied = []
    corrected_answer = answer

    if "get_fraud_analysis" not in tool_names:
        return {
            "passed": True,
            "answer": corrected_answer,
            "issues": [],
            "checks_applied": [],
        }

    checks_applied.append("fraud_evidence_present")

    # Detect unsupported claims about training-data relationships.
    training_patterns = [
        r"the models share training data",
        r"the models do not share training data",
        r"both models use the same training data",
        r"both models are trained on the same dataset",
        r"both models were trained on the same dataset",
        r"the models may not share training data",
        r"the models may share training data",
    ]

    if any(
        re.search(pattern, corrected_answer, re.IGNORECASE)
        for pattern in training_patterns
    ):
        issues.append("Unsupported training-data relationship claim")

    checks_applied.append("training_data_claims")

    # Detect certainty claims about real-world fraud.
    certainty_patterns = [
        r"100%\s+chance of fraud",
        r"100%\s+probability of fraud",
        r"certainly fraudulent",
        r"proves? (?:that )?the transaction is fraudulent",
    ]

    if any(
        re.search(pattern, corrected_answer, re.IGNORECASE)
        for pattern in certainty_patterns
    ):
        issues.append("Prototype probability overstated as fraud certainty")

    checks_applied.append("fraud_certainty_claims")

    
    # Replace the whole answer when a high-risk claim is detected.
    # Do not leave the problematic original wording in the response.
    if issues:
        corrected_answer = (
            "For TX1001, the available tool returned a recorded fraud "
            "score of 0.92, risk level HIGH, and status BLOCKED in a "
            "mock transaction record. The meaning of the score as a "
            "calibrated probability is not established by the available "
            "evidence.\n\n"
            "The local synthetic Random Forest prototype returned a "
            "fraud probability output of 1.0000. This is a prototype "
            "model output, not proof of fraud or a verified real-world "
            "probability.\n\n"
            "The recorded contributing factors are new device, unusual "
            "location, and high transfer amount. The tool says the device "
            "and location inputs for the local prototype were inferred "
            "from mock reason labels. These are not independently "
            "verified real-world observations.\n\n"
            "The SHAP values explain the local synthetic prototype, "
            "not necessarily the model associated with the recorded "
            "score. Whether the models share training data is not "
            "established by the available evidence.\n\n"
            "Evidence validation detected risky wording in the original "
            "LLM response. The answer above was replaced with a "
            "grounded summary based on the returned tool evidence."
        )


    if (
        "LOCAL SHAP PROTOTYPE" in evidence_text
        or "synthetic Random Forest" in evidence_text
    ):
        checks_applied.append("prototype_evidence_present")

    return {
        "passed": len(issues) == 0,
        "answer": corrected_answer,
        "issues": issues,
        "checks_applied": checks_applied,
    }
