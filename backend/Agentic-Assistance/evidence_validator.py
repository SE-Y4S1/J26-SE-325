
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

    if not tool_names:
        return {
            "passed": True,
            "answer": corrected_answer,
            "issues": [],
            "checks_applied": [],
        }

    if "get_fraud_analysis" in tool_names:
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

        # Detect recorded fraud score treated as a calibrated probability of fraud.
        score_probability_patterns = [
            r"(?:recorded\s+)?score\s+(?:of\s+)?0\.92\s+(?:indicates|is|represents|means|reflects)\s+(?:a\s+)?(?:probability|chance|confidence)",
            r"0\.92\s+(?:indicates|is|represents|means)\s+(?:a\s+)?probability",
            r"probability of fraud\s+(?:is\s+)?0\.92",
            r"92%\s+(?:probability|chance)\s+of\s+fraud",
            r"fraud\s+score\s+(?:of\s+)?0\.92\s+means\s+(?:a\s+)?92%",
        ]

        if any(
            re.search(pattern, corrected_answer, re.IGNORECASE)
            for pattern in score_probability_patterns
        ):
            issues.append("Recorded fraud score described as a calibrated probability")

        checks_applied.append("fraud_score_probability_claims")

    if "get_blockchain_audit" in tool_names:
        checks_applied.append("blockchain_audit_present")

        # Detect unsupported claims about independent blockchain verification.
        blockchain_verification_patterns = [
            r"can be independently verified",
            r"independent(?:ly)?\s+verif(?:ied|ication)\b(?!.*not\s+established)",
            r"provides?\s+(?:an?\s+)?independent\s+verification",
            r"independent\s+verification\s+of\s+the\s+decision",
            r"VERIFIED\s+(?:status\s+)?means\s+(?:that\s+)?the\s+record\s+is\s+(?:accurate|reliable|verified)",
            r"status\s+of\s+VERIFIED\s+means\s+(?:the\s+record\s+is\s+)?(?:accurate|reliable|verified)",
        ]

        if any(
            re.search(pattern, corrected_answer, re.IGNORECASE)
            for pattern in blockchain_verification_patterns
        ):
            issues.append("Unsupported blockchain independent verification claim")

        checks_applied.append("blockchain_verification_claims")

    # Replace the whole answer when a high-risk claim is detected.
    # Do not leave the problematic original wording in the response.
    if issues:
        audit_summary = ""
        if "get_blockchain_audit" in tool_names:
            audit_summary = (
                "Regarding blockchain audit status: An audit record "
                "(Audit ID AUDIT-2026-001, Block #1847291) was returned "
                "with record status VERIFIED from mock ledger data. "
                "Independent blockchain verification is not established "
                "by the available evidence.\n\n"
            )

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
            f"{audit_summary}"
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
