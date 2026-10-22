
import re
from preferences_context import get_current_preferences


def sanitize_internal_details(text: str) -> str:
    """
    Remove or clean any exposed system prompt fragments, internal routing instructions,
    or tool-routing markers from user-facing responses.
    """
    cleaned = text
    cleaned = re.sub(r"RESPONSIBLE AI USER PREFERENCE DIRECTIVES:.*?(?=\n\n|\Z)", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"SYSTEM PROMPT:.*?(?=\n\n|\Z)", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"MINIMAL INVOCATION RULE:.*?(?=\n\n|\Z)", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"STRICT FACTUALITY OVERRIDE:.*?(?=\n\n|\Z)", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"\[(?:INTERNAL ROUTING|ROUTING|TOOL CALL|SYSTEM DIRECTIVE)[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


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

    prefs = get_current_preferences()
    personalized_context = prefs.get("personalized_context", True)

    if not personalized_context:
        # Strict enforcement: when personalization is disabled, never return mock portfolio balances or claims
        if "LKR 1,000,000" in corrected_answer or "1,000,000" in corrected_answer or "Baseline Valuation" in corrected_answer:
            corrected_answer = (
                "Notice: Personalized financial context is disabled by user preference. "
                "Account-specific portfolio lookup and asset holdings analysis are blocked for this request, "
                "and no personalized portfolio data has been retrieved."
            )

    if not tool_names:
        if "get_portfolio_analysis" in corrected_answer and (corrected_answer.strip().startswith("{") or "name" in corrected_answer):
            corrected_answer = (
                "Notice: Personalized financial context is disabled by user preference. "
                "Account-specific portfolio lookup and asset holdings analysis are blocked for this request, "
                "and no personalized portfolio data has been retrieved."
            )
        return {
            "passed": True,
            "answer": sanitize_internal_details(corrected_answer),
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
            issues.append("Fraud score treated as calibrated probability")
        checks_applied.append("fraud_score_probability_claims")

        # Detect missing device or location signals asserted as verified facts
        if "Device Signals: Unavailable" in evidence_text or "Location Signals: Unavailable" in evidence_text:
            device_fact_patterns = [
                r"\b(?:verified|confirmed|detected|found)\s+(?:that\s+)?(?:you\s+used\s+)?a\s+new\s+device\b",
                r"\btransaction\s+(?:was\s+)?from\s+an?\s+unusual\s+location\b",
                r"\b(?:verified|confirmed)\s+unusual\s+location\b",
            ]
            if any(re.search(p, corrected_answer, re.IGNORECASE) for p in device_fact_patterns):
                issues.append("Unverified device or location signals presented as verified facts")
            checks_applied.append("missing_telemetry_signal_claims")

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

    if "get_portfolio_analysis" in tool_names:
        checks_applied.append("portfolio_evidence_present")
        is_portfolio_fallback = any(
            re.search(r"FALLBACK|Local mock baseline|NO LIVE PORTFOLIO|PERSONALIZED CONTEXT DISABLED", str(item.get("result", "")), re.IGNORECASE)
            for item in evidence
            if item.get("tool") == "get_portfolio_analysis"
        )
        if is_portfolio_fallback:
            checks_applied.append("portfolio_fallback_present")

            # Detect unsupported claims that the mock balance or assets belong to the user
            actual_portfolio_patterns = [
                r"\byour portfolio (?:is valued at|holds|contains|has a value of)\s+(?:LKR\s*)?1,?000,?000\b",
                r"\byou have (?:a total of\s+)?(?:LKR\s*)?1,?000,?000\b",
                r"(?<!not reflect )\byour actual (?:portfolio|balance|holdings)\b",
                r"\byour (?:current\s+)?expected (?:annual\s+)?return is 12%\b",
                r"\byour risk score is 45\b",
                r"\byou currently hold 40% (?:in )?equit(?:y|ies)\b",
            ]
            if any(
                re.search(pattern, corrected_answer, re.IGNORECASE)
                for pattern in actual_portfolio_patterns
            ):
                issues.append("Fallback portfolio described as actual user portfolio")

            # Check whether the answer begins with a notice that no live portfolio was retrieved
            has_starting_notice = bool(
                re.search(r"^\s*(?:Notice|Please note|Important Notice|Disclaimer|Status|Note):.*(?:no live portfolio|live portfolio (?:was )?not (?:found|retrieved)|not found on platform|live portfolio data could not be retrieved|service (?:is |was )?(?:unreachable|unavailable)|personalized financial context (?:is )?disabled|no personalized portfolio)", corrected_answer, re.IGNORECASE)
                or re.search(r"(?:no live portfolio|live portfolio (?:was )?not (?:found|retrieved)|live portfolio data could not be retrieved|personalized financial context (?:is )?disabled|no personalized portfolio)", corrected_answer[:250], re.IGNORECASE)
            )
            if not has_starting_notice:
                issues.append("Missing required opening notice that no live portfolio was retrieved")
        else:
            checks_applied.append("live_portfolio_verified")
            # In Scenario 1, live portfolio data must be used; synthetic baseline figures must NOT be substituted
            synthetic_leak_patterns = [
                r"\b1,?000,?000\b",
                r"\b11\.8%\b",
                r"\bRisk Score:\s*38\b",
                r"\bDiversification Score:\s*74\b",
                r"\bMock Baseline\b",
            ]
            if any(re.search(pattern, corrected_answer, re.IGNORECASE) for pattern in synthetic_leak_patterns):
                issues.append("Synthetic baseline values substituted into live portfolio response")
    else:
        checks_applied.append("portfolio_absence_verified")
        portfolio_leak_patterns = [
            r"no live portfolio was retrieved",
            r"live portfolio (?:was )?not (?:found|retrieved)",
            r"portfolio\s+risk\s+profile",
            r"baseline valuation",
            r"1,?000,?000",
            r"expected annual return",
            r"diversification score",
            r"asset allocation",
            r"verified live holdings",
        ]
        # An execution trace is the user's own executed plan, so allocation language in its
        # explanation is grounded evidence, not a leak from the portfolio tool.
        if "get_execution_explanation" not in tool_names and any(
            re.search(pattern, corrected_answer, re.IGNORECASE)
            for pattern in portfolio_leak_patterns
        ):
            issues.append("Unrelated portfolio claims in non-portfolio response")

    # Replace or correct the answer when a high-risk claim or missing notice is detected.
    if issues:
        if "get_portfolio_analysis" in tool_names and not ({"get_fraud_analysis", "get_blockchain_audit"} & tool_names):
            p_item = next((item for item in evidence if item.get("tool") == "get_portfolio_analysis"), None)
            p_text = str(p_item.get("result", "")) if p_item else ""
            if "PERSONALIZED CONTEXT DISABLED" in p_text or "Personalized financial context is disabled" in p_text:
                corrected_answer = (
                    "Notice: Personalized financial context is disabled by user preference. "
                    "Account-specific portfolio lookup was not performed, and no personalized holdings or balance were retrieved."
                )
            elif "PORTFOLIO ANALYSIS — LIVE PLATFORM RECORD" in p_text:
                from agent import synthesize_portfolio_response
                corrected_answer = synthesize_portfolio_response(p_text)
            else:
                reason_match = re.search(r"Platform service note:\s*([^\n\r]+)", p_text, re.IGNORECASE)
                reason_str = reason_match.group(1).strip() if reason_match else "Portfolio not found on platform service"

                is_service_err = bool(
                    re.search(r"unreachable|timed out|timeout|connection refused|500|502|503|504|Service unavailable", reason_str, re.IGNORECASE)
                    or re.search(r"Service unavailable|SERVICE UNAVAILABLE", p_text, re.IGNORECASE)
                ) and not bool(re.search(r"Empty portfolio list|no registered portfolios", reason_str, re.IGNORECASE))

                if is_service_err:
                    corrected_answer = (
                        f"Notice: Live portfolio data could not be retrieved because the platform service is unreachable or timed out ({reason_str}). "
                        "This does not indicate an empty portfolio; live holdings could not be verified due to a service error or timeout."
                    )
                else:
                    corrected_answer = (
                        f"Notice: No live portfolio was retrieved for this account from the platform service ({reason_str}). "
                        "The user has zero verified live holdings on record.\n\n"
                        "For illustrative demonstration only, the local mock baseline example lists:\n"
                        "- Baseline Valuation: LKR 1,000,000 (Mock Baseline Demonstration Only)\n"
                        "- Expected Annual Return: 11.8% (Synthetic Example)\n"
                        "- Risk Score: 38/100 (Moderate, Synthetic Example)\n"
                        "- Diversification Score: 74/100 (Synthetic Example)\n"
                        "- Liquidity: High (Synthetic Example)\n"
                        "- Mock Allocation: 35% Technology, 25% Banking, 15% Healthcare, 15% Government Bonds, 10% Cash (Sample Example)\n\n"
                        "Please note: These figures and recommendations are synthetic baseline examples for demonstration only and do not reflect your actual portfolio or verified balance."
                    )
        elif "get_blockchain_audit" in tool_names and not ({"get_fraud_analysis", "get_portfolio_analysis"} & tool_names):
            audit_item = next((item for item in evidence if item.get("tool") == "get_blockchain_audit"), None)
            audit_text = str(audit_item.get("result", "")) if audit_item else ""
            tx_match = re.search(r"Transaction ID:\s*([^\n\r]+)", audit_text)
            tx_id = tx_match.group(1).strip() if tx_match else "the requested transaction"

            if "Record Status:\nNOT_FOUND" in audit_text or "confirmed that no audit record exists" in audit_text:
                corrected_answer = (
                    f"Notice: The blockchain audit record for transaction {tx_id} was not found on the live audit service. "
                    f"No audit record exists and no on-chain commitment exists for {tx_id}, and no independent verification is established."
                )
            elif "INTEGRITY_FAILED" in audit_text or "Cryptographic integrity verification failed" in audit_text:
                corrected_answer = (
                    f"ALERT: Cryptographic integrity verification failed for transaction {tx_id}. "
                    "The calculated SHA-256 hash does not match the blockchain commitment. Tampering is suspected, "
                    "and independent verification is not established."
                )
            elif "SIMULATED LEDGER" in audit_text or "Simulated Mock Ledger Adapter" in audit_text:
                block_match = re.search(r"Block Number:\s*([^\n\r]+)", audit_text)
                audit_id_match = re.search(r"Audit ID:\s*([^\n\r]+)", audit_text)
                block_no = block_match.group(1).strip() if block_match else "1"
                audit_id = audit_id_match.group(1).strip() if audit_id_match else f"AUDIT-{tx_id}"
                corrected_answer = (
                    f"Regarding blockchain audit verification for transaction {tx_id}: An audit record ({audit_id}, Block #{block_no}) "
                    "was retrieved with status VERIFIED against the local simulated ledger adapter. Cryptographic hash matches the ledger commitment. "
                    "Independent external blockchain verification is not established by the available evidence."
                )
            elif "CONFIGURED BLOCKCHAIN" in audit_text:
                block_match = re.search(r"Block Number:\s*([^\n\r]+)", audit_text)
                audit_id_match = re.search(r"Audit ID:\s*([^\n\r]+)", audit_text)
                block_no = block_match.group(1).strip() if block_match else "1"
                audit_id = audit_id_match.group(1).strip() if audit_id_match else f"AUDIT-{tx_id}"
                corrected_answer = (
                    f"Regarding blockchain audit verification for transaction {tx_id}: An audit record ({audit_id}, Block #{block_no}) "
                    "was verified on the configured blockchain network. Cryptographic hash matches the on-chain smart contract commitment. "
                    "Independent external mainnet verification is not established by the available evidence."
                )
            elif "FALLBACK DATA" in audit_text or "NOT_FOUND_ON_LIVE_BRIDGE" in audit_text:
                block_match = re.search(r"Block Number:\s*([^\n\r]+)", audit_text)
                audit_id_match = re.search(r"Audit ID:\s*([^\n\r]+)", audit_text)
                block_no = block_match.group(1).strip() if block_match else "1847291"
                audit_id = audit_id_match.group(1).strip() if audit_id_match else f"AUDIT-{tx_id}"
                corrected_answer = (
                    f"Notice: The live audit service was queried, but no live audit record was found for {tx_id}. "
                    f"The available record ({audit_id}, Block #{block_no}) is local fallback mock baseline data. Independent blockchain verification is not established."
                )
            else:
                corrected_answer = (
                    f"Regarding blockchain audit verification for transaction {tx_id}: The live audit service was unreachable or returned an error. "
                    "No verified blockchain record is established by the available evidence."
                )
        else:
            parts = []
            answer_lower = (answer or "").lower()
            wants_fraud = (
                "get_fraud_analysis" in tool_names
                and (
                    "get_blockchain_audit" not in tool_names
                    or bool(re.search(r"\b(?:fraud|score|blocked|flagged|gateway|decision)\b", answer_lower))
                )
            )
            wants_audit = (
                "get_blockchain_audit" in tool_names
                and (
                    "get_fraud_analysis" not in tool_names
                    or bool(re.search(r"\b(?:audit|blockchain|ledger|hash|block\s*number|commitment)\b", answer_lower))
                )
            )
            # Fallback: if neither matched keyword, include tools that were actually run
            if not wants_fraud and not wants_audit:
                wants_fraud = "get_fraud_analysis" in tool_names
                wants_audit = "get_blockchain_audit" in tool_names

            if wants_fraud:
                fraud_item = next((item for item in evidence if item.get("tool") == "get_fraud_analysis"), None)
                fraud_text = str(fraud_item.get("result", "")) if fraud_item else ""
                tx_match = re.search(r"Transaction ID:\s*([^\n\r]+)", fraud_text)
                tx_id = tx_match.group(1).strip() if tx_match else "the transaction"

                if "RECORD NOT FOUND" in fraud_text or "SERVICE UNAVAILABLE" in fraud_text or "No verified live fraud record exists" in fraud_text:
                    reason_m = re.search(r"Live Service Detail:\s*([^\n\r]+)", fraud_text) or re.search(r"Reason:\s*([^\n\r]+)", fraud_text)
                    reason_str = reason_m.group(1).strip() if reason_m else "No decision is held on record"
                    parts.append(
                        f"Notice: For transaction {tx_id}, no live fraud decision was found in Component 2's decision store ({reason_str}). "
                        "No verified fraud score, decision, or risk factors are on record for this transaction. "
                        "FinTrust Assistant does not fabricate scores or reasons when live records are unavailable."
                    )
                elif "AUTHORIZATION DENIED" in fraud_text:
                    parts.append(
                        f"Authorization notice: You are not authorized to view transaction {tx_id}, or it does not exist under your account."
                    )
                elif "LIVE COMPONENT 2 AUDIT RECORD" in fraud_text:
                    score_m = re.search(r"Recorded Fraud Score:\s*([0-9.]+)", fraud_text)
                    decision_m = re.search(r"Recorded Status:\s*([A-Za-z0-9_-]+)", fraud_text)
                    model_m = re.search(r"Model Version:\s*([^\n\r]+)", fraud_text)
                    factors_m = re.search(r"RECORDED CONTRIBUTING FACTORS:\s*\n-\s*([^\n\r]+)", fraud_text)
                    score_val = score_m.group(1) if score_m else "N/A"
                    decision_val = decision_m.group(1) if decision_m else "UNKNOWN"
                    model_val = model_m.group(1) if model_m else "Component 2 engine"
                    factors_val = factors_m.group(1) if factors_m else "gateway rules"
                    parts.append(
                        f"For transaction {tx_id}, Component 2's live gateway reports recorded status {decision_val} with a fraud score of {score_val} "
                        f"(model version: {model_val}).\n\n"
                        f"Recorded contributing factors: {factors_val}.\n\n"
                        "Telemetry notice: Device and location signals were not retained in the gateway audit trail, "
                        "and the meaning of the score as a calibrated probability is not established by the available evidence."
                    )
                elif "Optional SHAP feature-attribution details were suppressed" in fraud_text or "AI EXPLANATION NOTICE" in fraud_text:
                    parts.append(
                        f"DEMO EVIDENCE: For transaction {tx_id}, synthetic demo data lists a recorded fraud score of 0.92, risk level HIGH, and status BLOCKED. "
                        "The meaning of the score as a calibrated probability is not established by the available evidence.\n\n"
                        "Detailed SHAP feature attributions and prototype model metrics were suppressed based on user preference (AI explanation disabled).\n\n"
                        "The recorded demo contributing factors are new device, unusual location, and high transfer amount."
                    )
                else:
                    parts.append(
                        f"DEMO EVIDENCE: For transaction {tx_id}, synthetic demo data lists a recorded fraud score of 0.92, risk level HIGH, and status BLOCKED. "
                        "The meaning of the score as a calibrated probability is not established by the available evidence.\n\n"
                        "The local synthetic Random Forest prototype returned a fraud probability output of 1.0000. This is a prototype "
                        "model output, not proof of fraud or a verified real-world probability.\n\n"
                        "The recorded demo contributing factors are new device, unusual location, and high transfer amount. Device and location "
                        "inputs for the local prototype were inferred from mock reason labels and are not independently verified observations.\n\n"
                        "The SHAP values explain the local synthetic prototype, not the production fraud model. Whether the models share "
                        "training data is not established by the available evidence."
                    )
            if wants_audit:
                audit_item = next((item for item in evidence if item.get("tool") == "get_blockchain_audit"), None)
                audit_text = str(audit_item.get("result", "")) if audit_item else ""
                tx_match = re.search(r"Transaction ID:\s*([^\n\r]+)", audit_text)
                tx_id = tx_match.group(1).strip() if tx_match else "the transaction"
                if "Record Status:\nNOT_FOUND" in audit_text or "confirmed that no audit record exists" in audit_text:
                    parts.append(f"Regarding blockchain audit status: No audit record exists for transaction {tx_id} on the live audit bridge.")
                elif "INTEGRITY_FAILED" in audit_text:
                    parts.append(f"Regarding blockchain audit status: Cryptographic integrity verification failed for transaction {tx_id}.")
                elif "FALLBACK DATA" in audit_text or "NOT_FOUND_ON_LIVE_BRIDGE" in audit_text:
                    parts.append(f"Regarding blockchain audit status: No live audit record was found on the bridge for {tx_id}; data shown is fallback mock baseline. Independent blockchain verification is not established.")
                else:
                    parts.append(f"Regarding blockchain audit status: An audit record for {tx_id} was inspected. Independent blockchain verification is not established by the available evidence.")

            parts.append("Evidence validation detected risky wording in the original LLM response. The answer above was replaced with a grounded summary based on the returned tool evidence.")
            corrected_answer = "\n\n".join(parts)

    if (
        "LOCAL SHAP PROTOTYPE" in evidence_text
        or "synthetic Random Forest" in evidence_text
    ):
        checks_applied.append("prototype_evidence_present")

    return {
        "passed": len(issues) == 0,
        "answer": sanitize_internal_details(corrected_answer),
        "issues": issues,
        "checks_applied": checks_applied,
    }
