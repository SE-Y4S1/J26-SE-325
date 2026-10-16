from .privacy import check_privacy
from .safety import check_financial_safety
from .fairness import fairness_check
from .research_telemetry import record_research_telemetry


def run_responsible_ai_checks(
    user_message: str,
    response: str,
    preferences: dict | None = None,
) -> dict:

    privacy = check_privacy(response)
    safety = check_financial_safety(response)
    fairness = fairness_check()

    prefs = preferences or {}
    explanation_enabled = prefs.get("explanation_enabled", True)
    personalized_context = prefs.get("personalized_context", True)
    data_sharing_consent = prefs.get("data_sharing_consent", True)

    transparency = (
        {
            "status": "enabled",
            "message": "The assistant provides explanations based on available tool evidence."
        }
        if explanation_enabled
        else {
            "status": "suppressed_by_preference",
            "message": "Detailed SHAP and model feature explanations were suppressed based on user preference."
        }
    )

    personalization = (
        {
            "status": "enabled",
            "message": "Personalized context is active for account-specific queries."
        }
        if personalized_context
        else {
            "status": "disabled_by_preference",
            "message": "Personalized financial context is disabled. Account-specific portfolio queries are blocked without fabricating anonymous holdings."
        }
    )

    data_sharing = (
        {
            "status": "consented",
            "message": "Consent granted for optional research and empirical evaluation telemetry. Upstream transaction and audit trail logging remain governed by their respective system policies."
        }
        if data_sharing_consent
        else {
            "status": "opted_out",
            "message": "User opted out: optional empirical research data sharing and telemetry logging are suppressed for this interaction. Note: this setting does not disable upstream gateway audit logs or required system operational logging."
        }
    )

    return {
        "privacy": privacy,
        "safety": safety,
        "fairness": fairness,
        "transparency": transparency,
        "user_control": {
            "status": "enforced",
            "message": "User preference controls actively enforce explanation suppression, personalization gating, and research consent limits. Financial recommendations require user confirmation before action."
        },
        "personalized_context": personalization,
        "data_sharing": data_sharing,
    }