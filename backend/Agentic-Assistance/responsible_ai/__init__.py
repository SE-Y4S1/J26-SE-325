from .privacy import check_privacy
from .safety import check_financial_safety
from .fairness import fairness_check


def run_responsible_ai_checks(
    user_message: str,
    response: str
) -> dict:

    privacy = check_privacy(response)
    safety = check_financial_safety(response)
    fairness = fairness_check()

    return {
        "privacy": privacy,
        "safety": safety,
        "fairness": fairness,
        "transparency": {
            "status": "enabled",
            "message": "The assistant provides explanations based on available tool evidence."
        },
        "user_control": {
            "status": "enabled",
            "message": "Financial recommendations require user confirmation before action."
        }
    }