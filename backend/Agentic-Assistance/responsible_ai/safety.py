def check_financial_safety(response: str) -> dict:

    risky_phrases = [
        "guaranteed profit",
        "guaranteed return",
        "you will definitely make money",
    ]

    detected = []

    response_lower = response.lower()

    for phrase in risky_phrases:
        if phrase in response_lower:
            detected.append(phrase)

    if detected:
        return {
            "status": "warning",
            "message": "Response contains potentially misleading financial claims.",
            "detected": detected
        }

    return {
        "status": "passed",
        "message": "No guaranteed financial claims detected.",
        "detected": []
    }