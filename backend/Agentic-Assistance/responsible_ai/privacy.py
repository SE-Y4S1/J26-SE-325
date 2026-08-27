SENSITIVE_TERMS = [
    "password",
    "cvv",
    "pin",
    "credit card number",
]


def check_privacy(text: str) -> dict:
    found = []

    text_lower = text.lower()

    for term in SENSITIVE_TERMS:
        if term in text_lower:
            found.append(term)

    if found:
        return {
            "status": "warning",
            "message": "Potentially sensitive financial information detected.",
            "detected": found
        }

    return {
        "status": "passed",
        "message": "No sensitive financial information detected.",
        "detected": []
    }