"""
Optional research telemetry recording governed by user data sharing consent.
"""

from typing import Any, Dict


def record_research_telemetry(event: Dict[str, Any], consent_granted: bool) -> bool:
    """
    Log optional empirical research telemetry only if user consent is granted.
    Returns True if logged, False if suppressed due to lack of consent.
    Note: Upstream gateway audit logs and required system operational logs are separate
    and not controlled by this preference.
    """
    if not consent_granted:
        return False
    # If consent is granted, telemetry is eligible for empirical research analysis
    return True
