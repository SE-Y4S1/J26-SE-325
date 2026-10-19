from pathlib import Path
from dotenv import load_dotenv

# Load trusted configuration from repository root .env if not already set in environment
env_candidates = [Path(__file__).resolve().parent / ".env"]
if len(Path(__file__).resolve().parents) > 2:
    env_candidates.append(Path(__file__).resolve().parents[2] / ".env")
for env_candidate in env_candidates:
    if env_candidate.is_file():
        load_dotenv(env_candidate, override=False)

from datetime import datetime, timezone
from typing import Any

from evidence_validator import validate_evidence_claims
from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator


from langchain_core.messages import HumanMessage

from agent import agent, get_deterministic_fast_path
from preferences_context import (
    get_current_preferences,
    reset_current_preferences,
    set_current_preferences,
)
from responsible_ai import record_research_telemetry, run_responsible_ai_checks
from timing_context import start_request_timing, end_request_timing
from request_auth import set_caller_authorization, reset_caller_authorization, decode_caller_identity
from governance_store import default_governance_store
from tools.portfolio_tools import init_request_portfolio_cache, clear_request_portfolio_cache

app = FastAPI(
    title="FinAgent - Agentic Financial Assistant",
    description="Component 4 - Agentic LLM Assistant & Responsible AI",
    version="0.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# INTEGRATION SURFACE ONLY
#
# CORS is handled above, by this component's own middleware. The only thing added here for
# the platform is a health endpoint; nothing below touches the agent, the tools or the
# responsible-AI checks.
# --------------------------------------------------


@app.get("/health")
def health():
    """Liveness only -- deliberately does NOT invoke the agent.

    docker-compose and the frontend need to distinguish "this service is not running" from
    "the agent failed on this request". Calling the LLM here would conflate the two and make
    the check slow and key-dependent.
    """
    return {"status": "ok", "component": "agentic-assistance"}


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class ResponsibleAIPreferences(BaseModel):
    model_config = ConfigDict(extra="ignore")

    explanation_enabled: bool = True
    personalized_context: bool = True
    data_sharing_consent: bool = True

    @model_validator(mode="before")
    @classmethod
    def normalize_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            exp = data.get("explanation_enabled")
            if exp is None:
                exp = data.get("explanationEnabled", True)
            pers = data.get("personalized_context")
            if pers is None:
                pers = data.get("personalizedContext", True)
            ds = data.get("data_sharing_consent")
            if ds is None:
                ds = data.get("dataSharingConsent", True)

            def to_bool(val: Any) -> bool:
                if isinstance(val, bool):
                    return val
                if isinstance(val, str):
                    if val.lower() in ("false", "0", "no"):
                        return False
                    if val.lower() in ("true", "1", "yes"):
                        return True
                return True

            return {
                "explanation_enabled": to_bool(exp),
                "personalized_context": to_bool(pers),
                "data_sharing_consent": to_bool(ds),
            }
        return data


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    message: str
    preferences: ResponsibleAIPreferences = Field(default_factory=ResponsibleAIPreferences)

    @model_validator(mode="before")
    @classmethod
    def validate_request(cls, data: Any) -> Any:
        if isinstance(data, dict):
            prefs = data.get("preferences")
            if prefs is None or not isinstance(prefs, (dict, ResponsibleAIPreferences)):
                data["preferences"] = ResponsibleAIPreferences()
        return data


# --------------------------------------------------
# ROOT
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "project": "FinAgent",
        "component": "Agentic LLM Assistant & Responsible AI",
        "version": "0.2.0",
        "status": "running"
    }


# --------------------------------------------------
# GOVERNANCE DECISION ENDPOINTS (HUMAN OVERSIGHT)
# --------------------------------------------------

class GovernanceDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    decision_type: str
    reference_id: str
    reference_type: str = "transaction"
    evidence_reference: dict[str, Any] = Field(default_factory=dict)
    rationale: str | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            dt = str(data.get("decision_type", "")).strip().lower()
            if dt not in {"confirmed", "rejected"}:
                raise ValueError("decision_type must be either 'confirmed' or 'rejected'")
            ref = str(data.get("reference_id", "")).strip()
            if not ref:
                raise ValueError("reference_id cannot be empty")
            data["decision_type"] = dt
            data["reference_id"] = ref
        return data


@app.post("/assistant/governance/decision")
def record_governance_decision(
    request: GovernanceDecisionRequest,
    authorization: str | None = Header(default=None),
):
    """
    Record a persistent human-in-the-loop oversight decision (Confirm Decision / Reject Action).
    Enforces authorization when AUTH_REQUIRED is active, extracts caller identity, and persists
    the record to the append-only governance store.
    """
    user_id, user_email = decode_caller_identity(authorization)
    record = default_governance_store.record_decision(
        decision_type=request.decision_type,
        reference_id=request.reference_id,
        reference_type=request.reference_type,
        user_id=user_id,
        user_email=user_email,
        evidence_reference=request.evidence_reference,
        rationale=request.rationale,
    )
    return {
        "status": record.get("status", "recorded"),
        "is_duplicate": record.get("is_duplicate", False),
        "decision": record,
    }


@app.get("/assistant/governance/decision/{reference_id}")
def get_governance_decision(
    reference_id: str,
    authorization: str | None = Header(default=None),
):
    """
    Retrieve the latest recorded human oversight decision for a transaction or portfolio.
    Used on page load / refresh to restore and display the persistent governance state.
    """
    decode_caller_identity(authorization)
    decision = default_governance_store.get_latest_decision(reference_id)
    return {
        "found": decision is not None,
        "decision": decision,
    }


@app.get("/assistant/governance/decisions")
def list_governance_decisions(
    limit: int = 50,
    authorization: str | None = Header(default=None),
):
    """List recent human oversight decisions for auditability and compliance reporting."""
    decode_caller_identity(authorization)
    return {
        "decisions": default_governance_store.list_decisions(limit=limit),
    }


# --------------------------------------------------
# CHAT
# --------------------------------------------------

@app.post("/assistant/chat")
def chat(request: ChatRequest, authorization: str | None = Header(default=None)):

    timing_ctx, timing_token = start_request_timing()
    cache_token = init_request_portfolio_cache()
    prefs = request.preferences or ResponsibleAIPreferences()
    pref_dict = {
        "explanation_enabled": prefs.explanation_enabled,
        "personalized_context": prefs.personalized_context,
        "data_sharing_consent": prefs.data_sharing_consent,
    }

    token = set_current_preferences(pref_dict)
    # The caller's own token, so get_execution_explanation reads their trace as them.
    auth_token = set_caller_authorization(authorization)
    timing_summary = None
    try:
        # ----------------------------------------------
        # Deterministic fast path for greetings, thanks,
        # and simple conversational messages.
        # These requests must not call Ollama or financial tools.
        # ----------------------------------------------
        fast_reply = get_deterministic_fast_path(request.message)
        if fast_reply is not None:
            tools_used = []
            evidence = []
            validation = validate_evidence_claims(fast_reply, evidence)
            response_text = validation["answer"]
            responsible_ai = run_responsible_ai_checks(
                request.message,
                response_text,
                preferences=pref_dict,
            )
            record_research_telemetry(
                event={
                    "message_length": len(request.message),
                    "tools_used": tools_used,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "fast_path": True,
                },
                consent_granted=prefs.data_sharing_consent,
            )
            timing_summary = end_request_timing(timing_token)
            return {
                "evidence_validation": {
                    "passed": validation["passed"],
                    "issues": validation["issues"],
                    "checks_applied": validation["checks_applied"]
                },
                "answer": response_text,
                "agent_execution": {
                    "tools_used": tools_used,
                    "number_of_tools": 0,
                    "status": "no_tools_required",
                    "timing": timing_summary,
                },
                "evidence": evidence,
                "responsible_ai": responsible_ai
            }

        # ----------------------------------------------
        # Send user message to agent
        # ----------------------------------------------

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content=request.message)
                ]
            }
        )

        messages = result["messages"]

        # ----------------------------------------------
        # Final answer
        # ----------------------------------------------

        final_message = messages[-1]

        response_text = final_message.content

        # ----------------------------------------------
        # Extract tools used
        # ----------------------------------------------

        tools_used = []

        for message in messages:

            # ToolMessage represents the result
            # returned by a tool
            if getattr(message, "type", None) == "tool":

                tool_name = getattr(message, "name", None)

                if tool_name and tool_name not in tools_used:
                    tools_used.append(tool_name)

        # ----------------------------------------------
        # Extract tool evidence
        # ----------------------------------------------

        evidence = []

        for message in messages:

            if getattr(message, "type", None) == "tool":

                tool_name = getattr(message, "name", "unknown_tool")

                tool_output = getattr(message, "content", "")

                evidence.append(
                    {
                        "tool": tool_name,
                        "result": tool_output
                    }
                )

        # ----------------------------------------------
        # Evidence validation
        # ----------------------------------------------

        validation = validate_evidence_claims(
            response_text,
            evidence
        )

        response_text = validation["answer"]

        # ----------------------------------------------
        # Responsible AI
        # ----------------------------------------------

        responsible_ai = run_responsible_ai_checks(
            request.message,
            response_text,
            preferences=pref_dict,
        )

        # ----------------------------------------------
        # Optional research telemetry (gated by consent)
        # ----------------------------------------------

        record_research_telemetry(
            event={
                "message_length": len(request.message),
                "tools_used": tools_used,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            consent_granted=prefs.data_sharing_consent,
        )

        # ----------------------------------------------
        # Return complete response
        # ----------------------------------------------

        timing_summary = end_request_timing(timing_token)

        return {
            "evidence_validation": {
                "passed": validation["passed"],
                "issues": validation["issues"],
                "checks_applied": validation["checks_applied"]
            },
            "answer": response_text,
            "agent_execution": {
                "tools_used": tools_used,
                "number_of_tools": len(tools_used),
                "status": (
                    "tools_executed"
                    if tools_used
                    else "no_tools_required"
                ),
                "timing": timing_summary,
            },
            "evidence": evidence,
            "responsible_ai": responsible_ai
        }
    finally:
        clear_request_portfolio_cache(cache_token)
        reset_current_preferences(token)
        reset_caller_authorization(auth_token)
        if timing_summary is None:
            end_request_timing(timing_token)