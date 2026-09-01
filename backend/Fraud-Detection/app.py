"""
app.py
======
FastAPI service for the Real-Time Fraud & Behavioral Anomaly Engine with
Deterministic Enforcement.

This file is deliberately thin. It only does HTTP: validation is in
schemas.py, detection is in fraud_model.py, enforcement is in gateway.py,
orchestration is in pipeline.py and logging is in audit_log.py.

Run it with:      uvicorn app:app --reload
Then open:        http://127.0.0.1:8000
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import adversarial_test
from audit_log import AuditLog
from config import SETTINGS
from fraud_model import MODEL_MODE
from pipeline import FraudPipeline
from schemas import (
    AttackRequest,
    AttackResponse,
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
    Transaction,
)

BASE_DIR = Path(__file__).resolve().parent
SERVICE_NAME = "fraud-detection-gateway"
VERSION = "1.0.0-prototype"

DISCLAIMER = (
    "PROTOTYPE: risk scores are produced by deterministic simulations of an "
    "LSTM autoencoder and a GNN. No neural network is trained or executed."
)

# --- shared application state ----------------------------------------------
AUDIT = AuditLog()
PIPELINE = FraudPipeline(audit=AUDIT)
STARTED_AT = time.time()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Announce the prototype status once, at start-up."""
    print(f"[{SERVICE_NAME}] {DISCLAIMER}", flush=True)
    print(f"[{SERVICE_NAME}] Demo UI : http://127.0.0.1:8000", flush=True)
    print(f"[{SERVICE_NAME}] API docs: http://127.0.0.1:8000/docs", flush=True)
    yield


app = FastAPI(
    title="Real-Time Fraud & Behavioral Anomaly Engine with Deterministic Enforcement",
    description=(
        "Dual-stream anomaly detection (simulated LSTM autoencoder + simulated GNN) "
        "feeding a deterministic security gateway that enforces ALLOW / STEP-UP / BLOCK.\n\n"
        f"**{DISCLAIMER}**"
    ),
    version=VERSION,
    lifespan=lifespan,
)

# Permissive CORS so the demo page also works if it is opened directly from
# disk (file://). Acceptable for a local prototype only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Error handling - the demo must never show a raw stack trace
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError):
    problems = [
        f"{'.'.join(str(p) for p in err.get('loc', [])[1:])}: {err.get('msg')}"
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"error": "Invalid transaction input", "detail": "; ".join(problems)},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal error in the fraud engine", "detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def demo_page():
    """Serve the presentation UI."""
    page = BASE_DIR / "demo.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="demo.html not found")
    return FileResponse(page)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Liveness plus the currently enforced policy - useful on stage."""
    return HealthResponse(
        status="ok",
        service=SERVICE_NAME,
        version=VERSION,
        model_mode=MODEL_MODE,
        uptime_seconds=round(time.time() - STARTED_AT, 2),
        transactions_scored=PIPELINE.transactions_scored,
        policy_version=SETTINGS.gateway.policy_version,
        thresholds={
            "allow_below": SETTINGS.gateway.allow_max,
            "block_at_or_above": SETTINGS.gateway.block_min,
        },
    )


@app.post("/score", response_model=ScoreResponse, tags=["detection"])
async def score(request: ScoreRequest) -> ScoreResponse:
    """
    Score one transaction and enforce a decision.

    Pipeline: behavioural analysis -> relational analysis -> fusion ->
    deterministic gateway -> audit log.
    """
    transaction = Transaction(**request.model_dump(exclude={"persist_state", "attack_type"}))
    return PIPELINE.evaluate(
        transaction,
        persist_state=request.persist_state,
        attack_type=request.attack_type,
    )


@app.post("/simulate-attack", response_model=AttackResponse, tags=["research"])
async def simulate_attack(request: AttackRequest) -> AttackResponse:
    """
    Run one of the three SAFE, SIMULATED evasion scenarios from the research:
    camouflage, slow drift, or structuring.

    Each scenario runs in its own isolated engine instance, so it cannot
    pollute the state of the live demo, and returns the before/after scores
    and gateway decisions for every stage.
    """
    try:
        result = adversarial_test.run_attack(request.attack_type, steps=request.steps)
    except KeyError as exc:  # unknown attack type
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Mirror the simulation into the shared audit trail so the log on the demo
    # page shows the attack alongside the ordinary transactions.
    for step in result["steps"]:
        AUDIT.record(
            transaction_id=f"{request.attack_type}_step_{step['step']}",
            user_id=result["user_id"],
            behavioral_score=step["behavioral_score"],
            graph_score=step["graph_score"],
            risk_score=step["risk_score"],
            decision=step["decision"],
            reason=step["reason"],
            attack_type=request.attack_type,
        )

    return AttackResponse(**{k: v for k, v in result.items() if k != "user_id"})


@app.get("/audit", tags=["governance"])
async def audit(limit: int = Query(default=25, ge=1, le=200)) -> Dict[str, Any]:
    """Most recent gateway decisions, newest first."""
    return {"count": AUDIT.count(), "entries": AUDIT.recent(limit)}


@app.get("/config", tags=["governance"])
async def config() -> Dict[str, Any]:
    """
    Expose the enforced policy. Showing this on stage proves the thresholds
    are configuration, not something the model invented.
    """
    gw = SETTINGS.gateway
    return {
        "model_mode": MODEL_MODE,
        "disclaimer": DISCLAIMER,
        "policy_version": gw.policy_version,
        "bands": {
            "ALLOW": f"risk < {gw.allow_max}",
            "STEP-UP": f"{gw.allow_max} <= risk < {gw.block_min}",
            "BLOCK": f"risk >= {gw.block_min}",
        },
        "fusion": {
            "behavioral_weight": SETTINGS.fusion.behavioral_weight,
            "graph_weight": SETTINGS.fusion.graph_weight,
            "corroboration_bonus": SETTINGS.fusion.corroboration_bonus,
        },
        "navigation_patterns": list(SETTINGS.behavioral.navigation_risk.keys()),
    }


@app.post("/reset", tags=["system"])
async def reset() -> Dict[str, str]:
    """Clear all learned user profiles and graph state, then reseed the demo graph."""
    PIPELINE.reset()
    return {"status": "reset", "detail": "User profiles, graph memory and audit view cleared."}


@app.get("/pipeline", tags=["system"])
async def pipeline_description() -> Dict[str, List[str]]:
    """The processing chain, for the diagram on the demo page."""
    return {
        "stages": [
            "Transaction received and validated",
            "Behavioural analysis (typing, navigation, frequency, amount, device)",
            "LSTM-like sequence anomaly score  [SIMULATED]",
            "GNN-like relational risk score over the entity graph  [SIMULATED]",
            "Late fusion -> final fraud risk score",
            "Deterministic security gateway (thresholds + policy rules)",
            "ALLOW / STEP-UP / BLOCK + audit log entry",
        ]
    }
