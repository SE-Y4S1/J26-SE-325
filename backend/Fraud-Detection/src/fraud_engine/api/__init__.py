"""FastAPI gateway for Component 02.

The endpoint returns a fraud score, an ALLOW / STEP_UP / BLOCK decision, and a
short explanation. The decision rules live in fraud_engine.gateway and do not
call a language model.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from fraud_engine import __version__
from fraud_engine.api.schemas import FeedbackRequest, TransactionRequest
from fraud_engine.config import load_settings
from fraud_engine.paths import ARTIFACT_DIR, RESULT_DIR
from fraud_engine.service import FraudService

# Tests pass bundle_path=None to force the untrained heuristic.
_AUTO_BUNDLE = object()

logger = logging.getLogger(__name__)


def default_trained_bundle() -> Path | None:
    """Prefer an explicit artifact, then the executed IEEE-CIS development bundle."""
    for candidate in (
        ARTIFACT_DIR / "fraud_bundle.joblib",
        RESULT_DIR / "ieee-cis" / "fraud_bundle.joblib",
    ):
        if candidate.is_file():
            return candidate
    return None


def create_app(settings: dict | None = None, bundle_path: Path | None | object = _AUTO_BUNDLE) -> FastAPI:
    resolved = settings or load_settings()
    if bundle_path is _AUTO_BUNDLE:
        bundle_path = default_trained_bundle()
    service = FraudService(resolved, bundle_path if isinstance(bundle_path, Path) else None)
    app = FastAPI(
        title="J26-SE-325 Component 02 Fraud Gateway",
        version=__version__,
        description=(
            "Scores one transaction and returns a deterministic ALLOW, STEP_UP, or BLOCK decision. "
            "Do not send names, emails, card numbers, or raw network addresses."
        ),
    )
    app.state.service = service
    app.state.settings = resolved
    # One CORS layer for every browser caller: two stacked CORSMiddleware instances reject
    # each other's preflights. The research dashboard (frontend/, Vite on 5173) is always
    # allowed; the platform frontend's origins come from ALLOWED_ORIGINS.
    dashboard_origins = ["http://127.0.0.1:5173", "http://localhost:5173"]
    configured = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=dashboard_origins + [o for o in configured if o not in dashboard_origins],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept"],
    )

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "component": "fraud-detection",
            "version": __version__,
            "model_version": service.model_version,
            "benchmark_model": service.benchmark_model,
            "kafka_enabled": bool(resolved.get("kafka", {}).get("enabled", False)),
        }

    @app.post("/v1/transactions/score")
    def score_transaction(body: TransactionRequest) -> dict:
        try:
            return service.score(body.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Scoring failed")
            raise HTTPException(status_code=503, detail="The fraud service could not score this transaction.") from exc

    @app.get("/v1/transactions/{transaction_id}")
    def get_transaction(transaction_id: str) -> dict:
        """The decision and explanation already returned for a transaction. Read-only."""
        found = service.lookup(transaction_id)
        if found is None:
            raise HTTPException(status_code=404, detail="No decision is held for this transaction.")
        return found

    @app.get("/v1/dashboard", response_class=HTMLResponse)
    def dashboard() -> str:
        return service.dashboard_html()

    @app.post("/v1/feedback")
    def feedback(body: FeedbackRequest) -> dict:
        service.feedback.add(body.transaction_id, body.fraud_score, body.label)
        return service.feedback.status()

    @app.post("/v1/admin/apply-feedback-calibration")
    def apply_feedback() -> dict:
        try:
            return service.apply_feedback()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/decisions/summary")
    def decision_summary() -> dict:
        with service.lock:
            return dict(service.decision_counts)

    @app.get("/v1/decisions")
    def decisions() -> dict:
        """Recent gateway decisions already stored by this process."""
        with service.lock:
            rows = list(service.recent)
            counts = dict(service.decision_counts)
        rows.reverse()
        return {
            "decisions": rows,
            "counts": counts,
            "scope": "this API process only",
        }

    return app
