"""Shared CORS setup, copied verbatim into every component service.

The frontend calls each backend DIRECTLY from the browser rather than proxying through
Next.js, so every service must return CORS headers or the browser refuses the response
before any application code runs.

This file is meant to be duplicated into Components 2-4 unchanged. One env var
(`ALLOWED_ORIGINS`) then configures the whole platform consistently, instead of four
services each inventing their own origin list and one of them getting it subtly wrong.

WHY NOT ALLOW `*`
-----------------
A wildcard origin cannot be combined with credentials, and more importantly it lets any
site a user visits issue authenticated cross-origin requests with their token. The list is
explicit and env-driven.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

DEFAULT_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")


def allowed_origins() -> list[str]:
    """Origins from ALLOWED_ORIGINS (comma separated), else the local Next.js dev server."""
    raw = os.getenv("ALLOWED_ORIGINS", "").strip()
    if not raw:
        return list(DEFAULT_ORIGINS)

    origins = [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]

    if "*" in origins:
        # Loud, because it is almost always someone debugging a CORS error who then forgets.
        logger.warning(
            "ALLOWED_ORIGINS contains '*'. Any site can then make cross-origin requests "
            "carrying the user's token. Acceptable only for throwaway local debugging."
        )
    return origins or list(DEFAULT_ORIGINS)


def install_cors(app: FastAPI) -> None:
    """Attach CORS middleware to a FastAPI app."""
    origins = allowed_origins()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        # True so the same setup still works if a component later moves to cookie auth.
        # Safe here only because the origin list is explicit, never '*'.
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        # Without this the browser hides the header from JavaScript even when it is sent.
        expose_headers=["X-Model-Version"],
        max_age=600,
    )
    logger.info("CORS enabled for: %s", ", ".join(origins))
