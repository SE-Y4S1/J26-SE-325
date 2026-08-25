"""FastAPI dependency providers: settings, model handles, producer.

Kept separate so tests can override each dependency without importing the whole app or
loading a real model.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Environment configuration. See .env.example."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        # `model_version` and friends collide with pydantic's reserved `model_` prefix.
        protected_namespaces=(),
    )

    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_enabled: bool = True
    mlflow_tracking_uri: str = "sqlite:///artifacts/mlflow.db"
    newsapi_key: str | None = None
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "gemma4-e4b"
    # auto|finbert|ollama. "auto" prefers FinBERT when its weights are already cached and
    # falls back to the local Ollama model, so no code path ever starts a large download.
    sentiment_backend: str = "auto"
    universe_path: Path = ROOT / "configs" / "universe.yaml"
    resolved_universe_path: Path = ROOT / "configs" / "resolved_universe.yaml"


@lru_cache
def get_settings() -> Settings:
    """Cached settings. Call get_settings.cache_clear() in tests that patch the environment."""
    return Settings()


_model_cache: dict[str, object] = {}


def get_active_forecaster():
    """Load the active hybrid model once and reuse it across requests.

    Returns None rather than raising when nothing is trained yet: /forecast should report
    "no model registered" as a clean 503, and /portfolio/withdraw must keep working
    regardless -- the fuzzy GA does not depend on a forecaster being present.
    """
    if "forecaster" in _model_cache:
        return _model_cache["forecaster"]

    try:
        from forecasting.base import get_forecaster

        forecaster = get_forecaster("hybrid")
    except Exception as exc:  # noqa: BLE001 - "not trained yet" is a normal early state
        logger.info("no active forecaster available: %s", exc)
        forecaster = None

    _model_cache["forecaster"] = forecaster
    return forecaster


def reset_caches() -> None:
    """Clear cached settings and models. Testing only."""
    get_settings.cache_clear()
    _model_cache.clear()
