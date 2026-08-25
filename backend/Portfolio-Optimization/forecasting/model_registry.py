"""Content-hashed model provenance registry.

REQUIRED, NOT OPTIONAL. Component 3 (blockchain auditability) anchors model-versioning
provenance on-chain and needs something concrete from this component to anchor. Its TAF task
list names "model-versioning and provenance tracking", and its AI-to-smart-contract bridge
consumes "AI risk/liquidity outputs" -- i.e. ours. Every /forecast, /portfolio/optimize and
/portfolio/withdraw response therefore carries a `model_version` resolvable here.

Each record ties a checkpoint to the exact conditions that produced it: content hash, git
commit, training window, data fingerprint, and MLflow run. That combination is what makes an
on-chain anchor meaningful -- a hash alone proves a file existed, not what produced it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "model_registry.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    model_version     TEXT PRIMARY KEY,
    model_name        TEXT NOT NULL,
    content_hash      TEXT NOT NULL,
    git_commit        TEXT NOT NULL,
    train_start       TEXT NOT NULL,
    train_end         TEXT NOT NULL,
    data_fingerprint  TEXT NOT NULL,
    mlflow_run_id     TEXT,
    created_at        TEXT NOT NULL,
    metrics           TEXT NOT NULL,
    is_active         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_models_name ON models(model_name);
CREATE INDEX IF NOT EXISTS idx_models_active ON models(model_name, is_active);
"""


@dataclass(frozen=True)
class ModelRecord:
    model_version: str        # e.g. "hybrid-timesfm-v0.3.1+a1b2c3d"
    model_name: str
    content_hash: str         # sha256 over the adapter/checkpoint bytes
    git_commit: str
    train_start: date
    train_end: date
    data_fingerprint: str     # hash of the resolved universe + feature schema
    mlflow_run_id: str | None
    created_at: datetime
    metrics: dict[str, float]
    is_active: bool


def _connect() -> sqlite3.Connection:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(REGISTRY_PATH)
    connection.executescript(_SCHEMA)
    return connection


def compute_content_hash(checkpoint_path: Path) -> str:
    """Deterministic sha256 over checkpoint bytes. Directories hash in sorted-name order.

    Sorting is what makes a directory hash reproducible -- filesystem iteration order is not
    guaranteed, and an unstable hash would make on-chain anchoring worthless. Relative paths
    are folded into the digest too, so renaming a file inside the adapter changes the hash.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")

    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()

    for child in sorted(path.rglob("*")):
        if child.is_file():
            digest.update(str(child.relative_to(path)).replace("\\", "/").encode("utf-8"))
            digest.update(child.read_bytes())
    return digest.hexdigest()


def compute_data_fingerprint(resolved_universe_path: Path, feature_columns: list[str]) -> str:
    """Hash the resolved universe and feature schema.

    Separate from the content hash so a DATA change is visible in the version even when the
    code and weights did not change -- retraining on a different window must not be
    mistakable for the same model.
    """
    payload = {
        "universe": (
            Path(resolved_universe_path).read_text(encoding="utf-8")
            if Path(resolved_universe_path).exists()
            else ""
        ),
        "features": sorted(feature_columns),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _git_commit() -> str:
    """Current commit, or 'unknown' outside a repo. Never raises -- provenance is
    best-effort metadata and must not block a training run from being recorded."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        return result.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _next_version(connection: sqlite3.Connection, model_name: str, git_commit: str) -> str:
    """Monotonic version string: <name>-v0.<n>+<commit>."""
    row = connection.execute(
        "SELECT COUNT(*) FROM models WHERE model_name = ?", (model_name,)
    ).fetchone()
    return f"{model_name}-v0.{int(row[0]) + 1}+{git_commit}"


def register(
    model_name: str,
    checkpoint_path: Path,
    *,
    train_start: date,
    train_end: date,
    metrics: dict[str, float],
    mlflow_run_id: str | None = None,
    activate: bool = False,
    resolved_universe_path: Path | None = None,
    feature_columns: list[str] | None = None,
) -> ModelRecord:
    """Record a trained checkpoint and return its record."""
    content_hash = compute_content_hash(checkpoint_path)
    git_commit = _git_commit()
    fingerprint = compute_data_fingerprint(
        resolved_universe_path or (REGISTRY_PATH.parents[1] / "configs" / "resolved_universe.yaml"),
        feature_columns or [],
    )

    with _connect() as connection:
        existing = connection.execute(
            "SELECT model_version FROM models WHERE content_hash = ? AND model_name = ?",
            (content_hash, model_name),
        ).fetchone()
        if existing:
            # Identical bytes: return the existing record rather than minting a second
            # version for the same artefact, which would break provenance one-to-one.
            logger.info("checkpoint already registered as %s", existing[0])
            return get_record(existing[0])  # type: ignore[return-value]

        version = _next_version(connection, model_name, git_commit)
        record = ModelRecord(
            model_version=version,
            model_name=model_name,
            content_hash=content_hash,
            git_commit=git_commit,
            train_start=train_start,
            train_end=train_end,
            data_fingerprint=fingerprint,
            mlflow_run_id=mlflow_run_id,
            created_at=datetime.now(timezone.utc),
            metrics=metrics,
            is_active=activate,
        )

        if activate:
            connection.execute(
                "UPDATE models SET is_active = 0 WHERE model_name = ?", (model_name,)
            )

        connection.execute(
            "INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                record.model_version, record.model_name, record.content_hash, record.git_commit,
                record.train_start.isoformat(), record.train_end.isoformat(),
                record.data_fingerprint, record.mlflow_run_id,
                record.created_at.isoformat(), json.dumps(record.metrics),
                int(record.is_active),
            ),
        )

    logger.info("registered %s (hash %s...)", record.model_version, content_hash[:12])
    return record


def _row_to_record(row: tuple) -> ModelRecord:
    return ModelRecord(
        model_version=row[0], model_name=row[1], content_hash=row[2], git_commit=row[3],
        train_start=date.fromisoformat(row[4]), train_end=date.fromisoformat(row[5]),
        data_fingerprint=row[6], mlflow_run_id=row[7],
        created_at=datetime.fromisoformat(row[8]), metrics=json.loads(row[9]),
        is_active=bool(row[10]),
    )


def get_active_version(model_name: str | None = None) -> str:
    """The `model_version` string the service stamps on every response.

    Returns 'unregistered' rather than raising when nothing is trained yet: the withdrawal
    endpoint does not need a model, and it must not 500 because none exists.
    """
    with _connect() as connection:
        if model_name:
            row = connection.execute(
                "SELECT model_version FROM models WHERE model_name = ? AND is_active = 1",
                (model_name,),
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT model_version FROM models WHERE is_active = 1 ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
    return row[0] if row else "unregistered"


def get_record(model_version: str) -> ModelRecord | None:
    """Look up a record -- the resolution step Component 3 needs to verify an anchor."""
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM models WHERE model_version = ?", (model_version,)
        ).fetchone()
    return _row_to_record(row) if row else None


def list_records(model_name: str | None = None) -> list[ModelRecord]:
    """All records, newest first."""
    with _connect() as connection:
        if model_name:
            rows = connection.execute(
                "SELECT * FROM models WHERE model_name = ? ORDER BY created_at DESC", (model_name,)
            ).fetchall()
        else:
            rows = connection.execute("SELECT * FROM models ORDER BY created_at DESC").fetchall()
    return [_row_to_record(row) for row in rows]


def activate(model_version: str) -> ModelRecord:
    """Promote a version to active for its model name."""
    record = get_record(model_version)
    if record is None:
        raise KeyError(f"no such model_version: {model_version}")

    with _connect() as connection:
        connection.execute(
            "UPDATE models SET is_active = 0 WHERE model_name = ?", (record.model_name,)
        )
        connection.execute(
            "UPDATE models SET is_active = 1 WHERE model_version = ?", (model_version,)
        )
    return get_record(model_version)  # type: ignore[return-value]


def export_for_anchoring(model_version: str) -> dict[str, str]:
    """The minimal provenance bundle Component 3 anchors on-chain.

    Deliberately small and flat: on-chain storage is expensive, so this is the digest a
    smart contract can hold, not the full record.
    """
    record = get_record(model_version)
    if record is None:
        raise KeyError(f"no such model_version: {model_version}")

    return {
        "model_version": record.model_version,
        "content_hash": record.content_hash,
        "data_fingerprint": record.data_fingerprint,
        "git_commit": record.git_commit,
        "created_at": record.created_at.isoformat(),
    }
