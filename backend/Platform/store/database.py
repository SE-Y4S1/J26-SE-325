"""SQLAlchemy engine, session factory, and schema creation.

SQLite is enough for this project -- the TAF scopes balances as "simulated platform balances
rather than real brokerage holdings", so there is no volume or durability requirement that
would justify running Postgres. The ORM is worth it here even though
Portfolio-Optimization's model registry uses raw sqlite3: that registry is one flat table,
whereas users own portfolios which own holdings, and hand-writing those joins and cascades
is where bugs live.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from store.models import Base

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "platform.sqlite"


def database_url() -> str:
    """Resolve the database URL, defaulting to a local SQLite file."""
    configured = os.getenv("DATABASE_URL")
    if configured:
        return normalise_url(configured)
    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DEFAULT_DB_PATH}"


def normalise_url(url: str) -> str:
    """Point a plain Postgres URL at the psycopg 3 driver.

    Hosted providers (Prisma Postgres, Render, Neon) hand out `postgres://` or
    `postgresql://`; SQLAlchemy would read either as psycopg2, which is not installed.
    Hosts with an ephemeral disk -- Render's free tier wipes it on every spin-down -- need
    one of these, or every account and portfolio vanishes with the instance.
    """
    if url.startswith("prisma+postgres://"):
        # Prisma's Accelerate URL speaks Prisma's own protocol, not Postgres wire protocol.
        raise ValueError(
            "DATABASE_URL is a Prisma Accelerate URL (prisma+postgres://). Use the database's "
            "direct connection string instead: postgres://USER:PASSWORD@db.prisma.io:5432/postgres?sslmode=require"
        )
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def create_db_engine(url: str | None = None):
    """Build an engine with the SQLite-specific settings this app needs."""
    resolved = normalise_url(url) if url else database_url()
    is_sqlite = resolved.startswith("sqlite")
    # "sqlite://" with no path is in-memory, and each connection would otherwise get its OWN
    # empty database -- so create_all() lands on one connection and requests hit another,
    # failing with "no such table". StaticPool holds a single connection open so the whole
    # app shares one in-memory database.
    is_memory = is_sqlite and resolved.replace("sqlite://", "").strip("/") in ("", ":memory:")

    kwargs: dict = {"echo": False, "future": True}
    if is_sqlite:
        # SQLite otherwise refuses to be used from a thread other than the creating one, and
        # FastAPI serves requests from a threadpool.
        kwargs["connect_args"] = {"check_same_thread": False}
    if is_memory:
        kwargs["poolclass"] = StaticPool
    if not is_sqlite:
        # Hosted Postgres closes idle connections, and a free-tier instance sleeps between
        # requests; checking each connection on checkout avoids a failed first request.
        kwargs["pool_pre_ping"] = True
        kwargs["pool_recycle"] = 300

    engine = create_engine(resolved, **kwargs)

    if is_sqlite:
        @event.listens_for(engine, "connect")
        def _enable_sqlite_pragmas(dbapi_connection, _record):  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            # OFF by default in SQLite, which silently permits orphaned holdings and
            # portfolios pointing at deleted users -- exactly the integrity the ORM
            # relationships are supposed to guarantee.
            cursor.execute("PRAGMA foreign_keys=ON")
            # Readers do not block the writer; matters as soon as the UI polls while a
            # withdrawal is being saved.
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    return engine


_engine = None
_SessionLocal: sessionmaker | None = None


def init_db(url: str | None = None) -> None:
    """Create the engine and tables. Idempotent; safe to call at startup and in tests."""
    global _engine, _SessionLocal

    _engine = create_db_engine(url)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
    Base.metadata.create_all(_engine)
    logger.info("database ready at %s", _engine.url)


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a session, rolled back on error and always closed."""
    if _SessionLocal is None:
        init_db()

    session = _SessionLocal()  # type: ignore[misc]
    try:
        yield session
    except Exception:
        # Without this an exception mid-request leaves a partial write visible to the next
        # caller on the same connection.
        session.rollback()
        raise
    finally:
        session.close()


def reset_db() -> None:
    """Drop and recreate every table. Testing only."""
    if _engine is None:
        init_db()
    Base.metadata.drop_all(_engine)
    Base.metadata.create_all(_engine)
