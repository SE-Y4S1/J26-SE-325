"""DATABASE_URL handling for hosted Postgres (Render free tier + Prisma Postgres)."""

from __future__ import annotations

import pytest

from store.database import normalise_url


@pytest.mark.parametrize("scheme", ["postgres://", "postgresql://"])
def test_plain_postgres_urls_use_the_psycopg_driver(scheme: str) -> None:
    url = f"{scheme}user:pw@db.prisma.io:5432/postgres?sslmode=require"
    assert normalise_url(url) == "postgresql+psycopg://user:pw@db.prisma.io:5432/postgres?sslmode=require"


def test_prisma_accelerate_url_is_refused_with_the_fix() -> None:
    with pytest.raises(ValueError, match="direct connection string"):
        normalise_url("prisma+postgres://accelerate.prisma-data.net/?api_key=abc")


@pytest.mark.parametrize("url", ["sqlite://", "sqlite:///x.db", "postgresql+psycopg://u@h/db"])
def test_other_urls_pass_through(url: str) -> None:
    assert normalise_url(url) == url
