# Platform service: identity and portfolio storage shared by all four components.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Dependencies in their own layer so source edits rebuild in seconds.
COPY pyproject.toml ./
RUN uv pip install --system \
      fastapi "uvicorn[standard]" pydantic pydantic-settings email-validator \
      sqlalchemy pyjwt "passlib[argon2]" argon2-cffi python-dotenv

COPY . .

# Non-root: this service holds every user's credentials and holdings.
RUN useradd --create-home --uid 1000 appuser \
 && mkdir -p /app/artifacts && chown -R appuser:appuser /app
USER appuser

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8100/health').read()" || exit 1

CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8100"]
