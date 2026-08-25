# Component 1 service image. Closes the TAF task "containerize and monitor the services".
#
# CPU-only by design: the dev machine has an AMD iGPU and no CUDA, and inference here is
# light (a small residual head over a frozen foundation model). LoRA fine-tuning happens in
# Colab, not in this container.
FROM python:3.12-slim

# uv gives the same resolver as local development, so the lockfile means the same thing here.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PIP_INDEX_URL=https://pypi.org/simple

WORKDIR /app

# Dependencies first, in their own layer: source changes then rebuild in seconds rather than
# re-resolving the whole ML stack.
COPY pyproject.toml ./
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cpu torch \
 && uv pip install --system \
      numpy pandas pyarrow scipy statsmodels pydantic pydantic-settings PyYAML python-dotenv \
      requests yfinance pandas-ta-openbb scikit-fuzzy networkx deap pymoo mlflow \
      fastapi "uvicorn[standard]" confluent-kafka prometheus-fastapi-instrumentator openai \
      transformers peft accelerate

COPY . .

# Non-root: this service is reachable from the other three components, so it should not run
# as root even on a testnet-grade deployment.
RUN useradd --create-home --uid 1000 appuser \
 && mkdir -p /app/artifacts && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# The container is unhealthy until /health answers, so compose will not route traffic to a
# service whose model registry failed to open.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
