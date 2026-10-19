FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies in a separate layer for build caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Run as non-root user for security best practices
RUN useradd --create-home --uid 1000 appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8003

HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8003/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
