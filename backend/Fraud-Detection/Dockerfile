FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
# Install the CPU wheel first so a later requirements install does not replace it.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY config ./config
COPY src ./src
COPY scripts ./scripts
ENV PYTHONPATH=/app/src
ENV FRAUD_CONFIG=/app/config/default.yaml
EXPOSE 8082
CMD ["python", "scripts/run_api.py"]
