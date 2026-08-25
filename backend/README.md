# Backend Services

Five FastAPI services: one per research component, plus a shared platform service that owns
identity and portfolio data.

| Directory | Service | Port | Status |
|---|---|---|---|
| [`Platform`](Platform) | Auth (JWT), users, portfolios — shared by all components | 8100 | **built** |
| [`Portfolio-Optimization`](Portfolio-Optimization) | Component 1 — forecasting + liquidity-aware optimization | 8000 | **built** |
| [`Fraud-Detection`](Fraud-Detection) | Component 2 — real-time fraud + security gateway | 8001 | not started |
| [`Blockchain-Auditability`](Blockchain-Auditability) | Component 3 — ZKP/Merkle anchoring, provenance | 8002 | not started |
| [`Agentic-Assistance`](Agentic-Assistance) | Component 4 — explainable agentic assistant | 8003 | not started |

Ports follow `8000 + component number`, with the shared service on 8100. 8001–8003 are
reserved by convention only — confirm before claiming one.

## Conventions

Each service is an independent uv project with a flat module tree imported via rootdir
(`[tool.uv] package = false`), its own `.venv`, and its own test suite. They are deliberately
**not** a shared Python package: the components have genuinely different dependency sets —
Component 1 alone pulls torch, pymoo, scikit-fuzzy and transformers — and forcing them into
one environment would make every teammate's install everyone else's problem.

Services communicate two ways:

- **Synchronously**, browser → service over HTTP. The frontend calls each service directly,
  so every service needs CORS (`service/cors.py`) and JWT verification (`service/auth.py`).
  Both files are meant to be copied verbatim rather than reimplemented.
- **Asynchronously**, over the Kafka topic `portfolio.decisions`. Component 1 publishes every
  decision with a top-level `model_version`; Components 3 and 4 consume it. The producer is
  fire-and-forget, so a consumer being down never blocks a user-facing request.

Tokens are **issued** by the platform service and only **verified** by each component, using a
shared `JWT_SECRET`. No service calls another on the request path, so one being slow or
restarting cannot take the others down.

## Running

The whole stack, from the repository root:

```bash
docker compose up --build
```

One service on its own:

```bash
cd Portfolio-Optimization
uv run uvicorn service.api:app --port 8000 --reload
```

## Adding a service

See [`../INTEGRATION.md`](../INTEGRATION.md). In short: copy `service/cors.py` and
`service/auth.py` from Component 1, share `JWT_SECRET`, register your base URL in
`frontend/lib/api/client.ts`, generate TypeScript types from your OpenAPI rather than writing
them, and replace your placeholder page.
