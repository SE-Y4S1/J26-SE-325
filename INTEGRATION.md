# Platform Integration Guide

How the four components of **J26-SE-325** connect, and what a teammate has to do to join
their service to the frontend.

## Architecture

```
frontend  :3000   Next.js 16 — the browser calls each service DIRECTLY (CORS)
   |
   +--> platform    :8100   auth (JWT), users, portfolios   <-- shared by all components
   +--> component1  :8000   forecasting + portfolio optimization      [built]
   +--> component2  :8001   fraud detection                           [reserved]
   +--> component3  :8002   blockchain auditability                   [reserved]
   +--> component4  :8003   agentic assistance                        [reserved]
```

Two deliberate choices shape everything else:

**The browser talks to each backend directly**, rather than proxying through Next.js. Every
service therefore needs CORS headers, and the access token lives in the browser. The
alternative — a Next.js backend-for-frontend — would have allowed an httpOnly cookie the
page's JavaScript cannot read. See *Security posture* below.

**The platform service owns identity and portfolios**, not Component 1. Fraud, Audit and
Assistance all need to know who the user is and what they hold, and none of them should have
to depend on another team's service to find out.

## Running it

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(48))"   # paste into .env
docker compose up --build
```

Then open <http://localhost:3000>, register, and create a portfolio.

Without Docker, in three terminals:

```bash
cd backend/Platform               && JWT_SECRET=... uv run uvicorn service.api:app --port 8100
cd backend/Portfolio-Optimization && JWT_SECRET=... AUTH_REQUIRED=true uv run uvicorn service.api:app --port 8000
cd frontend                       && npm run dev
```

## Adding your component

**All four are already wired.** This section is kept because the pattern is what a fifth
service would follow, and because it records what each component actually needed.

What integration turned out to require, per component:

| | Needed to join |
|---|---|
| **C1** Portfolio :8000 | nothing — CORS, JWT and `/health` were designed in |
| **C2** Fraud :8001 | **nothing** — already had `CORSMiddleware` and `/health` |
| **C3** Audit :8002 | **nothing** — already had CORS, `/api/health`, and reads `PORT` from the environment |
| **C4** Assistant :8003 | CORS middleware and a `/health` endpoint — 39 lines added, 0 removed |

No component's logic, schema or threshold was touched, and no build file was added to anyone
else's directory: in compose, Components 2 and 3 run from stock `python:3.12-slim` and
`node:20-slim` images with their source bind-mounted.

The two things a service genuinely needs are therefore small:

1. **CORS**, because the browser calls each service directly. Without it the browser discards
   the response before any handler runs. Read the origin list from `ALLOWED_ORIGINS` so one
   env var configures the whole platform.
2. **A health endpoint** that does *not* touch the expensive path. Component 4's returns a
   literal, deliberately: compose and the UI need to tell "this service is not running" from
   "the agent failed on this request", and calling an LLM to answer a healthcheck conflates
   the two.

Then on the frontend: add the base URL to `SERVICES` in `frontend/lib/api/client.ts`, add a
typed client beside `fraud.ts` / `audit.ts` / `assistant.ts`, replace the page under
`app/(platform)/<name>/`, and set `ready: true` in `NAV`.

**Failure isolation is a requirement, not a nicety.** Nothing `depends_on` a teammate's
component, each has its own healthcheck, and every screen renders an explanatory state via
`ApiError.isUnavailable` rather than an error when its service is down. `e2e/components.spec.ts`
asserts it: with all three teammate services down, it visits every one of their screens and
then still plans a withdrawal successfully.

## What Component 1 already gives you

**Portfolio data** — read it from the platform service. Holdings are stored in exactly the
shape Component 1 consumes (`symbol`, `quantity`, `current_price`, `avg_daily_volume`,
`cost_basis`), so no translation layer sits between them to drift.

**Decision events** — every `/portfolio/optimize` and `/portfolio/withdraw` call publishes to
the Kafka topic `portfolio.decisions`:

```json
{
  "event_id": "...", "event_type": "portfolio.withdraw",
  "schema_version": "1.0", "component": "component1_portfolio_optimization",
  "occurred_at": "...", "model_version": "...",
  "request": { ... }, "response": { ... }
}
```

`model_version` is top-level so **Component 3** can anchor provenance without parsing the
payload, and resolve it against the registry in `forecasting/model_registry.py` —
`export_for_anchoring(version)` returns the minimal on-chain bundle.

The producer is fire-and-forget: a consumer being down never blocks a withdrawal.

**Traces for explanation** — every withdrawal response carries `fuzzy_rule_trace` (which
rules fired and at what strength) and, when `use_agent` is set, `agent_reasoning_trace`.
**Component 4** turns those into user-facing explanation. Component 1 deliberately produces
no prose: duplicating that would create scope overlap that is hard to defend.

## Security posture

Worth stating plainly, because it is a deliberate trade-off rather than an oversight:

- **The access token is readable by JavaScript.** Direct browser→backend calls mean the token
  is held in browser storage and sent to every service, so an XSS would expose it. A
  backend-for-frontend would have permitted an httpOnly cookie. Given the TAF's Legal Impact
  section on auditability, and that Component 2 is itself a security gateway, this belongs in
  the dissertation as a stated choice.
- **Origins are explicit, never `*`.** A wildcard cannot be combined with credentials and
  would let any site the user visits issue authenticated requests on their behalf.
- **Passwords use argon2**, not bcrypt — no silent 72-byte truncation, so a long passphrase
  stays strong.
- **Login does not leak account existence.** Unknown email and wrong password return the same
  status and body; distinguishing them turns the form into an enumeration oracle.
- **Cross-user access returns 404, not 403.** A 403 would confirm the id exists.
- **JWT decoding pins one algorithm.** Accepting a caller-supplied list is how the `alg: none`
  and RS256→HS256 confusion attacks work.

## Behaviours that look like bugs and are not

| Observation | Explanation |
|---|---|
| `POST /forecast` returns **503** | No trained forecaster is registered. Expected until `experiments/colab_finetune.ipynb` has run; the UI renders an explanatory empty state. |
| A withdrawal returns **200** with `feasible: false` | "You can raise 80k of the 300k you asked for" is a real answer. RQ4 depends on infeasible plans being observable rather than thrown away. |
| `model_version` reads `unregistered` | No checkpoint registered yet. The withdrawal path does not need one. |
| `/portfolio/optimize` takes ~7s | MOEA/D runs 100 generations over 45 reference directions. Most of the time is pymoo's genetic operators, not the objective functions. |
| Kafka warnings with no broker | The producer is fire-and-forget by design; the service degrades to a no-op producer. |

## Ports

| Service | Port | Status |
|---|---|---|
| frontend | 3000 | built |
| component1 — portfolio optimization | 8000 | built |
| component2 — fraud detection | 8001 | reserved |
| component3 — blockchain auditability | 8002 | reserved |
| component4 — agentic assistance | 8003 | reserved |
| platform — auth and portfolios | 8100 | built |
| mlflow | 5000 | built |
| kafka (external listener) | 9094 | built |

8001–8003 are reserved by convention only — confirm before claiming one.
