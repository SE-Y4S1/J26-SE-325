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

Five steps. Nothing outside your own files needs to change.

**1. Copy two files from Component 1.**

| File | Why |
|---|---|
| `service/cors.py` | The browser calls you directly. Without CORS headers it discards your response before your code runs — and the failure looks like a network error, not a 403. |
| `service/auth.py` | Verifies the platform service's JWT with the shared secret. Verification only, so you never call the platform service on the request path and its downtime cannot become yours. |

Wire both in beside your `app = FastAPI(...)`:

```python
from service.cors import install_cors
install_cors(app)

from service.auth import CallerIdentity, require_user

@app.post("/your/endpoint")
def handler(request: YourRequest, caller: CallerIdentity | None = Depends(require_user)):
    ...
```

`require_user` returns `None` when `AUTH_REQUIRED` is off, so your existing tests keep
working unauthenticated. Compose sets it to `true`.

**2. Share `JWT_SECRET`.** One value in the repo-root `.env`, read by every service. The
platform service refuses to start if it is missing, under 32 characters, or still the
placeholder.

**3. Register your base URL** in `frontend/lib/api/client.ts`:

```ts
export const SERVICES = {
  platform:  process.env.NEXT_PUBLIC_PLATFORM_URL  ?? "http://localhost:8100",
  portfolio: process.env.NEXT_PUBLIC_PORTFOLIO_URL ?? "http://localhost:8000",
  fraud:     process.env.NEXT_PUBLIC_FRAUD_URL     ?? "http://localhost:8001",   // <- yours
} as const;
```

**4. Generate types, do not write them.** Add your schema to the `gen:api` script in
`package.json` and run `npm run gen:api` with your service up. Hand-written interfaces drift
from the backend silently; a generated file turns a contract change into a reviewable diff.

**5. Replace your placeholder page** at `app/(platform)/<name>/page.tsx` and flip
`ready: true` for your entry in `NAV` in `app/(platform)/layout.tsx`.

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
