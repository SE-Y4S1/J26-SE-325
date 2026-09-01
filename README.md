# An Integrated AI-Driven Smart Finance Platform

Portfolio optimization, fraud detection, blockchain auditability and explainable agentic
assistance for stock markets.

**SLIIT IT4010 · Project J26-SE-325**

> **Main objective (TAF):** Design, implement, and evaluate a prototype AI-Driven Smart
> Finance Platform that unifies portfolio management, liquidity optimization, fraud
> detection, blockchain-based auditability, and agentic AI assistance within a
> responsible-AI framework, demonstrating that these capabilities can operate as one
> coherent, empirically evaluated system rather than as isolated subsystems.

## Components

| # | Component | Owner | Location | Status |
|---|---|---|---|---|
| 1 | Liquidity-aware forecasting & portfolio optimization | Nivakaran S. | [`backend/Portfolio-Optimization`](backend/Portfolio-Optimization) | **built** |
| 2 | Real-time fraud detection | Dushanthini R. | [`backend/Fraud-Detection`](backend/Fraud-Detection) | built — FastAPI on :8001, wired to the frontend |
| 3 | Blockchain auditability | Abeysekara W.C.S.M | [`backend/Blockchain-Auditability`](backend/Blockchain-Auditability) | built — Express + Solidity on :8002, wired to the frontend |
| 4 | Explainable agentic assistance | W.V.A.D.K. Chamara | [`backend/Agentic-Assistance`](backend/Agentic-Assistance) | built — FastAPI on :8003, wired to the frontend (needs an LLM key) |
| — | Shared platform service (auth, portfolios) | — | [`backend/Platform`](backend/Platform) | **built** |
| — | Web frontend | — | [`frontend`](frontend) | **built** |

## Running the platform

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(48))"   # paste as JWT_SECRET
docker compose up --build
```

Then open <http://localhost:3000>, register an account, and create a portfolio.

| Service | Port |
|---|---|
| frontend | 3000 |
| component 1 — portfolio optimization | 8000 |
| components 2–4 | 8001–8003 |
| platform — auth and portfolios | 8100 |
| MLflow | 5000 |
| Kafka (external listener) | 9094 |

## Architecture

```
frontend  :3000   Next.js 16 — the browser calls each service directly (CORS)
   |
   +--> platform    :8100   auth (JWT), users, portfolios   <-- shared by all components
   +--> component1  :8000   forecasting + optimization
   +--> component2  :8001   fraud detection          FastAPI
   +--> component3  :8002   blockchain auditability  Express + Solidity
   +--> component4  :8003   agentic assistance       FastAPI + LangGraph
                       |
                       +--> Kafka topic `portfolio.decisions`
```

Per the TAF, the four subsystems communicate asynchronously through a central event broker so
each can be developed, evaluated and scaled independently. Component 1 publishes every
decision to `portfolio.decisions` with a `model_version`; Component 3 anchors that provenance
on-chain and Component 4 turns the accompanying traces into user-facing explanation.

## How the components are integrated

All four run under one `docker compose up`, behind one login, each with its own screen in the
frontend. The integration deliberately touched as little as possible:

| | What it needed to join |
|---|---|
| **C1** Portfolio | nothing — CORS, JWT and `/health` were built in |
| **C2** Fraud | **nothing** — already had CORS and `/health` |
| **C3** Audit | **nothing** — already had CORS, `/api/health`, and reads `PORT` from the environment |
| **C4** Assistant | CORS middleware and a `/health` endpoint: 39 added lines, 0 removed |

No component's logic, schema or threshold was altered, and no build file was added to anyone
else's folder — Components 2 and 3 run in compose from stock `python:3.12-slim` and
`node:20-slim` images with the source bind-mounted.

**One service being down never breaks another.** Nothing `depends_on` a teammate's component,
each has its own healthcheck, and every frontend screen renders an explanatory state rather
than an error when its service is absent. An E2E spec asserts exactly that: it visits all
three teammate screens with their services down, then plans a withdrawal successfully.

### What has actually been verified

| | |
|---|---|
| Component 1 | 401 tests pass after merging `main` and `Chenuli` |
| Component 2 | **runs** — `/health` OK, and `/score` correctly BLOCKs a high-risk transaction (risk 0.733) |
| Component 3 | **runs** — `/api/health` OK, records endpoint responds; starts in mock-blockchain mode with no `contractConfig.json` |
| Component 4 | **runs** — `/health` OK, CORS preflight returns the frontend origin, and a real agent round trip answers with its responsible-AI checks (~82s on CPU) |
| Frontend | 31 Jest, 16 Playwright, `tsc` clean |
| Compose | `docker compose config` validates all 8 services |

### Component 4 and the model alias

Component 4's agent uses **Ollama, not a hosted API** — `agent.py` constructs
`ChatOllama(model="llama3.2:3b")`. Two consequences worth knowing:

- **There is no API key to set.** An earlier version of `docker-compose.yml` passed
  `OPENAI_API_KEY` and installed `langchain-openai`; both were wrong, and it omitted
  `langchain-ollama`, which is the package actually imported. Fixed to install from the
  component's own `requirements.txt`.
- **Inside a container, `localhost:11434` is the container, not the host.** The compose
  service therefore sets `OLLAMA_HOST=http://host.docker.internal:11434` with a matching
  `extra_hosts` entry, or every chat call fails with a connection error.

> **On this machine `llama3.2:3b` is an alias for `gemma4-e4b`.** The real model is ~2GB,
> which is roughly a day of downloading on this connection, so it was aliased with
> `ollama cp gemma4-e4b:latest llama3.2:3b` — no code change, no download. **Do not attribute
> any Component 4 output here to Llama 3.2**; it came from Gemma. Remove the alias with
> `ollama rm llama3.2:3b`, and on a machine with the real model pulled the code uses the real
> one.

A round trip takes about **82 seconds** on CPU. The Assistant E2E spec allows for that; the
other specs use a 30-second budget, which is correct for them because a service that is *down*
fails immediately.

**Not yet verified:** a full `docker compose up`. Docker Desktop was not running on the
development machine, so all four services were started directly instead — same processes,
same ports, same endpoints, but the container images and healthchecks themselves are
unproven.

Running the real services immediately earned its keep: Component 3 returns
`{success, count, records}` rather than a bare array, so the first version of the frontend
client rendered an empty table no matter how many records existed. That is not a bug any
amount of reading the code would have found.

Identity and portfolios live in a **separate shared service** rather than inside any one
component, so no team has to depend on another's service to know who the user is or what they
hold.

**Adding your component:** [`INTEGRATION.md`](INTEGRATION.md) has the five steps. Each
placeholder page in the running app repeats them in context.

## Repository layout

```
backend/
  Platform/                shared auth + portfolio service
  Portfolio-Optimization/  Component 1  (built)
  Fraud-Detection/         Component 2
  Blockchain-Auditability/ Component 3  (Express + Solidity + its own Vite app)
  Agentic-Assistance/      Component 4
frontend/                  Next.js 16 web app
docs/                      TAF and research documents
docker-compose.yml         full platform stack
INTEGRATION.md             how components connect
```

## Component 1 at a glance

The one component built so far closes the TAF's stated gap for it — *"liquidity-aware
withdrawal planning has not been operationalized as a real-time, user-facing service"* — with
two **separate** optimizers, as the proposal specifies:

- **MOEA/D** for long-term allocation over the weight simplex, with three objectives: maximize
  return, minimize CVaR, minimize liquidity cost.
- **A fuzzy inference system plus a genetic algorithm** for instant liquidation under a hard
  cash deadline.

Measured results (details in the [component README](backend/Portfolio-Optimization/README.md)):

| RQ | Result |
|---|---|
| RQ1 | Baseline LSTM, walk-forward, pinball loss **0.009388**. All three quantiles sit below nominal — a level bias from trailing-window training through a rising market, which makes CVaR conservative. |
| RQ2 | MOEA/D matches Markowitz's expected return at **3.5× lower liquidity cost**. |
| RQ3 | Fuzzy GA beats pro-rata by **96.5%**, largest-first **26.2%**, most-liquid-first **8.7%**. |
| RQ4 | Best unstressed cost but degrades faster than most-liquid-first — traced to fuzzy-layer saturation. Input scaling fixed; the rule-base half is left open as a methodology decision. |

**402 tests** across the platform (362 Component 1, 40 platform service).

## Documents

- [`docs/TAF-J26-SE-325.pdf`](docs/TAF-J26-SE-325.pdf) — the Topic Assessment Form
- [`INTEGRATION.md`](INTEGRATION.md) — how components connect, and the security posture
