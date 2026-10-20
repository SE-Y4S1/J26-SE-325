# Nexa

*An Integrated AI-Driven Smart Finance Platform*

Liquidity-aware portfolio optimization, real-time fraud detection, blockchain auditability and
explainable agentic assistance for stock, crypto and forex markets, delivered as one running
system under a responsible-AI framework.

**SLIIT IT4010 · Project J26-SE-325**

> **Main objective (TAF):** Design, implement, and evaluate a prototype AI-Driven Smart
> Finance Platform that unifies portfolio management, liquidity optimization, fraud
> detection, blockchain-based auditability, and agentic AI assistance within a
> responsible-AI framework, demonstrating that these capabilities can operate as one
> coherent, empirically evaluated system rather than as isolated subsystems.

---

## Table of contents

1. [Overview](#1-overview)
2. [The four components](#2-the-four-components)
3. [System architecture](#3-system-architecture)
4. [Quick start](#4-quick-start)
5. [Methodology](#5-methodology)
   - [5.1 Research approach and evaluation discipline](#51-research-approach-and-evaluation-discipline)
   - [5.2 Component 1: forecasting and portfolio optimization](#52-component-1-liquidity-aware-forecasting-and-portfolio-optimization)
   - [5.3 Component 2: fraud detection](#53-component-2-real-time-fraud-detection-with-deterministic-enforcement)
   - [5.4 Component 3: blockchain auditability](#54-component-3-blockchain-auditability)
   - [5.5 Component 4: explainable agentic assistance](#55-component-4-explainable-agentic-assistance)
   - [5.6 Integration methodology](#56-integration-methodology)
   - [5.7 Responsible-AI framework](#57-responsible-ai-framework)
6. [Technology stack](#6-technology-stack)
7. [Repository layout](#7-repository-layout)
8. [Service and API reference](#8-service-and-api-reference)
9. [Frontend](#9-frontend)
10. [Configuration](#10-configuration)
11. [Testing and verification](#11-testing-and-verification)
12. [Results so far](#12-results-so-far)
13. [Security posture](#13-security-posture)
14. [Known limitations and honest caveats](#14-known-limitations-and-honest-caveats)
15. [Behaviours that look like bugs and are not](#15-behaviours-that-look-like-bugs-and-are-not)
16. [Documents and further reading](#16-documents-and-further-reading)

---

## 1. Overview

Commercial robo-advisors automate long-term allocation, typically with mean-variance
optimization. They offer no instant, loss-minimized withdrawal mechanism, no tamper-evident
record of why an AI made a decision, and little the end user can inspect. Research on each
of those problems exists, but usually in isolation.

This project builds the four capabilities as independent services and then integrates them
into a single product with one login, one portfolio store and one web interface:

| Concern | Question the platform answers | Where |
|---|---|---|
| Liquidity | How should a user raise cash *now* at the lowest realized loss, and how should the rest be allocated for the long term? | Component 1 |
| Security | Is this transaction fraudulent, and what action must be enforced regardless of what a model says? | Component 2 |
| Auditability | Can any AI decision be proven, later, to be unaltered, and which model and policy produced it? | Component 3 |
| Trust | Can the user understand *why* a decision was made, and stay in control of it? | Component 4 |

The research gap targeted by Component 1 (TAF section 6): *"Liquidity-aware withdrawal
planning has not been operationalized as a real-time, user-facing service."*

---

## 2. The four components

| # | Component | Owner | Location | Port | Stack | Status |
|---|---|---|---|---|---|---|
| 1 | Liquidity-aware forecasting and portfolio optimization | Nivakaran S. | [`backend/Portfolio-Optimization`](backend/Portfolio-Optimization) | 8000 | FastAPI, PyTorch, pymoo, scikit-fuzzy, Kafka | Built, 636 tests |
| 2 | Real-time fraud detection | Dhushanthini R. | [`backend/Fraud-Detection`](backend/Fraud-Detection) | 8001 | FastAPI, PyTorch (LSTM-AE, GNN) | Built, 33 tests; untrained heuristic until a bundle is trained (see 14) |
| 3 | Blockchain auditability | Abeysekara W.C.S.M | [`backend/Blockchain-Auditability`](backend/Blockchain-Auditability) | 8002 | Express, Solidity, Hardhat | Built; mock ledger when no node is running |
| 4 | Explainable agentic assistance | W.V.A.D.K. Chamara | [`backend/Agentic-Assistance`](backend/Agentic-Assistance) | 8003 | FastAPI, LangGraph, Ollama, SHAP | Built; needs a local Ollama model |
| - | Shared platform service (auth, portfolios, executions) | - | [`backend/Platform`](backend/Platform) | 8100 | FastAPI, SQLAlchemy, SQLite | Built, 64 tests |
| - | Web frontend | - | [`frontend`](frontend) | 3000 | Next.js 16, React 19, Tailwind v4 | Built |

Each component directory has its own README with deeper detail. Component 1's is the most
extensive because it carries the quantitative research questions.

---

## 3. System architecture

```
                          Browser  (Next.js 16, :3000)
                                 |
        calls each service DIRECTLY over HTTP (CORS, bearer JWT)
                                 |
   +-------------+---------------+----------------+----------------+----------------+
   |             |               |                |                |                |
platform      component1      component2       component3       component4
:8100         :8000           :8001            :8002            :8003
auth, users,  forecast +      fraud scoring    audit evaluate,  LangGraph agent
portfolios    optimize +      + deterministic  hash, anchor,    + SHAP + RAI
(SQLite)      withdraw +      gateway          verify           checks
              market data
                 |
                 +--> Kafka topic `portfolio.decisions`   (fire-and-forget producer)
                 +--> MLflow :5000   (experiment tracking, model registry)
```

### Design principles

1. **Independent services, one product.** Each component is its own process with its own
   dependencies and test suite. Component 1 alone pulls torch, pymoo, scikit-fuzzy and
   transformers; forcing the others into one Python environment would make every teammate
   carry everyone else's install.
2. **The browser calls each backend directly.** No Next.js proxy. Every service therefore
   needs CORS, and the access token lives in the browser (a deliberate trade-off, see 13).
3. **Tokens are issued once and verified everywhere.** The platform service signs JWTs; each
   component verifies them with the shared `JWT_SECRET`. Services do not call each other on
   the request path, with one deliberate exception: executing a plan (see *Cross-component
   data flow*), where the platform service must screen and anchor before holdings change.
4. **Asynchronous decision events.** Component 1 publishes every optimize and withdraw
   decision to Kafka with a top-level `model_version`. The producer is fire-and-forget; a
   consumer being down never blocks a user request.
5. **Failure isolation is a requirement.** Nothing in `docker-compose.yml` `depends_on` a
   teammate's component; each has its own healthcheck; every frontend screen renders an
   explanatory empty state, not an error, when its service is absent.
6. **Identity and portfolios live in a neutral service**, so no team depends on another's
   service to know who the user is or what they hold.
7. **Types are generated, not hand-copied.** The frontend's TypeScript types for Component 1
   and the platform come from each service's OpenAPI document (`npm run gen:api`), so a
   backend contract change appears as a reviewable diff.

### Decision events

```json
{
  "event_id": "...", "event_type": "portfolio.withdraw",
  "schema_version": "1.0", "component": "component1_portfolio_optimization",
  "occurred_at": "...", "model_version": "...",
  "request": { },  "response": { }
}
```

`model_version` is top-level so Component 3 can anchor provenance without parsing the
payload, and resolve it against the registry in `forecasting/model_registry.py`
(`export_for_anchoring(version)` returns the minimal on-chain bundle: content hash, data
fingerprint, git commit, creation time). Every withdrawal response also carries
`fuzzy_rule_trace` and, optionally, `agent_reasoning_trace`, which Component 4 turns into
user-facing explanation.

### Cross-component data flow

A plan from Component 1 is a recommendation until it is executed. Executing it is the one
path that crosses every component, and the platform service drives it because it owns the
holdings that change:

```
 Component 1  /portfolio/withdraw  -> assets_to_sell            (sells, native quantities)
              /portfolio/optimize  -> rebalance_orders          (buys and sells, FX-sized)
      |
      v   POST /portfolios/{id}/executions   {kind, plan}
 Platform     derives orders from the plan, checks them against holdings, hashes plan+orders
      |
      v   POST /v1/transactions/score        transaction_id = execution id, pseudonymous account_key
 Component 2  ALLOW / STEP_UP / BLOCK, score, gateway rules, feature attributions
      |
      v   POST /api/audit/evaluate           riskScore, upstreamDecision, planHash, c1+c2 model versions
 Component 3  final decision = stricter of P001 and Component 2; canonical hash anchored on-chain
      |
      v
 Platform     EXECUTED (holdings change) | PENDING_REVIEW | BLOCKED | HELD
      .
      .   read-only, by execution id
 Component 4  get_execution_explanation: why the plan (C1 rule trace), why the decision
              (C2 rules and attributions), proof it is unaltered (C3 verify)
```

| Outcome | When | Holdings |
|---|---|---|
| `EXECUTED` | Component 2 ALLOW and Component 3 anchored APPROVE | change |
| `BLOCKED` | Component 2 BLOCK (enforced even if Component 3 is down), or Component 3 REJECT | unchanged |
| `PENDING_REVIEW` | Component 2 STEP_UP, or P001 requires human review; the account holder re-enters their password to approve or reject, and the review is anchored before anything moves | unchanged until approved |
| `HELD` | Component 2 or 3 unreachable. Fail closed: nothing executes unscreened or unanchored. `POST /executions/{id}/retry` resumes without rescoring | unchanged |

The execution id doubles as Component 2's `transaction_id` and Component 3's `transactionId`,
so any component can join the three records without a lookup table.

**Corrections made to the originally proposed flow** (portfolio -> fraud -> blockchain, plus
explainable AI):

1. *Component 3 re-decided instead of recording Component 2's decision.* P001 looked only at
   risk score and confidence, so a transaction Component 2 BLOCKED on a rule (structuring,
   rate limit, reputation) at a moderate score was anchored as APPROVE: a ledger contradicting
   the enforcement it is meant to prove. Component 3 now takes `upstreamDecision` and anchors
   the stricter of the two, with the upstream decision inside the hash.
2. *Component 1's decision was not on the ledger at all; only the fraud verdict was.* The
   anchored record now carries the plan hash and Component 1's model version, so "which plan,
   produced by which model, was screened" is provable, not only "what the fraud score was".
3. *Explainable AI is a reader of all three, not a fourth stage after the blockchain.* It
   explains why the plan, why the decision and whether the record verifies, using each
   component's own evidence: Component 1's fuzzy rule trace, Component 2's attributions and
   Component 3's verification. Component 4's synthetic SHAP model explains none of these and
   is not used for them.
4. *Rebalance trades must be sized by Component 1.* Its weights are fractions of an
   FX-converted total; the platform stores no currency per holding, so it would mis-size every
   trade in a mixed-currency portfolio. `/portfolio/optimize` now returns `rebalance_orders`.

Not done: Kafka. Component 1 still publishes `portfolio.decisions` and Component 2 can consume
`finance.transactions.raw`, but the execution path is synchronous HTTP because the user waits
on its outcome. Wiring the topics together is an async alternative, not a dependency.

---

## 4. Quick start

Hosted (Render free plan + Prisma Postgres + Vercel): see [`DEPLOY.md`](DEPLOY.md).

### Whole platform with Docker

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(48))"   # paste as JWT_SECRET in .env
docker compose up --build
```

Open <http://localhost:3000>, register an account, create a portfolio, then try Withdraw,
Optimize, Forecast and Analysis.

| Service | Port |
|---|---|
| frontend | 3000 |
| component 1: portfolio optimization | 8000 |
| component 2: fraud detection | 8001 |
| component 3: blockchain auditability | 8002 |
| component 4: agentic assistance | 8003 |
| platform: auth and portfolios | 8100 |
| MLflow | 5000 |
| Kafka (external listener) | 9094 |

The platform service refuses to start if `JWT_SECRET` is unset, under 32 characters, or still
the placeholder.

Every setting for every service lives in the one root `.env` (see `.env.example`). Start any
part of the system from the repository root by service name:

```bash
docker compose up -d                          # everything
docker compose up -d platform component2      # just the platform and fraud detection
docker compose up -d component3               # one component
docker compose logs -f component1             # follow one service
docker compose down                           # stop (volumes keep accounts and records)
```

Service names: `platform`, `component1` (portfolio), `component2` (fraud), `component3` (audit),
`component4` (assistant, pulls its Ollama model on first start), `frontend`, `kafka`, `mlflow`.
Setting `DATABASE_URL` in `.env` moves the platform and Component 3 onto that Postgres
database; leave it unset to keep local data in Docker volumes.

### Without Docker

Component 1 installs torch and the ML stack, which is slow; running services directly is the
faster loop during development.

```bash
# platform (auth, portfolios)
cd backend/Platform && JWT_SECRET=... uv run uvicorn service.api:app --port 8100

# component 1
cd backend/Portfolio-Optimization
uv venv --python 3.12
uv pip install --index-url https://download.pytorch.org/whl/cpu torch   # CPU wheel first
uv pip install -r requirements.txt
JWT_SECRET=... AUTH_REQUIRED=true uv run uvicorn service.api:app --port 8000

# components 2-4
cd backend/Fraud-Detection      && pip install -r requirements.txt && FRAUD_API_PORT=8001 python scripts/run_api.py
cd backend/Blockchain-Auditability && npm install && PORT=8002 node src/index.js
cd backend/Agentic-Assistance   && pip install -r requirements.txt && uvicorn main:app --port 8003

# frontend
cd frontend && cp .env.example .env.local && npm install && npm run dev
```

Component 4 needs [Ollama](https://ollama.com) running with the model its code requests
(`llama3.2:3b`). Without it the service still starts and `/health` still answers; only
`/assistant/chat` fails.

---

## 5. Methodology

### 5.1 Research approach and evaluation discipline

The platform is evaluated as a **design-science artefact**: build the prototype, state the
research questions it must answer, measure it against named baselines, and record every
deviation from the proposal. Four disciplines run through all components.

**1. Every claim has a baseline.** Nothing is reported as "good" in isolation. Forecasting is
compared against a plain LSTM and a zero-shot foundation model; the MOEA/D optimizer against
Markowitz; the fuzzy-GA against three naive liquidation rules; the fraud engine against a
behaviour-only model under attack.

**2. No look-ahead.** Look-ahead bias is the usual way a financial ML result becomes
meaningless while looking excellent. The walk-forward harness (`evaluation/backtest.py`)
*asserts* the property rather than documenting it: it raises if any fold's feature rows carry
a timestamp at or after their target, an embargo of at least the forecast horizon separates
folds, indicator warm-up is offset per fold, and sentiment decay state is recomputed per
fold. A deliberate leak test confirms the guard actually fires. Fundamentals are
point-in-time: `fundamentals.pipeline.as_of` returns only reports available at a given date.

**3. Honest reporting of negative and partial results.** An under-calibrated forecaster, a
fuzzy layer that saturates under stress, and simulated fraud streams are all documented where
they occur rather than tuned away. The RQ4 rule-base decision was deliberately left for
supervisor discussion rather than tuned until the proposed method won.

**4. Decisions are reproducible.** Seeds are fixed (`seed=42` for trajectory generation),
LLM decoding for sentiment uses temperature 0 with JSON output, hashes over model
checkpoints are deterministic, and cached data is keyed by content hash.

#### Research questions (Component 1)

RQ1-RQ4 come from the build brief, not the written TAF, and are stated here so they become
the documented evaluation frame. RQ5 was added when the Laya decision model was introduced.

| RQ | Question | Metric | Comparison |
|----|----------|--------|------------|
| **RQ1** | Does the fine-tuned foundation + LSTM hybrid beat a plain LSTM and a zero-shot foundation model? | MAE, RMSE, pinball loss, quantile coverage | LSTM baseline vs zero-shot vs LoRA-tuned vs hybrid, split by asset class |
| **RQ2** | Does the MOEA/D liquidity-aware optimizer beat classic mean-variance at equal or lower realized cost? | Sharpe, Sortino, max drawdown, realized transaction cost | Markowitz vs MOEA/D across 3 Pareto selection rules |
| **RQ3** | Does the fuzzy-GA withdrawal module reduce realized slippage versus naive liquidation? | Realized slippage %, realized loss | Fuzzy GA vs pro-rata / largest-first / most-liquid-first |
| **RQ4** | How does withdrawal-plan quality degrade as liquidity worsens, and where does it break? | Degradation curve, breakdown point | RQ3 repeated across a stress-severity ladder |
| **RQ5** | Does a fine-tuned decision model improve liquidation decisions over zero-shot and rule-only? | Per-question accuracy, realized shortfall | Zero-shot vs fine-tuned Laya |

---

### 5.2 Component 1: liquidity-aware forecasting and portfolio optimization

> **Sub-objective (TAF):** Design a liquidity-aware forecasting and portfolio optimization
> engine that supports instant, loss-minimized user withdrawals alongside long-term return
> optimization.

#### Pipeline

```
ingestion --> features --> forecasting --> optimization --> service
 CSE            technical     TimesFM /       MOEA/D (allocation)     FastAPI
 Binance        sentiment     Chronos-Bolt    fuzzy GA (withdrawal)   Kafka events
 Yahoo          fundamentals  + LSTM residual        ^                MLflow
 FX             liquidity     head                   |
 news channels                                  agent tools --> grounding validator
```

Two optimizers, kept **separate** as the TAF specifies ("MOEA/D, fuzzy genetic algorithm").
They have different decision variables, constraints and algorithms and are never merged.

#### Step 1: Data ingestion and window selection

* **Instruments.** Any instrument on the Colombo Stock Exchange (all ~282 securities),
  Binance (all spot pairs) or Yahoo Finance (US equities, ETFs, FX). Analysis runs only for
  symbols a user holds, the *active set* (`data/active_set.py`).
* **Adapters.** `data/exchanges/{cse,binance,yahoo}.py` each answer "what instruments exist"
  and "give me daily bars" and normalise to one OHLCV schema, so everything above them is
  exchange-agnostic. A unified search ranks across all three (CSE wins ticker ties).
* **Currencies.** Each holding stays in its trading currency (LKR, USD, USDT). Plans are
  expressed in a base currency (default LKR) via `data/fx.py`, with a configurable
  conversion spread charged on foreign proceeds.
* **Per-symbol windows from explicit criteria** (`data/window_selector.py`), not one global
  window: an availability criterion (META resolves to its 2012 IPO), a signal-driven
  forecast horizon (autocorrelation-decay lag), regime coverage (at least 3 volatility
  regimes, via structural-break and volatility-regime detection) and a liquidity floor. Each
  entry records the criterion values that produced it, so no window is an unexplained constant.
* **Robustness.** yfinance signals rate limiting exactly like a delisting (an empty frame),
  so fetches retry with backoff and an empty OHLCV result is never cached. Raw data is cached
  on disk keyed by content hash.

#### Step 2: Features

| Family | Content | Notes |
|---|---|---|
| Technical | MACD (12/26/9), RSI(14), MFI(14), ATR(14), Bollinger width (20), trend and MA crosses, ADX, OBV, support/resistance, 52-week drawdown, relative strength vs ASPI / SPY / BTC | `pandas-ta-openbb`; lookbacks explicit so they can be cited |
| Sentiment | Per-tier, per-day, confidence-weighted polarity and attention spikes | FinBERT if cached, else local Ollama model; Sinhala/Tamil items scored only by Laya-multilingual |
| Fundamentals | Margins, ROE/ROA, growth, leverage, interest cover, P/E, P/B, bank ratios, Piotroski F, Altman Z''-EM, red flags | Extracted from CSE interim report PDFs; point-in-time via `available_from` |
| Liquidity | 20-day traded value, ADV, volatility, participation | Replaces the GA's hard-coded volatility for prepared symbols |

**News and social pipeline** (`news/`). A keyword profile is generated per symbol (the
exchange's own name, sector/macro templates, optional local-LLM expansion into aliases,
brands, people and Sinhala/Tamil names). Items are pulled from channels (Google News, GDELT,
RSS, CSE announcements, YouTube, NewsAPI, Jina web, and Agent-Reach social CLIs), filtered
for relevance (word-boundary matching; short tickers must be upper-case; exclusion terms
veto; Sri Lankan context required for CSE sector items), scored, and aggregated by tier.
Raw social text is never stored. Reviewed keyword profiles are never regenerated, and
`configs/keyword_overrides.yaml` holds version-controlled corrections.

**Fundamentals extraction** (`fundamentals/`). Statement pages are found by heading, split
into label / note reference / current / comparative, mapped through ranked synonym lists,
unit-detected (`Rs '000`) and bracket-negatives read correctly. Templates: general, bank,
finance company, insurance, detected from the income statement. Accounting identities are
checked on every report (assets = liabilities + equity, gross profit, profit after tax, net
interest income); a report with `confidence < 0.5` is kept out of decisions. A local-LLM read
is tried only when the parser fails; scanned PDFs are recorded as unusable (no OCR).
`evaluation/extraction_accuracy.py` scores the parser against hand-checked values.

#### Step 3: Forecasting (the hybrid)

```
final_forecast = base_forecast + residual_head(technical, sentiment, base_forecast, asset_class)
```

A foundation model pretrained on raw univariate series (TimesFM 2.5 or Chronos-Bolt)
captures trend and seasonality but has no channel for exogenous features. A small LSTM
**residual head** learns the systematic part of the base model's error as a function of those
features. The base stays frozen (or LoRA-adapted) and the head is zero-initialised, so the
hybrid starts exactly at the base model's accuracy. That gives RQ1 a clean answer to "did
the covariates help".

* **Probabilistic output.** Quantile forecasts (p10 / p50 / p90) trained with pinball loss,
  so the same output feeds CVaR downstream.
* **Head training data** is generated by walking a cut point forward: the base predicts from
  `group[:t+1]` and the forecast is stamped at `t`, aligned with the `target_return` it is
  predicting. The base never sees its own target. `HybridConfig.walk_forward_points` bounds
  cost; `_cut_sequences` cuts LSTM windows from the full consecutive history so training and
  serving see the same shape of sequence.
* **LoRA fine-tuning** (`forecasting/finetune_lora.py`) has a separate step per foundation
  model. Chronos-Bolt computes its own pinball loss; TimesFM 2.5 has no training entry point,
  so its differentiable prefill is mirrored line for line. Adapters only are exported (a few
  MB against ~821 MB of base weights), which is why LoRA is used.
* **Baseline**: `baseline_lstm`, trained locally, no weights download.
* **Model registry** (`forecasting/model_registry.py`) stores version, deterministic content
  hash, data fingerprint and git commit, and serves `/forecast` with the real `model_version`.
* **Empirical fallback.** Where no trained head is installed, the analysis pipeline uses the
  symbol's own past h-day return distribution, always labelled `method: "empirical"`.
* **Training happens on Colab** (`experiments/colab_hybrid_heads.ipynb`): one head per horizon
  (5 and 21 sessions) across every CSE security with at least 200 sessions, the top 50 Binance
  USDT pairs and the US universe.

#### Step 4a: Long-term allocation with MOEA/D

`optimization/moead_rebalance.py` searches the weight simplex with pymoo's MOEA/D over three
objectives:

| Objective | Direction | Definition |
|---|---|---|
| Expected return | maximise | `w . mu`, using the p50 forecast as `mu` |
| CVaR (alpha = 0.95) | minimise | Expected shortfall from scenarios expanded from the quantile forecasts |
| Liquidity cost | minimise | Square-root market-impact model charged on the **trade**, not the holding |

* **Why CVaR.** Variance penalises upside and downside equally, whereas a withdrawing user
  cares about the downside tail. CVaR is coherent (sub-additive), so it rewards
  diversification. CVaR is not named in the TAF; it is a deliberate strengthening and is
  flagged for supervisor sign-off.
* **Why quantile expansion.** CVaR of a portfolio is not the weighted sum of per-asset CVaRs;
  summing would ignore diversification and overstate risk.
* **Liquidity model.** Impact grows with the square root of participation (exponent 0.5,
  coefficient 0.30, calibrated to roughly 30 bp for a 10% ADV trade in a 20%-vol name).
  Both optimizers use the **same** cost model, so any RQ2 vs RQ3 difference is the
  algorithm, not the accounting.
* **Choosing one point from the Pareto front.** Any front-to-point rule embeds a value
  judgement, so three are implemented and reported as a sensitivity analysis: *knee*,
  *max-Sharpe* and *scalarized*. Objectives are min-max normalised first, because return
  (~0.001), CVaR (~0.02) and liquidity cost (~0.0005) differ by orders of magnitude.
* **Baseline.** `baseline_meanvariance.py` is Markowitz max-Sharpe.
* **Interactive latency.** The endpoint runs 8 partitions x 100 generations (the RQ2
  configuration, ~7 s) rather than the 12 x 200 research default (~30 s), because the TAF
  claims a real-time service.

#### Step 4b: Instant liquidation with a fuzzy inference system and a genetic algorithm

Raise `X` within `N` days at the lowest realized loss.

* **Fuzzy layer 1** (`optimization/fuzzy_withdrawal.py`). Mamdani system with three
  antecedents (urgency, market regime, position liquidity) over three terms each, giving 27
  rules, producing a per-holding `sell_priority`. *Why fuzzy:* "how urgently does this user
  need cash" is genuinely imprecise. Overlapping membership functions let a request be 0.7
  "high urgency" and 0.3 "medium" at once instead of forcing a false precision.
* **Fuzzy layer 2: market-signal adjustment.** An 18-rule zero-order Sugeno system over news
  sentiment x forecast outlook x attention spike outputs a signed priority adjustment. It is
  separate because adding three antecedents to the 27-rule grid would make 729 rules, unreadable
  in an audit trace. The adjustment is monotone: worse news or a weaker outlook never lowers
  sell priority.
* **Genetic algorithm** (`optimization/ga_withdrawal.py`, DEAP-style). Searches ordered
  `(asset, fraction)` schedules over multiple days under a hard cash deadline and a 10%-of-ADV
  daily participation cap.
  Population 120, 200 generations, tournament selection (size 3), elitism. Part of the initial
  population is seeded from the fuzzy ordering, and deviation from fuzzy priority is also
  penalised, so the fuzzy layer is provably load-bearing.
* **Fitness terms.** Realised market-impact cost, forecast return forgone (or loss avoided)
  on what is sold, timing risk from the forecast band (scaled by `risk_tolerance`), FX
  conversion cost, crystallised loss vs cost basis (weight 0.25), and penalties for
  shortfall and priority deviation.
* **Penalty scaling.** Shortfall and priority penalties are proportional to the target amount,
  so the GA behaves identically in USD and LKR (~300x). An absolute-constant version preferred
  under-raising and was feasible in only 6% of realistic scenarios.
* **Infeasibility is a result.** "You can raise 80k of the 300k you asked for" returns HTTP
  200 with `feasible: false` and the shortfall, because RQ4 depends on infeasible plans being
  observable.
* **Baselines** (`naive_liquidation.py`): pro-rata, largest-position-first, most-liquid-first,
  each costed by the same simulator and returned as `baseline_comparison`.
* **Stress testing** (`stress_scenarios.py`): three families with a severity knob in [0, 1],
  `ADV_COLLAPSE`, `VOLATILITY_SPIKE` and `COMPOUND`, swept to find the breakdown point (RQ4)
  because real crashes are too few to answer it.
* **Traceability.** Every response carries `fuzzy_rule_trace` (which rules fired and at what
  strength), `cost_breakdown`, `market_context` per holding and `effective_inputs`.

#### Step 5: The Laya two-stage decision model (optional)

`decision/` adds a small language-model decision layer. Stage 1 asks technical, fundamental
and news questions per holding (in the background). Stage 2, on the request path in one
batched pass, asks per holding *sell now / spread / defer / exclude* and *downside risk*, and
per portfolio *urgency* and *risk tolerance*. Answers with confidence >= 0.6 adjust fuzzy
priorities, cap an excluded holding at zero (only if the rest can cover the target), and
replace urgency and risk tolerance. **The GA still computes every amount.** Below the
threshold, or without Laya, the fuzzy rules decide and `market_context[symbol].decision` says
why. Training data is outcome-labelled and point-in-time (`decision/labels.py`), the encoder
is frozen, and a head-only checkpoint is exported from Colab Part C.

#### Step 6: The grounding constraint on the agent

An optional tool-calling agent may reason over signals and choose *how* to invoke the
optimizer. It may **never** emit a sell amount that did not come from `run_fuzzy_ga_withdrawal`.
This is enforced in code (`agent/reference_agent.py::enforce_grounding`): a transcript is valid
only if the tool was called **and** every number in the final decision matches that call's
output, keyed on `(symbol, execution_day)` so multi-day liquidations are verified correctly.
Requiring only the call would let a model invoke the tool, ignore it and invent figures.

The reason is regulatory (TAF Legal Impact): an LLM producing "sell 40% of AAPL" from raw
signals has no auditable derivation, whereas the fuzzy-GA output has one, with a rule trace.
Small local models fail the tool contract in three observed ways (looping, prose instead of
JSON, runaway wall-clock), each handled by a nudge, a one-shot re-ask and a turns-plus-time
budget. Bulk trajectory generation therefore uses a deterministic scripted policy, giving 767
grounded trajectories (from 800 scenarios) for supervised fine-tuning, all of which call the
optimizer.

#### Model policy: local first

Nothing starts a large download implicitly. Every model-backed step checks what is present
locally and either uses it or fails with the exact fetch command.

| Step | Preferred | Local fallback |
|---|---|---|
| Sentiment | `ProsusAI/finbert` (~420 MB) | Ollama `gemma4-e4b` |
| Forecasting | TimesFM / Chronos-Bolt (~821 MB each) | `baseline_lstm` |
| Agent | Fine-tuned Llama (Colab) | Ollama `gemma4-e4b` |

`available_foundation_models()` means the package imports; `usable_foundation_models()` also
requires weights on disk. A cache directory is not a cache hit: interrupted fetches leave
zero-byte `.incomplete` placeholders, so checks require a real multi-megabyte file.

#### Deviations from the build brief

Verified against primary sources on 2026-08-24; each would have caused a build failure if
followed as written.

| Brief said | Reality | What is done |
|---|---|---|
| Python 3.10 for timesfm | timesfm is `>=3.10`; `pandas-ta` and `jax` need `>=3.12` | Python 3.12 |
| `timesfm` v2.5 | PyPI tops out at 2.0.2; "2.5" is the *model* generation | Pin `timesfm==2.0.2`, use the 2.5 model class |
| TimesFM XReg for covariates | Pulls `jax[cuda]`, manylinux-only | Covariates go through the residual head |
| `pandas-ta` | 0.3.14b0 pulled; current release pins `numba` | `pandas-ta-openbb` |
| Alpha Vantage for forex | n/a | yfinance FX tickers: deeper history, no key |
| NewsAPI for sentiment | Free tier is ~30 days, cannot support a walk-forward backtest | GDELT for history, NewsAPI for the live window |
| Hosted API for the agent | n/a | Local Ollama, free, offline, reproducible |
| One global data window | n/a | Per-symbol windows from explicit criteria |

---

### 5.3 Component 2: real-time fraud detection with deterministic enforcement

> **Principle:** the machine-learning side never decides anything. It produces a score. A
> deterministic gateway is the only component allowed to choose an action.

Full method, datasets and executed benchmarks: [`backend/Fraud-Detection/README.md`](backend/Fraud-Detection/README.md)
and `experiments/EXPERIMENT_REPORT.md` there.

* **Models.** An LSTM autoencoder scores how badly a short per-account sequence reconstructs;
  GCN/GAT graph models pass messages from earlier transactions sharing an account or device.
  A fusion weight chosen on the validation slice combines them. Plain PyTorch, no PyTorch
  Geometric. Baselines: logistic regression, SVM, random forest.
* **Benchmarks.** IEEE-CIS (20k and 100k earliest rows by `TransactionDT`), Elliptic, with
  time-ordered splits, train-only scalers and an untouched test slice. Results in
  `experiments/results/`.
* **Gateway** (`fraud_engine/gateway`). Cut-points are chosen from **legitimate validation
  scores**: about 5% of legitimate traffic may reach STEP_UP, about 1% BLOCK. Amount, rate
  limit, IP reputation and concept drift can only make the action stricter.
* **Privacy.** Requests carrying a name, email, card number, password or raw network address
  are rejected; callers send pseudonymous `account_key` / `device_key` values.
* **Honesty.** Without a trained bundle the service answers with `heuristic-untrained` and
  says so in every response (`benchmark_model: false`); that score is not a research result.

---

### 5.4 Component 3: blockchain auditability

> **Sub-objective (TAF):** Design a privacy-preserving, tamper-evident blockchain audit layer
> that bridges continuous AI decisions to deterministic on-chain smart contract enforcement.

**Flow:** an AI output (risk score, confidence, amount, type, model version) enters
`POST /api/audit/evaluate`, is judged by a deterministic policy, canonicalised and hashed, and
only the hash plus provenance metadata is committed on-chain.

1. **Policy engine** (`policyEngine.js`, policy P001 v1.0), deterministic and versioned:
   `riskScore >= 80 and confidence >= 70` gives REJECT; `riskScore >= 80 and confidence < 70`
   gives HUMAN_REVIEW; below 80 gives APPROVE. Low confidence on a high-risk decision escalates
   to a human rather than acting automatically.
2. **Canonical hashing** (`hashService.js`). The record fields are placed in a fixed set with
   sorted keys and hashed with SHA-256, so identical inputs yield identical commitments across
   time and environments.
3. **Privacy-preserving anchoring.** `AuditRegistry.sol` stores only the hash, decision,
   policy version, model version, timestamp and submitter. Amounts, customer information and
   raw risk signals stay off-chain. This is a hash-commitment scheme; ZKP is not implemented.
4. **Human-in-the-loop.** `POST /api/audit/review` records a reviewer's decision
   (HUMAN_APPROVED / HUMAN_REJECTED) as a new canonical record and commitment.
5. **Verification.** `GET /api/audit/:id/verify` recomputes the canonical hash from the stored
   off-chain record and compares it with the on-chain commitment. `POST /:id/tamper` alters the
   stored record on purpose so the UI can demonstrate tamper detection.
6. **Ledger adapter** (`blockchainService.js`). Talks to a local Hardhat node when available
   and falls back to an in-memory mock ledger otherwise, so the platform runs without a chain.

Provenance of Component 1 models (`content_hash`, `data_fingerprint`, `git_commit`) is designed
to anchor through the same registry via `modelVersion`.

---

### 5.5 Component 4: explainable agentic assistance

> **Sub-objective (TAF):** Build a localized, explainable agentic LLM assistant, empirically
> evaluated for its effect on user trust and perceived control, within a responsible-AI
> governance framework.

* **Orchestration.** A LangGraph `StateGraph` with an LLM node and a tool node joined by
  `tools_condition`: the LLM decides whether to call a tool, the tool runs, and control
  returns to the LLM until it answers. The model is a **local Ollama** model (`ChatOllama`,
  temperature 0.1), so no data leaves the machine and no API key exists.
* **Tools** (`tools/`). `get_fraud_analysis`, `get_portfolio_analysis`, `get_blockchain_audit`.
  The system prompt forces tool use for fraud, portfolio and audit questions, forbids
  inventing financial data, forbids guaranteeing returns, and requires user confirmation for
  important actions.
* **Explainability (XAI).** `xai/shap_explainer.py` uses SHAP `TreeExplainer` on a fraud
  classifier to produce per-feature contributions to the fraud class (new device, unusual
  location, amount), sorted by absolute impact, each tagged "increases" or "decreases fraud
  risk". The LLM then translates this evidence into plain language.
* **Evidence-first responses.** `/assistant/chat` returns the answer, which tools ran, the
  raw tool evidence and the responsible-AI report, so each claim can be traced to a tool output.
* **Responsible-AI checks** run on every response (see 5.7): privacy, financial-safety,
  fairness, transparency and user control.
* **Scope boundary with Component 1.** Component 1's agent decides *which* liquidation strategy
  to execute and returns a structured decision with a short internal trace. Component 4
  explains *why*, to the end user, across the whole platform. Component 1 produces no
  user-facing prose, to avoid overlapping scope.
* **Planned.** Trust Panel confirmation and rollback workflows, consent and privacy settings,
  and the human-participant trust study (the only element needing ethical clearance).

---

### 5.6 Integration methodology

All four run under one `docker compose up`, behind one login, each with its own screen. The
integration touched as little as possible.

| Component | What it needed to join |
|---|---|
| C1 Portfolio | nothing: CORS, JWT and `/health` were designed in |
| C2 Fraud | CORS (from `ALLOWED_ORIGINS`) and a read-only `GET /v1/transactions/{id}` so Component 4 can read the enforced decision |
| C3 Audit | nothing: already had CORS, `/api/health`, and read `PORT` from the environment |
| C4 Assistant | CORS middleware and a `/health` endpoint: 39 added lines, 0 removed |

No component's logic, schema or threshold was altered, and no build file was added to anyone
else's directory: C2 and C3 run in compose from stock `python:3.12-slim` and `node:20-slim`
images with their source bind-mounted. The two things a service genuinely needs are CORS
(read from `ALLOWED_ORIGINS`) and a health endpoint that does not touch the expensive path;
Component 4's returns a literal so a healthcheck never calls an LLM.

Adding a service: copy `service/cors.py` and `service/auth.py` from Component 1, share
`JWT_SECRET`, register the base URL in `frontend/lib/api/client.ts`, write a typed client in
`frontend/lib/api/`, replace the page under `frontend/app/(platform)/<name>/`, and set
`ready: true` in `NAV`. Full steps in [`INTEGRATION.md`](INTEGRATION.md).

**Verification of failure isolation.** `frontend/e2e/components.spec.ts` visits all three
teammate screens with their services down, asserts each renders an explanatory state, and then
plans a withdrawal successfully.

Component 4's agent uses Ollama, so compose sets
`OLLAMA_HOST=http://host.docker.internal:11434` with a matching `extra_hosts` entry, because
`localhost` inside a container is the container.

---

### 5.7 Responsible-AI framework

| Principle | How it is implemented |
|---|---|
| **Explainability** | Fuzzy rule traces (C1), ranked plain-English fraud reasons (C2), SHAP attributions and tool evidence (C4) |
| **Auditability** | Append-only fraud audit log (C2), SHA-256 commitments on a registry contract with model and policy versions (C3), `model_version` on every event (C1) |
| **Human oversight** | HUMAN_REVIEW path for low-confidence high-risk decisions (C3), STEP-UP challenges (C2), confirmation required for financial actions (C4) |
| **Determinism where it matters** | Enforcement (C2 gateway, C3 policy engine) is rule-based and reproducible; ML only supplies scores |
| **Grounding** | The optimization agent may not state amounts the optimizer did not produce (C1) |
| **Privacy** | Only hashes on-chain (C3); raw social text never stored (C1); sensitive-term check on assistant output (C4); local LLM (C4) |
| **Safety** | Misleading-claim check (e.g. "guaranteed return") on assistant output (C4) |
| **Fairness** | Prototype hook in place (C4); real metrics are reserved for the evaluation phase |
| **Honesty about models** | Simulated streams are labelled in every response (C2); backends that produced results are recorded (C1) |

---

## 6. Technology stack

| Layer | Technology |
|---|---|
| Language runtimes | Python 3.12, Node 20, TypeScript |
| Backend frameworks | FastAPI, Uvicorn, Pydantic v2, Express |
| Forecasting | PyTorch (CPU wheel by default), TimesFM 2.5, Chronos-Bolt, PEFT/LoRA, HuggingFace Transformers |
| Optimization | pymoo (MOEA/D), scikit-fuzzy (Mamdani/Sugeno), DEAP, scipy, statsmodels |
| Features | pandas, pandas-ta-openbb, FinBERT, Ollama, pypdf |
| Data sources | Colombo Stock Exchange, Binance, Yahoo Finance (yfinance), GDELT, Google News, RSS, NewsAPI, CoinGecko |
| Messaging and tracking | Kafka (KRaft, `apache/kafka:3.7.2`), MLflow 2.16, Prometheus |
| Agent | LangGraph, LangChain, Ollama, SHAP, scikit-learn |
| Blockchain | Solidity ^0.8.20, Hardhat |
| Persistence | SQLite via SQLAlchemy (platform), SQLite (model registry), parquet caches |
| Auth | JWT (PyJWT, single pinned algorithm), Argon2id |
| Frontend | Next.js 16.3, React 19, Tailwind v4, openapi-typescript |
| Testing | pytest, Jest, Playwright |
| Packaging | uv, Docker Compose |

---

## 7. Repository layout

```
backend/
  Platform/                 shared auth + portfolio service (auth/, store/, service/)
  Portfolio-Optimization/   Component 1
    data/                     ingestion, exchanges/, fx, history, window selection, active set
    features/                 technical, sentiment, liquidity, enrichment, feature store
    news/                     keyword profiles, channels/, relevance, scoring, aggregation
    fundamentals/             report fetch, statement extraction, ratios, red flags
    forecasting/              baseline LSTM, TimesFM/Chronos adapters, LoRA, hybrid, registry
    optimization/             MOEA/D, Markowitz, fuzzy, GA, naive baselines, stress scenarios
    decision/                 Laya two-stage decision model
    agent/                    tools, grounding validator, scripted policy, SFT trajectories
    evaluation/               walk-forward backtest, metrics, stress runner, accuracy checks
    experiments/              RQ scripts, Colab notebooks
    service/                  FastAPI app, contracts (frozen), market routes, CORS, JWT
    configs/                  universe, news channels, keyword overrides
    docker/                   Dockerfile, Prometheus config
    tests/
  Fraud-Detection/          Component 2 (src/fraud_engine: models, gateway, api, evaluation; experiments)
  Blockchain-Auditability/  Component 3 (src/ Express API: policy, hashing, ledger adapter; tests/;
                              blockchain/ AuditRegistry.sol + Hardhat; scripts/). UI: frontend/app/(platform)/audit
  Agentic-Assistance/       Component 4 (agent, tools, xai, responsible_ai)
frontend/                   Next.js 16 web app (app/, components/, lib/api/, e2e/)
docs/                       TAF and research documents
docker-compose.yml          full stack
INTEGRATION.md              how components connect and the security posture
```

---

## 8. Service and API reference

All Component 1 and platform routes accept a bearer token. Interactive docs: `/docs` on each
FastAPI service.

### Platform (:8100)

| Method | Path | Purpose |
|---|---|---|
| POST | `/auth/register`, `/auth/login` | Create account / exchange credentials for a token |
| GET | `/auth/me` | Current user |
| GET, POST | `/portfolios` | List / create |
| GET, PUT, DELETE | `/portfolios/{id}` | Read / update / delete |
| POST | `/portfolios/{id}/executions` | Execute a Component 1 plan through Components 2 and 3 |
| GET | `/executions`, `/executions/{id}` | List / full cross-component trace |
| POST | `/executions/{id}/confirm` | Step-up: re-enter password to approve or reject a pending execution |
| POST | `/executions/{id}/retry` | Resume a HELD execution once Component 2 or 3 is back |
| GET | `/health` | Liveness and database check |

### Component 1 (:8000)

| Method | Path | Purpose |
|---|---|---|
| POST | `/portfolio/optimize` | MOEA/D allocation with a selection rule, plus the `rebalance_orders` that realise it |
| POST | `/portfolio/withdraw` | Fuzzy-GA liquidation plan (optionally agent-driven) |
| POST | `/forecast` | Quantile forecasts (503 until a head is registered) |
| GET | `/market/search?q=` | Search across CSE, Binance, Yahoo |
| POST | `/market/quote` | Price, traded value, volatility in native currency and LKR |
| POST | `/analysis/track`, `/analysis/refresh` | Register a portfolio's symbols / re-run preparation |
| GET | `/analysis/status` | Per-step preparation state: `preparing`, `ready`, `partial`, `failed` |
| GET | `/analysis/channels` | Every news channel and why any cannot run |
| GET, PUT | `/analysis/keywords/{symbol}` | Keyword profile; edits mark it reviewed |
| GET | `/analysis/sentiment/{symbol}` | Per-tier sentiment, attention spike, top sources |
| GET | `/analysis/fundamentals/{symbol}` | Statements, ratios, F/Z scores, red flags, outlook tone |
| GET | `/analysis/technical/{symbol}` | Trend, oscillators, support/resistance, relative strength |
| GET | `/analysis/forecast/{symbol}` | Forecast band and which method produced it |
| GET | `/health` | Liveness and usable model report |

### Component 2 (:8001)

`GET /health`, `POST /v1/transactions/score`, `GET /v1/transactions/{id}` (read back a
decision), `GET /v1/decisions/summary`, `POST /v1/feedback`,
`POST /v1/admin/apply-feedback-calibration`, `GET /v1/dashboard`.

### Component 3 (:8002)

`GET /api/health`, `POST /api/audit/evaluate` (optional `upstreamDecision` and `planHash`),
`POST /api/audit/review`,
`GET /api/audit/records`, `GET /api/audit/stats`, `GET /api/audit/:transactionId/verify`,
`POST /api/audit/:transactionId/tamper`, `POST /api/audit/reset`.

### Component 4 (:8003)

`GET /health`, `GET /`, `POST /assistant/chat` returning `answer`, `agent_execution`,
`evidence` and `responsible_ai`.

---

## 9. Frontend

Next.js 16 (App Router, Turbopack), React 19, Tailwind v4 (CSS-first, no config file).

| Route | Purpose |
|---|---|
| `/login`, `/register` | Authentication |
| `/portfolio` | Holdings CRUD, instrument search |
| `/withdraw` | Component 1 headline screen: plan, shortfall, rule trace, baseline comparison |
| `/optimize` | MOEA/D allocation and Pareto selection |
| `/forecast` | Quantile forecast bands |
| `/analysis` | Per-symbol sentiment, fundamentals, technicals, keyword review |
| `/fraud` | Component 2 scoring and attack simulator |
| `/audit` | Component 3 evaluation, records, verification and tamper demo |
| `/assistant` | Component 4 chat, explanation, trust panel, responsible-AI screens |

Notes: every API call passes `cache: "no-store"`; async request APIs are awaited; anything
using `useSearchParams` sits in a `<Suspense>` boundary. Read
[`frontend/README.md`](frontend/README.md) and `frontend/AGENTS.md` before editing, as this
Next.js version differs from older guides.

---

## 10. Configuration

| Variable | Used by | Meaning |
|---|---|---|
| `JWT_SECRET` | platform, C1 | Shared signing secret, 32+ characters, must not be the placeholder |
| `ALLOWED_ORIGINS` | all FastAPI services | Comma-separated browser origins; never `*` for authenticated services |
| `AUTH_REQUIRED` | C1 | `true` in compose; defaults off so the test suite runs unauthenticated |
| `DATABASE_URL` | platform | SQLite path, persisted in a volume |
| `KAFKA_BOOTSTRAP_SERVERS`, `KAFKA_ENABLED` | C1 | Event publishing; degrades to a no-op producer |
| `MLFLOW_TRACKING_URI` | C1 | `sqlite:///artifacts/mlflow.db` |
| `NEWSAPI_KEY`, `JINA_API_KEY` | C1 | Optional news channels |
| `FX_SPREAD_FIAT`, `FX_SPREAD_STABLECOIN` | C1 | Conversion spread charged by the GA |
| `OLLAMA_HOST` | C4 | Where the local model runs |
| `NEXT_PUBLIC_*_URL` | frontend | Backend URLs the *browser* uses, baked in at build time |

---

## 11. Testing and verification

| Suite | Count | Run |
|---|---|---|
| Component 1 (pytest) | 636 collected | `cd backend/Portfolio-Optimization && uv run pytest -q` |
| Platform (pytest) | 64 | `cd backend/Platform && uv run pytest -q` |
| Component 2 (pytest) | 33 | `cd backend/Fraud-Detection && pytest -q` |
| Component 3 backend (Jest) | 34 | `cd backend/Blockchain-Auditability && npm test` |
| Component 4 (pytest) | 107 | `cd backend/Agentic-Assistance && pytest -q tests` |
| Frontend unit (Jest) | 53 | `cd frontend && npm test` |
| Frontend end-to-end (Playwright) | | `npm run test:e2e` |
| Contract (Hardhat) | 12 | `cd backend/Blockchain-Auditability/blockchain && npx hardhat test` |

* Playwright runs against the **real** Platform and Component 1 services, which it starts
  itself; it does not use mocks.
* `tests/test_env.py` is the dependency gate. Run it first after installing.
* Regenerate the capability report after changing optional packages:
  `uv run python -m forecasting.env_report`. Without it a fresh checkout sees no foundation
  models, however the install went.
* Security tests assert what the API does *not* reveal: identical responses for unknown email
  and wrong password, 404 for another user's data, rejection of `alg: none` and foreign-secret
  tokens, and no response ever containing a password or hash.
* Counts are from test collection and drift as tests are added; re-run before quoting them.

---

## 12. Results so far

Full tables and analysis are in the [Component 1 README](backend/Portfolio-Optimization/README.md).

| RQ | Result |
|---|---|
| RQ1 | Baseline LSTM, walk-forward over 22 symbols and 4 non-overlapping folds (11,726 out-of-sample predictions): MAE 0.0288, RMSE 0.0386, pinball loss **0.009388**. All three quantiles sit below nominal coverage (p10 0.0695, p50 0.4513, p90 0.8598), a uniform level bias from trailing-window training through a rising market, which makes CVaR conservative. Foundation and hybrid rows await Colab. |
| RQ2 | MOEA/D matches Markowitz's expected return (0.000678) at **3.5x lower liquidity cost**, by leaving the illiquid position almost untouched because cost is charged on the trade, not the holding. |
| RQ3 | Fuzzy GA beats pro-rata by **96.5%**, largest-first by **26.2%** and most-liquid-first by **8.7%** in mean relative reduction of realized loss. A later indicative run with the forecast pathway shows the GA feasible as often as the best baselines after the penalty-scaling fix; the real ablation needs the trained hybrid. |
| RQ4 | Best unstressed cost but degrades faster than most-liquid-first, traced to fuzzy-layer saturation (identical priority 83.9 at severity 1.0). The scoring was fixed by log-scaling liquidity; the rule-base half is left open as a methodology decision. |

Other measured outcomes: per-symbol window selection resolved 26 of 26 symbols; 767 of 767
SFT trajectories grounded; fraud-engine adversarial results in 5.3; Component 4 round trip of
roughly 82 seconds on CPU.

---

## 13. Security posture

Stated plainly because each is a deliberate trade-off.

* **The access token is readable by JavaScript.** Direct browser-to-backend calls mean the
  token sits in browser storage, so an XSS would expose it. A backend-for-frontend proxy would
  have permitted an httpOnly cookie. This belongs in the dissertation as a stated choice.
* **Origins are explicit, never `*`** on authenticated services. Components 3 and 4 currently
  set a wildcard origin in their own code; they carry no platform tokens today, but this should
  be tightened before they do.
* **Argon2id passwords**, not bcrypt (no silent 72-byte truncation).
* **Login does not leak account existence.** Unknown email and wrong password return the same
  status and body.
* **Cross-user access returns 404, not 403**, so ids cannot be enumerated.
* **JWT decoding pins one algorithm**, preventing `alg: none` and RS256-to-HS256 confusion.
* **The platform refuses to start with a weak or placeholder secret.**
* **Fraud and audit components are demonstrators:** single process, in-memory state, no
  authentication of their own.

---

## 14. Known limitations and honest caveats

1. **Component 2 runs its untrained heuristic unless a bundle is present.** The LSTM
   autoencoder and GNN are real and benchmarked offline (`experiments/results/`), but trained
   weights are git-ignored, so a fresh checkout and the Docker image score with
   `heuristic-untrained` until `scripts/train_benchmark.py` writes `artifacts/fraud_bundle.joblib`.
   Every response says which one answered.
2. **Component 3 anchors hashes, not zero-knowledge proofs.** It uses a SHA-256 commitment
   scheme; Merkle trees and ZKPs are not implemented. Without a Hardhat node it uses an
   in-memory mock ledger.
3. **Kafka is not on the execution path.** Component 1 publishes `portfolio.decisions` and
   Component 2 can consume `finance.transactions.raw`, but plans move through Components 2 and
   3 by synchronous HTTP from the platform service (see *Cross-component data flow*).
   STEP_UP is satisfied by password re-entry; an OTP or device biometric is not built.
4. **Component 4's SHAP model is a prototype.** For executions and live Component 2
   decisions it reports the components' own evidence; the Random Forest trained on a dozen
   mock rows only explains its mock fallback records, the fairness check is a placeholder, and the trust study, Trust Panel
   confirmation/rollback and consent flows are not built.
5. **Component 4 model alias on the development machine.** `llama3.2:3b` is an alias for
   `gemma4-e4b` there; do not attribute outputs to Llama 3.2.
6. **No trained foundation forecaster yet.** TimesFM and Chronos weights (~821 MB each) and
   LoRA fine-tuning belong on Colab; `/forecast` returns 503 until a head is installed, and
   the analysis pipeline uses a labelled empirical band meanwhile.
7. **RQ1 evidence is limited.** Few walk-forward folds support "the hybrid beats the baseline
   here", not a significance claim.
8. **RQ4 is an open finding**, not a solved one (see 12).
9. **Extraction accuracy is checked on very few reports** (2 hand-checked; target ~30), and
   scanned PDFs are unsupported.
10. **Laya inference with real weights is unverified** on the development machine.
11. **Full `docker compose up` is not yet verified end to end**; services were verified as
    direct processes with the same ports and endpoints.
12. **Balances are simulated**, per the TAF, not real brokerage holdings.
13. **Not financial advice.** This is a research prototype.

### Open items for supervisor sign-off

1. CVaR is not named in the TAF; it is a deliberate strengthening.
2. Forex is outside the TAF's Component 1 dataset line ("liquid equities and/or ETFs"); adding
   it is a superset.
3. RQ1-RQ5 do not appear in the written TAF.
4. If TimesFM fails its gate, RQ1 runs on Chronos-Bolt alone, a deviation from a TAF-named
   method.
5. The TAF cites a "Section 13" and an "RQ4" that do not exist in the twelve-page document.
6. Whether to differentiate the (HIGH urgency, TURBULENT) fuzzy rule row for RQ4.

---

## 15. Behaviours that look like bugs and are not

| Observation | Explanation |
|---|---|
| `POST /forecast` returns **503** | No trained forecaster is registered. The UI renders an explanatory empty state. |
| A withdrawal returns **200** with `feasible: false` | A real answer: the participation cap limits what can be raised by the deadline. |
| `model_version` reads `unregistered` | No checkpoint registered yet; the withdrawal path does not need one. |
| `/portfolio/optimize` takes ~7 s | MOEA/D runs 100 generations over 45 reference directions; most time is pymoo's operators. |
| Kafka warnings with no broker | The producer is fire-and-forget and degrades to a no-op. |
| Assistant takes over a minute | Local model on CPU, ~82 s per round trip. |
| A foundation model reported "not installed" though it is | The capability report is missing or stale; run `python -m forecasting.env_report`. |
| yfinance returns an empty frame | Rate limiting looks like a delisting; the fetch retries and does not cache empties. |

---

## 16. Documents and further reading

| Document | Content |
|---|---|
| [`docs/TAF-J26-SE-325.pdf`](docs/TAF-J26-SE-325.pdf) | The approved Topic Assessment Form (a scan with no text layer) |
| [`INTEGRATION.md`](INTEGRATION.md) | How components connect, adding a service, security posture |
| [`backend/README.md`](backend/README.md) | Backend conventions and service table |
| [`backend/Portfolio-Optimization/README.md`](backend/Portfolio-Optimization/README.md) | Component 1 setup, deviations, findings, risks |
| [`backend/Fraud-Detection/README.md`](backend/Fraud-Detection/README.md) | Component 2 pipeline, gateway rules, adversarial results |
| [`backend/Blockchain-Auditability/README.md`](backend/Blockchain-Auditability/README.md) | Component 3 anchoring inputs |
| [`backend/Agentic-Assistance/README.md`](backend/Agentic-Assistance/README.md) | Component 4 scope and inputs |
| [`backend/Platform/README.md`](backend/Platform/README.md) | Auth and portfolio service design |
| [`frontend/README.md`](frontend/README.md) | Frontend structure, generated types, Next.js 16 notes |
