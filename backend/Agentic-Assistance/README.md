# Component 4 — Explainable Agentic Assistance

**Owner:** W.V.A.D.K. Chamara · **Port:** 8003 · **Status:** not started

> **Sub-objective (TAF):** Build a localized, explainable agentic LLM assistant, empirically
> evaluated for its effect on user trust and perceived control, within a responsible-AI
> governance framework.

Tasks listed against this component in the TAF:

- LangGraph-orchestrated agentic assistant backend
- Natural-language explainability layer using SHAP/LIME
- Responsible-AI guardrails (TRiSM, fairness metrics, bias mitigation)
- Trust Panel with confirmation and rollback workflows
- Consent flows and privacy settings
- User trust evaluation study

This is the only component requiring **ethical clearance**, because of the human-participant
trust study.

## Scope boundary with Component 1

Both components involve an agent, and the split is deliberate:

| | Component 1's agent | Component 4's agent |
|---|---|---|
| Decides | **which** liquidation strategy to execute | — |
| Explains | — | **why**, to the end user |
| Scope | this component's data only | the whole platform |
| Output | structured decision + short internal trace | natural-language explanation |

Component 1 deliberately produces **no user-facing prose** — duplicating that would create
scope overlap that is hard to defend to a supervisor. The explanation layer is yours.

## What Component 1 already provides to explain

Every withdrawal response carries two traces:

- **`fuzzy_rule_trace`** — which fuzzy rules fired and at what strength, per holding. This is
  the deterministic, auditable derivation behind every number in the plan, and the frontend
  already renders it raw on the Withdraw screen. Turning it into readable prose is exactly
  the gap you fill.
- **`agent_reasoning_trace`** — the tool-call sequence, when the request set `use_agent`.

Both arrive on the Kafka topic `portfolio.decisions` as well as in the HTTP response.

One constraint worth knowing: Component 1's agent may never state a sell amount that did not
come from its optimizer, and this is enforced in code rather than by convention — a decision
is rejected unless every figure matches the tool output exactly. Any explanation you generate
is therefore describing numbers with a verifiable derivation behind them.

## Wiring this up

Five steps, in [`../../INTEGRATION.md`](../../INTEGRATION.md): copy `service/cors.py` and
`service/auth.py` from Component 1, share `JWT_SECRET`, register your base URL in
`frontend/lib/api/client.ts`, generate types from your OpenAPI, and replace
`frontend/app/(platform)/assistant/page.tsx`.
