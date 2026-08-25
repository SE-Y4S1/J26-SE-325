# Component 2 — Fraud Detection

**Owner:** Dushanthini R. · **Port:** 8001 · **Status:** not started

> **Sub-objective (TAF):** Develop a real-time fraud and behavioral anomaly detection engine
> tightly coupled with a deterministic security enforcement gateway, robust to adversarial
> evasion.

Tasks listed against this component in the TAF:

- Streaming feature ingestion and dynamic transaction graph construction
- Hybrid LSTM autoencoder + GNN dual-stream anomaly classifier
- Behavioral biometrics and device fingerprinting
- Security gateway: rate limiting, IP reputation scoring, step-up authentication
- Concept-drift monitoring and adversarial robustness testing (camouflage, slow-drift, structuring)
- Analyst feedback dashboard and online learning pipeline

## Wiring this up

The frontend shell, auth and API client already exist, so joining the platform does not
require changing anything outside this directory. Five steps, detailed in
[`../../INTEGRATION.md`](../../INTEGRATION.md):

1. Copy `service/cors.py` and `service/auth.py` from
   [`../Portfolio-Optimization/service`](../Portfolio-Optimization/service) — unchanged.
2. Share the repo-root `JWT_SECRET`; tokens are issued by `backend/Platform` and only
   verified here.
3. Add your base URL to `SERVICES` in `frontend/lib/api/client.ts`.
4. Generate TypeScript types from your OpenAPI (`npm run gen:api`) rather than writing them.
5. Replace `frontend/app/(platform)/fraud/page.tsx` and set `ready: true` in `NAV`.

## What is already available to you

- **Portfolios and identity** from the shared platform service on 8100 — no dependency on
  Component 1.
- **Decision events** on the Kafka topic `portfolio.decisions`, each carrying the request,
  response and a top-level `model_version`.
