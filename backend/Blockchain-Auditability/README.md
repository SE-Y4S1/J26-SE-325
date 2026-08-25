# Component 3 — Blockchain Auditability

**Owner:** Abeysekara W.C.S.M · **Port:** 8002 · **Status:** not started

> **Sub-objective (TAF):** Design a privacy-preserving, tamper-evident blockchain audit layer
> that bridges continuous AI decisions to deterministic on-chain smart contract enforcement.

Tasks listed against this component in the TAF:

- Smart contracts for vault token management and event logging
- The AI-to-smart-contract bridge, translating AI risk/liquidity outputs into on-chain logic
- Privacy-preserving anchoring using ZKPs and/or Merkle tree commitments
- Model-versioning and provenance tracking for logged AI explanations

## What Component 1 already provides for anchoring

The TAF says this component's bridge consumes *"AI risk/liquidity outputs"* — that is
Component 1's, and it is already emitting them.

**Decision events** land on the Kafka topic `portfolio.decisions`:

```json
{
  "event_id": "...", "event_type": "portfolio.withdraw",
  "schema_version": "1.0", "component": "component1_portfolio_optimization",
  "occurred_at": "...", "model_version": "...",
  "request": { ... }, "response": { ... }
}
```

`model_version` is **top-level**, so anchoring does not require parsing the payload.

**Model provenance** is resolvable from Component 1's registry
(`forecasting/model_registry.py`). `export_for_anchoring(model_version)` returns the minimal
on-chain bundle:

```json
{
  "model_version": "...", "content_hash": "sha256 of the checkpoint bytes",
  "data_fingerprint": "hash of the resolved universe + feature schema",
  "git_commit": "...", "created_at": "..."
}
```

The content hash is deterministic over a directory (sorted, with relative paths folded in),
so re-hashing the same adapter always yields the same value — an unstable hash would make an
on-chain anchor worthless. `data_fingerprint` is separate on purpose: a data change is then
visible even when the code and weights did not change.

## Wiring this up

Five steps, in [`../../INTEGRATION.md`](../../INTEGRATION.md): copy `service/cors.py` and
`service/auth.py` from Component 1, share `JWT_SECRET`, register your base URL in
`frontend/lib/api/client.ts`, generate types from your OpenAPI, and replace
`frontend/app/(platform)/audit/page.tsx`.
