# Component 03 — AI Decision Assurance Engine with a Tamper-Evident Evidence Ledger

**Owner:** Abeysekara W.C.S.M · **Service port:** 8002 · **UI:** shared Nexa app, `frontend/app/(platform)/audit`

> **Research question.** Can the reliability of an AI decision be assessed from several signals —
> model confidence, disagreement between models, and contextual risk — to decide whether it
> should be processed automatically, verified further, or reviewed by a human, so that fewer
> incorrect decisions are automated at an acceptable human-review workload?

The contribution is the **assurance engine** (routing). The blockchain supports it: each decision,
its assurance result and the exact policy that produced it are committed as tamper-evident
evidence, with sensitive data kept off-chain.

---

## 1. Routes

| Route | What happens | Counted in the evaluation as |
|---|---|---|
| **AUTO** | The AI decision is executed and anchored as final. | An automatic decision (incorrect if it disagrees with the true label). |
| **VERIFY** | Held for step-up verification by the account holder (the platform's re-authentication flow). | Customer friction (verify rate), not analyst workload. How often VERIFY catches an error is an **assumption**, not a measurement. |
| **HUMAN_REVIEW** | Held for an authorised analyst who is not the subject of the decision. | Human-review workload. |

## 2. The engine (`src/services/assuranceEngine.js`)

Each signal is expressed as an uncertainty in [0, 1]:

| Signal | Computed from |
|---|---|
| Confidence `u_c` | Distance of the decision score from the decision threshold. Derived from the score, not from a caller-asserted confidence, so a caller cannot buy automation by claiming certainty. |
| Disagreement `u_d` | Spread (max − min) between independent model scores, e.g. Component 2's LSTM and GNN streams. |
| Context `u_r` | Weighted: amount (log scale), Component 2 rule hits, model validation status, transaction type. |

```
assuranceScore = 1 − (w_c·u_c + w_d·u_d + w_r·u_r)
AUTO if score ≥ autoMin · VERIFY if score ≥ verifyMin · otherwise HUMAN_REVIEW
```

**Vetoes** force HUMAN_REVIEW and are recorded: models disagreeing beyond `maxDisagreement`, and
an amount at or above `reviewAmount`.

**Conservative degradation.** The validator and the property tests enforce these:
- A missing signal is scored at its worst non-vetoed value (`missing.disagreement ≥ vetoes.maxDisagreement` is checked when the policy loads).
- An unvalidated model (`benchmarkModel: false`, or a heuristic/unregistered version) has its confidence uncertainty floored.
- Any DEGRADED assessment is capped below AUTO.

Property tests (2,000 random inputs each) check that a DEGRADED assessment is never AUTO, and
that removing a signal never makes a route more automated than any non-vetoed observation of it.

**Upstream decisions** never relax the outcome. A Component 2 `STEP_UP` holds an automatic
APPROVE for VERIFY. A `BLOCK` is an enforced rejection.

## 3. Policies (`src/policies/`)

Policies are data. Each document is hashed and that hash is registered on-chain, so every
decision commits to the exact rules that routed it.

| Policy | Engine | Hash scheme | Role |
|---|---|---|---|
| P001 v1.0 | threshold router (risk ≥ 80 and confidence ≥ 70 → REJECT; risk ≥ 80 and confidence < 70 → HUMAN_REVIEW; otherwise APPROVE) | v1 (legacy) | Baseline; used for requests that send no assurance signals |
| P002 v1.0 | assurance engine | v2 | Routes requests that send signals. **Status: untuned** (hand-set, conservative parameters until `research/` tunes them on IEEE-CIS) |

- **Hash v2** is SHA-256 over canonical JSON, with keys sorted at every depth, so every weight, threshold and veto is committed.
- **Hash v1** is kept only because P001's registration and the existing records use it. It hashed the nested `thresholds` object as `{}`, so it does not commit to the thresholds (`docs/BASELINE.md`).

`GET /api/audit/policies` returns each document with its hash and on-chain status, so anyone can
recompute it.

## 4. Evidence ledger (record v2)

```
decision ──► commitment #1  auditId       = sha256(txId:salt)         anchored immediately, also for held decisions
review   ──► commitment #2  reviewAuditId = sha256(txId:salt:review)  payload links to #1 via parentRecordHash
```

- **What goes on-chain:** only `bytes32` hashes (`auditId`, `recordHash`, `policyHash`). Amounts, scores and decisions stay off-chain.
- **What the decision commitment covers:**
  - the decision and the AI's inputs;
  - the full assurance result (route, score, components, signal quality, vetoes);
  - the policy hash and Component 2/1 provenance;
  - a keyed subject reference.
- **Verification statuses:** `VERIFIED_ON_CHAIN`, `VERIFIED_SIMULATED`, `MODIFIED`, `CHAIN_BROKEN` (the review does not link to the decision), `ON_CHAIN_MISSING`, `NOT_FOUND`, `BLOCKCHAIN_UNAVAILABLE`.
- **Legacy records:** v1 records (anchored before v2) still verify with the v1 rules.

**Independent verification.** `GET /api/audit/:tx/evidence` returns a bundle: the committed
payloads, the salt, the policy document and the anchors.
`node scripts/verify-evidence.js bundle.json --rpc <url>` recomputes every hash and reads the
contract directly, without the service.

## 5. Access control and privacy

- **Tokens.** The service verifies the platform's tokens exactly as Component 1 does: HS256 only, issuer `j26-se-325-platform`, and `exp`, `iat` and `sub` required. `AUTH_REQUIRED` defaults to off, which keeps existing callers working. **Turn it on in production.**
- **Roles.** `AUDIT_REVIEWER_IDS` lists the analysts and auditors. The platform service (`sub: service:platform`) can anchor decisions and resolve VERIFY. Only analysts can resolve HUMAN_REVIEW.
- **Separation of duties.** A reviewer cannot review a decision about themselves. This is checked with keyed references, HMAC-SHA256 under a key derived from `JWT_SECRET`. The unkeyed hash of a user id that it replaces could be reversed by enumeration.
- **Evidence bundles** are restricted, sent with `Cache-Control: no-store`, logged per export, and contain no name, email, account key or user id.
- **Public verification** returns statuses and hashes only when the caller may not see record contents.

## 6. API

| Method | Endpoint | Who (when `AUTH_REQUIRED=true`) |
|---|---|---|
| POST | `/api/audit/evaluate` | platform service, analyst |
| POST | `/api/audit/review` | platform service (VERIFY only), analyst |
| GET | `/api/audit/queue` | analyst |
| GET | `/api/audit/records`, `/api/audit/records/:tx`, `/api/audit/stats` | analyst |
| GET | `/api/audit/:tx/evidence` | analyst |
| POST | `/api/audit/:tx/retry` | platform service, analyst |
| GET | `/api/audit/policies`, `/api/audit/:tx/verify`, `/api/audit/by-audit-id/:id/verify` | public (contents redacted) |
| GET | `/api/audit/research/results` | public (aggregate evaluation results, or `NOT_RUN`) |
| POST | `/api/audit/:tx/tamper`, `/api/audit/reset`; DELETE `/api/audit/records` | analyst; demo only (`ALLOW_DEMO_TAMPER` in production) |
| GET | `/api/health` | public |

**Backward compatibility:**
- **Requests without `signals`:** their outcomes are unchanged.
- **Response fields:** every existing field keeps its name, type and value. This is checked against responses captured before the change (`tests/fixtures/api-contract-v1.json`).
- **VERIFY status:** VERIFY keeps the status `HUMAN_REVIEW_REQUIRED`, so unmodified callers hold it. The route is in the new `assurance` object.

## 7. Running

```bash
# Ledger (terminal 1)
cd backend/Blockchain-Auditability/blockchain && npx hardhat node
# Deploy and register every policy (terminal 2)
npx hardhat run scripts/deploy.js --network localhost
#   or, for an existing deployment after adding/tuning a policy:
npx hardhat run scripts/register-policies.js --network localhost
# Service
cd backend/Blockchain-Auditability && npm install && npm start      # http://localhost:8002
```

`BLOCKCHAIN_MODE=mock` runs without a chain. All settings are listed in `.env.example`.

**Database migration.** On first start, an existing v1 SQLite file is migrated in place:
- the change is additive, adding nullable columns only;
- a backup is written to `audits.sqlite.bak-v1-<timestamp>` first;
- `node scripts/db-rollback.js` restores it.

The Postgres store keeps records as JSONB and needs no migration.

## 8. Tests and measurements

| Suite | Command | Result (2026-10-21) |
|---|---|---|
| Service (Jest, 13 suites, incl. 400 Python↔JS golden vectors and missing/garbage-input cases) | `npm test` | 175 passed |
| Contract (Hardhat) | `cd blockchain && npx hardhat test` | 16 passing |
| Research pipeline (pytest, code tests on a synthetic fixture) | `cd research && uv run pytest -q` | 23 passed |

The IEEE-CIS evaluation itself (`research/`, see its README and METHODS.md) has **not been run
yet**: it needs the Kaggle data. Until then the Audit → Evaluation page says "Not run yet".

Gas on the Hardhat network (solc 0.8.20, optimizer 200 runs), from `blockchain/test/gas.test.js`:

| Operation | Gas |
|---|---|
| Deploy `AuditRegistry` | 582,076 |
| `registerPolicy` | 47,901 |
| `recordAudit` (one commitment) | 119,023 |
| A reviewed decision (two commitments) | 238,046 |

## 9. Limitations (stated, not hidden)

- **P002 is untuned.** No claim about its error/workload trade-off is made until the IEEE-CIS evaluation in `research/` has run on its held-out test split.
- **VERIFY effectiveness** (how often step-up catches an incorrect decision) is assumed in the evaluation, not measured.
- **Offline vs online signals differ.** Offline evaluation and online deployment use different models (an offline ensemble vs Component 2's LSTM/GNN) and different context signals. The evaluated policy is the enforced policy (same formula, weights and thresholds), but online performance is not inferred from offline results.
- **The local ledger is not persistent.** A local Hardhat node keeps its chain in memory. Persistent anchoring needs a public testnet (`DEPLOY.md`).
- **Defaults are permissive.** `AUTH_REQUIRED` and `ALLOWED_ORIGINS` keep permissive defaults for backward compatibility. Production must set them.
