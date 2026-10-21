# Baseline before the assurance-engine work (2026-10-20)

Recorded on the `Chenuli` branch before any Component 3 source change, so later results can be
compared against it. Every number is the output of a run on this machine; nothing is copied from
a README.

Environment: Windows 11, Node 24.14.1, npm 11.11.0, Python 3.12.15 (installed with uv 0.13.3).
Docker Desktop was installed but not running, so services were run directly.

| Suite | Command | Result |
|---|---|---|
| C3 backend (Jest) | `npm test` | 34 / 34 passed (4 suites). The Postgres suite only ran after `npm install` added the already-declared `pg-mem`; before that it failed to load. |
| C3 contract (Hardhat) | `cd blockchain && npx hardhat test` | 12 / 12 passing |
| Platform (pytest) | `uv sync && uv run pytest -q` | 70 passed |
| C1 Portfolio-Optimization (pytest) | `uv pip install -r requirements.txt` + CPU torch | 605 passed, 7 skipped. `PyJWT` had to be installed into the venv by hand: `service/auth.py` imports it but `requirements.txt` does not list it, and collection fails without it. |
| C2 Fraud-Detection (pytest) | `pip install -r requirements-dev.txt` | 34 passed |
| C4 Agentic-Assistance (pytest) | `pip install -r requirements.txt pytest httpx` | 256 passed, 6 skipped, 1 failed: `test_missing_transaction_not_treated_as_fraud` expects the `RECORD_NOT_FOUND` wording, but with Component 2 not running it reports `SERVICE_UNAVAILABLE`. Environmental; not touched. |
| Frontend typecheck | `npx tsc --noEmit` | clean |
| Frontend unit (Jest) | `npx jest` | 105 / 105 passed (13 suites) |
| Frontend lint | `npm run lint` | 7 errors, 19 warnings, all in Component 4 files (`assistant/page.tsx`, `ExplanationScreen.tsx`, `responsibleAIPreferences.test.tsx`) |

## Live ledger state at baseline

A local Hardhat node (chain 31337) was running with `AuditRegistry` at
`0x5FbDB2315678afecb367f032d93F642f64180aa3` and P001's hash
`0x6011145b5a10bf85f8b82c8daf9aec9b0a1c0590bba7d2b8d5663e67055c6096` registered. The off-chain
store held 3 records; re-verified against the chain after the layout change:

| Transaction | Status |
|---|---|
| TX-DEMO-001-1661 | `MODIFIED` (riskScore altered by the earlier tamper demonstration) |
| TX-DEMO-002-2034 | `VERIFIED_ON_CHAIN` |
| TX-DEMO-003-2187 | `VERIFIED_ON_CHAIN` |

A Hardhat node keeps its chain in memory, so these anchors exist only while that node runs.
`tests/fixtures/legacy-records-v1.json` keeps the three records so their off-chain hashes stay
checkable in CI.

## Defects found at baseline

- **Policy hash ignores the thresholds.** `computePolicyHash` passed a top-level key array as
  the `JSON.stringify` replacer, which filters keys at every depth, so `thresholds` hashed as
  `{}`. Changing a threshold did not change the hash. Kept as hash scheme v1 for P001 (its
  on-chain registration and stored records depend on it); new policies use v2.
- **No assurance engine.** Routing was P001 only.
- **Fabricated confidence upstream.** The Platform sends `confidence: 90 if benchmark_model else 50`.
- **HUMAN_REVIEW decisions were not anchored** until a reviewer acted.
- **No authentication** on review, reset or tamper, and CORS allowed `*`.
