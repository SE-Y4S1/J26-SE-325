# Privacy-Preserving AI-to-Smart-Contract Audit Bridge
## Component 03 — Technical Architecture, Formal Verification & Demonstration Report

> **Superseded — kept as the record of the first (ledger-only) iteration.** The current design,
> the AI Decision Assurance Engine (AUTO / VERIFY / HUMAN_REVIEW routing, policy P002, record v2
> with linked review commitments) and current test results are in [README.md](README.md).
> The hash formulas, contract and verification predicate below still hold for v1 records.

**Author / Researcher:** Chenuli Abeysekara (IT23225442) · **Component Port:** 8002 (UI in the shared Nexa app) · **Status:** Operational

---

## 1. Executive Summary & Research Scope

Automated AI decision-making in financial systems presents an acute trust dilemma: institutions must provide verifiable auditability to satisfy regulatory compliance without disclosing proprietary model internals or violating customer privacy regulations (e.g., GDPR, CCPA, GLBA). 

Component 03 establishes a **Privacy-Preserving AI-to-Smart-Contract Audit Bridge**. It cryptographically anchors AI risk evaluations, human review interventions, and policy provenance to an immutable EVM blockchain. By storing exclusively 32-byte cryptographic commitments on-chain while retaining complete financial records in durable off-chain SQLite storage, the bridge delivers mathematical proof of non-repudiation with **zero on-chain data leakage**.

```
┌────────────────────────┐      ┌──────────────────────────────┐      ┌─────────────────────────────┐
│  Upstream AI Decision  │ ───► │  Component 03 Audit Bridge   │ ───► │  AuditRegistry.sol (EVM)    │
│  (Risk, Conf, Model)   │      │  (Port 8002 / SQLite WAL)    │      │  (Hardhat Local Node: 8545) │
└────────────────────────┘      └──────────────────────────────┘      └─────────────────────────────┘
                                                │                                     ▲
                                                │ (Recompute & Verify)                │
                                                └─────────────────────────────────────┘
```

---

## 2. Mathematical Formalization & Cryptographic Commitments

### 2.1 Cryptographic Salt & Pseudonymous Audit Identifier
To eliminate linkability and transaction correlation on public or consortium ledgers, each decision is assigned an independent 32-byte cryptographically secure pseudorandom salt $S$:

$$S \leftarrow_{\$} \{0,1\}^{256}$$

The pseudonymous identifier $\text{auditId}$ anchored on-chain is defined as:

$$\text{auditId} = \text{0x} \parallel \text{SHA-256}(\text{transactionId} \parallel \text{":"} \parallel S)$$

Because $S$ is kept off-chain, an external blockchain observer cannot deduce $\text{transactionId}$ without knowledge of $S$, preventing rainbow-table and dictionary preimage attacks.

### 2.2 Canonical State Serialization & Record Commitment
To guarantee cross-platform determinism across varying CPU architectures, JavaScript runtimes, and database engines, the canonical payload encompasses exactly 16 sorted attributes:

$$\mathcal{K} = \{\text{aiDecision}, \text{amount}, \text{auditId}, \text{confidence}, \text{finalDecision}, \text{modelVersion}, \text{policyHash}, \text{policyId}, \text{policyVersion}, \text{reviewRequired}, \text{reviewerDecision}, \text{riskScore}, \text{salt}, \text{timestamp}, \text{transactionId}, \text{transactionType}\}$$

The canonical commitment hash is computed as:

$$\text{recordHash} = \text{0x} \parallel \text{SHA-256}(\text{JSON.stringify}(R, \text{sort}(\mathcal{K})))$$

### 2.3 Policy Specification Commitment
The governing policy specification ($P001$ v1.0: $\text{riskScore} \ge 80 \land \text{confidence} \ge 70 \implies \text{REJECT}$, $\text{riskScore} \ge 80 \land \text{confidence} < 70 \implies \text{HUMAN\_REVIEW}$, otherwise ($\text{riskScore} < 80$) $\text{APPROVE}$) is deterministically hashed. Note: this legacy scheme (v1) passes the sorted top-level keys as the `JSON.stringify` replacer, which filters keys at every depth, so the nested `thresholds` object is hashed as `{}`; policies from P002 on use scheme v2 (canonical JSON, keys sorted at every depth). See `README.md` §3:

$$\text{policyHash} = \text{0x} \parallel \text{SHA-256}(\text{JSON.stringify}(\text{PolicyDef}, \text{sort}(\text{Keys})))$$

### 2.4 Discrete Verification Predicate
Given an off-chain record $R$ and on-chain state $\mathcal{B}(\text{auditId}) = \langle \text{exists}, \text{onChainRecordHash}, \text{onChainPolicyHash}, \text{timestamp}, \text{submitter} \rangle$:

$$\mathcal{V}(R, \mathcal{B}) = \begin{cases} 
\text{NOT\_FOUND} & \text{if } R \notin \text{OffChainDB} \\
\text{MODIFIED} & \text{if } \text{auditId}(R) \neq \text{SHA-256}(R.\text{txId} \parallel \text{":"} \parallel R.\text{salt}) \\
\text{ON\_CHAIN\_MISSING} & \text{if } \neg \mathcal{B}(R.\text{auditId}).\text{exists} \\
\text{MODIFIED} & \text{if } \text{recordHash}(R) \neq \mathcal{B}(R.\text{auditId}).\text{onChainRecordHash} \\
\text{MODIFIED} & \text{if } R.\text{policyHash} \neq \mathcal{B}(R.\text{auditId}).\text{onChainPolicyHash} \\
\text{VERIFIED\_ON\_CHAIN} & \text{if hashes match and mined on live blockchain} \\
\text{VERIFIED\_SIMULATED} & \text{if hashes match in explicit mock mode}
\end{cases}$$

---

## 3. System Architecture & Component Design

### 3.1 Smart Contract Layer (`AuditRegistry.sol`)
- **OpenZeppelin AccessControl:** Implements `DEFAULT_ADMIN_ROLE` for policy registration and lifecycle governance, and `SUBMITTER_ROLE` for authorized backend signers.
- **Write-Once Idempotency:** Immutability enforced through custom Solidity errors:
  - `InvalidAuditId()`: Rejects `bytes32(0)`
  - `InvalidRecordHash()`: Rejects `bytes32(0)`
  - `PolicyNotActive()`: Rejects commitments for unregistered or deprecated policies
  - `AuditAlreadyExists()`: Rejects attempts to overwrite or replay existing audit records
- **Zero PII Exposure:** Plain financial amounts, scores, and decisions are never transmitted to EVM contract calldata.

### 3.2 Persistent Data Repository (`database.js`)
- **SQLite with WAL Mode:** Executes `PRAGMA journal_mode = WAL` and `PRAGMA synchronous = NORMAL` for crash-resilient write performance.
- **Dual-Engine Architecture:** Transparently supports Node.js native `node:sqlite` (Node 22+) and `better-sqlite3` (Node 20 / container environments).
- **Security & Privacy:** Public REST API endpoints return masked salts (`saltMasked: "a1b2c3d4…"`) while storing complete salts securely for integrity verification.

### 3.3 Live Blockchain Integration Service (`blockchainService.js`)
- **Strict Live-First Operation:** Connects via `ethers.JsonRpcProvider` directly to local Hardhat node on port 8545.
- **Error Gating:** If the blockchain node is offline or unreachable, returns HTTP 503 `BlockchainUnavailableError` rather than silently fabricating mock receipts.
- **Reconciliation & Retries:** Provides `POST /api/audit/:transactionId/retry` to publish offline records with on-chain deduplication.

---

## 4. Verification & Evaluation Benchmarks

### 4.1 Automated Test Suites Summary
| Test Suite | File Location | Framework | Total Tests | Status |
|---|---|---|---|---|
| Smart Contract Modernization | `blockchain/test/AuditRegistry.test.js` | Hardhat / Mocha / Chai | 12 | **12 / 12 PASSING** |
| Deterministic Hashing Service | `tests/hashService.test.js` | Jest | 9 | **9 / 9 PASSING** |
| End-to-End API Integration | `tests/e2e.test.js` | Supertest / Jest | 13 | **13 / 13 PASSING** |
| **Combined Total** | — | — | **34** | **34 / 34 PASSING** |

### 4.2 Live Scenario Benchmark Execution Output
Execution of `node scripts/demo-live-run.js`:

```
======================================================================
 EVALUATION BENCHMARK SUMMARY TABLE
======================================================================
┌─────────┬─────────────────────────────┬────────────────────┬──────────────────┬─────────┬─────────────────────────┬─────────────────────┐
│ (index) │ Scenario                    │ Tx ID              │ Decision         │ Block # │ Pre-Tamper Verification │ Integrity Preserved │
├─────────┼─────────────────────────────┼────────────────────┼──────────────────┼─────────┼─────────────────────────┼─────────────────────┤
│ 0       │ 'Scenario 1 (Low Risk)'     │ 'TX-DEMO-001-1475' │ 'APPROVE'        │ 3600    │ 'VERIFIED_ON_CHAIN'     │ 'YES'               │
│ 1       │ 'Scenario 2 (High Risk)'    │ 'TX-DEMO-002-1478' │ 'REJECT'         │ 3601    │ 'VERIFIED_ON_CHAIN'     │ 'YES'               │
│ 2       │ 'Scenario 3 (Human Review)' │ 'TX-DEMO-003-1479' │ 'HUMAN_APPROVED' │ 3602    │ 'VERIFIED_ON_CHAIN'     │ 'YES'               │
└─────────┴─────────────────────────────┴────────────────────┴──────────────────┴─────────┴─────────────────────────┴─────────────────────┘
```

### 4.3 Tamper Detection Validation
When an off-chain record field (`riskScore`) was altered from `20` to `99`:
- **Calculated Current Hash:** `0x4af32e742565cb9fccf4ed601708e229790b32550351d5da8aa430e8e5599a1e`
- **On-Chain Mined Hash:** `0xb5471ec0a1b9d1b1abe8b7807f8257a714004288f0a7611253729fb9131783bb`
- **Verification Status:** `MODIFIED`
- **Result:** Non-repudiation failure caught instantly with 100% precision.

---

## 5. Instructions for Running and Reproducing

### 1. Launch Hardhat Blockchain Node
```bash
cd backend/Blockchain-Auditability/blockchain
npx hardhat node
```

### 2. Deploy Smart Contract & Register Policy
```bash
cd backend/Blockchain-Auditability/blockchain
npx hardhat run scripts/deploy.js --network localhost
```

### 3. Run Automated Demonstration
```bash
cd backend/Blockchain-Auditability
node scripts/demo-live-run.js
```

### 4. Start Server & Interactive Dashboard
```bash
# Terminal 1: Backend (Port 8002)
cd backend/Blockchain-Auditability
npm start

# Terminal 2: shared Nexa web app (Port 3000)
cd frontend
npm run dev
```
Open `http://localhost:3000/audit` in your browser to access the audit console.
