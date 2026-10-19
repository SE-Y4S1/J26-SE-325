# Component 03 — Privacy-Preserving AI-to-Smart-Contract Audit Bridge

**Owner:** Abeysekara W.C.S.M · **Backend Port:** 8002 · **Frontend Port:** 3001 · **Status:** Operational

> **Sub-objective (TAF):** Design a privacy-preserving, tamper-evident blockchain audit layer that bridges continuous AI decisions to deterministic on-chain smart contract enforcement.

---

## 1. System Architecture Overview

Component 3 bridges AI decision evaluation outputs (such as fraud detection, risk scoring, and automated approvals) to an immutable EVM blockchain. It enforces non-repudiation and cryptographic integrity while ensuring **zero personally identifiable information (PII) or sensitive business logic is leaked on-chain**.

```
[ AI Engine / Simulator ]
          │
          ▼
[ Component 3 Backend (Port 8002) ]
   ├── Policy Engine (P001 v1.0 Thresholds)
   ├── Cryptographic Hash Engine (32-byte Salt + bytes32 auditId + 16-field Canonical SHA-256)
   ├── SQLite Audit Persistence (WAL mode, Dual-Engine: node:sqlite / better-sqlite3)
   └── Blockchain Bridge Service (ethers.js v6)
          │
          ▼
[ AuditRegistry.sol Smart Contract (Hardhat Local Node, Port 8545) ]
   ├── OpenZeppelin AccessControl (DEFAULT_ADMIN_ROLE, SUBMITTER_ROLE)
   ├── bytes32 auditId (sha256(transactionId + ":" + salt))
   ├── bytes32 recordHash (sha256(canonicalPayload))
   ├── bytes32 policyHash (sha256(canonicalPolicyDefinition))
   ├── Write-Once Immutability (Reject duplicates via AuditAlreadyExists)
   └── Policy Registry (registerPolicy, deactivatePolicy, isPolicyActive)
```

---

## 2. Key Modules & Implementation Details

### A. Smart Contract (`blockchain/contracts/AuditRegistry.sol`)
- **Privacy-Preserving:** Stores strictly `bytes32` cryptographic hashes. No plain transaction IDs, risk scores, or decisions ever touch the blockchain.
- **Write-Once Enforcement:** Custom errors (`InvalidAuditId`, `InvalidRecordHash`, `PolicyNotActive`, `AuditAlreadyExists`) guarantee records cannot be overwritten, modified, or deleted once mined.
- **Policy Registry:** Ensures an audit commitment can only be anchored if its evaluated policy specification (`policyHash`) is registered and active.

### B. Cryptographic Hash Service (`backend/src/services/hashService.js`)
- **Salt Generation:** Generates cryptographically secure 32-byte random hex salts.
- **Pseudonymous Audit ID:** `auditId = "0x" + sha256(transactionId + ":" + salt)`. Prevents on-chain transaction correlation.
- **Deterministic Canonical Serialization:** Canonical JSON string sorting 16 fields alphabetically to guarantee cross-platform hash determinism.

### C. Persistent Storage (`backend/src/db/database.js`)
- **Durable SQLite Storage:** Operates in `WAL` mode (`journal_mode = WAL`) with prepared statements.
- **Dual-Engine Adapter:** Supports both Node.js built-in `node:sqlite` (Node 22+) and `better-sqlite3` (Node 20 / Linux containers) transparently.
- **Data Protection:** API responses mask cryptographic salts (`saltMasked: "a1b2c3d4…"`) while storing the full salt securely off-chain.

### D. Strict Verification Engine (`backend/src/services/auditService.js`)
Evaluates integrity against 5 discrete states:
1. `VERIFIED_ON_CHAIN`: Calculated hash matches the immutable on-chain smart contract commitment.
2. `VERIFIED_SIMULATED`: Validated against mock adapter when `BLOCKCHAIN_MODE=mock`.
3. `MODIFIED`: Tampering detected! Off-chain data modified without smart contract consensus.
4. `ON_CHAIN_MISSING`: Audit record exists off-chain, but no anchor exists on the blockchain.
5. `NOT_FOUND`: Transaction record does not exist in the database.

---

## 3. API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/audit/evaluate` | Evaluates decision against policy rules, computes hashes, and anchors to blockchain. |
| `POST` | `/api/audit/review` | Submits human review approval/rejection and anchors the final consensus to blockchain. |
| `GET` | `/api/audit/:transactionId/verify` | Recomputes canonical hash and verifies against smart contract. |
| `GET` | `/api/audit/by-audit-id/:auditId/verify` | Verifies cryptographic integrity by on-chain `auditId`. |
| `POST` | `/api/audit/:transactionId/retry` | Retries publishing unanchored or failed commitments. |
| `POST` | `/api/audit/:transactionId/tamper` | Simulates unauthorized off-chain tampering (Demo/Research feature). |
| `GET` | `/api/audit/records` | Retrieves all off-chain audit records with masked salts. |
| `GET` | `/api/audit/stats` | Aggregated metrics and blockchain connectivity status. |
| `POST` | `/api/audit/reset` | Clears demo records for clean benchmark evaluation. |
| `GET` | `/api/health` | Health check probe. |

---

## 4. Quickstart Guide (Running Locally)

### Prerequisites
- Node.js >= 20
- npm >= 9

### Step 1: Start the Local Hardhat Blockchain Node
```bash
cd backend/Blockchain-Auditability/blockchain
npx hardhat node
```
*Listens on `http://127.0.0.1:8545` (ChainId: 31337).*

### Step 2: Deploy AuditRegistry Smart Contract
In a new terminal:
```bash
cd backend/Blockchain-Auditability/blockchain
npx hardhat run scripts/deploy.js --network localhost
```
*Deploys contract, grants `SUBMITTER_ROLE`, registers initial policy hash, and writes `contractConfig.json`.*

### Step 3: Start the Backend Service
```bash
cd backend/Blockchain-Auditability/backend
npm start
```
*Backend runs on `http://localhost:8002`.*

### Step 4: Start the Frontend UI
```bash
cd backend/Blockchain-Auditability/frontend
npm run dev
```
*Frontend runs on `http://localhost:3001` (proxies `/api` requests to `8002`).*

---

## 5. Automated Test Suites

### Running Smart Contract Unit Tests
```bash
cd backend/Blockchain-Auditability/blockchain
npx hardhat test
```
*Result: 12 passing tests covering access control, write-once semantics, and timestamp fidelity.*

### Running Backend Unit & E2E Integration Tests
```bash
cd backend/Blockchain-Auditability/backend
npm test
```
*Result: 22 passing tests covering deterministic hashing, duplicate prevention (HTTP 409), human review flows, and tamper detection.*
