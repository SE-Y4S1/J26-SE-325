# Real-Time Fraud & Behavioral Anomaly Engine with Deterministic Enforcement

Research prototype for a university group project on an integrated AI-driven finance
platform. This folder is **self-contained**: it does not read, import or modify any
other group member's code.

---

## Honest statement about the models

> **This prototype does not train or run any neural network.**
>
> The behavioural stream and the relational stream are **deterministic, hand-designed
> simulations** that reproduce the *shape of the output* a trained LSTM autoencoder and a
> trained GNN would produce. Every API response is tagged
> `"model_mode": "SIMULATED_PROTOTYPE"` so this can never be misread.
>
> | Layer | Proposed research implementation | What runs in this prototype |
> |---|---|---|
> | Behavioural sequence anomaly | LSTM autoencoder (PyTorch), reconstruction error | EWMA sequence memory + max-pooled deviation + drift term |
> | Relational fraud patterns | GNN (PyTorch Geometric), learned message passing | Real entity graph + decayed 2-hop risk propagation |
> | Behavioural biometrics | Learned keystroke / navigation embeddings | Rule-based deviation from a per-user baseline |
> | Device fingerprinting | Learned device embeddings | Device-to-user edges and fan-out counting |
> | **Deterministic gateway** | **Exactly as designed** | **Exactly as designed - this part is real** |
>
> The gateway, the thresholds, the policy rules, the explainability and the audit trail
> are the actual proposed design, fully implemented. Only the two score *producers* are
> simulated, and `fraud_model.py` documents precisely where the real PyTorch models plug in.

---

## The pipeline

```
                        Transaction (validated)
                                 |
                 +---------------+---------------+
                 |                               |
        Behavioural analysis            Relational analysis
   (amount, typing biometrics,      (users / devices / beneficiaries
    navigation, frequency,           / locations / IPs graph)
    device fingerprint)
                 |                               |
     LSTM-like anomaly score          GNN-like relational risk score
        [SIMULATED]                        [SIMULATED]
                 |                               |
                 +---------------+---------------+
                                 |
                    Late fusion -> final risk score
                                 |
                 DETERMINISTIC SECURITY GATEWAY
              (configurable thresholds + policy rules)
                                 |
                +----------------+----------------+
                |                |                |
             ALLOW           STEP-UP            BLOCK
                |                |                |
                +----------------+----------------+
                                 |
                            Audit log
```

**The machine-learning side never decides anything.** It produces a number. The gateway
in `gateway.py` is the only component allowed to choose an action.

---

## How to run it on Windows in VS Code

Open the `fraud-detection-gateway` folder in VS Code, open a terminal
(`Ctrl` + `` ` ``) and run:

```bat
python -m venv venv
```

```bat
venv\Scripts\activate
```

```bat
pip install -r requirements.txt
```

```bat
uvicorn app:app --reload
```

Then open <http://127.0.0.1:8000> in your browser. That is the demo UI.

### If PowerShell refuses to activate the venv

PowerShell blocks scripts by default. Either run this once:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

or simply skip activation and call the interpreter directly:

```powershell
.\venv\Scripts\python.exe -m uvicorn app:app --reload
```

### Running with F5 instead

A `.vscode/launch.json` is included. With the `fraud-detection-gateway` folder open in
VS Code, press `F5` and pick **"Run fraud gateway API (uvicorn)"**. Select the
`venv` interpreter first (`Ctrl+Shift+P` -> *Python: Select Interpreter*).

### Opening the demo UI

* **Recommended:** <http://127.0.0.1:8000> - the API serves `demo.html` itself, so
  everything is on one origin.
* Alternatively, double-click `demo.html`. It falls back to `http://127.0.0.1:8000`
  for its API calls, and the server allows cross-origin requests for this reason.
* Interactive API documentation: <http://127.0.0.1:8000/docs>

### Running the adversarial experiment on its own

```bat
python adversarial_test.py
```

This prints a full console report of all three simulated attacks. It needs no server.

---

## Files

| File | Responsibility |
|---|---|
| `app.py` | FastAPI layer only: routes, error handling, serving the UI |
| `fraud_model.py` | The two simulated streams (LSTM-like, GNN-like) and their fusion |
| `gateway.py` | The deterministic security gateway - thresholds and policy rules G1..G7 |
| `schemas.py` | Pydantic validation for every request and response |
| `pipeline.py` | Wires model -> gateway -> audit into one path shared by the API and the experiment |
| `config.py` | Every threshold and weight, in one editable place |
| `audit_log.py` | Append-only decision trail (memory + `audit_log.jsonl`) |
| `adversarial_test.py` | The three safe, simulated attack experiments |
| `demo.html` | Presentation UI |
| `requirements.txt` | FastAPI, Uvicorn, Pydantic - nothing else |
| `.vscode/launch.json` | F5 run configurations |
| `.gitignore` | Keeps `venv/` and generated files out of the group repository |

`audit_log.jsonl` is created automatically the first time a transaction is scored.

A working `venv/` with the dependencies already installed is present, so you can skip
straight to `uvicorn app:app --reload` if you want. Re-running the setup commands above
is harmless.

---

## API

### `GET /health`
Liveness, uptime, number of transactions scored, and the policy currently enforced.

### `POST /score`
Score one transaction and enforce a decision.

```json
{
  "user_id": "u_1001",
  "amount": 18000,
  "location": "Dubai, AE",
  "device_id": "dev_unknown_45",
  "device_change": true,
  "typing_speed": 400,
  "navigation_pattern": "exploratory",
  "transaction_frequency": 6,
  "beneficiary_change": false,
  "beneficiary_id": "ben_utility_22",
  "ip_address": "41.90.12.7",
  "previous_transaction_amount": 1200,
  "account_age": 900,
  "persist_state": false
}
```

Response (abbreviated):

```json
{
  "behavioral_score": 0.744,
  "graph_score": 0.363,
  "risk_score": 0.602,
  "decision": "STEP-UP",
  "reason": "STEP-UP - risk 0.60 in [0.35, 0.70) (medium band)",
  "reasons": ["Unusual typing pattern: ...", "New device detected ...", "..."],
  "rules_fired": [],
  "model_mode": "SIMULATED_PROTOTYPE"
}
```

`persist_state` controls whether the engine learns from the transaction. The demo UI
leaves it **off** so that every click produces an identical, reproducible result.

### `POST /simulate-attack`
`{"attack_type": "camouflage" | "slow_drift" | "structuring", "steps": 8}`
Returns the baseline stage, every attack stage, the scores and gateway decision at each
stage, and a written finding.

### `GET /audit`, `GET /config`, `POST /reset`, `GET /pipeline`
Audit trail, the enforced policy, a state reset for repeating a live demo, and the
pipeline stages used by the UI diagram.

---

## The deterministic gateway

Risk bands (`config.py` -> `GatewayConfig`):

| Band | Condition | Action |
|---|---|---|
| Low | `risk < 0.35` | **ALLOW** |
| Medium | `0.35 <= risk < 0.70` | **STEP-UP** (OTP / biometric) |
| High | `risk >= 0.70` | **BLOCK** |

On top of the bands, fixed policy rules run in a defined order. Escalation rules can
only raise severity; the single de-escalation rule can only lower it, and is never
allowed to undo a hard block.

| Rule | Meaning | Effect |
|---|---|---|
| G1 | Amount >= 250,000 on a new device/payee while risk is elevated | BLOCK |
| G2 | Device change with amount >= 25,000 | at least STEP-UP |
| G3 | Relational risk >= 0.75 | at least STEP-UP |
| G4 | Account <= 7 days old moving >= 10,000 | at least STEP-UP |
| G5 | Behavioural >= 0.90 **and** relational >= 0.60 | BLOCK |
| G7 | Split transfers to one payee total >= the single-transfer limit | BLOCK |
| G6 | Micro-payment from an established, clean account | capped at STEP-UP |

Every decision returns the rules that fired and the thresholds that applied, so it can
be reconstructed later from the audit log.

---

## Adversarial robustness results

Reproduce with `python adversarial_test.py`. These are the prototype's own numbers,
produced from synthetic sequences generated in-process. **No attack tooling exists in
this repository** - `adversarial_test.py` only builds dictionaries of made-up numbers
and feeds them to our own scoring function.

### 1. Camouflage attack
The attacker keeps the mule device, the mule beneficiary and the shared IP, but
progressively imitates normal behaviour.

| Stage | Behavioural | Relational | Risk | Decision |
|---|---|---|---|---|
| No camouflage | 0.976 | 0.903 | 1.000 | BLOCK |
| Camouflage 4/8 | 0.854 | 0.903 | 1.000 | BLOCK |
| Camouflage 6/8 | 0.522 | 0.903 | 0.797 | BLOCK |
| Camouflage 8/8 | 0.098 | 0.903 | 0.516 | STEP-UP |

**Finding:** behaviour can be faked, relationships cannot. A behaviour-only model would
have **ALLOWED** the final stage (0.098 is deep in the low band). The dual-stream design
still challenges it.

### 2. Slow-drift attack
Behaviour is nudged slightly every day; no single step is dramatic and the navigation
label never changes.

| Day | Behavioural | Risk | Decision |
|---|---|---|---|
| 0 (baseline) | 0.001 | 0.001 | ALLOW |
| 3 | 0.361 | 0.180 | ALLOW |
| 6 | 0.652 | 0.326 | ALLOW |
| 8 | 0.717 | 0.358 | STEP-UP |
| 10 | 0.797 | 0.399 | STEP-UP |

**Finding:** the drift term (short-term memory vs. long-term anchor) accumulates what the
reconstruction error alone would miss. The anchor profile only learns from **ALLOWed**
transactions, so an attacker cannot redefine "normal" for the account.

### 3. Structuring attack
One transfer of 360,000 is split into eight transfers of 45,000 to the same payee.

| Stage | Behavioural | Relational | Risk | Decision |
|---|---|---|---|---|
| Single 360,000 transfer | 0.572 | 0.205 | 0.410 | BLOCK (rule G1) |
| Split 1/8 | 0.209 | 0.205 | 0.215 | ALLOW |
| Split 3/8 | 0.319 | 0.411 | 0.388 | STEP-UP |
| Split 6/8 | 0.497 | 0.601 | 0.603 | **BLOCK (rule G7)** |
| Split 8/8 | 0.590 | 0.753 | 0.751 | BLOCK |

**Finding:** at split 6 the fused risk is only 0.603, which is still in the *medium*
band. The model alone would merely have asked for an OTP. The **deterministic gateway**
blocked it, because the running total had reached the same hard limit that applies to a
single transfer. This is the clearest argument for keeping enforcement out of the model.

A control test confirms the rule does not over-fire: eight genuinely small payments of
900 to the same payee stay **ALLOW**, because the aggregate never becomes material.

---

## Explainability

Every response carries plain-English reasons ranked by contribution, for example:

* New device detected (device fingerprint changed)
* Unusual typing pattern: typing is much higher than this user's biometric baseline
* Abnormal transaction frequency for this account
* Suspicious beneficiary relationship: 5 accounts send money to this same payee
* Impossible travel: Colombo, LK -> Dubai, AE within the hour
* High transaction amount: 12x this user's usual transfer of about 30,000
* Structuring pattern: 6 sub-threshold transfers to the same payee totalling 270,000 within 6h
* Connected (within 2 hops) to entities involved in blocked fraud

---

## Replacing the simulations with the real models

`fraud_model.py` defines a single, narrow contract:

```python
class Stream:
    def score(self, tx: Transaction) -> StreamResult: ...   # StreamResult(score, signals)
    def commit(self, tx: Transaction, flag: bool) -> None: ...
```

Write `TorchBehavioralStream` (a trained LSTM autoencoder producing a normalised
reconstruction error) and `TorchRelationalStream` (a trained GNN producing a node/edge
risk), then:

```python
model = DualStreamFraudModel(
    behavioral=TorchBehavioralStream("lstm_ae.pt"),
    relational=TorchRelationalStream("gnn.pt"),
)
```

Nothing else changes. The gateway, the API, the audit log and the UI only ever see a
float, which is exactly the separation of concerns the research proposes.

---

## Limitations (state these in the viva)

1. The scores are simulated heuristics, not learned functions; the weights were chosen by
   hand, not fitted to data.
2. All state is in memory. Restarting the server clears every learned profile and the
   graph (the audit file on disk survives).
3. The seeded relationship graph is fabricated demo data for the presentation.
4. Thresholds have not been calibrated against a labelled dataset, so no false-positive
   or false-negative rate is claimed.
5. Single process, no authentication, no persistence layer - it is a demonstrator, not a
   deployable service.
