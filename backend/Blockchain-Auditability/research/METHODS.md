# Methods

## Question and hypothesis

At a fixed share of decisions sent to human review, does routing on **confidence + model
disagreement + contextual risk** (the assurance engine) leave fewer incorrect decisions
automated than routing on risk alone, confidence alone, the legacy P001 thresholds, random
review, or a learned error predictor?

H1: at the 10% review budget, the assurance engine's incorrect-automatic rate is lower than
every simple baseline's (paired bootstrap, one-sided).

## Data

IEEE-CIS Fraud Detection (Kaggle, 2019), labelled `train_transaction.csv` left-joined with
`train_identity.csv` (~590k transactions, ~3.5% fraud). The competition test file has no labels
and is not used.

## Splits (time-ordered, `TransactionDT`)

| Split | Share | Used for, and nothing else |
|---|---|---|
| Train | 50% | preprocessing statistics, base models, context statistics |
| Calibration | 15% | isotonic calibration of each base model |
| Policy tuning | 15% | every router's parameters, including the assurance engine's and all baselines' |
| Test | 20% | final evaluation, once |

Split boundaries are moved past runs of equal timestamps, so no timestamp is in two splits. The
test split is fingerprinted (SHA-256 of its transaction ids) in every run log. `final` refuses to
run on an uncommitted policy and records each run; repeating it requires a logged reason.

## Preprocessing and leakage prevention

- **Features, fixed a priori** (Component 2's documented set): `TransactionAmt`, `dist1–2`,
  `C1–C14`, `D1–D15`, `V1–V20`, `M1–M9`, `id_01–id_11`, frequency encodings of `ProductCD`,
  `card4`, `card6`, `M4`, email domains, `card1`, `DeviceType`, `DeviceInfo`, three missingness
  flags and an identity flag. No data-driven feature selection.
- **Learned from train only:** medians, category frequencies (unseen → 0), scaler, the
  IsolationForest, the amount distribution.
- **Class imbalance:** `class_weight="balanced"` on train only; no resampling anywhere; the later
  splits keep their natural fraud rate.
- **Labels never feed a signal.** The first-seen-card flag looks strictly backwards in time.

## The AI decision

Calibrated ensemble probability `p` (mean of three members). Reject when `p ≥ 0.0909`, the
Bayes threshold for a 10:1 cost of a missed fraud to a false alarm (fixed, not tuned). A decision
is *incorrect* when it disagrees with `isFraud`.

## Signals (each an uncertainty in [0, 1])

| Signal | Definition | Limitation |
|---|---|---|
| Confidence | `1 − |p − t| / (t or 1 − t)`, the service engine's formula | calibration quality bounds its meaning |
| Disagreement | `max − min` of the three calibrated members | three models only; correlated errors stay invisible |
| Context | mean of the four below, equal weights fixed a priori | |
| · amount | percentile among training amounts | fraud is not monotone in amount |
| · novelty | IsolationForest outlier score | flags rare but legitimate behaviour |
| · new card | `card1` never seen in an earlier transaction | penalises new customers (fairness) |
| · no identity | no identity/device record | missing not at random, varies by product |

Online, the service's context uses different sub-signals (amount scale, fraud-gateway rule hits,
model validation status, transaction type) because the offline ones are not available there.

## Routes and metrics

AUTO executes the AI decision; VERIFY holds it for account-holder step-up; HUMAN_REVIEW holds it
for an analyst.

- **Incorrect automatic rate (primary)** = wrong decisions not sent to a human ÷ all decisions.
  VERIFY counts as automatic: the engine gets **no credit** for step-up.
- Human-review rate, verify rate, fraud auto-approved (count and amount), legitimate
  auto-rejected, AURC of each method's deferral ranking.
- 95% percentile bootstrap intervals (1,000 resamples); paired bootstrap of the difference
  against each comparator.
- **Assumptions, swept rather than claimed:** VERIFY catches 0/25/50/75% of errors; analysts are
  100/95/90% accurate.
- Stability: incorrect-automatic rate on the first and second half of the test period.

## Methods compared (all tuned on the policy-tuning split)

M0 automate everything · M1 risk only (highest `p` reviewed) · M2 confidence only ·
M3 P001 as deployed and P001 re-tuned · M4 random review (exact share) · **M5 assurance engine**
(weights on a 0.05 simplex grid, disagreement veto ∈ {0.2,…,0.6, none}, thresholds set so the
review budget is filled and 10% of decisions go to VERIFY) and its three ablations ·
M6 learned deferral (logistic regression predicting an incorrect decision from the same signals).

## Evaluated policy = enforced policy

The M5 parameters at the 10% budget are rounded to 4 decimals, written as
`src/policies/P002-assurance-v1.1.json`, and **the final evaluation uses those rounded values**.
The service loads the same file, hashes it (scheme v2) and commits that hash in every decision it
routes. Golden vectors (`tests/fixtures/golden-vectors.json`) prove the Python and JavaScript
engines produce the same route, score and flags for 400 inputs. What differs online: the base
models (Component 2's) and the context sub-signals — so online performance is not claimed from
these results.
