# IEEE-CIS evaluation of the assurance engine

Does the AI Decision Assurance Engine automate **fewer incorrect decisions** than simpler
decision routing, at the **same human-review workload**? Method details: [METHODS.md](METHODS.md).

## 1. Get the data (once)

The data is the Kaggle competition *IEEE-CIS Fraud Detection*; its rules must be accepted with
your own Kaggle account, and the files may not be committed or redistributed.

1. Log in at kaggle.com (verify your phone number if asked).
2. Open <https://www.kaggle.com/competitions/ieee-fraud-detection/rules> and accept the rules.
3. On the competition's **Data** tab, choose **Download All** (about 118 MB).
4. Extract `train_transaction.csv` and `train_identity.csv` into
   `backend/Fraud-Detection/data/raw/ieee-cis/` (git-ignored). Another folder works too:
   set `IEEE_CIS_DIR` to it.

## 2. Run

```bash
cd backend/Blockchain-Auditability/research
uv sync                                   # Python 3.12 environment (uv installs it if missing)

uv run python -m assurance.run tune       # ~5-15 min on a laptop CPU: models, signals, tuning
#   writes results/ieee-cis/tuning.json (tuning-split numbers only)
#   and the tuned policy src/policies/P002-assurance-v1.1.json

uv run python -m assurance.golden         # golden vectors for the tuned policy
cd .. && npm test                         # JS engine must reproduce them exactly

git add src/policies/P002-assurance-v1.1.json tests/fixtures/golden-vectors.json && git commit -m "Tuned P002 v1.1"

cd research && uv run python -m assurance.run final   # the held-out test split, once
#   writes results/ieee-cis/{metrics.json, REPORT.md, budgets.csv, figures/, RUNLOG.json}
```

Then register the tuned policy on-chain so new decisions can commit to it:

```bash
cd ../blockchain && npx hardhat run scripts/register-policies.js --network localhost
```

The service picks the highest P002 version automatically (`ACTIVE_ASSURANCE_POLICY` overrides),
and the **Audit → Evaluation** page shows the results.

Development runs: `--max-rows 100000 --fast` (results are not for reporting).

## 3. Tests

```bash
uv run pytest -q
```

The tests run on a small generated fixture (`tests/fixture.py`) only to check the pipeline's code
— splits, leakage controls, routers, metrics, the two stages. Fixture numbers are never results.

## What is trained, and what is not

- **Trained (here, for the evaluation):** three fraud models — logistic regression, random forest,
  gradient boosting — fitted on the training split and calibrated on the calibration split. They
  are the *AI* whose decisions are being assured; Component 2's models are not used or changed.
- **Tuned, not trained:** the assurance engine. It is an interpretable rule (a weighted score,
  two thresholds and a disagreement veto); its few parameters are chosen by grid search on the
  policy-tuning split, then frozen, exported as a policy document, hashed and registered
  on-chain.
