# Component 02 — Real-time fraud and behavioural anomaly engine

This folder is Dushanthini's component of J26-SE-325. It scores a transaction, explains the score, and returns one deterministic action: `ALLOW`, `STEP_UP`, or `BLOCK`.

The other project folders (`Portfolio-Optimization`, `Blockchain-Auditability`, and `Agentic-Assistance`) are not modified by this component. When the rest of the platform adds Kafka, this service already publishes and consumes the topic names `finance.transactions.raw` and `finance.fraud.decisions`.

## What each folder does

| Path | Purpose |
| --- | --- |
| `src/fraud_engine/data` | Download instructions, validation, time-based splits, and dataset loaders |
| `src/fraud_engine/models` | Logistic regression, SVM, random forest, LSTM autoencoder, GCN, GAT, and fusion |
| `src/fraud_engine/gateway` | Thresholds, rate limit, reputation band, drift flag, and analyst feedback |
| `src/fraud_engine/api` | FastAPI service and request checks |
| `src/fraud_engine/streaming` | In-process replay and an optional Kafka adapter |
| `src/fraud_engine/evaluation` | Metrics, adversarial simulations, charts, and the experiment runner |
| `scripts` | Commands for download, training, replay, and the API |
| `config` | Threshold budgets and model sizes |
| `tests` | Automated checks |
| `data/raw` | Downloaded public files. Git ignores this directory |
| `artifacts` | Trained weights. Git ignores this directory |
| `experiments` | Metrics, charts, and the experiment report |
| `docs` | Dataset licences and the demonstration steps |

## How a decision is made

1. The request is rejected if it contains a name, email, card number, password, or raw network address.
2. A trained model, when one has been saved, produces an LSTM reconstruction score and a graph score. A mixture weight chosen on the earlier validation slice combines them.
3. If no trained file exists, a fixed heuristic is used and the response says `heuristic-untrained`. That score is not a research result.
4. The gateway compares the score with two cut-points chosen from **legitimate validation transactions**:
   - about 5% of those legitimate scores may reach `STEP_UP` or `BLOCK`
   - about 1% may reach `BLOCK`
5. Amount, rate limit, reputation, and drift can only make the action stricter. They never turn a `BLOCK` into an `ALLOW`.
6. No language model chooses the action.

The LSTM autoencoder learns what a short legitimate sequence looks like and flags sequences it cannot rebuild. The graph models pass messages from earlier transactions that share an account or device to the current transaction. They are written in PyTorch so the project does not depend on PyTorch Geometric.

## Windows setup

Use PowerShell. Python 3.14 is installed on the development machine used for this component. If a wheel fails to install, install Python 3.12 from [python.org](https://www.python.org/downloads/windows/) and create the virtual environment with `py -3.12` instead of `py -3`.

```powershell
cd Fraud-Detection
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements-dev.txt
py -m pytest
py scripts\train_benchmark.py --profile synthetic
py scripts\run_api.py
```

The synthetic command checks that training and evaluation run. Its numbers are **not** IEEE-CIS, DGraphFin, or Elliptic results. `results_are_research_benchmark` in the JSON file is `false`.

OpenAPI documentation is at `http://127.0.0.1:8082/docs` after the API starts. The analyst page is `http://127.0.0.1:8082/v1/dashboard`.

## Public datasets

Read [docs/DATASETS.md](docs/DATASETS.md) before downloading anything. IEEE-CIS and Elliptic need a Kaggle account and accepted terms. DGraphFin needs an account on its own website. Do not paste a Kaggle token into code or chat.

After the files are on disk:

```powershell
py scripts\prepare_data.py ieee
py scripts\train_benchmark.py --profile dev --dataset ieee
py scripts\train_benchmark.py --profile wider --dataset ieee
py scripts\train_benchmark.py --profile dev --dataset elliptic
py scripts\train_benchmark.py --profile dev --dataset dgraphfin
```

`dev` uses the earliest 20,000 IEEE-CIS training rows after sorting by `TransactionDT`. That result is already in `experiments\results\ieee-cis\`. `wider` uses the earliest 100,000 rows and writes `experiments\results\ieee-cis-100000` so it does not replace the smaller run. `full` removes the row cap and writes `experiments\results\ieee-cis-full`. Graph datasets stay capped so a laptop does not load millions of nodes. Raise `graph_limits` in `config/full.yaml` only when you have the memory.

`py scripts\run_api.py` loads `experiments\results\ieee-cis\fraud_bundle.joblib` when no file is present in `artifacts\`. The health check then reports that model instead of the untrained heuristic.

The official IEEE-CIS test file has no public `isFraud` labels. Metrics use a later time slice of the labelled training file. A timestamp is never placed in two splits. Scalers, frequency counts, and class weights are fit on the training split only. The test split is not resampled.

## Docker

The repository does not yet contain a shared Compose file, so this folder has its own:

```powershell
docker compose up fraud-api
docker compose --profile kafka up
```

Kafka is optional. Without it, `scripts\replay_transactions.py` uses an in-process queue. From the host, the Compose Kafka listener is `localhost:9094`.

## Tests

```powershell
py -m pytest
```

A passing test means that command was run in this environment. Do not treat unread test names as results.

## Demonstration

Follow [docs/DEMO.md](docs/DEMO.md). The method notes and the latest executed run are in [experiments/EXPERIMENT_REPORT.md](experiments/EXPERIMENT_REPORT.md).
