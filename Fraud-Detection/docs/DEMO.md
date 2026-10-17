# Demonstrate Component 02

Run these steps from the `Fraud-Detection` folder after the Windows setup in the README. The first demonstration does not need Kaggle. It shows the gateway and a synthetic training run. Say clearly, if you present it, that the synthetic scores are a pipeline check. After the IEEE development bundle exists, `py scripts\run_api.py` loads `experiments\results\ieee-cis\fraud_bundle.joblib` and `/health` reports that model.

## 1. Tests

```powershell
.\.venv\Scripts\Activate.ps1
py -m pytest
```

## 2. Train the synthetic check

```powershell
py scripts\train_benchmark.py --profile synthetic
```

Open `experiments\results\synthetic-fixture\comparison.md` and `auroc.png`. The JSON field `results_are_research_benchmark` must be `false`.

## 3. Start the API

In a second terminal:

```powershell
cd Fraud-Detection
.\.venv\Scripts\Activate.ps1
py scripts\run_api.py
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8082/health
```

Score a small payment. The untrained heuristic should `ALLOW` it:

```powershell
Invoke-RestMethod http://127.0.0.1:8082/v1/transactions/score -Method Post -ContentType "application/json" -Body '{"transaction_id":"demo-1","amount":40,"account_key":"acct-demo","currency":"USD"}'
```

A very poor reputation score is a supplied number from 0 to 1. It is not a lookup of a real network address. This request should `BLOCK`:

```powershell
Invoke-RestMethod http://127.0.0.1:8082/v1/transactions/score -Method Post -ContentType "application/json" -Body '{"transaction_id":"demo-2","amount":40,"ip_reputation":0.05}'
```

A mid-range reputation should `STEP_UP`:

```powershell
Invoke-RestMethod http://127.0.0.1:8082/v1/transactions/score -Method Post -ContentType "application/json" -Body '{"transaction_id":"demo-3","amount":40,"ip_reputation":0.25}'
```

Missing amount:

```powershell
try { Invoke-RestMethod http://127.0.0.1:8082/v1/transactions/score -Method Post -ContentType "application/json" -Body '{"transaction_id":"demo-4"}' } catch { $_.Exception.Response.StatusCode.value__ }
```

That status should be 422.

The browser pages are:

- http://127.0.0.1:8082/docs
- http://127.0.0.1:8082/v1/dashboard

The dashboard lists transaction ids, scores, and actions. It does not list account keys.

## 4. Replay a small file

```powershell
py -c "import sys; sys.path.insert(0, 'src'); from fraud_engine.data.synthetic import make_synthetic_transactions; make_synthetic_transactions(30, seed=1).drop(columns=['isFraud']).to_csv('experiments/demo_stream.csv', index=False)"
py scripts\replay_transactions.py experiments\demo_stream.csv --limit 10
```

The replay script also drops `isFraud` if it is still present, so a label is not sent to the gateway.

## 5. Real benchmark, after you download a dataset

```powershell
py scripts\download_datasets.py ieee
py scripts\prepare_data.py ieee
py scripts\train_benchmark.py --profile dev --dataset ieee
```

Use the same pattern with `elliptic` or `dgraphfin` after those files are on disk. DGraphFin is a manual download. The steps are in `docs/DATASETS.md`.

Only a JSON file whose `results_are_research_benchmark` field is `true` may be presented as a dataset result.
