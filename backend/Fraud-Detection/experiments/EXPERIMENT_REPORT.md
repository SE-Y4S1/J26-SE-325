# Experiment report — Component 02

This report records runs that were executed on this machine. The IEEE-CIS and Elliptic sections are research results. The synthetic section is only a pipeline check.

## Question

Can a transaction be scored from its own features, from a short account sequence, and from earlier related transactions, and can those scores be turned into a fixed `ALLOW`, `STEP_UP`, or `BLOCK` action?

## IEEE-CIS development benchmark

Command, run from `Fraud-Detection` after the competition rules were accepted and the files were downloaded:

```powershell
py scripts\train_benchmark.py --profile dev --dataset ieee
```

`experiments/results/ieee-cis/metrics.json` has `results_are_research_benchmark: true`.

- Source files are in `data/raw/ieee-cis/`. Checksums are in `data/metadata/ieee_download.json`.
- Rows used: the earliest 20,000 labelled transactions after sorting by `TransactionDT`. The full labelled file is much larger. This subset is the early part of the year, not the whole competition.
- The official test CSV has no public `isFraud` labels, so the test metric uses a later time slice of the labelled file.
- Split: time-ordered, and no timestamp is shared across train, validation, and test.
- Fraud rates: training 0.0271, validation 0.0287, test 0.0320.
- Features: amount, distance, `C1`–`C14`, `D1`–`D15`, `V1`–`V20`, match flags, identity fields `id_01`–`id_11`, and train-only frequency encodings. `V21`–`V339` are not included.
- Imbalance: class weights on the training split only. The LSTM autoencoder is trained on legitimate training sequences. The test split is not resampled.
- Test size: 3,000 rows, of which 96 are fraud and 2,904 are legitimate.
- Gateway budgets, chosen on 1,457 legitimate validation rows: 5% may reach `STEP_UP` or `BLOCK`, and 1% may reach `BLOCK`.

| Model | AUROC | Average precision | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.7348 | 0.2600 | 0.1882 | 0.3646 | 0.2482 |
| Linear SVM | 0.7349 | 0.2423 | 0.1969 | 0.3958 | 0.2630 |
| Random forest | 0.8526 | 0.4812 | 0.2978 | 0.5521 | 0.3869 |
| Static GCN | 0.6331 | 0.0952 | 0.0952 | 0.2083 | 0.1307 |
| Static GAT | 0.6637 | 0.0902 | 0.0598 | 0.2188 | 0.0940 |
| LSTM autoencoder | 0.5954 | 0.0471 | 0.0590 | 0.1875 | 0.0898 |
| GNN-only | 0.5847 | 0.0743 | 0.0725 | 0.1979 | 0.1061 |
| Fused LSTM + GNN | 0.5847 | 0.0743 | 0.0725 | 0.1979 | 0.1061 |

Random forest is the strongest model on this subset. Precision stays modest because only about 3% of rows are fraud, so a useful detector still raises many legitimate alerts. AUROC is the more stable comparison here.

Logistic regression confusion matrix at its validation F1 threshold: TN 2,753, FP 151, FN 61, TP 35.

Gateway actions on the 3,000 test rows:

| Model | Fraud blocked | Legitimate challenged | Legitimate blocked |
| --- | ---: | ---: | ---: |
| Logistic regression | 18 / 96 | 177 / 2,904 | 39 / 2,904 |
| Linear SVM | 15 / 96 | 180 / 2,904 | 41 / 2,904 |
| Random forest | 45 / 96 | 212 / 2,904 | 45 / 2,904 |
| Static GCN | 12 / 96 | 154 / 2,904 | 32 / 2,904 |
| Static GAT | 6 / 96 | 207 / 2,904 | 44 / 2,904 |
| LSTM autoencoder | 18 / 96 | 287 / 2,904 | 287 / 2,904 |
| GNN-only and fused | 12 / 96 | 206 / 2,904 | 53 / 2,904 |

The fused mixture weight chosen on the earlier validation slice was 0 for the LSTM and 1 for the GNN, so the fused row copies the GNN-only scores. On this subset the validation slice preferred the weaker stream.

Adding the IEEE identity and device columns lowered logistic-regression test AUROC from 0.8033 to 0.7348, a lift of -0.0685. Those fields are anonymised benchmark fields. They did not help this particular model on the early 20,000-row window.

Adversarial rescoring of the trained logistic regression model:

| Simulation | Result |
| --- | --- |
| Camouflage | AUROC fell from 0.7348 to 0.6232. Fraud rows still blocked: 12 / 96. |
| Slow drift | AUROC fell from 0.7348 to 0.7297. Fraud rows still blocked: 18 / 96. |
| Structuring into 4 parts | 56 of 96 fraud cases became fully `ALLOW`. 18 cases still had at least one part `BLOCK`. |

Logistic single-row scoring on this test slice had a median latency of 0.136 ms over 30 calls. The batch figure is for 3,000 in-memory rows and is not an API load test.

The chart is `experiments/results/ieee-cis/auroc.png`. The trained gateway bundle is `experiments/results/ieee-cis/fraud_bundle.joblib`. Starting `scripts\run_api.py` loads that bundle when `artifacts\fraud_bundle.joblib` is absent.

## IEEE-CIS larger window

The same models were trained on the earliest 100,000 labelled rows. This does not replace the 20,000-row result.

```powershell
py scripts\train_benchmark.py --profile wider --dataset ieee
```

`experiments/results/ieee-cis-100000/metrics.json` has `results_are_research_benchmark: true` and `rows: 100000`. The subset note in that file says these are the earliest 100,000 labelled rows after sorting by `TransactionDT`, not the full competition year.

- Fraud rates: training 0.0268, validation 0.0251, test 0.0203.
- Test size: 15,000 rows, of which 305 are fraud and 14,695 are legitimate.
- Gateway budgets use 7,320 legitimate validation rows.
- Feature set and split rules match the 20,000-row run. `V21`–`V339` are still omitted.

| Model | AUROC | Average precision | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.7888 | 0.1357 | 0.1355 | 0.2918 | 0.1850 |
| Linear SVM | 0.7960 | 0.1416 | 0.1401 | 0.3115 | 0.1933 |
| Random forest | 0.8702 | 0.4113 | 0.2222 | 0.5770 | 0.3209 |
| Static GCN | 0.6108 | 0.0324 | 0.0470 | 0.2525 | 0.0793 |
| Static GAT | 0.5967 | 0.0339 | 0.0503 | 0.2623 | 0.0843 |
| LSTM autoencoder | 0.5138 | 0.0206 | 0.0214 | 0.0984 | 0.0351 |
| GNN-only | 0.6374 | 0.0387 | 0.0538 | 0.2721 | 0.0899 |
| Fused LSTM + GNN | 0.6374 | 0.0387 | 0.0538 | 0.2721 | 0.0899 |

Random forest remains the strongest model. The fused weight on the LSTM is again 0, and the LSTM score was not flipped, so the fused row copies the GNN-only scores. The validation slice still preferred the GNN stream.

Gateway actions on the 15,000 test rows:

| Model | Fraud blocked | Legitimate challenged | Legitimate blocked |
| --- | ---: | ---: | ---: |
| Logistic regression | 32 / 305 | 686 / 14,695 | 113 / 14,695 |
| Linear SVM | 34 / 305 | 699 / 14,695 | 114 / 14,695 |
| Random forest | 140 / 305 | 764 / 14,695 | 180 / 14,695 |
| Static GCN | 3 / 305 | 694 / 14,695 | 94 / 14,695 |
| Static GAT | 3 / 305 | 595 / 14,695 | 97 / 14,695 |
| LSTM autoencoder | 10 / 305 | 644 / 14,695 | 504 / 14,695 |
| GNN-only and fused | 2 / 305 | 647 / 14,695 | 87 / 14,695 |

The LSTM `BLOCK` threshold still flags many legitimate rows. That is a limitation of this small autoencoder, not a reporting error.

Adding the identity and device columns raised logistic-regression test AUROC from 0.7846 to 0.7888, a lift of +0.0042. The gain is small. No simulated biometric column was included.

Adversarial rescoring of the trained logistic regression model:

| Simulation | Result |
| --- | --- |
| Camouflage | AUROC fell from 0.7888 to 0.5869. Fraud rows still blocked: 0 / 305. |
| Slow drift | AUROC moved from 0.7888 to 0.7903. Fraud rows still blocked: 34 / 305. |
| Structuring into 4 parts | 46 of 100 sampled fraud cases became fully `ALLOW`. 18 cases still had at least one part `BLOCK`. |

Logistic single-row scoring on this test slice had a median latency of 0.380 ms over 30 calls. The batch figure is for 15,000 in-memory rows and is not an API load test.

The chart is `experiments/results/ieee-cis-100000/auroc.png`.

## Elliptic development benchmark

Command, after the Kaggle dataset terms were accepted and the archive was downloaded:

```powershell
py scripts\train_benchmark.py --profile dev --dataset elliptic
```

`experiments/results/elliptic/metrics.json` has `results_are_research_benchmark: true`.

- Source: `https://www.kaggle.com/datasets/ellipticco/elliptic-data-set`. Licence: CC BY-NC-ND 4.0. Do not publish modified copies.
- Files are under `data/raw/elliptic/elliptic_bitcoin_dataset/`. Checksums are in the metrics file and in `data/metadata/elliptic_download.json`.
- The development cap keeps labelled nodes from each time split. This run used an induced subgraph of 12,989 nodes, not the full Elliptic graph.
- Time split follows Weber et al.: steps 1–30 train, 31–34 validation, 35 and later test. Unknown nodes stay in the graph with label -1 and are left out of the metrics.
- Test labelled nodes in this subgraph: 401 illicit and 4,599 licit.
- The LSTM autoencoder and the fused model are skipped. Elliptic is a transaction graph, not a sequence of payments for one account. No score was invented for them.

| Model | AUROC | Average precision | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.5894 | 0.0869 | 0.1070 | 0.9751 | 0.1928 |
| Linear SVM | 0.6139 | 0.0929 | 0.1044 | 0.9052 | 0.1873 |
| Random forest | 0.7631 | 0.1468 | 0.1978 | 0.7556 | 0.3135 |
| Static GCN | 0.8628 | 0.3219 | 0.2722 | 0.6658 | 0.3864 |
| Static GAT | 0.7772 | 0.1778 | 0.1649 | 0.8279 | 0.2751 |
| GNN-only | 0.8246 | 0.3857 | 0.2525 | 0.5661 | 0.3492 |
| LSTM autoencoder | skipped |  |  |  |  |
| Fused LSTM + GNN | skipped |  |  |  |  |

Static GCN is the strongest ranker on this subgraph. Logistic regression and the linear SVM rank only modestly above chance, and their validation thresholds sit near the top of a saturated score, so the gateway blocks almost no illicit nodes (1 / 401 for both). The graph models block more illicit nodes at a lower legitimate-block rate: GCN 20 / 401 illicit blocked and 28 / 4,599 licit blocked; GNN-only 63 / 401 illicit blocked and 17 / 4,599 licit blocked.

The chart is `experiments/results/elliptic/auroc.png`.

## Synthetic pipeline check

An earlier command, `py scripts\train_benchmark.py --profile synthetic`, wrote `experiments/results/synthetic-fixture/`. That file has `results_are_research_benchmark: false`. Do not cite it as an IEEE result.

## What these runs do not cover

- DGraphFin is not on disk. Register at https://dgraph.xinye.com/, accept the terms, and place `DGraphFin.zip` in `data/raw/dgraphfin/`. Automatic download is not available, and no DGraphFin score was produced.
- The full IEEE labelled file, about 590,000 rows, was not used. `config/full.yaml` removes the row cap and would write `experiments/results/ieee-cis-full`. The largest executed IEEE window is 100,000 rows.
- `V21`–`V339` were left out to keep the runs small.
- Neural nets used hidden size 32 and at most 5 epochs.
- Elliptic metrics are for the 12,989-node development subgraph, not every node in the public graph.
- Structuring edits the scaled amount and scores each part without rebuilding the account graph. The 100,000-row structuring check used 100 fraud cases, not every fraud row in the test slice.
- The work is not committed yet.

## How to reproduce the IEEE development result

```powershell
cd J26-SE-325\Fraud-Detection
py scripts\train_benchmark.py --profile dev --dataset ieee
```

The IEEE CSV files must already be in `data\raw\ieee-cis\`.
