# Datasets, access, licences, and citations

Component 02 does not ship raw data. The download script refuses to run without credentials and never asks you to paste a token into source code.

Place the Kaggle token, if you create one, at:

`%USERPROFILE%\.kaggle\kaggle.json`

or set the environment variable `KAGGLE_API_TOKEN`. Both are git-ignored. Do not commit them and do not paste them into a chat.

## A. IEEE-CIS Fraud Detection — primary benchmark

- Page: https://www.kaggle.com/competitions/ieee-fraud-detection/data
- Rules: https://www.kaggle.com/competitions/ieee-fraud-detection/rules
- Why it is primary: it is a labelled e-commerce transaction table with an optional identity file that contains anonymised device fields.

Access:

1. Create a free Kaggle account.
2. Open the rules page while logged in and accept the competition rules. The competition ended in 2019; Kaggle still shows a join or late-submission control.
3. In Kaggle account settings, create an API token. This downloads `kaggle.json`.
4. Move that file to `%USERPROFILE%\.kaggle\kaggle.json`.
5. From `Fraud-Detection`, with the virtual environment active:

```powershell
py -m pip install kaggle
py scripts\download_datasets.py ieee
py scripts\prepare_data.py ieee
```

Expected files inside `data/raw/ieee-cis/`:

- `train_transaction.csv` — labelled transactions
- `train_identity.csv` — anonymised device and identity fields for some of those transactions
- `test_transaction.csv` and `test_identity.csv` — no public fraud labels
- `sample_submission.csv`

The competition rules you accept control how the files may be used. Read them. Do not upload the CSV files or a modified copy to Git or to a public website. This academic project uses a time-based holdout of the labelled file because the official test labels are not public.

Citation:

Addison Howard, Bernadette Bouchon-Meunier, Catherine Huang, inversion, John Lei, Lynn@Vesta, Marcus2010, and Prof. Hussein Abbass. IEEE-CIS Fraud Detection. Kaggle, 2019. https://www.kaggle.com/competitions/ieee-fraud-detection

Columns used as model features include `TransactionAmt`, `C1`–`C14`, `D1`–`D15`, `V1`–`V20`, `M1`–`M9`, selected categoricals, and identity fields `id_01`–`id_11`, `DeviceType`, and `DeviceInfo` when the identity file is present. `TransactionDT` is used only to order the split. `card1` is a graph key and a train-only frequency feature, not a value printed in the decision log. `V21`–`V339` are left out of the development run to keep memory small; document that limit if you cite the score.

## B. DGraphFin — graph experiment, manual download

- Site: https://dgraph.xinye.com/
- Dataset page: https://dgraph.xinye.com/dataset
- Terms: https://dgraph.xinye.com/clause
- Paper: Xuanwen Huang, Yang Yang, Yang Wang, Chunping Wang, Zhisheng Zhang, Jiarong Xu, Lei Chen, and Michalis Vazirgiannis. DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection. NeurIPS Datasets and Benchmarks, 2022.

Access is not anonymous. PyTorch Geometric also refuses to download this file automatically.

1. Register on https://dgraph.xinye.com/.
2. Read and accept the dataset terms. Use is limited to non-commercial research and education. If a for-profit employer is involved, that employer must also be bound by the terms.
3. Download `DGraphFin.zip` from the dataset page after logging in.
4. Put the zip in `Fraud-Detection/data/raw/dgraphfin/DGraphFin.zip`.
5. Run:

```powershell
py scripts\download_datasets.py dgraphfin
py scripts\prepare_data.py dgraphfin
py scripts\train_benchmark.py --profile dev --dataset dgraphfin
```

The archive contains `dgraphfin.npz` with node features `x`, labels `y`, `edge_index`, and official `train_mask`, `valid_mask`, and `test_mask` index arrays. The published graph has about 3.7 million nodes. The default config keeps at most 10,000 labelled nodes, spread across the official masks, and induces edges inside that sample. A sampled score is not comparable to a paper that trains on the full graph. Labels other than 0 and 1 are excluded from the metrics.

## C. Elliptic Bitcoin transaction graph

- Research note: https://www.elliptic.co/insights/elliptic-dataset-cryptocurrency-financial-crime/
- Distribution used here: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- Licence: Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)

A Hugging Face reupload advertises an MIT licence. This project does not use that copy, because the Kaggle page states CC BY-NC-ND 4.0. Do not publish a modified version of the files. Local preprocessing for a private experiment stays on disk and is git-ignored.

Access:

1. Log in to Kaggle and open the dataset page.
2. Accept the dataset terms.
3. Use the same `kaggle.json` file as for IEEE-CIS.
4. Run:

```powershell
py scripts\download_datasets.py elliptic
py scripts\prepare_data.py elliptic
py scripts\train_benchmark.py --profile dev --dataset elliptic
```

Expected files:

- `elliptic_txs_features.csv` — transaction id, time step, then features, usually without a header
- `elliptic_txs_classes.csv` — `1` illicit, `2` licit, `unknown`
- `elliptic_txs_edgelist.csv`

Kaggle unpacks these into `data/raw/elliptic/elliptic_bitcoin_dataset/`. The loader searches that folder. Do not publish modified copies of the files. The licence is CC BY-NC-ND 4.0.

The loader maps illicit to the positive class and licit to the negative class. Unknown nodes can remain in the graph but are omitted from precision, recall, F1, and AUROC. Time steps 1–30 are training, 31–34 are validation, and 35 onward are test. Weber et al. tested from time step 35 and trained on 1–34. This project holds out 31–34 for threshold selection so the published test window is not used to choose the cut-point.

Citation:

M. Weber, G. Domeniconi, J. Chen, D. K. I. Weidele, C. Bellei, T. Robinson, and C. E. Leiserson. Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics. KDD Workshop on Anomaly Detection in Finance, 2019.

Also cite Elliptic, https://www.elliptic.co/.

The LSTM autoencoder is not trained on Elliptic or DGraphFin. Those tables are node graphs, not a sequence of payments for one account. The comparison file records that the LSTM and the fused model were skipped, with the reason. It does not invent a score.

## What is stored after a real download

`data/metadata/` receives a JSON file with the source URL, byte size, SHA-256 checksum, row count, missing-value fractions, duplicate identifiers, and the fraud rate. Raw and processed tables stay under `data/raw/` and `data/processed/`, which are git-ignored.

## Simulated signals

`add_label_free_simulated_biometrics` creates typing, pointer, and session columns from the transaction id only. Every such column is named `simulated_*` and the row is marked `origin=simulated`. These columns are not mixed into the IEEE feature matrix unless you add them yourself, and they must not be described as collected biometrics.

The API rejects a simulated block whose origin is anything other than `simulated`.

## Not used as substitutes

PaySim and the Kaggle credit-card fraud table are mentioned in the project topic form as examples. They are not downloaded here, and they are not used to fill in a missing IEEE, DGraphFin, or Elliptic file.
