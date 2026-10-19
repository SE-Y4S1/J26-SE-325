# Experiment outputs

Each finished run writes a folder here:

- `metrics.json` — full executed summary
- `metrics.csv` — one row per model
- `comparison.md` — a short table
- `auroc.png` — a bar chart of scores that were actually computed
- `fraud_bundle.joblib` — trained weights, when training produced a bundle

`results_are_research_benchmark` is true only for an executed IEEE-CIS, Elliptic, or DGraphFin run. A synthetic folder is a pipeline check.

Folders written on this machine:

- `ieee-cis` — earliest 20,000 IEEE-CIS rows. The API loads this bundle.
- `ieee-cis-100000` — earliest 100,000 IEEE-CIS rows.
- `elliptic` — 12,989-node Elliptic development subgraph.
- `synthetic-fixture` — pipeline check. Do not cite it as a public-dataset result.

Model weight files are git-ignored. Small metric files can be kept locally. Do not commit raw datasets.
