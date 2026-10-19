# Dataset metadata

JSON files in this folder are written by `scripts/prepare_data.py` and `scripts/download_datasets.py` after a real file is on disk. They store source URLs, byte sizes, SHA-256 checksums, and validation summaries.

They do not store transaction rows. Do not copy raw CSV or NPZ files into this folder.
