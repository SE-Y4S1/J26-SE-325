"""Download public datasets when access has already been granted."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_engine.data.download import download_elliptic, download_ieee, prepare_dgraph_manual
from fraud_engine.paths import METADATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or validate Component 02 datasets.")
    parser.add_argument("dataset", choices=["ieee", "elliptic", "dgraphfin"])
    args = parser.parse_args()
    try:
        if args.dataset == "ieee":
            record = download_ieee()
        elif args.dataset == "elliptic":
            record = download_elliptic()
        else:
            record = prepare_dgraph_manual()
    except RuntimeError as exc:
        print(exc)
        print("No dataset was downloaded.")
        sys.exit(2)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    target = METADATA_DIR / f"{args.dataset}_download.json"
    target.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Wrote {target}")
    if record["missing"]:
        print(f"Missing expected files: {record['missing']}")
        sys.exit(2)
    print("Download step finished and the expected files are present.")


if __name__ == "__main__":
    main()
