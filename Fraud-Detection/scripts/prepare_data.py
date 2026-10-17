"""Validate a local dataset and write metadata. This does not invent rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_engine.data.dgraphfin import load_dgraphfin
from fraud_engine.data.elliptic import load_elliptic
from fraud_engine.data.ieee import load_ieee_frame, write_ieee_metadata
from fraud_engine.paths import METADATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a dataset that is already on disk.")
    parser.add_argument("dataset", choices=["ieee", "elliptic", "dgraphfin"])
    parser.add_argument("--max-rows", type=int, default=20000)
    args = parser.parse_args()
    try:
        if args.dataset == "ieee":
            _frame, metadata = load_ieee_frame(max_rows=args.max_rows)
            target = write_ieee_metadata(metadata)
        elif args.dataset == "elliptic":
            prepared = load_elliptic(max_nodes=args.max_rows)
            target = _write("elliptic.json", prepared)
        else:
            prepared = load_dgraphfin(max_nodes=args.max_rows)
            target = _write("dgraphfin.json", prepared)
    except FileNotFoundError as exc:
        print(exc)
        print("Nothing was validated because the source files are not on disk.")
        sys.exit(2)
    print(f"Wrote {target}")


def _write(name: str, prepared: dict) -> str:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    safe = {
        key: prepared[key]
        for key in ("dataset", "license", "source", "subset", "files", "notes")
        if key in prepared
    }
    safe["rows"] = int(len(prepared["y"]))
    target = METADATA_DIR / name
    target.write_text(json.dumps(safe, indent=2), encoding="utf-8")
    return str(target)


if __name__ == "__main__":
    main()
