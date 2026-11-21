"""Build bars.parquet for the hybrid and Laya trainers from the local OHLCV cache only.

The Colab trainers fetch a broad CSE + Binance + Yahoo universe over the network. On the
development laptop that link is too slow, so this seeds their --out directory with the
26 resolved RQ1 symbols already cached on disk. collect_bars() then skips the network.

    uv run python -m experiments.build_local_bars --out artifacts/hybrid_training
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

from data.ingestion import bars_to_frame, fetch_ohlcv
from data.schema import AssetClass
from experiments.train_hybrid_heads import YEARS_OF_HISTORY


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("artifacts/hybrid_training"))
    args = parser.parse_args(argv)

    resolved = yaml.safe_load(Path("configs/resolved_universe.yaml").read_text(encoding="utf-8"))["symbols"]
    frames = []
    for symbol, entry in sorted(resolved.items()):
        asset_class = AssetClass(entry["asset_class"])
        # Same history depth the Colab trainer fetches, so the two runs are comparable.
        start = max(date.fromisoformat(entry["history_start"]),
                    date.fromisoformat(entry["history_end"]) - timedelta(days=365 * YEARS_OF_HISTORY))
        rows = fetch_ohlcv(symbol, asset_class, start, date.fromisoformat(entry["history_end"]))
        if rows:
            frame = bars_to_frame(rows)
            frame["symbol"] = symbol
            frame["asset_class"] = asset_class.value
            frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out / "bars.parquet", index=False)
    print(f"bars: {len(table)} rows, {table['symbol'].nunique()} symbols")
    print(table.groupby("asset_class")["symbol"].nunique().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
