"""The hybrid heads' holdout table (RQ1 input), shared by the Colab notebooks.

Each trained head writes `h<horizon>/meta.json` with pinball loss for the base model and for the
hybrid on a time-held-out final 20% of dates, overall and per asset class. This turns those into
one row per (horizon, asset class).

Two things are attached to the row itself rather than left to the reader:
  * a SMALL holdout ("indicative only"): a ratio of two pinball losses over a few dozen rows is
    noise, and it is the CSE class that lands here
  * the CSE one-year history limit: the chart endpoint returns about 250 sessions, so even a
    large CSE holdout comes from a single year (see data/cse_history.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MIN_HOLDOUT_ROWS = 100
LEGACY_GENERATION = "2.5 (legacy, unrecorded)"


def _note(asset_class: str, rows: float | None, min_rows: int) -> str:
    notes = []
    if rows is not None and rows < min_rows:
        notes.append(f"indicative only: {int(rows)} holdout rows (<{min_rows})")
    if asset_class == "cse_equity":
        notes.append("CSE history is about one year, so this holdout covers a single year")
    return "; ".join(notes)


def holdout_table(heads_dir: Path | str, *, min_rows: int = MIN_HOLDOUT_ROWS) -> pd.DataFrame:
    """One row per (horizon, asset class) from every `h*/meta.json` directly under `heads_dir`.

    Only top-level `h<digits>` directories are read, so a `smoke/` run alongside never leaks in.
    """
    rows = []
    for meta_path in sorted(Path(heads_dir).glob("h[0-9]*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        metrics = meta.get("metrics", {})
        generation = meta.get("base_generation") or (
            LEGACY_GENERATION if meta.get("base_model", "timesfm") == "timesfm" else "n/a"
        )
        for key in metrics:
            if not key.startswith("pinball_base_"):
                continue
            asset_class = key.removeprefix("pinball_base_")
            base = metrics[key]
            hybrid = metrics.get(f"pinball_hybrid_{asset_class}")
            holdout = metrics.get(f"holdout_rows_{asset_class}")
            rows.append({
                "horizon": meta["horizon"],
                "asset_class": asset_class,
                "base": base,
                "hybrid": hybrid,
                "rows": holdout,
                "improvement_%": 100 * (1 - hybrid / base) if hybrid is not None and base else np.nan,
                "base_model": meta.get("base_model"),
                "base_generation": generation,
                "note": _note(asset_class, holdout, min_rows),
            })
    return pd.DataFrame(rows)
