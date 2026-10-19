"""Phase 6: precision of the per-stock news filter (keywords + relevance check).

Two steps, because precision needs a person:

    python -m evaluation.sentiment_precision export --symbols JKH.N0000 COMB.N0000 --per-symbol 40
        -> artifacts/sentiment_precision/sample.csv  (title, url, tier, matched terms, label="")
    # a person fills `label` with 1 (about the stock / its price drivers at that tier) or 0
    python -m evaluation.sentiment_precision score
        -> precision overall, per symbol and per tier, plus how many items were labelled

Items are sampled from what the filter KEPT (precision), at random, so the estimate is not
biased toward the highest-ranked sources. Social post text is not exported (titles of
news/official items only), matching the storage rule in news/aggregate.py.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "sentiment_precision"


def export(symbols: list[str], per_symbol: int, seed: int = 5) -> Path:
    from data import active_set
    from data.instruments import resolve
    from news.keywords import ensure_profile, load_config
    from news.pipeline import collect
    from news.relevance import filter_relevant

    rng = random.Random(seed)
    window = int(load_config().get("aggregation", {}).get("window_days", 14))
    rows = []
    for symbol in symbols:
        instrument = active_set.get_instrument(symbol) or resolve(symbol)
        profile = ensure_profile(instrument)
        raw, _ = collect(profile, window_days=window)
        kept, method = filter_relevant(raw, profile)
        for item in rng.sample(kept, k=min(per_symbol, len(kept))):
            social = item.item.kind.value == "social"
            rows.append({"symbol": symbol, "channel": item.item.channel, "kind": item.item.kind.value,
                         "tier": item.tier.value, "matched": "; ".join(item.matched), "relevance_method": method,
                         "title": None if social else item.item.title, "url": item.item.url, "label": ""})
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "sample.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def score(path: Path | None = None) -> pd.DataFrame:
    table = pd.read_csv(path or OUT / "sample.csv")
    labelled = table[table["label"].isin([0, 1, "0", "1"])].copy()
    labelled["label"] = labelled["label"].astype(int)
    parts = [("all", labelled)] + [(f"symbol={s}", g) for s, g in labelled.groupby("symbol")] + \
        [(f"tier={t}", g) for t, g in labelled.groupby("tier")]
    return pd.DataFrame([{"slice": name, "labelled": len(g), "precision": g["label"].mean()} for name, g in parts])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("--symbols", nargs="+", required=True)
    exp.add_argument("--per-symbol", type=int, default=40)
    sco = sub.add_parser("score")
    sco.add_argument("--path", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "export":
        print("wrote", export([s.upper() for s in args.symbols], args.per_symbol))
    else:
        print(score(args.path).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
