"""Score every RQ1 model on the same rows.

A walk-forward table compares models only if each is scored on the same (symbol, date) rows. The
first Colab RQ1 table was not: the LSTM scored 15,488 predictions, the foundation models 30,008 and
the hybrids 13,310. They forecast different dates (the hybrid needs a minimum context and strides
its walk-forward points; the foundation models roll per symbol), so a gap between two models mixed
how good each is with which rows each happened to cover.

`align_predictions` reduces every model to the rows ALL of them forecast and that have a realised
target; `aligned_metrics` scores them there and reports both row counts so the cost of aligning is
visible.
"""

from __future__ import annotations

import pandas as pd

from evaluation.metrics import forecast_metrics

KEYS = ["symbol", "timestamp"]
QUANTILE_COLUMNS = ["p10", "p50", "p90"]


def _usable(frame: pd.DataFrame, quantile_columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.dropna(subset=["target_return", *quantile_columns])
    return frame.drop_duplicates(KEYS, keep="last")


def align_predictions(
    predictions: dict[str, pd.DataFrame],
    *,
    quantile_columns: list[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], int]:
    """({model: its rows restricted to the common set}, number of common rows).

    A row is common when every model forecast it AND it has a realised target (the final rows of
    a series have none). Duplicate (symbol, date) rows are collapsed, so a model cannot count a
    row twice.
    """
    if not predictions:
        raise ValueError("need at least one model's predictions")
    columns = quantile_columns or QUANTILE_COLUMNS

    usable = {name: _usable(frame, columns) for name, frame in predictions.items()}
    common = None
    for frame in usable.values():
        index = pd.MultiIndex.from_frame(frame[KEYS])
        common = index if common is None else common.intersection(index)

    aligned = {}
    for name, frame in usable.items():
        keyed = frame.set_index(KEYS)
        kept = keyed.loc[keyed.index.isin(common)].reset_index().sort_values(KEYS).reset_index(drop=True)
        aligned[name] = kept
    return aligned, len(common)


def aligned_metrics(
    predictions: dict[str, pd.DataFrame],
    *,
    quantile_columns: list[str] | None = None,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> pd.DataFrame:
    """One row per model: the RQ1 metric bundle on the common rows, plus n_original and n_common."""
    columns = quantile_columns or QUANTILE_COLUMNS
    aligned, n_common = align_predictions(predictions, quantile_columns=columns)

    rows = []
    for name, frame in aligned.items():
        row = {"model": name, "n_original": len(_usable(predictions[name], columns)), "n_common": n_common}
        if n_common:
            row.update(forecast_metrics(frame["target_return"].to_numpy(), frame[columns].to_numpy(), quantiles))
        rows.append(row)
    return pd.DataFrame(rows)
