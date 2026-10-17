"""Comparison charts. Titles state whether the rows are a real benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_metric_chart(table: pd.DataFrame, metric: str, output: Path, title: str) -> Path | None:
    if metric not in table.columns or table.empty:
        return None
    usable = table.dropna(subset=[metric])
    if usable.empty:
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    positions = range(len(usable))
    axis.bar(positions, usable[metric].astype(float))
    axis.set_xticks(list(positions), list(usable["model"]), rotation=30, ha="right")
    axis.set_ylabel(metric)
    axis.set_title(title)
    axis.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(output, dpi=120)
    plt.close(figure)
    return output
