"""Human-readable outputs of a final evaluation: a budget table, a Markdown report and figures.
Aggregates only -- no row-level data leaves the cache."""

from __future__ import annotations

import csv
from pathlib import Path

LABELS = {
    "assurance": "Assurance engine (P002)",
    "assurance_no_confidence": "  ablation: no confidence",
    "assurance_no_disagreement": "  ablation: no disagreement",
    "assurance_no_context": "  ablation: no context",
    "learned_deferral": "Learned deferral (logistic)",
    "risk_band": "Risk only (fraud-gateway style)",
    "confidence_only": "Confidence only",
    "p001_retuned": "P001 thresholds, re-tuned",
    "p001_deployed": "P001 as deployed",
    "random_review": "Random review",
    "auto_all": "Automate everything",
}
ORDER = list(LABELS)


def pct(x: float, digits: int = 2) -> str:
    return f"{100 * x:.{digits}f}%"


def write_all(payload: dict, results: Path) -> None:
    write_budgets_csv(payload, results / "budgets.csv")
    write_report(payload, results / "REPORT.md")
    write_figures(payload, results / "figures")


def write_budgets_csv(payload: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["budget", "method", "incorrect_automatic_rate", "ci_low", "ci_high", "human_review_rate",
                         "verify_rate", "fraud_auto_approved", "fraud_auto_approved_amount", "legit_auto_rejected", "aurc"])
        for budget, block in payload["results"].items():
            for name in ORDER:
                m = block["methods"].get(name)
                if not m:
                    continue
                ci = m["ci95"]["incorrect_automatic_rate"]
                writer.writerow([budget, name, m["incorrect_automatic_rate"], ci[0], ci[1], m["human_review_rate"],
                                 m["verify_rate"], m["fraud_auto_approved"], round(m["fraud_auto_approved_amount"], 2),
                                 m["legit_auto_rejected"], m["aurc"]])


def write_report(payload: dict, path: Path) -> None:
    meta = payload["meta"]
    lines = [
        "# IEEE-CIS evaluation of the AI Decision Assurance Engine",
        "",
        f"Status: **{payload['status']}** · generated {payload['generated_at']} · policy {payload['policy']['id']} "
        f"v{payload['policy']['version']} `{payload['policy']['hash']}`",
        "",
        "## Data",
        f"- Rows per split: {meta['rows']} (time-ordered, no shared timestamps)",
        f"- Fraud rate per split: { {k: round(v, 4) for k, v in meta['fraud_rate'].items()} }",
        f"- Test set fingerprint: `{meta['test_index_digest']}`",
        f"- AI decision: reject when the calibrated ensemble probability ≥ {meta['decision_threshold']} (Bayes threshold for a 10:1 miss:false-alarm cost)",
        "",
        "Base-model quality on the calibration split (sanity check, not a result of this work):",
        "",
        "| Model | AUROC | Average precision |",
        "|---|---:|---:|",
    ]
    for name, q in meta["calibration_split_model_quality"].items():
        lines.append(f"| {name} | {q['auroc']:.4f} | {q['average_precision']:.4f} |")

    for budget, block in payload["results"].items():
        lines += ["", f"## Human-review budget {pct(float(budget), 0)}", "",
                  "| Method | Incorrect automatic decisions (95% CI) | Human review | Verify | Fraud auto-approved | Legit auto-rejected | AURC |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for name in ORDER:
            m = block["methods"].get(name)
            if not m:
                continue
            ci = m["ci95"]["incorrect_automatic_rate"]
            aurc = f"{m['aurc']:.4f}" if m["aurc"] is not None else "—"
            lines.append(f"| {LABELS[name].strip()} | {pct(m['incorrect_automatic_rate'], 3)} ({pct(ci[0], 3)}–{pct(ci[1], 3)}) | "
                         f"{pct(m['human_review_rate'])} | {pct(m['verify_rate'])} | {m['fraud_auto_approved']} | "
                         f"{m['legit_auto_rejected']} | {aurc} |")
        lines += ["", "Paired bootstrap, assurance engine minus comparator (negative = fewer incorrect automatic decisions):", "",
                  "| Comparator | Difference | 95% CI | P(assurance not better) |", "|---|---:|---|---:|"]
        for name, p in block["paired_vs_assurance"].items():
            lines.append(f"| {LABELS.get(name, name).strip()} | {pct(p['difference'], 3)} | {pct(p['ci95'][0], 3)} to {pct(p['ci95'][1], 3)} | {p['p_reference_not_better']:.3f} |")

    lines += [
        "",
        "## Assumptions (not measured)",
        "- VERIFY effectiveness: the primary metric counts VERIFY decisions as automatic (effectiveness 0). "
        "`verify_effectiveness_sweep` in metrics.json shows the rate if step-up caught 25/50/75% of errors.",
        "- Analyst accuracy: `analyst_accuracy_sweep` shows final error rates if analysts were 100/95/90% accurate.",
        "- Human-review rates on the test split differ slightly from the budget because thresholds were set on the tuning split.",
        "",
        "## Limitations",
        "- The offline AI is a calibrated ensemble trained here; online, Component 2's models make the decisions. "
        "The exported policy keeps the same formula, weights, thresholds and veto, but its online context sub-signals differ.",
        "- One dataset, one time period (IEEE-CIS 2019). The drift halves in metrics.json show stability within the test period only.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figures(payload: dict, folder: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    styles = {  # identity by line style and label, not colour alone; the method under test is emphasised
        "assurance": ("#047857", "-", 2.4),
        "learned_deferral": ("#404040", "--", 1.6),
        "confidence_only": ("#737373", "-.", 1.6),
        "risk_band": ("#737373", ":", 1.8),
        "assurance_no_disagreement": ("#a3a3a3", (0, (5, 2, 1, 2)), 1.4),
    }
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for name, curve in payload.get("curves_at_policy_budget", {}).items():
        if name not in styles or not curve:
            continue
        colour, dash, width = styles[name]
        xs = [100 * p["human_review_rate"] for p in curve]
        ys = [100 * p["incorrect_automatic_rate"] for p in curve]
        ax.plot(xs, ys, color=colour, linestyle=dash, linewidth=width, label=LABELS[name].strip())
        ax.annotate(LABELS[name].strip(), (xs[-1], ys[-1]), fontsize=7, color="#262626", xytext=(4, 0), textcoords="offset points", va="center")
    ax.set_xlabel("Decisions sent to human review (%)")
    ax.set_ylabel("Incorrect automatic decisions (% of all)")
    ax.set_title("Deferral curves on the held-out test split", fontsize=10, loc="left")
    ax.grid(color="#e5e5e5", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(folder / "deferral_curves.png")
    plt.close(fig)

    block = payload["results"][str(payload["policy_budget"])]
    names = [n for n in ORDER if n in block["methods"]]
    values = [100 * block["methods"][n]["incorrect_automatic_rate"] for n in names]
    lows = [100 * block["methods"][n]["ci95"]["incorrect_automatic_rate"][0] for n in names]
    highs = [100 * block["methods"][n]["ci95"]["incorrect_automatic_rate"][1] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150)
    colours = ["#047857" if n == "assurance" else "#a3a3a3" for n in names]
    ax.barh(range(len(names)), values, color=colours, height=0.6,
            xerr=[[v - lo for v, lo in zip(values, lows)], [hi - v for v, hi in zip(values, highs)]],
            error_kw={"linewidth": 0.8, "ecolor": "#404040"})
    ax.set_yticks(range(len(names)), [LABELS[n].strip() for n in names], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Incorrect automatic decisions (% of all), 95% CI")
    ax.set_title(f"At a {int(100 * payload['policy_budget'])}% human-review budget", fontsize=10, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(folder / "incorrect_automatic_at_budget.png")
    plt.close(fig)
