"""Phase 6: field-level accuracy of the CSE statement extractor against hand-checked values.

    python -m evaluation.extraction_accuracy [--spec evaluation/handchecked/extraction.yaml]

Per report: template detected correctly?, share of hand-checked fields extracted within 0.5%,
consistency checks passed, and the extractor's own confidence -- so the dissertation can show
both that the parser is accurate and that its confidence flag tracks its accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from fundamentals.extract import PER_SHARE_FIELDS, extract_statements

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 0.005


def _pages(entry: dict) -> dict[int, str]:
    if entry.get("pages_fixture"):
        raw = json.loads((ROOT / entry["pages_fixture"]).read_text(encoding="utf-8"))
        return {int(k): v for k, v in raw.items()}
    from datetime import datetime, timezone

    from fundamentals.extract import pdf_pages
    from fundamentals.reports import ReportRef, download_report

    ref = ReportRef(entry["symbol"], entry["id"], "quarterly", entry["id"], entry["url"],
                    datetime.now(timezone.utc), None)
    return pdf_pages(download_report(ref))


def evaluate(spec_path: Path) -> pd.DataFrame:
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    rows = []
    for entry in spec["reports"]:
        statement = extract_statements(_pages(entry), company_name=entry.get("company", ""))
        hits = []
        for field, printed in entry["fields"].items():
            expected = printed if field in PER_SHARE_FIELDS else printed * entry.get("unit", 1)
            got = statement.current.get(field)
            hits.append(got is not None and abs(got - expected) <= TOLERANCE * max(abs(expected), 1e-9))
        rows.append({
            "report": entry["id"], "template_expected": entry.get("template"), "template_detected": statement.template,
            "fields": len(hits), "field_accuracy": sum(hits) / len(hits) if hits else float("nan"),
            "checks_passed": sum(statement.checks.values()), "checks": len(statement.checks),
            "confidence": round(statement.confidence, 3), "usable": statement.usable,
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, default=ROOT / "evaluation" / "handchecked" / "extraction.yaml")
    args = parser.parse_args(argv)
    table = evaluate(args.spec)
    print(table.to_string(index=False))
    print(f"\nmean field accuracy {table['field_accuracy'].mean():.3f}; "
          f"template accuracy {(table['template_expected'] == table['template_detected']).mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
