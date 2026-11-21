"""Part D is resumable: a finished report is skipped on re-run. That is right for a disconnect and
WRONG after the extractor improves, because the 7,602 CSE records already extracted would keep
their old, mostly unusable output forever. Records are therefore stamped with the extractor
version, and a report extracted by an older one is redone.
"""

from __future__ import annotations

import json

from experiments.fundamentals_backfill import Checkpoint
from fundamentals import extract


def _write(path, rows) -> None:  # noqa: ANN001
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _checkpoint(tmp_path, rows):  # noqa: ANN001
    _write(tmp_path / "backfill_checkpoint.jsonl", rows)
    return Checkpoint(tmp_path)


def test_the_extractor_is_versioned_and_the_version_travels_in_every_record() -> None:
    assert extract.EXTRACTOR_VERSION >= 2, "version 1 is the extractor that produced the original backfill"
    statement = extract.Statement(template="general", period_months=3, unit=1.0)
    assert statement.to_dict()["extractor_version"] == extract.EXTRACTOR_VERSION


def test_cse_reports_from_an_older_extractor_are_redone_but_current_ones_are_skipped(tmp_path) -> None:
    v = extract.EXTRACTOR_VERSION
    checkpoint = _checkpoint(tmp_path, [
        {"key": "cse:AAA.N0000:1", "record": {"usable": False}},                                    # original: no version
        {"key": "cse:AAA.N0000:2", "record": {"usable": True, "extractor_version": v}},             # current
        {"key": "cse:AAA.N0000:3", "record": {"usable": True, "extractor_version": v - 1}},         # older
    ])

    assert "cse:AAA.N0000:2" in checkpoint.done
    assert "cse:AAA.N0000:1" not in checkpoint.done
    assert "cse:AAA.N0000:3" not in checkpoint.done


def test_failed_cse_reports_are_retried_after_an_extractor_upgrade_only(tmp_path) -> None:
    v = extract.EXTRACTOR_VERSION
    checkpoint = _checkpoint(tmp_path, [
        {"key": "cse:AAA.N0000:old-error", "error": "ValueError: boom"},
        {"key": "cse:AAA.N0000:new-error", "error": "ValueError: boom", "extractor_version": v},
    ])

    assert "cse:AAA.N0000:old-error" not in checkpoint.done          # a better extractor may now cope
    assert "cse:AAA.N0000:new-error" in checkpoint.done              # same extractor: would fail again


def test_us_filings_are_unaffected_by_the_pdf_extractor_version(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path, [{"key": "us:AAPL:0000320193-26-000001", "record": {"usable": True}}])
    assert "us:AAPL:0000320193-26-000001" in checkpoint.done


def test_new_rows_are_stamped_and_then_skipped(tmp_path) -> None:
    checkpoint = Checkpoint(tmp_path)
    checkpoint.add("cse:BBB.N0000:9", {"record": {"usable": True}})

    assert json.loads((tmp_path / "backfill_checkpoint.jsonl").read_text().splitlines()[-1])[
        "extractor_version"] == extract.EXTRACTOR_VERSION
    assert "cse:BBB.N0000:9" in Checkpoint(tmp_path).done


def test_the_export_keeps_only_the_newest_row_for_a_report(tmp_path) -> None:
    """A redone report appends a second row; the stale first one must not be exported too."""
    v = extract.EXTRACTOR_VERSION
    checkpoint = _checkpoint(tmp_path, [
        {"key": "cse:AAA.N0000:1", "record": {"usable": False, "confidence": 0.2}},
        {"key": "cse:BBB.N0000:5", "record": {"usable": True}, "extractor_version": v},
        {"key": "cse:AAA.N0000:1", "record": {"usable": True, "confidence": 0.9}, "extractor_version": v},
    ])

    rows = {row["key"]: row for row in checkpoint.rows()}
    assert len(checkpoint.rows()) == 2
    assert rows["cse:AAA.N0000:1"]["record"]["confidence"] == 0.9
