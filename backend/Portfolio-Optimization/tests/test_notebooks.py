"""Guards for the Colab notebooks.

A notebook only runs on a GPU runtime we do not have in CI, so these are string- and
syntax-level checks. They exist because each one pins something an audit found wrong:

  * a notebook promised "same code as Part A" but silently dropped the fundamentals input
  * the RQ1 table had no valid LoRA row, and the adapters were scored on their own training
    period
  * the SFT cell asked a T4 for bf16 and passed a deprecated 4-bit flag
  * a register snippet carried placeholder metrics that would have been written as provenance
  * the paired bootstrap resampled rows that share a date
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
NOTEBOOKS = sorted(p.name for p in EXPERIMENTS.glob("colab_*.ipynb"))


def _cells(name: str) -> list[dict]:
    return json.loads((EXPERIMENTS / name).read_text(encoding="utf-8"))["cells"]


def _code(name: str) -> str:
    return "\n".join("".join(c["source"]) for c in _cells(name) if c["cell_type"] == "code")


def _markdown(name: str) -> str:
    return "\n".join("".join(c["source"]) for c in _cells(name) if c["cell_type"] == "markdown")


def test_there_are_notebooks_to_check() -> None:
    assert len(NOTEBOOKS) >= 5


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_every_code_cell_is_valid_python(name: str) -> None:
    for index, cell in enumerate(_cells(name)):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # Jupyter shell/magic lines are not Python; blank them so the rest still compiles.
        cleaned = "\n".join("" if line.lstrip().startswith(("!", "%")) else line for line in source.splitlines())
        try:
            compile(cleaned, f"{name}[{index}]", "exec")
        except SyntaxError as exc:  # pragma: no cover - only on failure
            pytest.fail(f"{name} cell {index}: {exc}")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_every_cli_flag_a_notebook_passes_exists_in_a_script(name: str) -> None:
    known = set()
    for script in EXPERIMENTS.glob("*.py"):
        known |= set(re.findall(r"add_argument\(\s*[\"'](--[a-z0-9-]+)[\"']", script.read_text(encoding="utf-8")))
    # Flags of git and pip, which the bootstrap and install cells pass; not ours to define.
    external = {"--depth", "--branch", "--hard", "--force-reinstall", "--index-url"}
    used = set(re.findall(r"[\"'](--[a-z][a-z0-9-]+)[\"']", _code(name))) - external
    assert used <= known, f"{name} passes flags no script defines: {sorted(used - known)}"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_laya_is_pinned_to_the_version_whose_api_was_checked(name: str) -> None:
    for match in re.findall(r"PIP_PKGS\s*=\s*[\"']([^\"']*)[\"']", _code(name)):
        packages = match.split()
        if any(p.split("==")[0] == "laya" for p in packages):
            assert "laya==0.4.1" in packages, f"{name} leaves laya unpinned"


def test_pp1_passes_the_fundamentals_backfill_to_both_trainers() -> None:
    code = _code("colab_pp1.ipynb")
    assert code.count('"--backfill"') >= 2, "hybrid-heads and Laya must both receive --backfill"
    assert "NOT FOUND" in code, "a missing backfill must be reported, not silently zero-filled"


def test_finetune_scores_lora_only_on_folds_after_the_adapters_training_period() -> None:
    code = _code("colab_finetune.ipynb")
    assert "min_test_start" in code
    assert "adapters_meta.json" in code
    # Adapters carry their cutoff in the directory name, so a leaky full-history adapter left
    # on Drive by an earlier run can never be picked up and "skipped as done".
    assert "-pre" in code
    for label in ("timesfm_lora", "chronos_bolt_lora", "hybrid_lora"):
        assert label in code, f"RQ1 table is missing the {label} row"


def test_finetune_no_longer_trains_the_adapters_on_the_full_history() -> None:
    code = _code("colab_finetune.ipynb")
    assert not re.search(r"finetune\(\s*model_name,\s*features,", code), (
        "LoRA must be fit on the pre-cutoff slice, not the full feature table"
    )


def test_finetune_sft_is_t4_safe() -> None:
    code = _code("colab_finetune.ipynb")
    assert "bf16=True" not in code, "a T4 has no native bf16"
    assert "load_in_4bit=True," not in code.replace("four_bit_params", ""), "pass BitsAndBytesConfig instead"
    assert "BitsAndBytesConfig" in code
    assert "choose_precision" in code
    assert "assert_template_keeps_tool_turns" in code
    assert "SFT_SMOKE" in code


def test_finetune_register_snippet_carries_no_placeholder_provenance() -> None:
    text = _code("colab_finetune.ipynb") + _markdown("colab_finetune.ipynb")
    assert "'val_pinball': 0.0" not in text
    assert "2011" not in text, "hard-coded training dates would contradict the real window"
    assert "no evaluated" in text, "the snippet must refuse to register an unevaluated adapter"


def test_finetune_part_b_text_matches_where_the_trajectories_live() -> None:
    assert "Upload `artifacts/trajectories" not in _markdown("colab_finetune.ipynb")


def test_laya_notebook_bootstraps_by_date() -> None:
    code = _code("colab_laya_decision.ipynb")
    assert "paired_block_bootstrap" in code
    assert "rng.choice(d" not in code, "resampling rows that share a date understates the interval"
    assert 'groups=frame["date"]' in code
