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


def test_laya_notebook_defaults_to_a_real_run_not_the_smoke_preset() -> None:
    code = _code("colab_laya_decision.ipynb")
    assert re.search(r'^RUN_SIZE\s*=\s*"pp1"', code, re.M), (
        'Run-all on "smoke" produced numbers the notebook itself labels "not a result"'
    )
    assert "smoke" in code, "the smoke preset must still exist for a pipeline check"


def test_finetune_records_library_versions_with_the_adapters() -> None:
    code = _code("colab_finetune.ipynb")
    assert "library_versions" in code
    assert "'versions'" in code, "adapters_meta.json must carry the library versions the run used"


# --------------------------------------------------------------------------------------
# TimesFM 3.0 and CSE data
# --------------------------------------------------------------------------------------

TIMESFM_NOTEBOOKS = ["colab_finetune.ipynb", "colab_hybrid_heads.ipynb", "colab_pp1.ipynb"]
HEAD_TRAINING_NOTEBOOKS = ["colab_hybrid_heads.ipynb", "colab_pp1.ipynb"]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_no_notebook_installs_the_superseded_timesfm_package(name: str) -> None:
    assert "timesfm[torch]==2.0.2" not in _code(name)


@pytest.mark.parametrize("name", TIMESFM_NOTEBOOKS)
def test_timesfm_notebooks_install_the_release_that_ships_3_0(name: str) -> None:
    assert "timesfm[torch]==3.0.2" in _code(name)


@pytest.mark.parametrize("name", HEAD_TRAINING_NOTEBOOKS)
def test_head_training_notebooks_choose_the_timesfm_generation_and_feed_cse_history(name: str) -> None:
    code = _code(name)
    assert '"--timesfm-generation"' in code
    assert '"--cse-history"' in code
    assert "TIMESFM_GENERATION" in code, "the generation is a named, visible choice"
    assert "non-commercial" in (code + _markdown(name)).lower(), "the 3.0 licence must be stated"


@pytest.mark.parametrize("name", HEAD_TRAINING_NOTEBOOKS)
def test_head_training_notebooks_report_through_the_shared_holdout_table(name: str) -> None:
    assert "holdout_table" in _code(name)


def test_laya_notebook_reads_the_accumulated_cse_history() -> None:
    code = _code("colab_laya_decision.ipynb")
    assert "cse_history=" in code and "j26_cse_history" in code


def test_finetune_scores_both_timesfm_generations_like_for_like() -> None:
    code = _code("colab_finetune.ipynb")
    assert "'timesfm3'" in code, "the latest generation needs its own zero-shot row"
    assert "'hybrid_2p5'" in code, "hybrid_lora is on 2.5, so it needs a 2.5 hybrid to be compared with"
    # `timesfm` and `timesfm_lora` must share a generation, or they no longer isolate fine-tuning.
    assert "'timesfm': ('timesfm', {'generation': '2.5'})" in code


def test_finetune_sft_falls_back_to_an_ungated_base_instead_of_crashing_on_a_gated_one() -> None:
    code = _code("colab_finetune.ipynb")
    assert "resolve_base_model" in code
    assert "meta-llama/Llama-3.2-3B-Instruct" in code, "Llama stays the preferred base (the TAF names it)"
    assert "Qwen/Qwen2.5-3B-Instruct" in code, "the ungated base the existing agent adapter was trained on"
    assert "base_model.txt" in code, "the base actually used must be recorded next to the adapter"


def test_finetune_sft_casts_trainable_weights_to_float32_before_training() -> None:
    """fp16 mixed precision unscales gradients with a GradScaler, which cannot handle bf16 grads."""
    code = _code("colab_finetune.ipynb")
    assert "ensure_fp32_trainable" in code and "parameter_dtypes" in code
    assert code.index("ensure_fp32_trainable") < code.index("trainer.train()")


def test_finetune_sft_does_not_block_on_a_browser_login_unless_asked() -> None:
    """notebook_login() waits for a browser approval; Qwen needs no token, so the default run must not."""
    code = _code("colab_finetune.ipynb")
    assert "HF_LOGIN = False" in code
    assert "if HF_LOGIN:" in code
    call = "    notebook_login()"                      # the indented call, not the comment that names it
    assert code.index("if HF_LOGIN:") < code.index(call) < code.index("resolve_base_model(BASE_MODELS)")


# --------------------------------------------------------------------------------------
# CSE training coverage: a smoke run's cache must never feed a full run
# --------------------------------------------------------------------------------------

def test_hybrid_heads_notebook_writes_to_a_fresh_directory_and_splits_per_class() -> None:
    code = _code("colab_hybrid_heads.ipynb")
    assert "j26_hybrid_v2" in code, "j26_hybrid holds caches built by a Laya smoke run (20 CSE names)"
    assert '"--split", "per-class"' in code


def test_pp1_notebook_splits_per_class() -> None:
    assert '"--split", "per-class"' in _code("colab_pp1.ipynb")


def test_laya_notebook_fetches_its_own_bars_instead_of_sharing_the_heads_cache() -> None:
    """Both notebooks used to share j26_hybrid/bars.parquet; whichever ran first fixed the CSE
    universe for the other. A Laya smoke run (20 CSE names) therefore capped the full head run."""
    code = _code("colab_laya_decision.ipynb")
    assert "collect_bars(HYBRID_DIR" not in code
    assert "LAYA_BARS_DIR" in code
    live = "\n".join(line for line in code.splitlines() if not line.lstrip().startswith("#"))   # comments may name it
    assert "j26_hybrid" not in live


def test_laya_notebook_states_the_real_size_of_the_head() -> None:
    assert "a few MB" not in _markdown("colab_laya_decision.ipynb")


def test_finetune_rescores_rq1_on_identical_rows_from_the_saved_fold_predictions() -> None:
    code = _code("colab_finetune.ipynb")
    assert "aligned_metrics" in code
    assert "rq1_forecast_aligned_" in code
    assert "_folds_" in code, "the rescoring reads the per-fold predictions already on Drive"
