"""Tests for the code behind the Colab-notebook audit fixes.

  * RQ5's paired bootstrap resampled ROWS, but test rows share dates and overlapping label
    windows, so its intervals were too narrow
  * the RQ1 table had no valid LoRA row: adapters were fit on the same period they would be
    scored on, and `get_forecaster("hybrid")` could not take an adapter at all
  * the SFT cell asked a T4 for bf16, passed a deprecated 4-bit flag, and never checked that
    the chat template kept the tool turns
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# --------------------------------------------------------------------------------------
# Paired bootstrap by group
# --------------------------------------------------------------------------------------


def test_block_bootstrap_is_much_wider_than_the_row_bootstrap_when_rows_share_a_date_effect() -> None:
    from evaluation.stats import paired_block_bootstrap

    rng = np.random.default_rng(0)
    dates = np.repeat(np.arange(20), 50)                     # 20 dates x 50 holdings
    effect = rng.normal(0.0, 1.0, 20)[dates]                  # everything on a date moves together
    a = effect + rng.normal(0.0, 0.01, dates.size)
    b = np.zeros(dates.size)

    _, lo_block, hi_block = paired_block_bootstrap(a, b, groups=dates, seed=1)
    _, lo_rows, hi_rows = paired_block_bootstrap(a, b, groups=np.arange(dates.size), seed=1)

    assert (hi_block - lo_block) > 3 * (hi_rows - lo_rows)


def test_block_bootstrap_mean_is_the_plain_mean_difference() -> None:
    from evaluation.stats import paired_block_bootstrap

    rng = np.random.default_rng(2)
    a, b = rng.normal(0.3, 1.0, 400), rng.normal(0.0, 1.0, 400)
    mean, lo, hi = paired_block_bootstrap(a, b, groups=np.repeat(np.arange(40), 10), seed=3)
    assert mean == pytest.approx(float((a - b).mean()))
    assert lo < mean < hi


def test_block_bootstrap_excludes_zero_for_a_consistent_effect() -> None:
    from evaluation.stats import paired_block_bootstrap

    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(30), 5)
    a = 0.5 + rng.normal(0.0, 0.05, groups.size)
    _, lo, hi = paired_block_bootstrap(a, np.zeros(groups.size), groups=groups, seed=5)
    assert lo > 0


def test_block_bootstrap_declines_to_answer_with_too_few_groups() -> None:
    from evaluation.stats import paired_block_bootstrap

    result = paired_block_bootstrap(np.ones(50), np.zeros(50), groups=np.repeat(np.arange(5), 10), seed=1)
    assert all(np.isnan(v) for v in result)


def test_block_bootstrap_is_deterministic_for_a_seed() -> None:
    from evaluation.stats import paired_block_bootstrap

    a = np.random.default_rng(6).normal(size=300)
    groups = np.repeat(np.arange(30), 10)
    assert paired_block_bootstrap(a, np.zeros(300), groups=groups, seed=9) == paired_block_bootstrap(
        a, np.zeros(300), groups=groups, seed=9
    )


def test_block_bootstrap_accepts_pandas_series_with_string_dates() -> None:
    from evaluation.stats import paired_block_bootstrap

    frame = pd.DataFrame({
        "a": np.random.default_rng(7).normal(0.2, 1.0, 200),
        "b": np.random.default_rng(8).normal(0.0, 1.0, 200),
        "date": np.repeat([f"2024-01-{d:02d}" for d in range(1, 21)], 10),
    })
    mean, lo, hi = paired_block_bootstrap(frame["a"], frame["b"], groups=frame["date"], seed=1)
    assert np.isfinite([mean, lo, hi]).all()


# --------------------------------------------------------------------------------------
# Walk-forward: evaluate only after a cutoff
# --------------------------------------------------------------------------------------


@pytest.fixture
def recorder(monkeypatch):
    from forecasting.base import ForecastResult

    class _Recorder:
        name, version = "recorder", "v1"

        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            self.kwargs = kwargs

        def fit(self, features, *, horizon, **kwargs) -> None:  # noqa: ANN001, ARG002
            return None

        def predict_quantiles(self, features, *, horizon, **kwargs):  # noqa: ANN001, ARG002
            frame = features.sort_values("timestamp")
            return ForecastResult(
                symbol=str(frame["symbol"].iloc[0]), horizon=horizon,
                quantiles=(0.1, 0.5, 0.9), values=np.zeros((1, 3)),
                timestamps=pd.DatetimeIndex([frame["timestamp"].iloc[-1]]),
                model_name=self.name, model_version=self.version,
            )

    monkeypatch.setattr("forecasting.base.get_forecaster", lambda name, **kw: _Recorder(**kw))
    return _Recorder


def _features(n: int = 2200) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.bdate_range("2015-01-01", periods=n),
        "symbol": "AAA", "close": 100.0, "target_return": 0.001,
    })


def _config():
    from evaluation.backtest import BacktestConfig

    return BacktestConfig(train_window_days=365, test_window_days=180, step_days=180, embargo_days=5)


def test_walk_forward_min_test_start_drops_earlier_folds_and_keeps_fold_numbers(recorder) -> None:
    import evaluation.backtest as bt

    everything = bt.run_walk_forward(_features(), "recorder", horizon=5, config=_config(), log_to_mlflow=False)
    cutoff = date(2019, 1, 1)
    later = bt.run_walk_forward(
        _features(), "recorder", horizon=5, config=_config(), log_to_mlflow=False, min_test_start=cutoff
    )

    assert later["fold"].nunique() < everything["fold"].nunique()
    assert pd.to_datetime(later["timestamp"]).min() >= pd.Timestamp(cutoff)
    assert pd.to_datetime(everything["timestamp"]).min() < pd.Timestamp(cutoff)
    # Fold numbers keep their meaning, so a checkpoint written with a cutoff still resumes.
    assert set(later["fold"]) <= set(everything["fold"])


def test_walk_forward_min_test_start_after_every_fold_is_an_error(recorder) -> None:
    import evaluation.backtest as bt

    with pytest.raises(ValueError, match="min_test_start"):
        bt.run_walk_forward(
            _features(), "recorder", horizon=5, config=_config(), log_to_mlflow=False,
            min_test_start=date(2099, 1, 1),
        )


# --------------------------------------------------------------------------------------
# Hybrid on a LoRA base
# --------------------------------------------------------------------------------------


def test_hybrid_forwards_adapter_path_to_its_base_model(monkeypatch) -> None:
    import forecasting.base as base

    seen: dict = {}

    class _Base:
        name = "timesfm"

        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            seen.update(kwargs)

    monkeypatch.setattr(base, "available_foundation_models", lambda: {"timesfm": True, "chronos": True})
    monkeypatch.setattr("forecasting.timesfm_adapter.TimesFMForecaster", _Base)

    hybrid = base.get_forecaster("hybrid", base_model="timesfm", adapter_path="/adapters/timesfm")
    assert seen == {"adapter_path": "/adapters/timesfm"}
    assert isinstance(hybrid.base, _Base)


def test_hybrid_without_an_adapter_still_builds_a_zero_shot_base(monkeypatch) -> None:
    import forecasting.base as base

    seen: dict = {"called": False}

    class _Base:
        name = "timesfm"

        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            seen["called"] = True
            seen.update(kwargs)

    monkeypatch.setattr(base, "available_foundation_models", lambda: {"timesfm": True, "chronos": True})
    monkeypatch.setattr("forecasting.timesfm_adapter.TimesFMForecaster", _Base)

    base.get_forecaster("hybrid", base_model="timesfm")
    assert seen == {"called": True}


# --------------------------------------------------------------------------------------
# SFT helpers
# --------------------------------------------------------------------------------------


def test_t4_is_given_fp16_never_bf16() -> None:
    from agent.sft import choose_precision

    precision = choose_precision((7, 5))                       # Tesla T4
    assert precision.fp16 and not precision.bf16
    assert precision.compute_dtype == "float16"


def test_ampere_and_newer_are_given_bf16() -> None:
    from agent.sft import choose_precision

    for capability in ((8, 0), (8, 9), (9, 0)):
        precision = choose_precision(capability)
        assert precision.bf16 and not precision.fp16 and precision.compute_dtype == "bfloat16"


def test_no_gpu_means_full_precision() -> None:
    from agent.sft import choose_precision

    precision = choose_precision(None)
    assert not precision.bf16 and not precision.fp16 and precision.compute_dtype == "float32"


def test_four_bit_params_are_nf4_with_the_requested_compute_dtype() -> None:
    from agent.sft import four_bit_params

    params = four_bit_params("float16")
    assert params["load_in_4bit"] is True
    assert params["bnb_4bit_quant_type"] == "nf4"
    assert params["bnb_4bit_compute_dtype"] == "float16"


@pytest.mark.parametrize(("version", "expected"), [
    ("4.44.2", "torch_dtype"), ("4.55.9", "torch_dtype"), ("4.56.0", "dtype"),
    ("4.57.1", "dtype"), ("5.0.0", "dtype"), ("4.9.0", "torch_dtype"),
])
def test_dtype_keyword_depends_on_the_transformers_version(version: str, expected: str) -> None:
    from agent.sft import dtype_kwarg

    assert dtype_kwarg(version) == expected


def _trajectory() -> dict:
    path = next(iter(sorted((Path(__file__).resolve().parents[1] / "artifacts" / "trajectories").glob("*.jsonl"))), None)
    if path is None:
        pytest.skip("no committed trajectories")
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


def test_render_messages_serialises_tool_calls_and_keeps_tool_results() -> None:
    from agent.sft import render_messages

    record = _trajectory()
    messages = render_messages(record)

    assert [m["role"] for m in messages] == [m["role"] for m in record["messages"]]
    called = next(m for m, raw in zip(messages, record["messages"], strict=True) if raw.get("tool_calls"))
    assert json.loads(called["content"]) == next(r["tool_calls"] for r in record["messages"] if r.get("tool_calls"))
    tool = next(m for m in messages if m["role"] == "tool")
    assert tool["content"]
    assert all(isinstance(m["content"], str) for m in messages)


def test_template_check_fails_when_the_template_drops_tool_turns() -> None:
    from agent.sft import assert_template_keeps_tool_turns

    record = _trajectory()

    def drops_tools(messages):
        return "\n".join(m["content"] for m in messages if m["role"] != "tool")

    with pytest.raises(ValueError, match="tool"):
        assert_template_keeps_tool_turns(drops_tools, record)


def test_template_check_passes_when_tool_text_survives() -> None:
    from agent.sft import assert_template_keeps_tool_turns

    assert_template_keeps_tool_turns(lambda messages: "\n".join(m["content"] for m in messages), _trajectory())


def test_library_versions_reports_installed_packages_and_none_for_missing_ones() -> None:
    from agent.sft import library_versions

    versions = library_versions(["numpy", "definitely-not-an-installed-package"])
    assert isinstance(versions["numpy"], str) and versions["numpy"]
    assert versions["definitely-not-an-installed-package"] is None


# --------------------------------------------------------------------------------------
# Choosing an SFT base model that the account can actually download
# --------------------------------------------------------------------------------------

GATED = "gated: accept the licence at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct and log in"


def test_the_first_accessible_candidate_wins_and_nothing_is_reported_skipped() -> None:
    from agent.sft import resolve_base_model

    model, skipped = resolve_base_model(["org/a", "org/b"], check=lambda repo: None)
    assert model == "org/a" and skipped == {}


def test_a_gated_candidate_is_skipped_with_its_reason_and_the_next_one_is_used() -> None:
    from agent.sft import resolve_base_model

    reasons = {"meta-llama/Llama-3.2-3B-Instruct": GATED}
    model, skipped = resolve_base_model(
        ["meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct"], check=lambda repo: reasons.get(repo)
    )
    assert model == "Qwen/Qwen2.5-3B-Instruct"
    assert skipped == {"meta-llama/Llama-3.2-3B-Instruct": GATED}


def test_no_accessible_candidate_is_an_error_that_lists_every_reason() -> None:
    from agent.sft import resolve_base_model

    with pytest.raises(RuntimeError) as excinfo:
        resolve_base_model(["org/a", "org/b"], check=lambda repo: f"{repo} is gated")
    assert "org/a is gated" in str(excinfo.value) and "org/b is gated" in str(excinfo.value)


def test_the_default_check_names_gating_as_the_reason(monkeypatch) -> None:
    import huggingface_hub
    from huggingface_hub.errors import GatedRepoError

    from agent.sft import repo_access_problem

    import httpx

    url = "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/resolve/main/config.json"
    response = httpx.Response(403, request=httpx.Request("HEAD", url), headers={"X-Error-Code": "GatedRepo"})

    def gated(repo_id, filename, **kwargs):  # noqa: ANN001, ANN003
        raise GatedRepoError("403 gated", response=response)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", gated)
    reason = repo_access_problem("meta-llama/Llama-3.2-3B-Instruct")

    assert reason and "gated" in reason
    assert "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct" in reason


def test_the_default_check_reports_no_problem_when_the_download_works(monkeypatch) -> None:
    import huggingface_hub

    from agent.sft import repo_access_problem

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo_id, filename, **kw: "/cache/config.json")
    assert repo_access_problem("Qwen/Qwen2.5-3B-Instruct") is None
