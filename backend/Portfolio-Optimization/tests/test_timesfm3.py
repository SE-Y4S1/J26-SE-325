"""TimesFM 3.0 support: generation choice, the adapter, head/base compatibility, LoRA scope.

TimesFM 3.0 (August 2026) ships in the same PyPI package as 2.5 (`timesfm` 3.0.x provides the
`timesfm` module for 2.5 and a new `timesfm3` module). Its weights are licensed
NON-COMMERCIAL / NON-PRODUCTION; 2.5's remain Apache-2.0. So the generation is an explicit,
recorded choice rather than something that changes silently with an upgrade.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from forecasting import timesfm_adapter as adapter


# --------------------------------------------------------------------------------------
# Choosing a generation
# --------------------------------------------------------------------------------------

def test_auto_prefers_3_0_when_it_is_installed() -> None:
    assert adapter.resolve_generation(None, has_v3=True, has_v25=True, env=None) == "3.0"
    assert adapter.resolve_generation("auto", has_v3=True, has_v25=True, env=None) == "3.0"


def test_auto_falls_back_to_2_5_when_only_the_older_code_is_installed() -> None:
    assert adapter.resolve_generation(None, has_v3=False, has_v25=True, env=None) == "2.5"


def test_the_environment_can_pin_the_generation_for_a_deployment() -> None:
    assert adapter.resolve_generation(None, has_v3=True, has_v25=True, env="2.5") == "2.5"
    assert adapter.resolve_generation(None, has_v3=True, has_v25=True, env="auto") == "3.0"


def test_an_explicit_request_beats_the_environment() -> None:
    assert adapter.resolve_generation("3.0", has_v3=True, has_v25=True, env="2.5") == "3.0"


def test_requesting_a_generation_that_is_not_installed_is_an_error_naming_the_fix() -> None:
    with pytest.raises(RuntimeError, match=r"timesfm\[torch\]>=3\.0\.2"):
        adapter.resolve_generation("3.0", has_v3=False, has_v25=True, env=None)


def test_no_timesfm_at_all_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="not installed"):
        adapter.resolve_generation(None, has_v3=False, has_v25=False, env=None)


def test_an_unknown_generation_is_rejected() -> None:
    with pytest.raises(ValueError, match="generation"):
        adapter.resolve_generation("4.0", has_v3=True, has_v25=True, env=None)


def test_repo_ids_and_licences_are_recorded_per_generation() -> None:
    assert adapter.REPO_IDS["3.0"] == "google/timesfm-3.0-pytorch"
    assert adapter.REPO_IDS["2.5"] == "google/timesfm-2.5-200m-pytorch"
    assert "non-commercial" in adapter.LICENCES["3.0"].lower()
    assert "apache" in adapter.LICENCES["2.5"].lower()


# --------------------------------------------------------------------------------------
# The 3.0 adapter, against a stand-in for the timesfm3 package
# --------------------------------------------------------------------------------------

@pytest.fixture
def fake_timesfm3(monkeypatch):
    """A `timesfm3` module with the real package's surface (ModelConfig, TimesFM3Forecaster,
    predict_batch yielding objects with `.forecast` and `.quantiles` of shape (H, 9))."""
    seen: dict = {"configs": [], "calls": []}

    class _Output:
        def __init__(self, forecast, quantiles) -> None:
            self.forecast, self.quantiles = forecast, quantiles

    class _Forecaster:
        def __init__(self, config) -> None:
            seen["configs"].append(config)

        def predict_batch(self, contexts, horizon, return_quantiles=False, **kwargs):  # noqa: ANN001, ANN003
            seen["calls"].append({"n": len(contexts), "horizon": horizon, **kwargs})
            for index, _ in enumerate(contexts):
                grid = np.tile(np.arange(1, 10) * 0.001 * (index + 1), (horizon, 1))
                yield _Output(grid[:, 4], grid)

    module = types.ModuleType("timesfm3")
    module.ModelConfig = lambda **kw: types.SimpleNamespace(**kw)
    module.TimesFM3Forecaster = _Forecaster
    monkeypatch.setitem(sys.modules, "timesfm3", module)
    monkeypatch.setattr(adapter, "_installed_generations", lambda: (True, True))
    monkeypatch.delenv("TIMESFM_GENERATION", raising=False)
    return seen


def _bars(symbol: str = "AAA", n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "timestamp": pd.bdate_range("2024-01-01", periods=n), "symbol": symbol,
        "close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
    })


def test_3_0_adapter_reports_its_generation_version_and_licence(fake_timesfm3) -> None:
    model = adapter.TimesFMForecaster()
    assert model.generation == "3.0"
    assert model.version == "timesfm-3p0-zeroshot"
    assert "non-commercial" in model.licence.lower()


def test_3_0_adapter_loads_the_3_0_checkpoint_and_returns_requested_quantiles(fake_timesfm3) -> None:
    model = adapter.TimesFMForecaster()
    result = model.predict_quantiles(_bars(), horizon=5, quantiles=(0.1, 0.5, 0.9))

    config = fake_timesfm3["configs"][0]
    assert config.checkpoint_path == "google/timesfm-3.0-pytorch"
    # The stand-in's last-step grid is 0.001..0.009 for the first series.
    np.testing.assert_allclose(result.values[0], [0.001, 0.005, 0.009])
    assert result.model_version == "timesfm-3p0-zeroshot"
    assert fake_timesfm3["calls"][0]["horizon"] == 5


def test_3_0_adapter_forecasts_a_batch_in_one_call_and_keeps_series_apart(fake_timesfm3) -> None:
    model = adapter.TimesFMForecaster()
    first, second = model.predict_quantiles_batch([_bars("AAA"), _bars("BBB")], horizon=5, quantiles=(0.5,))

    assert len(fake_timesfm3["calls"]) == 1 and fake_timesfm3["calls"][0]["n"] == 2
    assert (first.symbol, second.symbol) == ("AAA", "BBB")
    assert second.values[0, 0] == pytest.approx(2 * first.values[0, 0])    # per-series, not shared


def test_3_0_adapter_rejects_a_malformed_quantile_grid(monkeypatch, fake_timesfm3) -> None:
    class _Bad:
        forecast = np.zeros(5)
        quantiles = np.zeros((5, 3))                                      # not 9 levels

    monkeypatch.setattr(sys.modules["timesfm3"].TimesFM3Forecaster, "predict_batch",
                        lambda self, contexts, horizon, **kw: iter([_Bad()]))
    with pytest.raises(ValueError, match="quantile"):
        adapter.TimesFMForecaster().predict_quantiles(_bars(), horizon=5)


def test_a_lora_adapter_pins_the_generation_it_was_trained_on(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_installed_generations", lambda: (True, True))
    monkeypatch.delenv("TIMESFM_GENERATION", raising=False)

    # LoRA is implemented and verified for 2.5 only, so an adapter implies 2.5...
    assert adapter.TimesFMForecaster(adapter_path="/adapters/x").generation == "2.5"
    # ...and asking to put it on 3.0 is refused rather than silently loading mismatched weights.
    with pytest.raises(ValueError, match="LoRA"):
        adapter.TimesFMForecaster(adapter_path="/adapters/x", generation="3.0")


def test_2_5_keeps_its_original_version_string(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_installed_generations", lambda: (False, True))
    monkeypatch.delenv("TIMESFM_GENERATION", raising=False)
    assert adapter.TimesFMForecaster().version == "timesfm-2p5-zeroshot"


# --------------------------------------------------------------------------------------
# Hybrid heads must meet the base they were trained on
# --------------------------------------------------------------------------------------

class _Base:
    def __init__(self, name: str = "timesfm", generation: str | None = None) -> None:
        self.name = name
        if generation is not None:
            self.generation = generation


def test_a_head_trained_on_2_5_refuses_a_3_0_base() -> None:
    from forecasting.hybrid_model import BaseMismatchError, check_base_compatibility

    with pytest.raises(BaseMismatchError, match=r"2\.5.*3\.0"):
        check_base_compatibility({"base_model": "timesfm", "base_generation": "2.5"}, _Base(generation="3.0"))


def test_a_legacy_head_without_the_field_is_treated_as_2_5() -> None:
    from forecasting.hybrid_model import BaseMismatchError, check_base_compatibility

    with pytest.raises(BaseMismatchError, match="before TimesFM 3.0"):
        check_base_compatibility({"base_model": "timesfm"}, _Base(generation="3.0"))
    check_base_compatibility({"base_model": "timesfm"}, _Base(generation="2.5"))


def test_matching_or_generationless_bases_are_accepted() -> None:
    from forecasting.hybrid_model import check_base_compatibility

    check_base_compatibility({"base_model": "timesfm", "base_generation": "3.0"}, _Base(generation="3.0"))
    check_base_compatibility({"base_model": "chronos_bolt"}, _Base(name="chronos_bolt"))
    check_base_compatibility({"base_model": "baseline_lstm"}, _Base(name="baseline_lstm"))


def test_serving_builds_the_base_a_head_was_trained_on() -> None:
    from forecasting.serving import base_kwargs_for

    assert base_kwargs_for({"base_model": "timesfm"}) == {"generation": "2.5"}                 # legacy head
    assert base_kwargs_for({"base_model": "timesfm", "base_generation": "3.0"}) == {"generation": "3.0"}
    assert base_kwargs_for({"base_model": "chronos_bolt"}) == {}


def test_hybrid_forwards_a_generation_to_its_base(monkeypatch) -> None:
    import forecasting.base as base

    seen: dict = {}

    class _Base2:
        name = "timesfm"

        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            seen.update(kwargs)

    monkeypatch.setattr(base, "available_foundation_models", lambda: {"timesfm": True, "chronos": True})
    monkeypatch.setattr("forecasting.timesfm_adapter.TimesFMForecaster", _Base2)
    base.get_forecaster("hybrid", base_model="timesfm", generation="2.5")
    assert seen == {"generation": "2.5"}


# --------------------------------------------------------------------------------------
# Weights cache and environment report follow the generation
# --------------------------------------------------------------------------------------

def test_the_weights_check_looks_for_the_generation_that_will_actually_load(monkeypatch) -> None:
    import forecasting.base as base

    monkeypatch.setattr(base, "available_foundation_models", lambda: {"timesfm": True, "chronos": False})
    monkeypatch.setattr(adapter, "_installed_generations", lambda: (True, True))
    monkeypatch.delenv("TIMESFM_GENERATION", raising=False)
    cached = {"google/timesfm-3.0-pytorch"}
    monkeypatch.setattr(base, "weights_cached", lambda repo: repo in cached)

    assert base.usable_foundation_models()["timesfm"] is True            # auto -> 3.0, which is cached

    monkeypatch.setenv("TIMESFM_GENERATION", "2.5")
    assert base.usable_foundation_models()["timesfm"] is False           # 2.5 is not cached


def test_the_environment_report_records_both_generations() -> None:
    from forecasting.env_report import probe_forecasters

    probes = probe_forecasters()
    assert "timesfm3" in probes
    assert "timesfm[torch]>=3.0.2" in str(probes["timesfm3"]["distribution"])


# --------------------------------------------------------------------------------------
# LoRA stays on the verified generation
# --------------------------------------------------------------------------------------

def test_finetune_builds_timesfm_on_2_5_and_leaves_other_models_alone(monkeypatch) -> None:
    import forecasting.base as base
    from forecasting.finetune_lora import finetune

    recorded: list[tuple[str, dict]] = []

    class _Stop(Exception):
        pass

    def fake_get_forecaster(name, **kwargs):  # noqa: ANN001, ANN003
        recorded.append((name, kwargs))
        raise _Stop

    monkeypatch.setattr(base, "get_forecaster", fake_get_forecaster)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        for model in ("timesfm", "chronos_bolt"):
            with pytest.raises(_Stop):
                finetune(model, pd.DataFrame(), horizon=5, output_dir=Path(tmp) / model,
                         register=False, log_to_mlflow=False)

    assert recorded == [("timesfm", {"generation": "2.5"}), ("chronos_bolt", {})]


# --------------------------------------------------------------------------------------
# The REAL timesfm3 code, on a tiny random-weight model (no download, numbers meaningless)
# --------------------------------------------------------------------------------------

@pytest.fixture
def tiny_real_forecaster(tmp_path):
    """A genuine TimesFM3Forecaster with 2 small layers, loaded through the package's own
    local-safetensors path. Weights are random, so forecasts are noise; what is real is the
    package's batching, stitching and output shapes -- the contract the adapter relies on."""
    pytest.importorskip("timesfm3")
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file
    from timesfm3 import ModelConfig, TimesFM3Forecaster
    from timesfm3.torch import configs
    from timesfm3.torch.timesfm3_forecaster import _make_torch_model

    tiny = dict(
        residual_block_config=configs.ResidualBlockConfig(hidden_dims=32, output_dims=32, use_bias=False, activation="relu"),
        transformer_config=configs.StackedTransformersConfig(
            num_layers=2,
            transformer=configs.TransformerConfig(
                model_dims=32, hidden_dims=32, num_heads=4, attention_norm="rms", feedforward_norm="rms",
                qk_norm="rms", use_rope_seq=True, use_rope_var=True, use_bias=False, ff_activation="relu",
                deterministic=True,
            ),
        ),
    )
    torch.manual_seed(0)
    path = tmp_path / "tiny.safetensors"
    save_file({k: v.contiguous() for k, v in _make_torch_model(ModelConfig(**tiny)).state_dict().items()}, str(path))
    return TimesFM3Forecaster(config=ModelConfig(checkpoint_path=str(path), device="cpu", per_core_batch_size=4, **tiny))


@pytest.mark.parametrize("horizon", [5, 21, 70])      # 70 > one 64-step output patch: stitching
def test_the_adapter_handles_the_real_3_0_output_contract(monkeypatch, tiny_real_forecaster, horizon) -> None:
    monkeypatch.setattr(adapter, "_installed_generations", lambda: (True, True))
    monkeypatch.delenv("TIMESFM_GENERATION", raising=False)
    monkeypatch.setattr(adapter.TimesFMForecaster, "_load_v3", lambda self: tiny_real_forecaster)

    model = adapter.TimesFMForecaster()
    results = model.predict_quantiles_batch(
        [_bars("AAA", 200), _bars("BBB", 150), _bars("CCC", 90)], horizon=horizon, quantiles=(0.1, 0.5, 0.9)
    )

    assert [r.symbol for r in results] == ["AAA", "BBB", "CCC"]
    for result in results:
        assert result.values.shape == (1, 3)
        assert np.isfinite(result.values).all()
        assert (np.diff(result.values[0]) >= 0).all()          # quantiles never cross
        assert result.model_version == "timesfm-3p0-zeroshot"


def test_a_series_is_forecast_the_same_alone_or_in_a_batch(monkeypatch, tiny_real_forecaster) -> None:
    """The adapter's docstring promises batching changes nothing per series; with the real
    package code that is now checked rather than assumed."""
    monkeypatch.setattr(adapter, "_installed_generations", lambda: (True, True))
    monkeypatch.delenv("TIMESFM_GENERATION", raising=False)
    monkeypatch.setattr(adapter.TimesFMForecaster, "_load_v3", lambda self: tiny_real_forecaster)

    model = adapter.TimesFMForecaster()
    alone = model.predict_quantiles(_bars("AAA", 200), horizon=5, quantiles=(0.1, 0.5, 0.9))
    together = model.predict_quantiles_batch([_bars("ZZZ", 120), _bars("AAA", 200)], horizon=5,
                                             quantiles=(0.1, 0.5, 0.9))
    np.testing.assert_allclose(alone.values, together[1].values, rtol=1e-4, atol=1e-6)
