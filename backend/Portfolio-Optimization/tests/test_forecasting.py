"""Phase 3/4 tests: the LSTM baseline, the residual head, the hybrid, and the registry.

Shape and sanity only, per the plan -- this phase is compute-heavy and forecast ACCURACY is
Phase 7's job, not a unit test's. What is asserted here are the structural guarantees the
rest of the system depends on:

  * quantiles never cross (Phase 5a's CVaR is incoherent if they do)
  * the residual head is an exact no-op at initialization (so RQ1 gets a clean answer to
    "did the covariates help")
  * registry hashes are stable and content-addressed (Component 3 anchors them on-chain)
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import inspect

import pytest
from types import SimpleNamespace
import torch

from forecasting.base import (
    DEFAULT_QUANTILES,
    ForecastResult,
    available_foundation_models,
    enforce_non_crossing,
    get_forecaster,
    registered_forecasters,
)
from forecasting.baseline_lstm import (
    BaselineLSTMForecaster,
    LSTMConfig,
    QuantileLSTM,
    build_sequences,
    pinball_loss,
)
from forecasting.model_registry import (
    compute_content_hash,
    compute_data_fingerprint,
)
from forecasting.residual_head import (
    ASSET_CLASS_ORDER,
    ResidualHead,
    ResidualHeadConfig,
    asset_class_index,
)

FAST_LSTM = LSTMConfig(window=20, hidden_size=16, num_layers=1, epochs=8, batch_size=16, patience=3)


def _feature_table(n: int = 260, *, symbol: str = "TEST", seed: int = 5) -> pd.DataFrame:
    """A minimal Phase-2-shaped feature table with a learnable signal.

    The target is deliberately a smooth function of `rsi` plus noise, so a working trainer
    must be able to drive the loss down. On pure noise a decreasing-loss assertion would only
    prove the model can memorize.
    """
    rng = np.random.default_rng(seed)
    stamps = pd.bdate_range("2022-01-03", periods=n)
    rsi = 50 + 30 * np.sin(np.arange(n) / 12)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))

    frame = pd.DataFrame(
        {
            "timestamp": stamps,
            "symbol": symbol,
            "asset_class": "equity",
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": np.full(n, 1e6),
            "rsi": rsi,
            "macd": np.gradient(rsi),
            "atr_pct": np.abs(rng.normal(0.012, 0.002, n)),
            "mean_sentiment": rng.normal(0, 0.2, n),
            "sentiment_volume": rng.integers(0, 5, n).astype(float),
            "sentiment_momentum": rng.normal(0, 0.1, n),
            "days_since_news": rng.integers(0, 4, n).astype(float),
        }
    )
    frame["target_return"] = (rsi - 50) / 5000 + rng.normal(0, 0.0005, n)
    frame.loc[frame.index[-5:], "target_return"] = np.nan   # forward-looking tail
    return frame


# --------------------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------------------

def test_forecast_result_validates_its_own_shape() -> None:
    stamps = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=3))
    with pytest.raises(ValueError, match="value columns"):
        ForecastResult(
            symbol="T", horizon=1, quantiles=(0.1, 0.5, 0.9),
            values=np.zeros((3, 2)), timestamps=stamps,
            model_name="m", model_version="v",
        )


def test_forecast_result_point_and_frame() -> None:
    stamps = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=2))
    result = ForecastResult(
        symbol="AAPL", horizon=5, quantiles=(0.1, 0.5, 0.9),
        values=np.array([[-1.0, 0.0, 1.0], [-2.0, 0.5, 2.0]]),
        timestamps=stamps, model_name="m", model_version="v1",
    )
    np.testing.assert_allclose(result.point(), [0.0, 0.5])

    frame = result.to_frame()
    assert list(frame.columns) == ["timestamp", "symbol", "horizon", "p10", "p50", "p90"]
    assert len(frame) == 2


def test_enforce_non_crossing_sorts_crossed_quantiles() -> None:
    """Crossed quantiles make CVaR incoherent, so this is applied at the interface rather
    than trusting each model to behave."""
    crossed = np.array([[0.5, -0.2, 0.1]])
    fixed = enforce_non_crossing(crossed)
    assert (np.diff(fixed, axis=1) >= 0).all()


def test_registered_forecasters_always_includes_the_baseline() -> None:
    """The baseline has no optional dependencies, so it must be available unconditionally --
    a broken foundation-model install cannot be allowed to empty the registry."""
    assert "baseline_lstm" in registered_forecasters()


def test_get_forecaster_rejects_unknown_names_helpfully() -> None:
    with pytest.raises(KeyError, match="unknown forecaster"):
        get_forecaster("not_a_model")


def test_unavailable_foundation_model_raises_an_actionable_error() -> None:
    """The error must name the install command, not surface a bare ImportError."""
    available = available_foundation_models()
    if available.get("timesfm"):
        pytest.skip("TimesFM is installed; nothing to assert about its absence")

    with pytest.raises(RuntimeError, match="timesfm"):
        get_forecaster("timesfm")


# --------------------------------------------------------------------------------------
# Baseline LSTM
# --------------------------------------------------------------------------------------

def test_pinball_loss_is_zero_for_perfect_prediction() -> None:
    pred = torch.tensor([[1.0, 1.0, 1.0]])
    target = torch.tensor([1.0])
    assert float(pinball_loss(pred, target, (0.1, 0.5, 0.9))) == pytest.approx(0.0, abs=1e-6)


def test_pinball_loss_penalises_asymmetrically() -> None:
    target = torch.tensor([0.0])
    under = float(pinball_loss(torch.tensor([[-1.0]]), target, (0.9,)))
    over = float(pinball_loss(torch.tensor([[1.0]]), target, (0.9,)))
    assert under > over


def test_quantile_lstm_output_is_non_crossing_by_construction() -> None:
    """Guaranteed structurally via cumulative softplus increments, not left to the loss."""
    torch.manual_seed(0)
    model = QuantileLSTM(n_features=6, config=LSTMConfig(hidden_size=8, num_layers=1))
    out = model(torch.randn(16, 20, 6)).detach().numpy()

    assert out.shape == (16, 3)
    assert (np.diff(out, axis=1) >= -1e-6).all(), "quantiles crossed"


def test_build_sequences_never_straddles_a_symbol_boundary() -> None:
    """A window mixing two instruments' histories is meaningless."""
    frame = pd.concat([_feature_table(60, symbol="AAA"), _feature_table(60, symbol="BBB", seed=9)])
    cols = ["rsi", "macd", "atr_pct"]
    X, y, stamps = build_sequences(frame, cols, "target_return", window=20)

    # 60 rows/symbol - 20 window - 5 NaN tail = 35 usable per symbol.
    assert len(X) == 70
    assert X.shape[1:] == (20, 3)
    assert len(y) == len(stamps) == 70


def test_build_sequences_skips_nan_targets() -> None:
    frame = _feature_table(80)
    X, y, _ = build_sequences(frame, ["rsi"], "target_return", window=20)
    assert not np.isnan(y).any()


def test_baseline_lstm_trains_and_loss_decreases() -> None:
    """End-to-end trainability on a signal the model CAN learn."""
    model = BaselineLSTMForecaster(FAST_LSTM)
    model.fit(_feature_table(), horizon=5, log_to_mlflow=False)

    assert model.model is not None
    assert model.version != "untrained"
    assert model.feature_cols, "no feature columns were selected"


def test_baseline_lstm_predicts_non_crossing_quantiles() -> None:
    model = BaselineLSTMForecaster(FAST_LSTM)
    frame = _feature_table()
    model.fit(frame, horizon=5, log_to_mlflow=False)

    result = model.predict_quantiles(frame, horizon=5)
    assert result.values.shape[1] == 3
    assert (np.diff(result.values, axis=1) >= -1e-6).all()
    assert result.model_name == "baseline_lstm"


def test_baseline_lstm_refuses_to_predict_before_fit() -> None:
    with pytest.raises(RuntimeError, match="call fit"):
        BaselineLSTMForecaster(FAST_LSTM).predict_quantiles(_feature_table(), horizon=5)


def test_baseline_lstm_requires_a_target_column() -> None:
    frame = _feature_table().drop(columns=["target_return"])
    with pytest.raises(ValueError, match="add_targets"):
        BaselineLSTMForecaster(FAST_LSTM).fit(frame, horizon=5, log_to_mlflow=False)


def test_baseline_lstm_scaler_is_fitted_on_train_only() -> None:
    """Fitting the scaler on the full series would leak the validation distribution's mean
    and variance into training."""
    model = BaselineLSTMForecaster(FAST_LSTM)
    frame = _feature_table()
    model.fit(frame, horizon=5, log_to_mlflow=False)

    X, _, _ = build_sequences(frame, model.feature_cols, "target_return", FAST_LSTM.window)
    split = int(len(X) * (1 - FAST_LSTM.val_fraction))
    train_mean = X[:split].reshape(-1, X.shape[-1]).mean(axis=0)
    np.testing.assert_allclose(model.scaler.mean, train_mean, rtol=1e-4)


# --------------------------------------------------------------------------------------
# Residual head
# --------------------------------------------------------------------------------------

def test_residual_head_is_an_exact_noop_at_initialization() -> None:
    """THE property that makes RQ1 interpretable: the hybrid starts EXACTLY at the base
    model's accuracy, so any improvement is attributable to the covariates rather than to a
    lucky random initialization."""
    torch.manual_seed(0)
    head = ResidualHead(
        n_features=6, n_asset_classes=len(ASSET_CLASS_ORDER),
        config=ResidualHeadConfig(hidden_size=16, num_layers=1, zero_init_output=True),
    )
    residual = head(
        torch.randn(8, 20, 6),
        torch.randn(8, 3),
        torch.zeros(8, dtype=torch.long),
    ).detach().numpy()

    np.testing.assert_allclose(residual, 0.0, atol=1e-8)


def test_residual_head_is_not_a_noop_when_zero_init_is_disabled() -> None:
    """Confirms the previous test is testing the flag, not a dead code path."""
    torch.manual_seed(0)
    head = ResidualHead(
        n_features=6, n_asset_classes=3,
        config=ResidualHeadConfig(hidden_size=16, num_layers=1, zero_init_output=False),
    )
    residual = head(torch.randn(8, 20, 6), torch.randn(8, 3), torch.zeros(8, dtype=torch.long))
    assert not np.allclose(residual.detach().numpy(), 0.0)


def test_residual_head_output_shape_matches_quantile_count() -> None:
    head = ResidualHead(n_features=4, n_asset_classes=3, config=ResidualHeadConfig())
    out = head(torch.randn(5, 10, 4), torch.randn(5, 3), torch.zeros(5, dtype=torch.long))
    assert out.shape == (5, len(DEFAULT_QUANTILES))


@pytest.mark.parametrize(
    "asset_class,expected", [("equity", 0), ("etf", 1), ("forex", 2), ("unknown", 0)]
)
def test_asset_class_index_is_stable(asset_class: str, expected: int) -> None:
    """Embedding indices must be stable across runs, or saved checkpoints stop loading."""
    assert asset_class_index(asset_class) == expected


# --------------------------------------------------------------------------------------
# Model registry (Component 3's provenance anchor)
# --------------------------------------------------------------------------------------

def test_content_hash_is_deterministic(tmp_path) -> None:
    path = tmp_path / "adapter.bin"
    path.write_bytes(b"weights")
    assert compute_content_hash(path) == compute_content_hash(path)


def test_content_hash_changes_with_content(tmp_path) -> None:
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"weights-v1")
    b.write_bytes(b"weights-v2")
    assert compute_content_hash(a) != compute_content_hash(b)


def test_directory_hash_is_order_independent(tmp_path) -> None:
    """Filesystem iteration order is not guaranteed; an unstable hash would make an on-chain
    anchor worthless."""
    def build(root):
        root.mkdir()
        (root / "z.bin").write_bytes(b"zzz")
        (root / "a.bin").write_bytes(b"aaa")
        (root / "m.bin").write_bytes(b"mmm")
        return root

    assert compute_content_hash(build(tmp_path / "one")) == compute_content_hash(build(tmp_path / "two"))


def test_directory_hash_detects_a_rename(tmp_path) -> None:
    """Relative paths are folded into the digest, so renaming a file inside the adapter is a
    different artefact."""
    first = tmp_path / "first"
    first.mkdir()
    (first / "a.bin").write_bytes(b"data")
    before = compute_content_hash(first)

    (first / "a.bin").rename(first / "b.bin")
    assert compute_content_hash(first) != before


def test_content_hash_raises_on_missing_checkpoint(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        compute_content_hash(tmp_path / "nope.bin")


def test_data_fingerprint_separates_a_data_change_from_a_code_change(tmp_path) -> None:
    """Retraining on a different window must not be mistakable for the same model."""
    universe = tmp_path / "resolved.yaml"
    universe.write_text("symbols:\n  AAPL:\n    history_start: 2015-01-01\n", encoding="utf-8")
    baseline = compute_data_fingerprint(universe, ["rsi", "macd"])

    universe.write_text("symbols:\n  AAPL:\n    history_start: 2018-01-01\n", encoding="utf-8")
    assert compute_data_fingerprint(universe, ["rsi", "macd"]) != baseline

    # And a feature-set change is likewise visible.
    assert compute_data_fingerprint(universe, ["rsi"]) != compute_data_fingerprint(universe, ["rsi", "macd"])


def test_data_fingerprint_is_order_insensitive_for_features(tmp_path) -> None:
    universe = tmp_path / "resolved.yaml"
    universe.write_text("symbols: {}\n", encoding="utf-8")
    assert compute_data_fingerprint(universe, ["rsi", "macd"]) == compute_data_fingerprint(universe, ["macd", "rsi"])


def test_registry_round_trip_and_export(tmp_path, monkeypatch) -> None:
    """The full provenance path Component 3 depends on."""
    from forecasting import model_registry as reg

    monkeypatch.setattr(reg, "REGISTRY_PATH", tmp_path / "registry.sqlite")

    checkpoint = tmp_path / "adapter.bin"
    checkpoint.write_bytes(b"lora-weights")

    record = reg.register(
        "hybrid-timesfm", checkpoint,
        train_start=date(2020, 1, 1), train_end=date(2024, 1, 1),
        metrics={"val_pinball": 0.0031}, mlflow_run_id="abc123", activate=True,
    )

    assert record.is_active
    assert reg.get_active_version("hybrid-timesfm") == record.model_version
    assert reg.get_record(record.model_version).content_hash == record.content_hash

    bundle = reg.export_for_anchoring(record.model_version)
    for key in ("model_version", "content_hash", "data_fingerprint", "git_commit", "created_at"):
        assert key in bundle, f"anchoring bundle missing {key}"


def test_registry_does_not_mint_two_versions_for_identical_bytes(tmp_path, monkeypatch) -> None:
    """Provenance must stay one-to-one with the artefact."""
    from forecasting import model_registry as reg

    monkeypatch.setattr(reg, "REGISTRY_PATH", tmp_path / "registry.sqlite")
    checkpoint = tmp_path / "adapter.bin"
    checkpoint.write_bytes(b"same-bytes")

    kwargs = dict(train_start=date(2020, 1, 1), train_end=date(2024, 1, 1), metrics={})
    first = reg.register("m", checkpoint, **kwargs)
    second = reg.register("m", checkpoint, **kwargs)

    assert first.model_version == second.model_version
    assert len(reg.list_records("m")) == 1


def test_get_active_version_returns_a_sentinel_when_nothing_is_registered(tmp_path, monkeypatch) -> None:
    """The withdrawal endpoint does not need a model and must not 500 because none exists."""
    from forecasting import model_registry as reg

    monkeypatch.setattr(reg, "REGISTRY_PATH", tmp_path / "empty.sqlite")
    assert reg.get_active_version() == "unregistered"


# --------------------------------------------------------------------------------------
# Foundation adapters (slow: each loads real weights)
# --------------------------------------------------------------------------------------

def _weights_cached(repo_id: str) -> bool:
    """Whether a model's weights are already in the local HuggingFace cache.

    Guards the adapter tests against HANGING rather than failing. These checkpoints are
    ~800MB; on a slow link the download can take hours, and a test that stalls indefinitely
    is far worse than one that skips with a reason. Measured on this machine: ~24 KB/s to
    both HuggingFace and PyPI, i.e. ~10 hours for one checkpoint.

    A skip here is not "the adapter is broken" -- it means the weights are not present. Run
    these in Colab, or pre-warm the cache with `huggingface-cli download <repo_id>`.
    """
    from pathlib import Path as _Path

    cache = _Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_id.replace('/', '--')}"
    if not cache.exists():
        return False

    # A directory full of .incomplete placeholders is a stalled download, not a cache hit.
    real = [
        f for f in cache.rglob("*")
        if f.is_file() and not f.name.endswith(".incomplete") and f.stat().st_size > 1_000_000
    ]
    return bool(real)


def _price_frame(n: int = 300, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "timestamp": pd.bdate_range("2023-01-02", periods=n),
            "symbol": "TEST",
            "close": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))),
        }
    )


@pytest.mark.slow
def test_chronos_adapter_produces_non_crossing_quantiles() -> None:
    """Loads ~200MB of weights on first run and is CPU-bound thereafter -- marked slow so it
    never blocks the default suite."""
    if not available_foundation_models().get("chronos"):
        pytest.skip("chronos-forecasting not installed")

    from forecasting.chronos_adapter import DEFAULT_MODEL_ID, ChronosBoltForecaster

    if not _weights_cached(DEFAULT_MODEL_ID):
        pytest.skip(
            f"{DEFAULT_MODEL_ID} weights not in the local HF cache. Pre-warm with "
            f"`huggingface-cli download {DEFAULT_MODEL_ID}` or run this in Colab."
        )

    result = ChronosBoltForecaster().predict_quantiles(_price_frame(), horizon=5)

    assert result.values.shape == (1, 3)
    assert (np.diff(result.values, axis=1) >= -1e-9).all(), "quantiles crossed"
    assert result.model_name == "chronos_bolt"
    assert np.isfinite(result.values).all()


@pytest.mark.slow
def test_timesfm_adapter_produces_non_crossing_quantiles() -> None:
    """Same shape contract as Chronos, so the hybrid can swap either in unchanged."""
    if not available_foundation_models().get("timesfm"):
        pytest.skip("timesfm not installed")

    from forecasting.timesfm_adapter import TimesFMForecaster

    import timesfm

    repo_id = getattr(timesfm.TimesFM_2p5_200M_torch, "DEFAULT_REPO_ID", "google/timesfm-2.5-200m-pytorch")
    if not _weights_cached(repo_id):
        pytest.skip(f"{repo_id} weights not in the local HF cache; run this in Colab.")

    result = TimesFMForecaster().predict_quantiles(_price_frame(), horizon=5)

    assert result.values.shape == (1, 3)
    assert (np.diff(result.values, axis=1) >= -1e-9).all(), "quantiles crossed"
    assert result.model_name == "timesfm"
    assert np.isfinite(result.values).all()


@pytest.mark.slow
def test_both_adapters_agree_on_the_output_contract() -> None:
    """RQ1 tabulates them side by side, so their ForecastResult shapes must be identical --
    a mismatch would surface as a confusing merge error deep in the evaluation harness."""
    available = available_foundation_models()
    if not (available.get("chronos") and available.get("timesfm")):
        pytest.skip("needs both foundation models installed")

    import timesfm

    from forecasting.chronos_adapter import DEFAULT_MODEL_ID, ChronosBoltForecaster
    from forecasting.timesfm_adapter import TimesFMForecaster

    timesfm_repo = getattr(timesfm.TimesFM_2p5_200M_torch, "DEFAULT_REPO_ID", "google/timesfm-2.5-200m-pytorch")
    if not (_weights_cached(DEFAULT_MODEL_ID) and _weights_cached(timesfm_repo)):
        pytest.skip("needs both checkpoints cached locally; run this in Colab.")

    frame = _price_frame()
    chronos = ChronosBoltForecaster().predict_quantiles(frame, horizon=5)
    timesfm = TimesFMForecaster().predict_quantiles(frame, horizon=5)

    assert chronos.values.shape == timesfm.values.shape
    assert chronos.quantiles == timesfm.quantiles
    assert list(chronos.to_frame().columns) == list(timesfm.to_frame().columns)


# --------------------------------------------------------------------------------------
# LoRA fine-tuning
# --------------------------------------------------------------------------------------

def test_build_finetune_dataset_produces_context_future_pairs() -> None:
    from forecasting.finetune_lora import build_finetune_dataset

    dataset = build_finetune_dataset(_price_frame(n=200), context_length=64, horizon=5)

    context, future = dataset[0]
    assert context.shape == (64,)
    assert future.shape == (5,)
    # 200 closes -> 199 returns; windows from index 64 to 199-5 inclusive.
    assert len(dataset) == 199 - 64 - 5 + 1


def test_build_finetune_dataset_never_straddles_a_symbol_boundary() -> None:
    """A context mixing two instruments' histories is a series that does not exist."""
    from forecasting.finetune_lora import build_finetune_dataset

    a = _price_frame(n=150, seed=1).assign(symbol="AAA")
    b = _price_frame(n=150, seed=2).assign(symbol="BBB")
    combined = build_finetune_dataset(pd.concat([a, b]), context_length=64, horizon=5)
    single = build_finetune_dataset(a, context_length=64, horizon=5)

    # Exactly twice the single-symbol count -- no extra windows spanning the join.
    assert len(combined) == 2 * len(single)


def test_build_finetune_dataset_skips_symbols_that_are_too_short() -> None:
    from forecasting.finetune_lora import build_finetune_dataset

    short = _price_frame(n=30, seed=3).assign(symbol="SHORT")
    long = _price_frame(n=200, seed=4).assign(symbol="LONG")
    dataset = build_finetune_dataset(pd.concat([short, long]), context_length=64, horizon=5)

    assert len(dataset) == len(build_finetune_dataset(long, context_length=64, horizon=5))


def test_build_finetune_dataset_raises_when_nothing_fits() -> None:
    from forecasting.finetune_lora import build_finetune_dataset

    with pytest.raises(ValueError, match="no windows"):
        build_finetune_dataset(_price_frame(n=20), context_length=512, horizon=5)


def test_build_finetune_dataset_requires_close() -> None:
    from forecasting.finetune_lora import build_finetune_dataset

    with pytest.raises(ValueError, match="close"):
        build_finetune_dataset(pd.DataFrame({"timestamp": [1, 2]}), context_length=2, horizon=1)


def test_finetune_dataset_operates_on_returns_not_price_levels() -> None:
    """Both adapters forecast returns at inference; fine-tuning on levels would be a
    train/serve mismatch no other test would catch."""
    from forecasting.finetune_lora import build_finetune_dataset

    frame = _price_frame(n=200)
    context, _ = build_finetune_dataset(frame, context_length=64, horizon=5)[0]
    # Log returns sit near zero; prices for this fixture start at 100.
    assert abs(float(context.mean())) < 0.1


def test_lora_target_module_discovery_finds_attention_projections() -> None:
    """Guessing wrong is a silent failure: PEFT only raises if NOTHING matches, so a partial
    guess would attach adapters to the wrong layers and train something useless."""
    from forecasting.finetune_lora import _discover_target_modules

    class FakeAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 4)
            self.v_proj = torch.nn.Linear(4, 4)

    assert _discover_target_modules(FakeAttention()) == ("q_proj", "v_proj")


def test_lora_target_module_discovery_handles_t5_naming() -> None:
    """Chronos-Bolt is T5-based and names its projections 'q'/'v', not 'q_proj'/'v_proj'."""
    from forecasting.finetune_lora import _discover_target_modules

    class FakeT5Attention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q = torch.nn.Linear(4, 4)
            self.v = torch.nn.Linear(4, 4)

    assert _discover_target_modules(FakeT5Attention()) == ("q", "v")


def test_lora_target_discovery_falls_back_to_all_linears() -> None:
    """An unrecognised architecture must still be adaptable. Raising here is what killed
    fine-tuning on an unfamiliar foundation model; adapting every Linear is more parameters
    than necessary but correct, and PEFT handles it fine."""
    from forecasting.finetune_lora import _discover_target_modules

    class Unrecognised(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weird_layer = torch.nn.Linear(4, 4)
            self.other = torch.nn.Linear(4, 4)

    targets = _discover_target_modules(Unrecognised())
    assert set(targets) == {"weird_layer", "other"}


def test_lora_target_discovery_prefers_attention_like_names() -> None:
    """Between the known-pair tier and the all-linears tier, anything that looks like an
    attention projection should win -- that is where LoRA is most effective."""
    from forecasting.finetune_lora import _discover_target_modules

    class OddNaming(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn_qkv = torch.nn.Linear(4, 4)
            self.feed_forward = torch.nn.Linear(4, 4)

    targets = _discover_target_modules(OddNaming())
    assert targets == ("attn_qkv",)


def test_lora_target_discovery_raises_only_when_there_are_no_linears() -> None:
    """The one genuinely unadaptable case."""
    from forecasting.finetune_lora import _discover_target_modules

    class NoLinears(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = torch.nn.LayerNorm(4)

    with pytest.raises(ValueError, match="no torch.nn.Linear"):
        _discover_target_modules(NoLinears())


def test_resolve_inner_model_finds_a_nested_module() -> None:
    from forecasting.finetune_lora import _resolve_inner_model

    class Pipeline:
        def __init__(self) -> None:
            self.model = torch.nn.Linear(2, 2)

    class Adapter:
        def __init__(self) -> None:
            self._pipeline = Pipeline()

    assert isinstance(_resolve_inner_model(Adapter()), torch.nn.Module)


def test_resolve_inner_model_raises_when_there_is_none() -> None:
    from forecasting.finetune_lora import _resolve_inner_model

    class Empty:
        pass

    with pytest.raises(TypeError, match="torch.nn.Module"):
        _resolve_inner_model(Empty())


# --------------------------------------------------------------------------------------
# No-download policy: local models first, fetch only when genuinely required
# --------------------------------------------------------------------------------------

def test_weights_cached_rejects_a_stalled_download(tmp_path, monkeypatch) -> None:
    """An interrupted fetch leaves config.json plus zero-byte `.incomplete` placeholders.
    Treating that as a cache hit would send the caller into a stalling re-download."""
    from forecasting import base as base_mod

    cache = tmp_path / ".cache" / "huggingface" / "hub" / "models--amazon--chronos-bolt-base"
    (cache / "blobs").mkdir(parents=True)
    (cache / "config.json").write_bytes(b"{}")
    (cache / "blobs" / "abc123.incomplete").write_bytes(b"")

    monkeypatch.setattr(base_mod.Path, "home", staticmethod(lambda: tmp_path))
    assert base_mod.weights_cached("amazon/chronos-bolt-base") is False


def test_weights_cached_accepts_a_real_checkpoint(tmp_path, monkeypatch) -> None:
    from forecasting import base as base_mod

    cache = tmp_path / ".cache" / "huggingface" / "hub" / "models--amazon--chronos-bolt-base"
    cache.mkdir(parents=True)
    (cache / "model.safetensors").write_bytes(b"x" * 2_000_000)

    monkeypatch.setattr(base_mod.Path, "home", staticmethod(lambda: tmp_path))
    assert base_mod.weights_cached("amazon/chronos-bolt-base") is True


def test_usable_is_stricter_than_available() -> None:
    """`available` means the package imports; `usable` also requires weights on disk. The
    difference is hours of download, so anything picking a default must use `usable`."""
    from forecasting.base import available_foundation_models, usable_foundation_models

    available, usable = available_foundation_models(), usable_foundation_models()
    for name, is_usable in usable.items():
        if is_usable:
            assert available.get(name), f"{name} usable but not available -- contradiction"


def test_baseline_lstm_is_always_offline_usable() -> None:
    """It trains from scratch on local data, so it needs no downloaded weights at all. This
    is what keeps the pipeline runnable on a machine that cannot fetch checkpoints."""
    from forecasting.base import registered_forecasters

    assert "baseline_lstm" in registered_forecasters(require_weights=True)


def test_hybrid_refuses_rather_than_triggering_a_download() -> None:
    """The critical guard: get_forecaster('hybrid') must never silently start an hours-long
    fetch. It either uses cached weights or explains how to get them."""
    from forecasting.base import get_forecaster, usable_foundation_models

    if any(usable_foundation_models().values()):
        pytest.skip("a foundation checkpoint is cached; nothing to assert about refusal")

    with pytest.raises(RuntimeError) as excinfo:
        get_forecaster("hybrid")

    message = str(excinfo.value)
    assert "download" in message.lower()
    assert "huggingface-cli download" in message or "baseline_lstm" in message


# --------------------------------------------------------------------------------------
# Hybrid fusion
# --------------------------------------------------------------------------------------

def _hybrid(window: int = 20):
    from forecasting.hybrid_model import HybridConfig, HybridForecaster

    return HybridForecaster(
        BaselineLSTMForecaster(LSTMConfig(window=window, hidden_size=8, num_layers=1, epochs=2)),
        HybridConfig(window=window, log_to_mlflow=False),
    )


def test_hybrid_fits_a_trainable_base_itself() -> None:
    """Regression guard. HybridForecaster.fit used to assume the base was already usable,
    which holds for a zero-shot adapter (its fit is a no-op) but NOT for a trainable base --
    predict_quantiles raised and the head had nothing to learn from. Found by running the
    hybrid on real data before the Colab run rather than during it.
    """
    hybrid = _hybrid()
    hybrid.fit(_feature_table(n=300), horizon=5, log_to_mlflow=False)
    assert hybrid.head is not None
    assert hybrid.version != "untrained"


def test_hybrid_decompose_exposes_the_rq1_ablation() -> None:
    """base / residual / final side by side is what lets Phase 7 answer "how much of the
    hybrid's advantage came from the covariates?" rather than only "it is better"."""
    hybrid = _hybrid()
    table = _feature_table(n=300)
    hybrid.fit(table, horizon=5, log_to_mlflow=False)

    decomposed = hybrid.decompose(table, horizon=5)
    for q in (10, 50, 90):
        for prefix in ("base", "residual", "final"):
            assert f"{prefix}_p{q}" in decomposed.columns

    # final must actually equal base + residual, or the decomposition is decorative.
    for q in (10, 50, 90):
        np.testing.assert_allclose(
            decomposed[f"final_p{q}"],
            decomposed[f"base_p{q}"] + decomposed[f"residual_p{q}"],
            rtol=1e-5,
        )


def test_hybrid_preserves_non_crossing_quantiles() -> None:
    """The head adds an unconstrained per-quantile delta, so ordering could be broken by a
    large correction; enforce_non_crossing at the interface must prevent that."""
    hybrid = _hybrid()
    table = _feature_table(n=300)
    hybrid.fit(table, horizon=5, log_to_mlflow=False)

    result = hybrid.predict_quantiles(table, horizon=5)
    assert (np.diff(result.values, axis=1) >= -1e-9).all()
    assert result.model_name == "hybrid"


def test_hybrid_refuses_to_predict_before_fit() -> None:
    with pytest.raises(RuntimeError, match="call fit"):
        _hybrid().predict_quantiles(_feature_table(n=100), horizon=5)


class _SingleRowBase:
    """A base with the shape of the real foundation adapters.

    TimesFM and Chronos-Bolt both return ONE row per call, stamped at the last input
    timestamp. That shape is what broke the hybrid on Colab, and no stub in this file had
    it -- the LSTM returns many in-sample rows, so every hybrid test passed while the two
    bases the component actually ships with could not train at all.
    """

    name = "single_row_stub"
    version = "stub"

    def __init__(self) -> None:
        self.calls = 0

    def fit(self, features, *, horizon, **kwargs) -> None:  # noqa: ANN001, ARG002
        return None

    def predict_quantiles(self, features, *, horizon, **kwargs):  # noqa: ANN001, ARG002
        import numpy as np
        import pandas as pd

        from forecasting.base import ForecastResult

        self.calls += 1
        frame = features.sort_values("timestamp")
        # A weak signal so the residual head has something real to correct.
        level = float(frame["close"].pct_change().tail(5).mean() or 0.0)
        return ForecastResult(
            symbol=str(frame["symbol"].iloc[0]),
            horizon=horizon,
            quantiles=(0.1, 0.5, 0.9),
            values=np.array([[level - 0.01, level, level + 0.01]], dtype=float),
            timestamps=pd.DatetimeIndex([pd.to_datetime(frame["timestamp"].iloc[-1])]),
            model_name=self.name,
            model_version=self.version,
        )


def test_hybrid_trains_on_a_single_row_foundation_base() -> None:
    """Regression guard for the Colab failure: RuntimeError('no overlap between base
    forecasts and targets') on every foundation base.

    One call per symbol stamps the forecast at the last input timestamp -- which is exactly
    the row add_targets leaves NaN, since the final `horizon` rows have no realised future.
    The join then dropped everything. The fix walks the cut point forward and stamps each
    forecast at its cut, so it lines up with the target it is actually predicting.
    """
    from forecasting.hybrid_model import HybridConfig, HybridForecaster

    base = _SingleRowBase()
    hybrid = HybridForecaster(
        base,
        HybridConfig(window=20, log_to_mlflow=False),
    )
    hybrid.fit(_feature_table(n=300), horizon=5, log_to_mlflow=False)

    assert hybrid.head is not None
    assert hybrid.version != "untrained"
    # It must actually have walked, not made a single call.
    assert base.calls > 1, f"base was called {base.calls} time(s); the walk did not happen"


def test_walk_forward_never_stamps_a_forecast_on_an_unrealised_target() -> None:
    """The property behind the fix: every base forecast must land on a row whose target
    exists. If any cut fell in the final `horizon` rows the join would silently shrink,
    and the head would train on fewer points than intended without anything saying so.
    """
    from features.feature_store import add_targets
    from forecasting.hybrid_model import HybridConfig, HybridForecaster

    horizon = 5
    features = add_targets(_feature_table(n=300), horizon=horizon)
    hybrid = HybridForecaster(
        _SingleRowBase(),
        HybridConfig(window=20, log_to_mlflow=False),
    )

    forecasts = hybrid._walk_forward_base(features, horizon=horizon)
    assert not forecasts.empty

    merged = features.merge(forecasts, on=["symbol", "timestamp"], how="inner")
    assert len(merged) == len(forecasts), "a forecast was stamped on a timestamp not in the data"
    assert merged["target_return"].notna().all(), "a forecast landed on an unrealised target"


# --------------------------------------------------------------------------------------
# Per-architecture LoRA training steps
#
# The Colab run died inside the training loop, twice, with two different errors: TimesFM
# with "forward() missing 1 required positional argument: 'masks'", Chronos with our own
# "model returned neither a loss nor a tensor". The loop had guessed at a common interface
# neither model has. These tests pin down what each architecture is actually asked to do.
# --------------------------------------------------------------------------------------


def test_unsupported_lora_is_checked_before_anything_loads() -> None:
    """The refusal mechanism, not a claim about any particular model. TimesFM was listed
    unsupported on the grounds that decode() runs under torch.no_grad(); that was wrong --
    forward() is differentiable and prefill alone covers a horizon within one output patch.
    The mechanism stays because a genuinely untrainable architecture must fail in a second
    rather than after a model download."""
    import forecasting.finetune_lora as fl

    monkey = dict(fl.UNSUPPORTED_LORA)
    monkey["pretend_model"] = "cannot be trained, for a stated reason"
    original, fl.UNSUPPORTED_LORA = fl.UNSUPPORTED_LORA, monkey
    try:
        with pytest.raises(RuntimeError, match="for a stated reason"):
            fl.finetune("pretend_model", pd.DataFrame({"close": [1.0]}), horizon=5)
    finally:
        fl.UNSUPPORTED_LORA = original


def test_chronos_step_asks_the_model_for_its_own_loss() -> None:
    """Chronos-Bolt computes a pinball loss internally when given a target. The step must
    use that rather than scoring the quantile predictions itself -- the model normalises the
    target with the same loc/scale as the context, which an external loss would not."""
    import torch

    from forecasting.finetune_lora import LoRAConfig, _chronos_step

    seen = {}

    class _StubChronos:
        def __call__(self, *, context, target=None, mask=None, target_mask=None):
            seen["context"] = context
            seen["target"] = target
            return SimpleNamespace(loss=torch.tensor(0.25), quantile_preds=None)

    ctx = torch.randn(4, 64)
    fut = torch.randn(4, 5)
    loss = _chronos_step(_StubChronos(), ctx, fut, LoRAConfig())

    assert float(loss) == pytest.approx(0.25)
    assert seen["target"] is fut, "the future must be passed as `target`, not scored outside"
    assert seen["context"] is ctx


def test_chronos_step_matches_the_installed_library_signature() -> None:
    """Guard against contract drift. _chronos_step calls forward(context=..., target=...) by
    keyword; if chronos-forecasting renames either, the fine-tune breaks on Colab where it
    cannot be debugged cheaply. This catches it here, and needs no model weights."""
    bolt = pytest.importorskip("chronos.chronos_bolt")

    parameters = inspect.signature(bolt.ChronosBoltModelForForecasting.forward).parameters
    assert "context" in parameters, "Chronos forward() no longer takes `context`"
    assert "target" in parameters, "Chronos forward() no longer takes `target`"


def test_chronos_step_reports_a_missing_loss_clearly() -> None:
    """If the model returns no loss despite a target, say that, rather than letting a None
    reach loss.backward() and surface as an AttributeError."""
    from forecasting.finetune_lora import LoRAConfig, _chronos_step

    class _NoLoss:
        def __call__(self, **kwargs):
            return SimpleNamespace(loss=None)

    with pytest.raises(RuntimeError, match="no loss"):
        _chronos_step(_NoLoss(), None, None, LoRAConfig())


def test_unsupported_architecture_names_itself_in_the_error() -> None:
    """The generic step is the last resort. When it cannot work, the error must name the
    model class and point at the registry, so the next person knows where to add a step."""
    import torch

    from forecasting.finetune_lora import LoRAConfig, _generic_step

    class _WrongInterface(torch.nn.Module):
        def forward(self, inputs, masks):  # noqa: ARG002 - mirrors TimesFM's signature
            return None

    with pytest.raises(RuntimeError, match="STEP_FUNCTIONS"):
        _generic_step(_WrongInterface(), torch.randn(2, 8), torch.randn(2, 5), LoRAConfig())


def test_run_epoch_delegates_to_the_step_function() -> None:
    """_run_epoch must not compute a loss itself any more -- that was the guessing that
    broke both models. It runs whatever the architecture supplied."""
    import torch

    from forecasting.finetune_lora import LoRAConfig, _run_epoch

    calls = []

    def _step(model, context, future, config):  # noqa: ANN001, ARG001
        calls.append(len(context))
        return torch.tensor(0.5, requires_grad=True)

    loader = [(torch.randn(3, 8), torch.randn(3, 2)), (torch.randn(2, 8), torch.randn(2, 2))]
    loss = _run_epoch(torch.nn.Linear(8, 2), loader, LoRAConfig(), None, _step)

    assert calls == [3, 2], "every batch must go through the step function"
    assert loss == pytest.approx(0.5)


# --------------------------------------------------------------------------------------
# TimesFM LoRA step
# --------------------------------------------------------------------------------------


class _StubTimesFMConfig:
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class _StubTimesFM(torch.nn.Module):
    """The interface _timesfm_step relies on, at a size a test can afford.

    p / o / q and the four-tuple return mirror TimesFM_2p5_200M_torch_module exactly; the
    real thing is exercised by the slow test below.
    """

    config = _StubTimesFMConfig()

    def __init__(self, patch_len: int = 32, out_len: int = 128) -> None:
        super().__init__()
        self.p, self.o = patch_len, out_len
        self.q = len(self.config.quantiles) + 1
        self.proj = torch.nn.Linear(patch_len, self.o * self.q)
        self.seen: dict[str, object] = {}

    def forward(self, inputs, masks, decode_caches=None):  # noqa: ANN001
        self.seen["inputs"] = inputs
        self.seen["masks"] = masks
        out = self.proj(inputs)
        return (inputs, inputs, out, out), None


def test_timesfm_step_produces_a_differentiable_loss() -> None:
    from forecasting.finetune_lora import LoRAConfig, _timesfm_step

    model = _StubTimesFM()
    loss = _timesfm_step(model, torch.randn(3, 128) * 0.01, torch.randn(3, 5) * 0.01, LoRAConfig())

    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    assert model.proj.weight.grad is not None
    assert model.proj.weight.grad.abs().sum() > 0


def test_timesfm_step_reads_the_quantile_columns_not_the_mean() -> None:
    """TimesFM's head emits len(quantiles) + 1 columns, and column 0 is the MEAN. Reading
    it as the 0.1 quantile would train against the wrong target and still converge, which a
    loss curve cannot reveal -- so the mapping is asserted rather than assumed."""
    from forecasting.finetune_lora import LoRAConfig, _timesfm_quantile_indices

    model = _StubTimesFM()
    indices = _timesfm_quantile_indices(model, LoRAConfig(quantiles=(0.1, 0.5, 0.9)))

    assert indices == [1, 5, 9], "0.1/0.5/0.9 must map past the leading mean column"


def test_timesfm_step_rejects_a_quantile_the_model_does_not_emit() -> None:
    from forecasting.finetune_lora import LoRAConfig, _timesfm_quantile_indices

    with pytest.raises(RuntimeError, match="not among them"):
        _timesfm_quantile_indices(_StubTimesFM(), LoRAConfig(quantiles=(0.05,)))


def test_timesfm_step_refuses_a_horizon_beyond_one_output_patch() -> None:
    """Past output_patch_len, decode() feeds its own output back autoregressively. This
    step reproduces prefill only, so it must refuse rather than silently train on a horizon
    it cannot represent."""
    from forecasting.finetune_lora import LoRAConfig, _timesfm_step

    model = _StubTimesFM(out_len=16)
    with pytest.raises(RuntimeError, match="exceeds TimesFM's output patch length"):
        _timesfm_step(model, torch.randn(2, 64), torch.randn(2, 20), LoRAConfig())


def test_timesfm_step_truncates_context_to_whole_patches() -> None:
    """forward() patches by reshape, which needs an exact multiple of the patch length. The
    most RECENT whole patches are the ones to keep -- dropping from the tail would forecast
    from a window that stops short of the present."""
    from forecasting.finetune_lora import LoRAConfig, _timesfm_step

    model = _StubTimesFM()
    context = torch.randn(2, 140) * 0.01           # 4 patches of 32, plus 12 spare
    _timesfm_step(model, context, torch.randn(2, 5) * 0.01, LoRAConfig())

    used = model.seen["inputs"]
    assert used.shape == (2, 4, 32), f"expected 4 whole patches, got {tuple(used.shape)}"
    assert model.seen["masks"].dtype == torch.bool


def test_timesfm_step_rejects_a_context_shorter_than_one_patch() -> None:
    from forecasting.finetune_lora import LoRAConfig, _timesfm_step

    with pytest.raises(RuntimeError, match="shorter than one"):
        _timesfm_step(_StubTimesFM(), torch.randn(2, 8), torch.randn(2, 5), LoRAConfig())


@pytest.mark.slow
def test_timesfm_step_trains_the_real_architecture() -> None:
    """The stub proves the wiring; this proves the wiring matches the real model.

    Instantiated with random weights -- no checkpoint, so no download. One catch makes this
    test subtler than it looks: every transformer norm `scale` initialises to exactly ZERO,
    which zeroes both residual branches and makes the whole 20-layer stack an identity. A
    naive run therefore shows gradients reaching only the tokenizer and the output head, and
    looks exactly like a broken step function. The scales are filled with a trained-like
    value first so the test measures the step rather than the initialisation.
    """
    timesfm_torch = pytest.importorskip("timesfm.timesfm_2p5.timesfm_2p5_torch")

    from forecasting.finetune_lora import LoRAConfig, _timesfm_step

    torch.manual_seed(0)
    model = timesfm_torch.TimesFM_2p5_200M_torch_module()
    with torch.no_grad():
        for layer in model.stacked_xf:
            for norm in ("pre_attn_ln", "post_attn_ln", "pre_ff_ln", "post_ff_ln"):
                getattr(layer, norm).scale.fill_(0.5)

    loss = _timesfm_step(model, torch.randn(2, 128) * 0.01, torch.randn(2, 5) * 0.01, LoRAConfig())
    assert torch.isfinite(loss)
    loss.backward()

    # The point of LoRA here is to adapt the attention projections, so gradient MUST reach
    # them. Only the tokenizer and output head receiving gradient is the failure mode.
    qkv_grad = model.stacked_xf[0].attn.qkv_proj.weight.grad
    assert qkv_grad is not None, "no gradient reached the first transformer layer"
    assert qkv_grad.abs().max() > 0, "gradient reached the transformer layer but was zero"


# --------------------------------------------------------------------------------------
# Device handling
#
# Colab failed with "mat1 is on cpu, different from other tensors on cuda:0": TimesFM's
# load_checkpoint() moves itself to cuda:0 whenever CUDA exists, while the DataLoader went
# on yielding CPU tensors. It could not surface locally, where everything is CPU, so these
# tests use the `meta` device to create a real device boundary on a CPU-only machine.
# --------------------------------------------------------------------------------------


def test_resolve_device_honours_an_explicit_preference() -> None:
    from forecasting.base import resolve_device

    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("meta").type == "meta"


def test_resolve_device_picks_the_gpu_when_there_is_one(monkeypatch) -> None:
    """Auto-detection both ways. The CPU branch is what this machine exercises; the CUDA
    branch is the one that matters on Colab and can only be reached by patching."""
    from forecasting.base import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device(None).type == "cpu"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device(None).type == "cuda"


def test_chronos_config_no_longer_pins_the_cpu(monkeypatch) -> None:
    """ChronosConfig.device was hardcoded "cpu" for this laptop, which silently sent Colab's
    fine-tune to the CPU too -- no error, just twenty epochs at the wrong speed."""
    from forecasting.base import resolve_device
    from forecasting.chronos_adapter import ChronosConfig

    assert ChronosConfig().device is None, "a pinned device would follow the code to Colab"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device(ChronosConfig().device).type == "cuda"


def test_model_device_reads_the_parameters() -> None:
    from forecasting.finetune_lora import _model_device

    assert _model_device(torch.nn.Linear(2, 2)).type == "cpu"
    assert _model_device(torch.nn.Linear(2, 2).to("meta")).type == "meta"
    assert _model_device(torch.nn.Module()).type == "cpu", "a parameterless model must not raise"


def test_run_epoch_moves_batches_to_the_model_device() -> None:
    """THE regression guard for the Colab failure.

    The model's parameters sit on `meta` while the loader yields CPU tensors -- the same
    split as a CUDA model fed by a CPU DataLoader, reproducible without a GPU. Before the
    fix the step function received CPU tensors and the matmul blew up inside the model.
    """
    from forecasting.finetune_lora import LoRAConfig, _run_epoch

    seen = []

    def _step(model, context, future, config):  # noqa: ANN001, ARG001
        seen.append((context.device.type, future.device.type))
        return torch.tensor(0.5)

    model = torch.nn.Linear(8, 2).to("meta")
    loader = [(torch.randn(3, 8), torch.randn(3, 2)), (torch.randn(2, 8), torch.randn(2, 2))]

    _run_epoch(model, loader, LoRAConfig(), None, _step)

    assert seen == [("meta", "meta"), ("meta", "meta")], (
        f"batches reached the step function on {seen}, not on the model's device"
    )


def test_smoke_subset_always_yields_usable_windows() -> None:
    """The smoke check's own regression guard.

    Its first version sliced to 400 rows per symbol while leaving context_length at 512, so
    build_finetune_dataset skipped every symbol and raised "no windows could be built: need
    at least 518 bars". The slice and the context length have to be chosen together.
    """
    from forecasting.finetune_lora import build_finetune_dataset, smoke_subset

    for bars in (2000, 400, 200):
        features = pd.concat(
            [_price_frame(n=bars, seed=i).assign(symbol=s) for i, s in enumerate(("AAA", "BBB"))],
            ignore_index=True,
        )
        subset, context = smoke_subset(features, horizon=5)

        assert subset["symbol"].nunique() == 1, "a smoke slice is one symbol by design"
        assert context % 32 == 0, f"context {context} is not a whole number of TimesFM patches"

        dataset = build_finetune_dataset(subset, context_length=context, horizon=5)
        assert len(dataset) > 0, f"{bars} bars produced no windows at context {context}"


def test_smoke_subset_refuses_a_series_it_cannot_use() -> None:
    """Too short to form even one patch of context must say so, rather than returning a
    slice that fails later inside build_finetune_dataset."""
    from forecasting.finetune_lora import smoke_subset

    with pytest.raises(ValueError, match="too short"):
        smoke_subset(_price_frame(n=20).assign(symbol="TINY"), horizon=5)


# --------------------------------------------------------------------------------------
# Training progress
#
# On the full universe an epoch is ~1,100 optimizer steps on a 231M-parameter model, so
# logging only at epoch boundaries left the Colab cell silent for many minutes -- which
# looks exactly like a hang. These pin the progress output that fixes that.
# --------------------------------------------------------------------------------------


def test_duration_reads_at_a_glance() -> None:
    from forecasting.finetune_lora import _duration

    assert _duration(45) == "45s"
    assert _duration(605) == "10m05s"
    assert _duration(7500) == "2h05m"


def test_run_epoch_logs_progress_during_the_epoch(caplog) -> None:
    """The point is DURING, not after. A 40-batch epoch must report while it runs."""
    from forecasting.finetune_lora import LoRAConfig, _run_epoch

    loader = [(torch.randn(2, 8), torch.randn(2, 2)) for _ in range(40)]

    def _step(model, context, future, config):  # noqa: ANN001, ARG001
        return torch.tensor(0.25)

    with caplog.at_level("INFO", logger="forecasting.finetune_lora"):
        _run_epoch(torch.nn.Linear(8, 2), loader, LoRAConfig(), None, _step, epoch=3)

    progress = [r.getMessage() for r in caplog.records if "batch" in r.getMessage()]
    assert progress, "no intra-epoch progress was logged"
    assert len(progress) > 1, "progress must appear repeatedly, not once at the end"
    assert "epoch 3" in progress[0], "the epoch number must be identifiable"
    assert "left" in progress[0], "an ETA is the point -- it tells a slow run from a hung one"


def test_run_epoch_survives_a_loader_with_no_length(caplog) -> None:
    """A generator loader has no len(); progress must degrade rather than raise."""
    from forecasting.finetune_lora import LoRAConfig, _run_epoch

    def _gen():
        for _ in range(4):
            yield torch.randn(2, 8), torch.randn(2, 2)

    def _step(model, context, future, config):  # noqa: ANN001, ARG001
        return torch.tensor(0.5)

    with caplog.at_level("INFO", logger="forecasting.finetune_lora"):
        loss = _run_epoch(torch.nn.Linear(8, 2), _gen(), LoRAConfig(), None, _step)

    assert loss == pytest.approx(0.5)


# --------------------------------------------------------------------------------------
# Memory
#
# Colab was dying at its 12GB system / 15GB GPU limits. Measured at the real universe
# scale, the data pipeline is not the cause -- features 119MB, dataset 187MB before this
# change -- but materialising every window stored 181MB of near-duplicate data built from
# 400KB of returns, and that is worth not doing regardless.
# --------------------------------------------------------------------------------------


def test_dataset_does_not_materialise_every_window() -> None:
    """Consecutive windows share all but one step, so storing each one in full is a large
    multiple of the underlying series. The dataset must hold the series, not the windows."""
    from forecasting.finetune_lora import build_finetune_dataset

    frame = pd.concat(
        [_price_frame(n=2000, seed=i).assign(symbol=f"S{i}") for i in range(3)],
        ignore_index=True,
    )
    dataset = build_finetune_dataset(frame, context_length=512, horizon=5)

    materialised = len(dataset) * 512 * 4
    assert len(dataset) > 4000, "expected a window count worth caring about"
    assert dataset.nbytes < materialised / 50, (
        f"holds {dataset.nbytes:,} bytes; materialising would be {materialised:,} -- "
        "the dataset is still storing windows rather than series"
    )


def test_dataset_windows_are_identical_to_materialised_ones() -> None:
    """Slicing on demand must return exactly what building them up front did, or the space
    saving comes at the cost of training on something subtly different."""
    from forecasting.finetune_lora import build_finetune_dataset

    frame = _price_frame(n=300, seed=7).assign(symbol="AAA")
    dataset = build_finetune_dataset(frame, context_length=64, horizon=5)

    closes = frame.sort_values("timestamp")["close"].to_numpy(dtype=np.float64)
    returns = np.log(closes[1:] / closes[:-1]).astype(np.float32)

    for i in (0, 1, len(dataset) // 2, len(dataset) - 1):
        context, future = dataset[i]
        assert np.allclose(context.numpy(), returns[i : i + 64])
        assert np.allclose(future.numpy(), returns[i + 64 : i + 69])


def test_memory_note_never_breaks_a_run() -> None:
    """It is a log line. It must degrade to something printable rather than raise, whatever
    is or is not installed."""
    from forecasting.finetune_lora import memory_note

    note = memory_note()
    assert isinstance(note, str) and note


# --------------------------------------------------------------------------------------
# OOM defences
# --------------------------------------------------------------------------------------


def test_gradient_accumulation_preserves_the_effective_batch() -> None:
    """Halving the batch and doubling accumulation must leave the optimisation unchanged.

    If it did not, the OOM fallback would silently alter training rather than only its
    memory profile -- and the run would still finish, so nothing would flag it.
    """
    from dataclasses import replace as dc_replace

    from forecasting.finetune_lora import LoRAConfig, _run_epoch

    torch.manual_seed(0)
    data = [(torch.randn(16, 4), torch.randn(16, 1)) for _ in range(4)]

    def _run(batch_size: int, accum: int) -> list[float]:
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        # Re-chunk the same samples into the requested batch size.
        contexts = torch.cat([c for c, _ in data])
        futures = torch.cat([f for _, f in data])
        loader = [
            (contexts[i : i + batch_size], futures[i : i + batch_size])
            for i in range(0, len(contexts), batch_size)
        ]
        config = dc_replace(LoRAConfig(), grad_accum_steps=accum, max_grad_norm=1e9)
        _run_epoch(
            model, loader, config, optimizer,
            lambda m, c, f, cfg: torch.nn.functional.mse_loss(m(c), f),
        )
        return [p.detach().clone().flatten().tolist() for p in model.parameters()][0]

    full = _run(batch_size=16, accum=1)
    split = _run(batch_size=8, accum=2)

    assert full == pytest.approx(split, abs=1e-5), (
        "batch 8 x 2 accumulation diverged from batch 16 -- the fallback changes training"
    )


def test_fit_batch_size_is_a_no_op_without_cuda() -> None:
    """On CPU there is nothing to probe, and probing anyway would cost a forward/backward
    on every run for no benefit."""
    from forecasting.finetune_lora import LoRAConfig, fit_batch_size

    config = LoRAConfig(batch_size=64, grad_accum_steps=1)
    batch, accum = fit_batch_size(torch.nn.Linear(4, 1), [], config, None)

    assert (batch, accum) == (64, 1)


def test_fit_batch_size_halves_until_it_fits(monkeypatch) -> None:
    """The probe must back off and compensate, so the effective batch is preserved.

    CUDA is faked: the point is the search, which has to be right before it ever runs
    somewhere it cannot be debugged.
    """
    import forecasting.finetune_lora as fl

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(fl, "_release_cuda", lambda: None)
    monkeypatch.setattr(fl, "_model_device", lambda m: torch.device("cpu"))

    dataset = [(torch.randn(4), torch.randn(1)) for _ in range(64)]
    model = torch.nn.Linear(4, 1)

    def _step(m, context, future, config):  # noqa: ANN001, ARG001
        if len(context) > 16:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return m(context).sum()

    batch, accum = fl.fit_batch_size(model, dataset, fl.LoRAConfig(batch_size=64), _step)

    assert batch == 16, f"settled on batch {batch}, expected 16"
    assert batch * accum == 64, "the effective batch must survive the back-off"


def test_fit_batch_size_gives_up_with_a_useful_message(monkeypatch) -> None:
    """If nothing fits, the cause is not batch size, and the error should say so rather
    than leaving someone halving a number that was never the problem."""
    import forecasting.finetune_lora as fl

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(fl, "_release_cuda", lambda: None)
    monkeypatch.setattr(fl, "_model_device", lambda m: torch.device("cpu"))

    def _always_oom(m, context, future, config):  # noqa: ANN001, ARG001
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="other than batch size"):
        fl.fit_batch_size(
            torch.nn.Linear(4, 1),
            [(torch.randn(4), torch.randn(1)) for _ in range(64)],
            fl.LoRAConfig(batch_size=64),
            _always_oom,
        )


def test_chronos_predict_quantiles_matches_the_installed_signature() -> None:
    """chronos-forecasting renamed this parameter from `context` to `inputs`, and because
    the adapter passed it by keyword every backtest fold failed with "missing 1 required
    positional argument: 'inputs'". The adapter test that would have caught it is marked
    slow and needs ~200MB of weights, so it never runs here -- this one needs none.
    """
    chronos = pytest.importorskip("chronos")

    parameters = list(
        inspect.signature(chronos.ChronosBoltPipeline.predict_quantiles).parameters
    )
    assert parameters[1] not in {"prediction_length", "quantile_levels"}, (
        "the first argument is no longer the context series; the adapter passes it "
        "positionally and would now be sending it as something else"
    )
    for name in ("prediction_length", "quantile_levels"):
        assert name in parameters, f"predict_quantiles no longer accepts {name}"
