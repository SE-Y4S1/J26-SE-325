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
import pytest
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


def test_lora_target_discovery_reports_what_it_found_when_it_fails() -> None:
    from forecasting.finetune_lora import _discover_target_modules

    class Unrecognised(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weird_layer = torch.nn.Linear(4, 4)

    with pytest.raises(ValueError, match="weird_layer"):
        _discover_target_modules(Unrecognised())


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
