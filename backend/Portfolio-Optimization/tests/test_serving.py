"""Phase 3 tests: enrichment inputs, hybrid head persistence, install/registry, the forecast
service (hybrid vs labelled empirical), the forecast step and /forecast.

No foundation weights are touched: the base model is a stub Forecaster.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features import enrichment
from forecasting import model_registry, serving
from forecasting.base import DEFAULT_QUANTILES, ForecastResult
from forecasting.hybrid_model import HybridConfig, HybridForecaster
from forecasting.residual_head import ResidualHead, ResidualHeadConfig


class StubBase:
    name = "timesfm"
    version = "stub"

    def fit(self, features, *, horizon, **kwargs):  # noqa: ANN001
        return None

    def predict_quantiles(self, features, *, horizon, quantiles=DEFAULT_QUANTILES, **kwargs):  # noqa: ANN001
        stamp = pd.DatetimeIndex([pd.to_datetime(features["timestamp"]).max()])
        return ForecastResult(symbol=str(features["symbol"].iloc[0]), horizon=horizon, quantiles=tuple(quantiles),
                              values=np.array([[-0.05, 0.01, 0.06]]), timestamps=stamp,
                              model_name=self.name, model_version=self.version)


def bars(n: int = 320, symbol: str = "JKH.N0000", asset_class: str = "cse_equity") -> pd.DataFrame:
    rng = np.random.default_rng(1)
    close = 20 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    return pd.DataFrame({
        "timestamp": pd.bdate_range("2025-01-01", periods=n), "symbol": symbol, "asset_class": asset_class,
        "open": np.nan, "high": close * 1.01, "low": close * 0.99, "close": close,
        "volume": rng.integers(1_000, 50_000, n).astype(float),
    })


# --------------------------------------------------------------------------------------
# Enrichment
# --------------------------------------------------------------------------------------

def test_enrichment_always_has_every_column(tmp_path) -> None:
    table = bars(30)
    out = enrichment.add_enrichment(table, sentiment_dir=tmp_path, fundamentals_loader=lambda s: [])
    assert set(enrichment.ENRICHMENT_COLUMNS) <= set(out.columns)
    assert (out[enrichment.ENRICHMENT_COLUMNS] == 0).all().all()


def test_tier_sentiment_decays_in_trading_days(tmp_path) -> None:
    table = bars(10)
    from data.cache import _safe_symbol

    day = table["timestamp"].iloc[3]
    pd.DataFrame([{"symbol": "JKH.N0000", "date": str(day.date()), "tier": "company", "items": 4, "scored": 4,
                   "mean_sentiment": -0.8}]).to_parquet(tmp_path / f"{_safe_symbol('JKH.N0000')}.parquet")
    out = enrichment.add_tier_sentiment(table, sentiment_dir=tmp_path)
    company = out["sent_company"].to_numpy()
    assert company[:3].tolist() == [0.0, 0.0, 0.0]               # nothing before the news
    assert company[3] == pytest.approx(-0.8)
    assert company[3 + 3] == pytest.approx(-0.8 * np.exp(-1))     # tau = 3 trading days
    assert out["attention_company"].iloc[3] == pytest.approx(np.log1p(4))
    assert out["has_tier_sentiment"].tolist() == [0.0] * 3 + [1.0] * 7


def test_fundamentals_are_point_in_time() -> None:
    table = bars(60)
    available = table["timestamp"].iloc[30]
    record = {"usable": True, "available_from": available.tz_localize("UTC").isoformat(),
              "current": {"naps": 10.0},
              "ratios": {"roe": 0.12, "debt_to_equity": 0.5, "earnings_growth_yoy": 0.2, "f_score": 6,
                         "f_score_max": 9, "z_score": 5.1, "eps_annualised": 2.0}}
    out = enrichment.add_fundamentals(table, loader=lambda s: [record])
    before, after = out.iloc[29], out.iloc[30]
    assert before["has_fundamentals"] == 0 and before["fund_roe"] == 0
    assert after["has_fundamentals"] == 1 and after["fund_roe"] == pytest.approx(0.12)
    assert after["fund_f_score"] == pytest.approx(6 / 9)
    # Multiples move with the price between reports.
    assert after["fund_pb"] == pytest.approx(min(after["close"] / 10.0, 20.0))
    assert out.iloc[59]["fund_earnings_yield"] == pytest.approx(2.0 / out.iloc[59]["close"])


# --------------------------------------------------------------------------------------
# Head persistence
# --------------------------------------------------------------------------------------

def trained_like(feature_cols: list[str], horizon: int = 5) -> HybridForecaster:
    model = HybridForecaster(StubBase(), HybridConfig(window=20, log_to_mlflow=False))
    model.feature_cols = feature_cols
    model.horizon = horizon
    model.version = f"hybrid-timesfm-h{horizon}"
    head_config = ResidualHeadConfig(zero_init_output=False)
    model.head_config = head_config
    model.head = ResidualHead(len(feature_cols), model.n_asset_classes, head_config)
    return model


@pytest.fixture
def features() -> pd.DataFrame:
    return serving.symbol_features(bars(), "JKH.N0000")


def test_save_load_round_trip(tmp_path, features) -> None:
    from features.feature_store import feature_columns

    model = trained_like(feature_columns(features))
    before = model.predict_latest(features, horizon=5)
    model.save(tmp_path / "h5")
    loaded = HybridForecaster.load(tmp_path / "h5", StubBase())
    after = loaded.predict_latest(features, horizon=5)
    assert after["final"] == pytest.approx(before["final"])
    assert after["base"] == pytest.approx([-0.05, 0.01, 0.06])
    assert sorted(after["final"]) == after["final"]                # non-crossing
    meta = json.loads((tmp_path / "h5" / "meta.json").read_text())
    assert meta["horizon"] == 5 and "cse_equity" in meta["asset_class_order"]


def test_missing_feature_column_is_zero_filled(tmp_path, features) -> None:
    from features.feature_store import feature_columns

    model = trained_like([*feature_columns(features), "a_column_this_env_cannot_build"])
    assert len(model.predict_latest(features, horizon=5)["final"]) == 3


# --------------------------------------------------------------------------------------
# Install + service
# --------------------------------------------------------------------------------------

@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(model_registry, "REGISTRY_PATH", tmp_path / "registry.sqlite")
    monkeypatch.setattr(serving, "CHECKPOINT_ROOT", tmp_path / "checkpoints")
    serving.reset()
    yield tmp_path
    serving.reset()


def make_zip(tmp_path: Path, features: pd.DataFrame, horizons=(5, 21)) -> Path:
    from features.feature_store import feature_columns

    export = tmp_path / "export"
    for h in horizons:
        model = trained_like(feature_columns(features), horizon=h)
        model.save(export / f"h{h}")
        meta = json.loads((export / f"h{h}" / "meta.json").read_text())
        meta.update({"train_window": {"start": "2020-01-01", "end": "2026-06-30"}, "metrics": {"pinball": 0.01}})
        (export / f"h{h}" / "meta.json").write_text(json.dumps(meta))
    archive = tmp_path / "hybrid_heads.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for file in export.rglob("*"):
            zf.write(file, file.relative_to(export))
    return archive


def test_install_registers_and_activates(registry, features) -> None:
    archive = make_zip(registry, features)
    versions = serving.install(archive)
    assert sorted(v.split("-v")[0] for v in versions) == ["hybrid_h21", "hybrid_h5"]
    directory, version = serving.active_head_dir(5)
    assert (directory / "head.pt").exists() and version.startswith("hybrid_h5-v0.1")
    # Re-installing identical bytes mints no new version.
    assert serving.install(archive) == versions


def test_service_without_head_serves_labelled_empirical(registry) -> None:
    result = serving.get_service().forecast("JKH.N0000", bars(), 5)
    assert result["method"] == "empirical" and result["model_version"] is None
    assert "no trained head" in result["reason"]
    assert result["p10"] < result["p50"] < result["p90"]


def test_service_without_base_weights_says_so(registry, features, monkeypatch) -> None:
    serving.install(make_zip(registry, features))
    monkeypatch.setattr("forecasting.base.usable_foundation_models", lambda: {"timesfm": False, "chronos": False})
    result = serving.get_service().forecast("JKH.N0000", bars(), 5)
    assert result["method"] == "empirical"
    assert "weights" in result["reason"]


def test_service_serves_hybrid_when_runnable(registry, features, monkeypatch) -> None:
    serving.install(make_zip(registry, features))
    monkeypatch.setattr("forecasting.base.usable_foundation_models", lambda: {"timesfm": True, "chronos": False})
    monkeypatch.setattr("forecasting.base.get_forecaster", lambda name, **kw: StubBase())
    result = serving.get_service().forecast("JKH.N0000", bars(), 21)
    assert result["method"] == "hybrid"
    assert result["model_version"].startswith("hybrid_h21-v0.1")
    assert result["base"] == pytest.approx([-0.05, 0.01, 0.06])


def test_empirical_quantiles() -> None:
    close = pd.Series(np.full(300, 100.0) * np.exp(np.linspace(0, 0.3, 300)))
    q = serving.empirical_quantiles(close, 5)
    assert q == pytest.approx([np.exp(0.3 / 299 * 5) - 1] * 3, rel=1e-6)
    assert serving.empirical_quantiles(close.head(20), 5) is None


def test_forecast_step(registry, monkeypatch) -> None:
    from data import history
    from data.instruments import Exchange, Instrument
    from data.schema import AssetClass

    monkeypatch.setattr(history, "load_stored", lambda instrument: bars())
    jkh = Instrument(symbol="JKH.N0000", name="JKH", exchange=Exchange.CSE, asset_class=AssetClass.CSE_EQUITY,
                     currency="LKR", provider_id="508")
    ok, block = serving.forecast_step(jkh)
    assert ok and set(block) == {"h5", "h21"}
    assert block["h21"]["method"] == "empirical"


def test_forecast_endpoint_serves_hybrid(registry, features, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from data import active_set, history
    from data.history import History
    from service import deps
    from service.api import app

    serving.install(make_zip(registry, features))
    monkeypatch.setattr("forecasting.base.usable_foundation_models", lambda: {"timesfm": True, "chronos": False})
    monkeypatch.setattr("forecasting.base.get_forecaster", lambda name, **kw: StubBase())
    monkeypatch.setattr(deps, "_model_cache", {})
    monkeypatch.setattr(active_set, "touch", lambda symbols: None)
    from data.instruments import Exchange, Instrument
    from data.schema import AssetClass

    monkeypatch.setattr(active_set, "get_instrument", lambda symbol: Instrument(
        symbol="JKH.N0000", name="JKH", exchange=Exchange.CSE, asset_class=AssetClass.CSE_EQUITY,
        currency="LKR", provider_id="508"))
    monkeypatch.setattr(history, "get_history", lambda instrument, **kw: History(bars=bars(), stale=False))
    body = TestClient(app).post("/forecast", json={"symbols": ["JKH.N0000"], "horizon": 5}).json()
    assert body["model_version"].startswith("hybrid_h5-v0.1")
    assert body["forecasts"][0]["p10"] <= body["forecasts"][0]["p50"] <= body["forecasts"][0]["p90"]

    response = TestClient(app).post("/forecast", json={"symbols": ["JKH.N0000"], "horizon": 3})
    assert response.status_code == 503 and "no trained head registered for h=3" in response.json()["detail"]


def test_colab_training_exports_installable_heads(registry, monkeypatch) -> None:
    """Part A end to end with a stub base: features -> fit -> holdout metrics -> zip -> install."""
    from experiments import train_hybrid_heads as part_a
    from features.enrichment import add_enrichment
    from features.feature_store import add_targets, build_feature_table
    from features.technical import compute_universe_indicators

    monkeypatch.setattr("forecasting.base.get_forecaster", lambda name, **kw: StubBase())
    by_symbol = {"JKH.N0000": bars(260), "BTCUSDT": bars(260, "BTCUSDT", "crypto")}
    table = build_feature_table(by_symbol, compute_universe_indicators(by_symbol))
    table = add_targets(add_enrichment(table, sentiment_dir=registry, fundamentals_loader=lambda s: []), horizon=5)

    out = registry / "training"
    out.mkdir()
    meta = part_a.train_head(table, 5, out, walk_forward_points=400, epochs=1)
    assert meta["horizon"] == 5 and meta["universe"] == {"crypto": 1, "cse_equity": 1}
    assert "pinball_hybrid_all" in meta["metrics"] and "pinball_base_all" in meta["metrics"]
    # Resume: a finished horizon is not retrained.
    monkeypatch.setattr("forecasting.base.get_forecaster", lambda name, **kw: pytest.fail("retrained"))
    assert part_a.train_head(table, 5, out, walk_forward_points=400, epochs=1)["horizon"] == 5

    [version] = serving.install(part_a.export(out))
    assert version.startswith("hybrid_h5-v0.1")
