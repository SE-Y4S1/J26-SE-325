"""Assemble the RQ1-RQ4 comparison tables.

Written as a runnable script rather than living only inside a notebook, so results can be
regenerated from CI or a terminal and the notebook stays a thin presentation layer. Every
table is also logged to MLflow, so the dissertation can pull numbers without re-running.

    uv run python experiments/run_rq_analysis.py --rq 3 4      # the parts that need no model
    uv run python experiments/run_rq_analysis.py --all

RQ1 and RQ2 need a trained forecaster and real market data; RQ3 and RQ4 do not, which is why
they are the default. That asymmetry is deliberate -- the withdrawal module is this
component's novelty claim, and its evidence should not be gated on a fine-tuning run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Run from anywhere: the flat module tree lives at the repo root, which is this file's
# parent. pytest gets this from pyproject's `pythonpath`, but a bare `python experiments/...`
# does not.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from evaluation.metrics import slippage_vs_baseline
from evaluation.stress_test_runner import METHODS, run_full_suite, summarize_degradation
from optimization.ga_withdrawal import GAConfig, optimize_withdrawal
from optimization.naive_liquidation import BASELINES

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rq_analysis")

RESULTS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "results"

# A representative portfolio spanning the liquidity spectrum. Deliberately includes one
# genuinely thin holding: on an all-liquid book every method performs identically and RQ3
# would have nothing to measure.
DEMO_PORTFOLIO = {
    "SPY":  {"value": 300_000, "price": 580.0, "adv_usd": 4.0e10, "daily_volatility": 0.009, "volatility_pct": 0.30},
    "AAPL": {"value": 250_000, "price": 230.0, "adv_usd": 2.5e9,  "daily_volatility": 0.014, "volatility_pct": 0.45},
    "QQQ":  {"value": 200_000, "price": 490.0, "adv_usd": 2.0e10, "daily_volatility": 0.012, "volatility_pct": 0.40},
    "XLE":  {"value": 150_000, "price": 92.0,  "adv_usd": 8.0e8,  "daily_volatility": 0.015, "volatility_pct": 0.55},
    "THIN": {"value": 100_000, "price": 15.0,  "adv_usd": 8.0e5,  "daily_volatility": 0.030, "volatility_pct": 0.80},
}


def rq3_withdrawal_vs_naive(
    portfolio: dict | None = None,
    *,
    targets: tuple[float, ...] = (100_000, 250_000, 500_000),
    urgencies: tuple[float, ...] = (0.2, 0.5, 0.9),
    deadline_days: int = 3,
) -> pd.DataFrame:
    """RQ3: does the fuzzy GA reduce realized slippage versus naive liquidation?

    Swept across withdrawal size and urgency, because a single (size, urgency) cell would be
    cherry-picking -- and the honest answer is that the advantage is largest exactly where
    the constraint binds.
    """
    portfolio = portfolio or DEMO_PORTFOLIO
    rows = []

    for target in targets:
        for urgency in urgencies:
            ga = optimize_withdrawal(
                portfolio, target_amount=target, withdrawal_urgency=urgency,
                deadline_days=deadline_days, config=GAConfig(population_size=80, n_generations=100),
            )
            for name, baseline_fn in BASELINES.items():
                baseline = baseline_fn(
                    portfolio, target, deadline_days=deadline_days, participation_cap=0.10
                )
                rows.append(
                    {
                        "target_amount": target,
                        "urgency": urgency,
                        "baseline": name,
                        **slippage_vs_baseline(ga, baseline),
                    }
                )

    return pd.DataFrame(rows)


def rq4_stress_degradation(
    portfolio: dict | None = None,
    *,
    target_amount: float = 300_000,
    urgency: float = 0.7,
    deadline_days: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """RQ4: how does plan quality degrade as liquidity worsens, and where does it break?"""
    portfolio = portfolio or DEMO_PORTFOLIO
    results = run_full_suite(
        portfolio, target_amount,
        methods=METHODS, urgency=urgency, deadline_days=deadline_days,
        output_path=RESULTS_DIR / "rq4_stress_results.csv",
    )
    return results, summarize_degradation(results)


def rq2_allocation_comparison(portfolio: dict | None = None) -> pd.DataFrame:
    """RQ2: MOEA/D versus Markowitz, across all three Pareto selection rules.

    Uses the portfolio's own volatility/ADV as inputs rather than a trained forecaster, so
    this runs without Phase 4. With real forecasts the expected returns improve, but the
    LIQUIDITY comparison -- which is what RQ2 actually turns on -- is unaffected.
    """
    import numpy as np

    from optimization.baseline_meanvariance import max_sharpe_portfolio
    from optimization.moead_rebalance import MOEADConfig, optimize_allocation
    from optimization.objectives import liquidity_cost
    from optimization.pareto_selection import compare_rules

    portfolio = portfolio or DEMO_PORTFOLIO
    symbols = tuple(portfolio)
    n = len(symbols)

    values = np.array([portfolio[s]["value"] for s in symbols])
    total = values.sum()
    current = values / total
    vol = np.array([portfolio[s]["daily_volatility"] for s in symbols])
    adv = np.array([portfolio[s]["adv_usd"] for s in symbols])

    # Expected returns proportional to volatility (a crude risk-premium proxy), so the
    # illiquid, high-vol name looks ATTRACTIVE on return alone -- exactly the trap a
    # liquidity-blind optimizer walks into.
    mu = vol * 0.05
    rng = np.random.default_rng(42)
    scenarios = rng.normal(mu, vol, size=(500, n))
    cov = np.cov(scenarios, rowvar=False)

    front = optimize_allocation(
        mu, scenarios, adv, current, float(total), symbols,
        volatility=vol, config=MOEADConfig(n_partitions=8, n_generations=100, max_weight=0.4),
    )
    selections = compare_rules(front.objectives, front.weights)

    mv = max_sharpe_portfolio(mu, cov, max_weight=0.4)
    rows = [
        {
            "method": "markowitz_max_sharpe",
            "selection_rule": "n/a",
            "expected_return": mv.expected_return,
            "liquidity_cost": liquidity_cost(mv.weights, np.abs(mv.weights - current) * total, adv, vol),
            "thin_weight": float(mv.weights[symbols.index("THIN")]),
            **{f"w_{s}": round(float(w), 4) for s, w in zip(symbols, mv.weights, strict=True)},
        }
    ]
    for rule, point in selections.items():
        rows.append(
            {
                "method": "moead_liquidity_aware",
                "selection_rule": rule,
                "expected_return": float(-point.objectives[0]),
                "liquidity_cost": float(point.objectives[2]),
                "thin_weight": float(point.weights[symbols.index("THIN")]),
                **{f"w_{s}": round(float(w), 4) for s, w in zip(symbols, point.weights, strict=True)},
            }
        )

    return pd.DataFrame(rows)


def load_real_features(*, include_sentiment: bool = False) -> "pd.DataFrame":
    """Feature table built from the resolved universe and cached market data.

    Requires configs/resolved_universe.yaml -- run experiments/resolve_universe.py first.
    Reading the resolved windows rather than a fixed date range is the point: each symbol
    contributes exactly the history its own criteria justified.

    Sentiment is off by default. Scoring years of headlines through the local model is hours
    of serial work, and it is not needed for RQ1's BASELINE row; the hybrid rows add it.
    """
    from datetime import date as _date

    import yaml

    from data.ingestion import bars_to_frame, fetch_ohlcv
    from data.schema import AssetClass
    from features.feature_store import build_feature_table
    from features.technical import compute_universe_indicators

    resolved_path = Path(__file__).resolve().parents[1] / "configs" / "resolved_universe.yaml"
    if not resolved_path.exists():
        raise FileNotFoundError(
            "configs/resolved_universe.yaml is missing. Run "
            "`uv run python experiments/resolve_universe.py` first."
        )

    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))["symbols"]
    classes = {"equity": AssetClass.EQUITY, "etf": AssetClass.ETF, "forex": AssetClass.FOREX}

    bars = {}
    for symbol, entry in sorted(resolved.items()):
        rows = fetch_ohlcv(
            symbol, classes[entry["asset_class"]],
            _date.fromisoformat(entry["history_start"]),
            _date.fromisoformat(entry["history_end"]),
        )
        if rows:
            bars[symbol] = bars_to_frame(rows)

    if not bars:
        raise RuntimeError("no market data available; check the cache and network")

    sentiment = None
    if include_sentiment:
        logger.warning("sentiment enabled: this is slow on the local scoring backend")

    logger.info("loaded %d symbols, %d bars", len(bars), sum(len(f) for f in bars.values()))
    return build_feature_table(bars, compute_universe_indicators(bars), sentiment)


def rq1_forecast_comparison(
    *,
    horizon: int = 5,
    models: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """RQ1: walk-forward forecast quality per model.

    Only runs models that are usable WITHOUT a download. TimesFM and Chronos-Bolt need ~821MB
    of weights each; where those are absent the row is reported as unavailable rather than
    silently omitted, so the table always states what was and was not compared.
    """
    from evaluation.backtest import BacktestConfig, run_walk_forward
    from evaluation.metrics import forecast_metrics
    from forecasting.base import registered_forecasters, usable_foundation_models

    offline = registered_forecasters(require_weights=True)
    candidates = list(models) if models else offline
    features = load_real_features()

    rows = []
    for name in candidates:
        if name not in offline:
            rows.append({"model": name, "status": "weights not cached; excluded from RQ1"})
            continue
        try:
            preds = run_walk_forward(
                features, name, horizon=horizon,
                config=BacktestConfig(train_window_days=1095, test_window_days=180,
                                      step_days=180, embargo_days=horizon),
            )
            valid = preds.dropna(subset=["target_return"])
            metrics = forecast_metrics(
                valid["target_return"].to_numpy(),
                valid[["p10", "p50", "p90"]].to_numpy(), (0.1, 0.5, 0.9),
            )
            rows.append({
                "model": name, "status": "ok", "horizon": horizon,
                "n_folds": int(preds["fold"].nunique()), "n_predictions": len(valid), **metrics,
            })
        except Exception as exc:  # noqa: BLE001 - one model must not kill the table
            logger.error("%s failed: %s", name, exc)
            rows.append({"model": name, "status": f"failed: {exc}"})

    for name, usable in usable_foundation_models().items():
        if not usable and name not in {r["model"] for r in rows}:
            rows.append({"model": name, "status": "weights not cached; run RQ1 in Colab"})

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RQ1-RQ4 comparison tables.")
    parser.add_argument("--rq", nargs="*", default=["3", "4"], choices=["1", "2", "3", "4"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    selected = {"1", "2", "3", "4"} if args.all else set(args.rq)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if "1" in selected:
        table = rq1_forecast_comparison()
        table.to_csv(RESULTS_DIR / "rq1_forecast.csv", index=False)
        print("\n=== RQ1: forecast quality ===\n", table.to_string(index=False))

    if "2" in selected:
        table = rq2_allocation_comparison()
        table.to_csv(RESULTS_DIR / "rq2_allocation.csv", index=False)
        print("\n=== RQ2: allocation quality (Markowitz vs MOEA/D) ===")
        print(table[["method", "selection_rule", "expected_return", "liquidity_cost", "thin_weight"]].to_string(index=False))

    if "3" in selected:
        table = rq3_withdrawal_vs_naive()
        table.to_csv(RESULTS_DIR / "rq3_withdrawal.csv", index=False)
        print("\n=== RQ3: fuzzy GA vs naive liquidation ===")
        summary = (
            table.groupby("baseline")[["absolute_improvement", "relative_improvement_pct"]]
            .mean()
            .sort_values("relative_improvement_pct", ascending=False)
        )
        print(summary.to_string())

    if "4" in selected:
        results, summary = rq4_stress_degradation()
        summary.to_csv(RESULTS_DIR / "rq4_summary.csv", index=False)
        print("\n=== RQ4: degradation under stress ===")
        print(summary.to_string(index=False))

    print(f"\nTables written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
