"""RQ3 / RQ5: liquidation ablation scored on REALISED outcomes.

Each configuration plans the same historical withdrawal scenarios with only the information
available on the scenario date, then is scored against the prices that actually followed:

    implementation shortfall = 1 - proceeds / (arrival value of what was sold)
    proceeds = sum over fills of quantity x realised close on the fill day x (1 - modelled impact)
    excess wealth vs hold   = (proceeds + retained holdings at the deadline - all holdings at
                              the deadline) / initial value   -- credits WHICH names were sold

Scoring on realised prices rather than on the planner's own expected cost matters: a planner
judged by its own cost model wins by construction.

CONFIGURATIONS (RQ3: does forecast/sentiment information help? RQ5: which decision layer?)
  pro_rata, largest_first, most_liquid_first   naive baselines
  fuzzy_ga                                     liquidity/volatility fuzzy rules + GA, no signals
  fuzzy_ga+forecast                            + forecast drift/spread (hybrid when a head is
                                                 installed, else the empirical band: recorded)
  fuzzy_ga+oracle                              + the REALISED drift (an upper bound on what any
                                                 forecast could add; not a deployable method)
  laya                                         + Laya Stage 1/2 (zero-shot or fine-tuned;
                                                 only when an engine is available)
Scenarios are stratified by liquidity: each holding's position is sized in days of its own
traded value, so thin CSE names really are slow to exit.

    python -m experiments.rq_liquidation --bars bars.parquet --scenarios 300 --out artifacts/rq_liquidation
"""

from __future__ import annotations

import argparse
import logging
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEADLINES = (1, 3, 5, 10, 21)
PARTICIPATION_CAP = 0.10
WARMUP = 80


def sample_scenarios(panel: dict[str, pd.DataFrame], n: int, rng: random.Random) -> list[dict[str, Any]]:
    """Point-in-time scenarios: holdings at date t, a target, a deadline, and the realised path."""
    symbols = [s for s, f in panel.items() if len(f) > WARMUP + max(DEADLINES) + 2]
    scenarios: list[dict[str, Any]] = []
    attempts = 0
    while len(scenarios) < n and attempts < n * 20 and len(symbols) >= 2:
        attempts += 1
        chosen = rng.sample(symbols, k=min(len(symbols), rng.randint(2, 5)))
        deadline = rng.choice(DEADLINES)
        common = set.intersection(*[set(pd.to_datetime(panel[s]["timestamp"]).iloc[WARMUP:-deadline - 1]) for s in chosen])
        if not common:
            continue
        when = rng.choice(sorted(common))
        holdings, paths, history = {}, {}, {}
        for symbol in chosen:
            frame = panel[symbol]
            stamps = pd.to_datetime(frame["timestamp"])
            t = int((stamps <= when).sum()) - 1
            close = frame["close"].to_numpy(dtype=float)
            returns = np.diff(np.log(close[max(0, t - 60) : t + 1]))
            adv = float((frame["close"] * frame["volume"]).iloc[max(0, t - 20) : t + 1].mean())
            if not adv or not math.isfinite(adv):
                break
            days_of_volume = rng.choice([0.05, 0.3, 1.0, 3.0])     # liquidity stratum
            value = days_of_volume * adv
            holdings[symbol] = {
                "value": value, "price": close[t], "quantity": value / close[t], "adv_usd": adv,
                "daily_volatility": float(returns.std()) if len(returns) > 5 else 0.02,
                "volatility_pct": 0.5, "cost_basis": close[t] * rng.uniform(0.7, 1.3),
                "liquidity_days": days_of_volume,
            }
            paths[symbol] = close[t : t + deadline + 1]
            history[symbol] = frame.iloc[: t + 1]
        else:
            total = sum(h["value"] for h in holdings.values())
            scenarios.append({"date": str(pd.Timestamp(when).date()), "deadline": deadline, "holdings": holdings,
                              "paths": paths, "history": history, "target": total * rng.uniform(0.1, 0.6),
                              "urgency": round(rng.uniform(0, 1), 2)})
    return scenarios


def realised(plan, scenario: dict[str, Any]) -> dict[str, float]:  # noqa: ANN001 - WithdrawalPlan
    """Score a plan against the realised path."""
    proceeds = arrival = 0.0
    for step in plan.steps:
        path = scenario["paths"][step.symbol]
        fill = path[min(step.execution_day, len(path) - 1)]
        proceeds += step.quantity * fill * (1 - step.expected_slippage_pct)
        arrival += step.quantity * path[0]
    shortfall = 1 - proceeds / arrival if arrival else 0.0
    # Terminal wealth vs simply holding everything to the deadline: credits WHICH holdings
    # were sold (sell what then fell, keep what then rose), which shortfall on the sold
    # shares alone cannot see. Higher is better.
    sold: dict[str, float] = {}
    for step in plan.steps:
        sold[step.symbol] = sold.get(step.symbol, 0.0) + step.quantity
    initial = sum(h["quantity"] * h["price"] for h in scenario["holdings"].values()) if "holdings" in scenario else 0.0
    hold_all = retained = 0.0
    for symbol, holding in scenario.get("holdings", {}).items():
        end = scenario["paths"][symbol][-1]
        hold_all += holding["quantity"] * end
        retained += max(0.0, holding["quantity"] - sold.get(symbol, 0.0)) * end
    excess = (proceeds + retained - hold_all) / initial if initial else 0.0
    return {"implementation_shortfall": shortfall, "excess_wealth_vs_hold": excess, "raised_realised": proceeds,
            "feasible": float(plan.feasible), "days_required": plan.days_required,
            "crystallized_loss": plan.crystallized_loss}


def forecast_signals(history: pd.DataFrame, deadline: int, forecast_fn: Callable | None) -> tuple[dict[str, float], str]:
    """Point-in-time forecast signals for one holding (hybrid via forecast_fn, else empirical)."""
    from forecasting.serving import empirical_quantiles
    from optimization.signals import holding_signals

    horizon = 5 if deadline <= 10 else 21
    if forecast_fn is not None:
        band = forecast_fn(history, horizon)
        if band and band.get("method") == "hybrid":
            return holding_signals({"metrics": {"forecast": {f"h{horizon}": band}}}, deadline), "hybrid"
    values = empirical_quantiles(history["close"], horizon)
    if values is None:
        return {}, "none"
    band = {"horizon": horizon, "method": "empirical", "p10": values[0], "p50": values[1], "p90": values[2]}
    return holding_signals({"metrics": {"forecast": {f"h{horizon}": band}}}, deadline), "empirical"


def run(scenarios: list[dict[str, Any]], *, forecast_fn: Callable | None = None, laya_engine=None,  # noqa: ANN001
        ga_config=None) -> pd.DataFrame:  # noqa: ANN001
    from optimization.ga_withdrawal import GAConfig, optimize_withdrawal
    from optimization.naive_liquidation import BASELINES

    config = ga_config or GAConfig()
    rows = []
    for index, scenario in enumerate(scenarios):
        base = scenario["holdings"]
        common = {"deadline": scenario["deadline"], "scenario": index, "date": scenario["date"],
                  "min_liquidity_days": min(h["liquidity_days"] for h in base.values())}

        def record(name: str, plan, extra: dict | None = None) -> None:  # noqa: ANN001
            rows.append({"config": name, **common, **realised(plan, scenario), **(extra or {})})

        for name, baseline in BASELINES.items():
            record(name, baseline(base, scenario["target"], deadline_days=scenario["deadline"],
                                  participation_cap=PARTICIPATION_CAP))

        def ga(holdings: dict, urgency: float = scenario["urgency"], risk: float = 0.5):  # noqa: ANN202
            return optimize_withdrawal(holdings, scenario["target"], withdrawal_urgency=urgency, risk_tolerance=risk,
                                       deadline_days=scenario["deadline"], participation_cap=PARTICIPATION_CAP,
                                       config=config)

        record("fuzzy_ga", ga(base))

        with_forecast, sources = {}, set()
        for symbol, holding in base.items():
            signals, source = forecast_signals(scenario["history"][symbol], scenario["deadline"], forecast_fn)
            with_forecast[symbol] = {**holding, **signals}
            sources.add(source)
        record("fuzzy_ga+forecast", ga(with_forecast), {"forecast_source": "+".join(sorted(sources))})

        oracle = {}
        for symbol, holding in base.items():
            path = scenario["paths"][symbol]
            drift = math.log(path[-1] / path[0]) / max(len(path) - 1, 1)
            oracle[symbol] = {**holding, "drift_daily": drift, "forecast_outlook": 0.5 + 0.5 * math.tanh(drift * 50)}
        record("fuzzy_ga+oracle", ga(oracle))

        if laya_engine is not None:
            record("laya", _laya_plan(scenario, with_forecast, laya_engine, ga), {"laya_version": laya_engine.version})
    return pd.DataFrame(rows)


def _laya_plan(scenario, holdings, engine, ga):  # noqa: ANN001, ANN202
    from decision import engine as decision_engine
    from features.technical import technical_snapshot

    snapshots = {}
    for symbol, history in scenario["history"].items():
        metrics = {"technical": technical_snapshot(history)}
        snapshots[symbol] = {"metrics": {"laya_analysis": decision_engine.analyse(engine, metrics, "equity")}}
    decision_engine.set_engine(engine)
    try:
        inputs = {"urgency": scenario["urgency"], "risk_tolerance": 0.5, "liquidity_target": scenario["target"],
                  "deadline_days": scenario["deadline"]}
        state, inputs, _, _ = decision_engine.apply(holdings, snapshots, inputs)
    finally:
        decision_engine.reset()
    return ga(state, inputs["urgency"], inputs["risk_tolerance"])


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean realised shortfall (bps), feasibility and win rate vs pro-rata, per configuration,
    overall and for scenarios containing a thin holding (>= 1 day of volume)."""
    pivot = results.pivot_table(index="scenario", columns="config", values="implementation_shortfall")
    out = []
    for subset_name, mask in (("all", results["min_liquidity_days"] >= 0),
                              ("has_thin_holding", results.groupby("scenario")["min_liquidity_days"].transform("max") >= 1)):
        part = results[mask]
        scenarios = part["scenario"].unique()
        for config, group in part.groupby("config"):
            wins = (pivot.loc[scenarios, config] < pivot.loc[scenarios, "pro_rata"]).mean() if "pro_rata" in pivot else np.nan
            feasible = group[group["feasible"] == 1.0]
            out.append({"subset": subset_name, "config": config, "scenarios": len(group),
                        "shortfall_bps": group["implementation_shortfall"].mean() * 1e4,
                        # Only plans that raised the cash: a plan that sells less always looks
                        # cheaper, so shortfall must be read together with feasibility.
                        "shortfall_bps_feasible": feasible["implementation_shortfall"].mean() * 1e4,
                        "excess_wealth_bps_feasible": feasible["excess_wealth_vs_hold"].mean() * 1e4,
                        "shortfall_bps_p90": group["implementation_shortfall"].quantile(0.9) * 1e4,
                        "feasible_rate": group["feasible"].mean(), "beats_pro_rata": wins})
    return pd.DataFrame(out).sort_values(["subset", "shortfall_bps"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bars", type=Path, required=True, help="bars.parquet (Part A) or any OHLCV panel")
    parser.add_argument("--scenarios", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("artifacts/rq_liquidation"))
    parser.add_argument("--laya", action="store_true", help="include Laya (needs a loadable engine)")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    bars = pd.read_parquet(args.bars)
    panel = {str(s): f.sort_values("timestamp").reset_index(drop=True) for s, f in bars.groupby("symbol")}
    scenarios = sample_scenarios(panel, args.scenarios, random.Random(args.seed))

    forecast_fn = None
    from forecasting.serving import get_service

    service = get_service()
    if service.available():
        def forecast_fn(history: pd.DataFrame, horizon: int) -> dict:
            return service.forecast(str(history["symbol"].iloc[-1]), history, horizon)

    engine = None
    if args.laya:
        from decision.engine import get_engine

        engine, reason = get_engine(block=True)
        logger.info("laya: %s", reason if engine is None else engine.version)

    results = run(scenarios, forecast_fn=forecast_fn, laya_engine=engine)
    args.out.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / "results.csv", index=False)
    summary = summarise(results)
    summary.to_csv(args.out / "summary.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
