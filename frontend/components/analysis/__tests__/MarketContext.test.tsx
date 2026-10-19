/**
 * The planner's context panels. What matters: an empirical forecast never reads as a model
 * forecast, fallback to fuzzy rules says why, and the baseline table marks the GA's row.
 */

import { render, screen } from "@testing-library/react";

import { BaselineComparison, CostBreakdownCard, HoldingContext } from "../MarketContext";

const CONTEXT = {
  status: "ready",
  data_age_hours: 2.5,
  currency: "LKR",
  signals_used: { forecast_sigma_daily: 0.012 },
  forecast: { horizon: 5, method: "empirical", p10: -0.04, p50: 0.002, p90: 0.05, reason: "no trained head registered for h=5" },
  fundamentals: { kind: "statements", roe: 0.12, pb: 0.78, f_score: 7, f_score_max: 9, z_zone: "grey", red_flags: ["going_concern_doubt"] },
  decision: { source: "fuzzy", reason: "base checkpoint not cached" },
};

it("labels an empirical band as not a model forecast", () => {
  render(<HoldingContext symbol="JKH.N0000" context={CONTEXT} />);
  expect(screen.getByText("empirical (no model)")).toBeInTheDocument();
  expect(screen.getByText("no trained head registered for h=5")).toBeInTheDocument();
});

it("shows red flags and why Laya did not decide", () => {
  render(<HoldingContext symbol="JKH.N0000" context={CONTEXT} />);
  expect(screen.getByText("going concern doubt")).toBeInTheDocument();
  expect(screen.getByText(/Fuzzy rules only — base checkpoint not cached/)).toBeInTheDocument();
});

it("shows a confident Laya action", () => {
  render(
    <HoldingContext
      symbol="A"
      context={{ ...CONTEXT, decision: { source: "laya", action: { choice: "sell_now", confidence: 0.81 }, applied: ["action sell_now"] } }}
    />,
  );
  expect(screen.getByText(/sell now/)).toBeInTheDocument();
  expect(screen.getByText(/confidence 0.81/)).toBeInTheDocument();
});

it("tells an untracked holding how to get analysis", () => {
  render(<HoldingContext symbol="X" context={{ status: "untracked", signals_used: {} }} />);
  expect(screen.getByText(/save the portfolio on the Portfolio screen/)).toBeInTheDocument();
});

it("compares the GA with the baselines in the plan currency", () => {
  const row = { feasible: true, raised_amount: 1e6, expected_slippage_pct: 0.001, crystallized_loss: 0, days_required: 2 };
  render(
    <BaselineComparison
      currency="LKR"
      rows={[
        { ...row, method: "fuzzy_ga", expected_realized_loss: 1200 },
        { ...row, method: "pro_rata", expected_realized_loss: 3400 },
      ]}
    />,
  );
  expect(screen.getByText("Fuzzy GA (this plan)")).toBeInTheDocument();
  expect(screen.getByText("Pro-rata")).toBeInTheDocument();
  expect(screen.getByText(/1,200\.00/)).toBeInTheDocument();
});

it("breaks the expected cost down", () => {
  render(
    <CostBreakdownCard
      currency="LKR"
      breakdown={{ slippage: 10, timing_risk: 20, forecast_drift: -5, fx_conversion: 7, realized_pnl: -300, crystallized_loss: 300 }}
    />,
  );
  expect(screen.getByText(/FX conversion/)).toBeInTheDocument();
  expect(screen.getByText(/-LKR\s?300\.00|LKR\s?-300\.00/)).toBeInTheDocument();
});
