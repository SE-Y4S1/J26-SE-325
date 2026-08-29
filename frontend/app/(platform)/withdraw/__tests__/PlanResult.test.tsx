/**
 * The withdrawal result panel.
 *
 * The behaviour that matters here is research-relevant rather than cosmetic: an infeasible
 * plan comes back as HTTP 200 with feasible=false, and it must read as an ANSWER with a
 * shortfall, not as a failure. RQ4 depends on infeasible plans being visible.
 */

import { render, screen } from "@testing-library/react";

import { PlanResult } from "../page";

const BASE = {
  assets_to_sell: [
    {
      symbol: "THIN",
      fraction: 0.17106,
      quantity: 120,
      expected_price: 41.2,
      expected_slippage_pct: 0.0123,
      execution_day: 0,
    },
  ],
  raised_amount: 40_000,
  target_amount: 50_000,
  shortfall: 10_000,
  expected_slippage_pct: 0.0123,
  expected_realized_loss: -615,
  residual_portfolio_weights: { THIN: 0.4 },
  days_required: 3,
  feasible: false,
  model_version: "unregistered",
  fuzzy_rule_trace: [],
  agent_reasoning_trace: [],
  generated_at: "2026-08-29T00:00:00Z",
};

it("presents an infeasible plan as a limit, not an error", () => {
  render(<PlanResult plan={BASE as never} />);

  expect(screen.getByText(/Cannot raise the full amount/)).toBeInTheDocument();
  expect(screen.getByText(/shortfall of \$10,000\.00/)).toBeInTheDocument();
  // The wording has to attribute it to the participation cap rather than a fault, or a
  // reader takes a legitimate result for a bug.
  expect(screen.getByText(/not by an error/)).toBeInTheDocument();
});

it("still shows the numbers that were achieved when infeasible", () => {
  // A shortfall does not make the rest of the plan meaningless -- the raised amount and the
  // execution schedule are exactly what the user acts on.
  render(<PlanResult plan={BASE as never} />);

  expect(screen.getByText("$40,000.00")).toBeInTheDocument();
  expect(screen.getByText("1.230%")).toBeInTheDocument();
  expect(screen.getByText("THIN")).toBeInTheDocument();
});

it("reports success when the full amount can be raised", () => {
  render(
    <PlanResult
      plan={{ ...BASE, feasible: true, raised_amount: 50_000, shortfall: 0 } as never}
    />,
  );

  expect(screen.getByText("Feasible")).toBeInTheDocument();
  expect(screen.queryByText(/Cannot raise the full amount/)).not.toBeInTheDocument();
});

it("renders the fuzzy rule trace when the backend supplies one", () => {
  // This trace is the auditability evidence the TAF's legal-impact section calls for, and
  // Component 4 explains it. Silently dropping it would remove the justification for every
  // number on the screen.
  render(
    <PlanResult
      plan={{
        ...BASE,
        // Shape copied from the backend's rule_trace_to_dict, not invented: it emits
        // `rules` with rule_id / if / then / strength. An invented fixture would let the
        // test pass while the real payload rendered nothing.
        fuzzy_rule_trace: [
          {
            symbol: "THIN",
            sell_priority: 83.9,
            rules: [
              {
                rule_id: "R21",
                if: "urgency[high] AND market_volatility[turbulent] AND position_liquidity[illiquid]",
                then: "sell_priority[very_high]",
                strength: 0.8,
              },
            ],
          },
        ],
      } as never}
    />,
  );

  expect(screen.getByText(/83\.9/)).toBeInTheDocument();
});

it("survives a missing trace rather than crashing the screen", () => {
  // fuzzy_rule_trace is declared list[dict] on the backend, so it is not covered by the
  // generated types and nothing else would catch an undefined here.
  const { container } = render(
    <PlanResult plan={{ ...BASE, fuzzy_rule_trace: undefined } as never} />,
  );
  expect(container).not.toBeEmptyDOMElement();
});
