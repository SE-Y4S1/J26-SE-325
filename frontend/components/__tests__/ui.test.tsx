/**
 * Formatting helpers. Every number the user reads goes through these, and the withdrawal
 * screen shows losses and slippage, so negatives and very small fractions matter.
 */

import { compact, money, percent } from "../ui";

describe("money", () => {
  it("formats as USD currency", () => {
    expect(money(1234.5)).toBe("$1,234.50");
  });

  it("shows a negative as negative", () => {
    // expected_realized_loss is displayed with this. A loss rendered as a positive number
    // would invert the meaning of the whole panel.
    expect(money(-250)).toBe("-$250.00");
  });

  it("handles zero", () => {
    expect(money(0)).toBe("$0.00");
  });
});

describe("percent", () => {
  it("scales a fraction and keeps three digits by default", () => {
    // Slippage arrives as a fraction (0.0123 = 1.23%). Rendering the fraction raw would
    // understate market impact by two orders of magnitude.
    expect(percent(0.0123)).toBe("1.230%");
  });

  it("does not round a small but non-zero impact away to nothing", () => {
    expect(percent(0.00008)).toBe("0.008%");
  });

  it("respects an explicit digit count", () => {
    expect(percent(0.5, 1)).toBe("50.0%");
  });
});

describe("compact", () => {
  it("shortens large values", () => {
    expect(compact(1_500_000)).toBe("1.5M");
  });
});
