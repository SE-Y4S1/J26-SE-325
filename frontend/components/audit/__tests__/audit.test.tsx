/**
 * Component 3's audit console pieces. What matters: a route is always named in words (never
 * colour alone), legacy records are not presented as assurance decisions, missing signals and
 * vetoes are visible, and the record table links each decision to its detail page.
 */

import { render, screen, within } from "@testing-library/react";

import { DecisionTable } from "../DecisionTable";
import { DeferralChart } from "../DeferralChart";
import { RouteDistribution } from "../RouteDistribution";
import { SignalBreakdown } from "../SignalBreakdown";
import { IntegrityBadge, RouteBadge, outcomeLabel, routeOf } from "../badges";
import type { Assurance, AuditRecord } from "@/lib/api/audit";

function assurance(overrides: Partial<Assurance> = {}): Assurance {
  return {
    engine: "assurance-engine/1.0", policyId: "P002", policyVersion: "1.0", policyHash: "0xabc",
    aiDecision: "APPROVE", route: "VERIFY", assessedRoute: "VERIFY", assuranceScore: 0.52,
    components: { confidence: 0.5, disagreement: 0.5, context: 0.4 },
    context: { amount: 0.4, upstreamRules: 0, modelValidation: 1, transactionType: 0.5 },
    vetoes: [], signalQuality: "DEGRADED", degradedSignals: ["disagreement"], unreliableSignals: ["confidence"],
    reasons: ["Assurance score 0.52 …"], escalation: null,
    ...overrides,
  };
}

describe("route labels", () => {
  it("name every route in words", () => {
    render(<><RouteBadge route="AUTO" /><RouteBadge route="VERIFY" /><RouteBadge route="HUMAN_REVIEW" /><RouteBadge route="LEGACY" /></>);
    for (const label of ["Automatic", "Verification required", "Human review", "Legacy policy"]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
  });

  it("treat a P001-routed or v1 record as legacy, not as an assurance decision", () => {
    expect(routeOf(null)).toBe("LEGACY");
    expect(routeOf(assurance({ signalQuality: "LEGACY", route: "AUTO" }))).toBe("LEGACY");
    expect(routeOf(assurance({ route: "HUMAN_REVIEW", signalQuality: "FULL" }))).toBe("HUMAN_REVIEW");
  });

  it("describe integrity and outcomes in plain words", () => {
    render(<><IntegrityBadge status="CHAIN_BROKEN" /><IntegrityBadge status="SOMETHING_NEW" /></>);
    expect(screen.getByText("Review link broken")).toBeInTheDocument();
    expect(screen.getByText("something new")).toBeInTheDocument();
    expect(outcomeLabel("PENDING_HUMAN_REVIEW")).toBe("Held");
    expect(outcomeLabel("HUMAN_APPROVED")).toBe("Approved after review");
  });
});

describe("RouteDistribution", () => {
  it("labels each route with its count and share", () => {
    render(<RouteDistribution routes={{ AUTO: 6, VERIFY: 3, HUMAN_REVIEW: 1, LEGACY: 0 }} />);
    const auto = screen.getByText("Automatic").closest("li")!;
    expect(within(auto).getByText("6")).toBeInTheDocument();
    expect(within(auto).getByText("60%")).toBeInTheDocument();
    expect(screen.getByRole("img")).toHaveAccessibleName("Automatic 6, Verification required 3, Human review 1");
  });

  it("says so when nothing has been routed", () => {
    render(<RouteDistribution routes={{}} />);
    expect(screen.getByText("No decisions routed yet.")).toBeInTheDocument();
  });
});

describe("SignalBreakdown", () => {
  it("shows each signal's value and flags missing and unvalidated signals", () => {
    render(<SignalBreakdown assurance={assurance()} thresholds={{ autoMin: 0.7, verifyMin: 0.45 }} />);
    expect(screen.getByText("0.52")).toBeInTheDocument();
    expect(screen.getByText("missing — scored conservatively")).toBeInTheDocument();
    expect(screen.getByText("unvalidated model — floored")).toBeInTheDocument();
    expect(screen.getByLabelText(/Uncertainty by signal/)).toBeInTheDocument();
  });

  it("lists vetoes with their reason", () => {
    render(<SignalBreakdown assurance={assurance({ vetoes: [{ code: "MODEL_DISAGREEMENT", detail: "Models disagree by 0.65 (> 0.5)." }] })} />);
    expect(screen.getByText("MODEL_DISAGREEMENT")).toBeInTheDocument();
    expect(screen.getByText(/Models disagree by 0.65/)).toBeInTheDocument();
  });

  it("explains a legacy decision instead of inventing a breakdown", () => {
    render(<SignalBreakdown assurance={assurance({ components: null, assuranceScore: null, signalQuality: "LEGACY" })} />);
    expect(screen.getByText(/legacy P001 policy/)).toBeInTheDocument();
  });
});

describe("DeferralChart", () => {
  const curve = (slope: number) => [0, 0.1, 0.2].map((h) => ({ human_review_rate: h, incorrect_automatic_rate: 0.03 - slope * h }));

  it("labels every plotted method directly, so lines are not told apart by colour alone", () => {
    render(<DeferralChart curves={{ assurance: curve(0.1), risk_band: curve(0.05), confidence_only: curve(0.07) }} />);
    expect(screen.getByRole("img")).toHaveAccessibleName(/Incorrect automatic decisions by human-review rate/);
    for (const label of ["Assurance engine", "Risk only", "Confidence only"]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
    expect(screen.queryByText("Learned deferral")).not.toBeInTheDocument();
  });

  it("says so when there is nothing to plot", () => {
    render(<DeferralChart curves={{}} />);
    expect(screen.getByText("No curves in this result.")).toBeInTheDocument();
  });
});

describe("DecisionTable", () => {
  it("links each decision to its detail page and shows route, score and integrity", () => {
    const record = {
      transactionId: "EXE-123", amount: 2500, timestamp: "2026-10-20T10:00:00.000Z", finalDecision: "PENDING_HUMAN_REVIEW",
      verificationStatus: "VERIFIED_ON_CHAIN", assurance: assurance({ route: "HUMAN_REVIEW", signalQuality: "FULL", assuranceScore: 0.31 }),
    } as unknown as AuditRecord;
    render(<DecisionTable records={[record]} />);
    expect(screen.getByRole("link", { name: "EXE-123" })).toHaveAttribute("href", "/audit/decisions/EXE-123");
    expect(screen.getByText("Human review")).toBeInTheDocument();
    expect(screen.getByText("0.31")).toBeInTheDocument();
    expect(screen.getByText("Held")).toBeInTheDocument();
    expect(screen.getByText("Verified on-chain")).toBeInTheDocument();
  });
});
