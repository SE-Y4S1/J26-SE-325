import React from "react";
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import { TrustPanelScreen } from "../TrustPanelScreen";
import { parsePortfolioResult } from "../explanationParser";

// Helper evidence fixtures for the 5 scenarios
const LIVE_PORTFOLIO_EVIDENCE = `
PORTFOLIO ANALYSIS — LIVE PLATFORM RECORD

Data Source:
Live platform portfolio store (http://localhost:8100).

Resolved User ID:
1

Portfolio ID:
101

Live Lookup Status:
Active live portfolio record retrieved from platform service.

Portfolio Name:
Main Growth Portfolio

Base Currency:
USD

Total Portfolio Value:
USD 750,000.00

Holding Count:
2

RECORDED HOLDINGS:
- AAPL: 1,500.00 units @USD 220.00 = USD 330,000.00 (44.0% of portfolio); Cost Basis: USD 180.00; ADV: 45,000,000
- MSFT: 1,000.00 units @USD 420.00 = USD 420,000.00 (56.0% of portfolio); Cost Basis: USD 350.00; ADV: 28,000,000

OBSERVATIONS:
- Live holdings and asset valuations retrieved directly from the platform portfolio store.
- Total recorded market value: USD 750,000.00.
- Largest recorded positions: AAPL, MSFT.

METRIC NOTICE:
The platform portfolio service stores live asset holdings and valuation.
`;

const EMPTY_PORTFOLIO_EVIDENCE = `
PORTFOLIO ANALYSIS — FALLBACK DATA (NO LIVE RECORD RETRIEVED)

Data Source:
Local mock baseline (live platform service at http://localhost:8100: No portfolios registered for user on platform service (http://localhost:8100)).

STATUS:
NO LIVE PORTFOLIO RECORD RETRIEVED for user ID '1'.
Platform service note: No portfolios registered for user on platform service (http://localhost:8100).

Live Lookup Status:
Empty portfolio list (User has no portfolios on platform)

Resolved User Identity:
1 (Requested: '1')

FALLBACK NOTICE:
No live portfolio was retrieved for user '1' from the platform service (account has no registered portfolios) (No portfolios registered for user on platform service (http://localhost:8100)).
The platform service confirmed that no portfolios or active holdings are registered for this account.
This record is NOT the user's actual portfolio. All metrics, valuations, and allocations below are synthetic baseline estimates for demonstration only. The user does NOT have a verified balance of LKR 1,000,000 or active live holdings on record.

User ID:
1

ILLUSTRATIVE MOCK BASELINE ESTIMATES (NOT ACTUAL USER PORTFOLIO):
- Illustrative Baseline Valuation: LKR 1,000,000 (Mock Baseline Demonstration Only)
- Illustrative Expected Annual Return: 11.8% (Synthetic Estimate)
- Illustrative Risk Score: 38/100 (Synthetic Estimate)
- Illustrative Risk Level: Moderate (Synthetic Estimate)
- Illustrative Diversification Score: 74/100 (Synthetic Estimate)
- Illustrative Liquidity: High (Synthetic Estimate)

ILLUSTRATIVE CURRENT ALLOCATION (Mock Baseline Example Only):
- Technology: 35% (Mock Baseline)
- Banking: 25% (Mock Baseline)

IMPORTANT DISCLAIMER:
Do not describe these synthetic metrics, asset allocations, or observations as the user's actual portfolio.
Live holdings: Unverified (No live portfolio registered on platform).
Live market value: Unverified (No live portfolio record).
`;

const SERVICE_UNAVAILABLE_EVIDENCE = `
PORTFOLIO ANALYSIS — FALLBACK DATA (NO LIVE RECORD RETRIEVED)

Data Source:
Local mock baseline (live platform service at http://localhost:8100: Request timed out (connection error)).

STATUS:
NO LIVE PORTFOLIO RECORD RETRIEVED for user ID '1'.
Platform service note: Request timed out (connection error).

Live Lookup Status:
Service unavailable (Request timed out (connection error))

Resolved User Identity:
1 (Requested: '1')

FALLBACK NOTICE:
Platform service unreachable for user '1' (Request timed out (connection error)).
This does not indicate an empty portfolio; live holdings could not be retrieved due to a platform service connection failure or timeout.
This record is NOT the user's actual portfolio. All metrics, valuations, and allocations below are synthetic baseline estimates for demonstration only.

IMPORTANT DISCLAIMER:
Live holdings: Unverified (Platform service unreachable or timed out).
Live market value: Unverified (No live portfolio record).
`;

const PARTIAL_PORTFOLIO_EVIDENCE = `
PORTFOLIO ANALYSIS — LIVE PLATFORM RECORD

Data Source:
Live platform portfolio store (http://localhost:8100).

Resolved User ID:
1

Portfolio ID:
103

Live Lookup Status:
Active live portfolio record retrieved (partial fields marked)

Portfolio Name:
Partial Tech Holdings

Base Currency:
USD

Total Portfolio Value:
Unavailable / Not recorded

Holding Count:
2

RECORDED HOLDINGS:
- NVDA: 100.00 units @ Price: Unavailable = Value: Unavailable (Weight: Unavailable)
- TSLA: 50.00 units @USD 250.00 = USD 12,500.00 (Weight: Unavailable); Cost Basis: Unavailable;

OBSERVATIONS:
- Live holdings and asset valuations retrieved directly from the platform portfolio store.
- Total recorded market value: Unavailable / Not recorded.
- Largest recorded positions: NVDA, TSLA.
- Notice: Some fields (such as valuation, price, or cost basis) are not recorded on the platform and are explicitly marked as Unavailable.
`;

const DISABLED_CONTEXT_EVIDENCE = `
PORTFOLIO ANALYSIS — PERSONALIZED CONTEXT DISABLED

Data Source:
None (Personalized financial context is disabled by user preference).

STATUS:
Personalized financial context is disabled by user preference.
Account-specific portfolio lookup was not performed, and no personalized data was retrieved.

NOTICE:
Under your current Responsible AI settings, access to account-specific investment portfolios and personal asset holdings is disabled.
No anonymous or fallback portfolio data has been substituted.
Live holdings: Unverified (Personalized context disabled).
`;

describe("Component 4 Portfolio Analysis Scenarios & Fallback Audit", () => {
  // ---------------------------------------------------------------------------
  // Scenario 1: Live portfolio exists
  // ---------------------------------------------------------------------------
  it("Scenario 1: Live portfolio displays verified holdings and does not substitute synthetic baseline values", () => {
    const parsed = parsePortfolioResult(LIVE_PORTFOLIO_EVIDENCE);
    expect(parsed.statusType).toBe("live");
    expect(parsed.isPartial).toBe(false);
    expect(parsed.provenanceLabel).toBe("Live Service Evidence (Platform Portfolio)");
    expect(parsed.totalValue).toBe("750,000.00");
    expect(parsed.holdingCount).toBe(2);

    render(
      <TrustPanelScreen
        latestEvidence={[{ tool: "get_portfolio_analysis", result: LIVE_PORTFOLIO_EVIDENCE }]}
      />
    );

    // Status badge must be LIVE PORTFOLIO
    expect(screen.getAllByText("LIVE PORTFOLIO").length).toBeGreaterThan(0);
    expect(screen.getByText(/Live portfolio record retrieved from the platform portfolio store/)).toBeInTheDocument();
    expect(screen.getAllByText(/USD 750,000.00/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/2 verified holdings/).length).toBeGreaterThan(0);
    // Must NOT display synthetic demo baseline valuation as actual
    expect(screen.queryByText(/LKR 1,000,000 \(Illustrative\)/)).not.toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Scenario 2: No portfolio exists
  // ---------------------------------------------------------------------------
  it("Scenario 2: Empty portfolio result reports zero holdings and labels mock fallback values", () => {
    const parsed = parsePortfolioResult(EMPTY_PORTFOLIO_EVIDENCE);
    expect(parsed.statusType).toBe("empty");
    expect(parsed.isFallback).toBe(true);
    expect(parsed.provenanceLabel).toBe("Tool Evidence — Synthetic Demo Data (0 Live Holdings)");

    render(
      <TrustPanelScreen
        latestEvidence={[{ tool: "get_portfolio_analysis", result: EMPTY_PORTFOLIO_EVIDENCE }]}
      />
    );

    // Status badge must be DEMO BASELINE
    expect(screen.getAllByText("DEMO BASELINE").length).toBeGreaterThan(0);
    // Explanation must state 0 verified holdings and label mock data
    expect(screen.getAllByText(/0 verified holdings/).length).toBeGreaterThan(0);
    expect(screen.getByText(/synthetic baseline demonstrations shown for system demonstration purposes/)).toBeInTheDocument();
    // Checklist confirms 0 active holdings
    expect(screen.getByText(/Platform service query: No live portfolio record found on platform service \(0 verified holdings on record\)/)).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Scenario 3: Platform unavailable
  // ---------------------------------------------------------------------------
  it("Scenario 3: Platform unavailable reports service outage and does NOT claim zero holdings", () => {
    const parsed = parsePortfolioResult(SERVICE_UNAVAILABLE_EVIDENCE);
    expect(parsed.statusType).toBe("service_error");
    expect(parsed.provenanceLabel).toBe("Service Unavailable (Platform Error / Timeout)");
    expect(parsed.totalValue).toBe("Unavailable");

    render(
      <TrustPanelScreen
        latestEvidence={[{ tool: "get_portfolio_analysis", result: SERVICE_UNAVAILABLE_EVIDENCE }]}
      />
    );

    // Status badge must be SERVICE UNAVAILABLE
    expect(screen.getAllByText("SERVICE UNAVAILABLE").length).toBeGreaterThan(0);
    // Must NOT claim user has 0 holdings
    expect(screen.queryByText(/0 verified holdings on record/)).not.toBeInTheDocument();
    // Explanation must state holdings remain unverified due to platform outage
    expect(screen.getByText(/Live holdings remain unverified due to platform service unavailability; this does NOT indicate an empty portfolio or zero holdings/)).toBeInTheDocument();
    // Checklist reflects outage
    expect(screen.getByText(/Service unreachable or timed out — live holdings could not be retrieved/)).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Scenario 4: Partial portfolio data
  // ---------------------------------------------------------------------------
  it("Scenario 4: Partial portfolio data marks missing fields as Unavailable without inventing values", () => {
    const parsed = parsePortfolioResult(PARTIAL_PORTFOLIO_EVIDENCE);
    expect(parsed.statusType).toBe("live");
    expect(parsed.isPartial).toBe(true);
    expect(parsed.provenanceLabel).toBe("Live Service Evidence (Platform Portfolio — Partial Data)");
    expect(parsed.totalValue).toContain("Unavailable");

    render(
      <TrustPanelScreen
        latestEvidence={[{ tool: "get_portfolio_analysis", result: PARTIAL_PORTFOLIO_EVIDENCE }]}
      />
    );

    // Status badge must be LIVE (PARTIAL DATA)
    expect(screen.getAllByText("LIVE (PARTIAL DATA)").length).toBeGreaterThan(0);
    // Explanation must state unrecorded fields are marked as unavailable
    expect(screen.getByText(/Verified fields are displayed; unrecorded valuation\/cost fields are marked as unavailable rather than estimated/)).toBeInTheDocument();
    // Checklist reflects partial data handling
    expect(screen.getByText(/Verified fields displayed; unrecorded valuation\/price fields marked Unavailable/)).toBeInTheDocument();
    expect(screen.getByText(/Synthetic baseline values: Strictly omitted from live record/)).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Scenario 5: Personalized context disabled
  // ---------------------------------------------------------------------------
  it("Scenario 5: Personalized context disabled suppresses lookup and mock portfolio values", () => {
    const parsed = parsePortfolioResult(DISABLED_CONTEXT_EVIDENCE);
    expect(parsed.statusType).toBe("disabled");
    expect(parsed.hasEvidence).toBe(false);
    expect(parsed.provenanceLabel).toBe("Personalized Context Disabled");

    render(
      <TrustPanelScreen
        latestEvidence={[{ tool: "get_portfolio_analysis", result: DISABLED_CONTEXT_EVIDENCE }]}
      />
    );

    // Status badge must be CONTEXT DISABLED
    expect(screen.getAllByText("CONTEXT DISABLED").length).toBeGreaterThan(0);
    // Explanation must confirm blocking without fabricating anonymous holdings
    expect(screen.getByText(/Personalized financial context is disabled by user preference\. Account-specific portfolio queries are blocked without fabricating anonymous holdings/)).toBeInTheDocument();
    // Checklist reflects suppressed lookup
    expect(screen.getByText(/Holdings lookup: Suppressed \(No mock holdings substituted\)/)).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Scenario 6: Evidence & Trust Panel consistent provenance separation
  // ---------------------------------------------------------------------------
  it("Scenario 6: Evidence and Trust Panel consistently distinguish all 5 conditions without ambiguity", () => {
    const live = parsePortfolioResult(LIVE_PORTFOLIO_EVIDENCE);
    const empty = parsePortfolioResult(EMPTY_PORTFOLIO_EVIDENCE);
    const outage = parsePortfolioResult(SERVICE_UNAVAILABLE_EVIDENCE);
    const partial = parsePortfolioResult(PARTIAL_PORTFOLIO_EVIDENCE);
    const disabled = parsePortfolioResult(DISABLED_CONTEXT_EVIDENCE);

    // Verify distinct statusType classification
    expect(live.statusType).toBe("live");
    expect(empty.statusType).toBe("empty");
    expect(outage.statusType).toBe("service_error");
    expect(partial.statusType).toBe("live");
    expect(partial.isPartial).toBe(true);
    expect(disabled.statusType).toBe("disabled");

    // Verify distinct provenance labels
    const labels = [
      live.provenanceLabel,
      empty.provenanceLabel,
      outage.provenanceLabel,
      partial.provenanceLabel,
      disabled.provenanceLabel,
    ];
    // Each scenario must have an honest, distinct provenance label
    expect(new Set(labels).size).toBe(5);
  });
});
