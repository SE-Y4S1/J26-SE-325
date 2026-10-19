import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { SettingsScreen, clearAssistantSessionCache } from "../SettingsScreen";
import { TrustPanelScreen } from "../TrustPanelScreen";
import * as assistantApi from "@/lib/api/assistant";

jest.mock("@/lib/api/assistant", () => ({
  recordGovernanceDecision: jest.fn(),
  getGovernanceDecision: jest.fn(),
}));

jest.mock("@/lib/auth/context", () => ({
  useAuth: () => ({
    user: {
      id: 42,
      display_name: "Test Research User",
      email: "researcher@university.edu",
    },
    logout: jest.fn(),
  }),
}));

jest.mock("@/lib/api/client", () => ({
  getToken: jest.fn(() => "mock-valid-token-xyz"),
  SERVICES: {
    platform: "http://localhost:8100",
    portfolio: "http://localhost:8000",
    fraud: "http://localhost:8001",
    audit: "http://localhost:8002",
    assistant: "http://localhost:8003",
  },
}));

const mockRecordGovernance = assistantApi.recordGovernanceDecision as jest.MockedFunction<
  typeof assistantApi.recordGovernanceDecision
>;
const mockGetGovernance = assistantApi.getGovernanceDecision as jest.MockedFunction<
  typeof assistantApi.getGovernanceDecision
>;

describe("Negative and Safety Scenarios — Component 4 Assistant", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    sessionStorage.clear();
    localStorage.clear();
    mockGetGovernance.mockResolvedValue({ found: false, decision: null });
  });

  describe("Storage exceptions during cache clearing", () => {
    it("handles sessionStorage.removeItem exceptions safely without crashing", () => {
      sessionStorage.setItem("finagent_session_cache", "test-val");
      const spy = jest.spyOn(Storage.prototype, "removeItem").mockImplementation(() => {
        throw new DOMException("Access is denied", "SecurityError");
      });

      const res = clearAssistantSessionCache();
      expect(res.success).toBe(false);
      expect(res.error).toBe("Access is denied");

      spy.mockRestore();
    });

    it("displays error banner on SettingsScreen when cache clearing fails", async () => {
      sessionStorage.setItem("finagent_session_cache", "test-val");
      const spy = jest.spyOn(Storage.prototype, "removeItem").mockImplementation(() => {
        throw new DOMException("Quota exceeded or storage disabled", "QuotaExceededError");
      });

      render(<SettingsScreen />);
      const clearBtn = screen.getByRole("button", { name: /Clear Session Cache/i });
      fireEvent.click(clearBtn);

      await waitFor(() => {
        expect(screen.getByText(/Quota exceeded or storage disabled/i)).toBeInTheDocument();
      });

      spy.mockRestore();
    });
  });

  describe("Governance persistence failures", () => {
    it("displays error and does not pretend success when persistence fails", async () => {
      mockRecordGovernance.mockRejectedValueOnce(new Error("Network timeout: 504 Gateway Timeout"));

      render(
        <TrustPanelScreen
          latestEvidence={[
            {
              tool: "get_fraud_analysis",
              result: "Transaction ID: TX1001\nRecorded Fraud Score: 0.88\nRecorded Status: BLOCKED",
            },
          ]}
        />
      );

      const confirmBtn = screen.getByRole("button", { name: /Confirm Decision/i });
      fireEvent.click(confirmBtn);

      await waitFor(() => {
        expect(screen.getByText(/Network timeout: 504 Gateway Timeout/i)).toBeInTheDocument();
      });

      // Verify state was not falsely set to confirmed
      expect(screen.queryByText(/CONFIRMED BY HUMAN REVIEW/i)).not.toBeInTheDocument();
    });
  });

  describe("Rejected action policy enforcement", () => {
    it("displays rejected status and enforces blocked action policy", async () => {
      mockRecordGovernance.mockResolvedValueOnce({
        status: "recorded",
        is_duplicate: false,
        decision: {
          decision_id: "gov-rej-9999",
          decision_type: "rejected",
          reference_id: "TX1001",
          reference_type: "transaction",
          user_id: "42",
          user_email: "auditor@university.edu",
          timestamp: "2026-10-19T14:30:00Z",
          evidence_reference: {},
          status: "recorded",
        },
      });

      render(
        <TrustPanelScreen
          latestEvidence={[
            {
              tool: "get_fraud_analysis",
              result: "Transaction ID: TX1001\nRecorded Fraud Score: 0.95\nRecorded Status: BLOCKED",
            },
          ]}
        />
      );

      const rejectBtn = screen.getByRole("button", { name: /Reject Action/i });
      fireEvent.click(rejectBtn);

      await waitFor(() => {
        expect(screen.getByText(/REJECTED BY HUMAN OVERSIGHT/i)).toBeInTheDocument();
        expect(screen.getByText(/Execution Halted \(Human Rejected\)/i)).toBeInTheDocument();
        expect(screen.getByText(/Record ID: gov-rej-9999/i)).toBeInTheDocument();
      });
    });
  });

  describe("Upstream platform outage vs empty portfolio in Trust Panel", () => {
    it("displays SERVICE UNAVAILABLE and does not misrepresent outage as empty portfolio", () => {
      render(
        <TrustPanelScreen
          latestEvidence={[
            {
              tool: "get_portfolio_analysis",
              result:
                "PORTFOLIO ANALYSIS — FALLBACK DATA (NO LIVE RECORD RETRIEVED)\n" +
                "Live platform service at http://localhost:8100: Service unavailable (Connection refused).\n" +
                "This does not indicate an empty portfolio; live holdings could not be retrieved due to a platform service connection failure.",
            },
          ]}
        />
      );

      expect(screen.getAllByText("SERVICE UNAVAILABLE").length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText(/Active Session: Service Unavailable \(Platform Outage\)/i)).toBeInTheDocument();
      expect(screen.getByText(/Live holdings remain unverified due to platform service unavailability; this does NOT indicate an empty portfolio/i)).toBeInTheDocument();
      // Must not display "0 active holdings"
      expect(screen.queryByText(/0 Active Holdings/i)).not.toBeInTheDocument();
    });

    it("displays DEMO BASELINE and illustrative warnings when demo fallback is used", () => {
      render(
        <TrustPanelScreen
          latestEvidence={[
            {
              tool: "get_portfolio_analysis",
              result:
                "PORTFOLIO ANALYSIS — FALLBACK DATA (NO LIVE RECORD RETRIEVED)\n" +
                "Platform service note: No portfolios registered for user on platform service (http://localhost:8100).\n" +
                "Illustrative Baseline Valuation: LKR 1,000,000\n" +
                "Empty portfolio list (User has no portfolios on platform)",
            },
          ]}
        />
      );

      expect(screen.getAllByText("DEMO BASELINE").length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText(/Notice: No live portfolio record was retrieved/i)).toBeInTheDocument();
      expect(screen.getByText(/The displayed metrics and valuation are synthetic baseline demonstrations/i)).toBeInTheDocument();
    });
  });

  describe("Stale evidence replacement", () => {
    it("updates displayed data cleanly when new evidence is passed", () => {
      const { rerender } = render(
        <TrustPanelScreen
          latestEvidence={[
            {
              tool: "get_fraud_analysis",
              result: "Transaction ID: TX-FIRST\nRecorded Fraud Score: 0.35\nRecorded Status: ALLOW",
            },
          ]}
        />
      );

      expect(screen.getByText("Transaction TX-FIRST")).toBeInTheDocument();

      // Rerender with a new turn's evidence
      rerender(
        <TrustPanelScreen
          latestEvidence={[
            {
              tool: "get_fraud_analysis",
              result: "Transaction ID: TX-SECOND\nRecorded Fraud Score: 0.95\nRecorded Status: BLOCK",
            },
          ]}
        />
      );

      expect(screen.getByText("Transaction TX-SECOND")).toBeInTheDocument();
      expect(screen.queryByText("Transaction TX-FIRST")).not.toBeInTheDocument();
    });
  });
});
