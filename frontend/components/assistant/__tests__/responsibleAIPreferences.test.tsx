import React from "react";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { ResponsibleAIScreen } from "../ResponsibleAIScreen";
import { getValidatedResponsibleAIPreferences, ChatScreen } from "../ChatScreen";
import { request } from "@/lib/api/client";

jest.mock("@/lib/api/client", () => ({
  request: jest.fn(),
  SERVICES: { assistant: "http://localhost:8003" },
}));

const STORAGE_KEY = "finagent_responsible_ai_preferences";

describe("Responsible AI Preferences - Frontend Enforcement", () => {
  beforeEach(() => {
    window.localStorage.clear();
    jest.clearAllMocks();
  });

  describe("getValidatedResponsibleAIPreferences contract", () => {
    it("returns default true for all preferences when localStorage is empty", () => {
      const prefs = getValidatedResponsibleAIPreferences();
      expect(prefs).toEqual({
        explanationEnabled: true,
        personalizedContext: true,
        dataSharingConsent: true,
      });
    });

    it("handles malformed JSON in localStorage safely and falls back to defaults", () => {
      window.localStorage.setItem(STORAGE_KEY, "invalid-json{}[");
      const prefs = getValidatedResponsibleAIPreferences();
      expect(prefs).toEqual({
        explanationEnabled: true,
        personalizedContext: true,
        dataSharingConsent: true,
      });
    });

    it("handles non-boolean fields safely", () => {
      window.localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          explanationEnabled: "not-a-bool",
          personalizedContext: null,
          dataSharingConsent: 123,
        })
      );
      const prefs = getValidatedResponsibleAIPreferences();
      expect(prefs).toEqual({
        explanationEnabled: true,
        personalizedContext: true,
        dataSharingConsent: true,
      });
    });

    it("faithfully returns explicit false values", () => {
      window.localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          explanationEnabled: false,
          personalizedContext: false,
          dataSharingConsent: false,
        })
      );
      const prefs = getValidatedResponsibleAIPreferences();
      expect(prefs).toEqual({
        explanationEnabled: false,
        personalizedContext: false,
        dataSharingConsent: false,
      });
    });
  });

  describe("ResponsibleAIScreen Reporting and Enforcement", () => {
    it("reflects default active states when initialized", () => {
      render(<ResponsibleAIScreen />);

      expect(screen.getByText("Status: Active")).toBeInTheDocument();
      expect(screen.getByText("Status: Enabled")).toBeInTheDocument();
      expect(screen.getAllByText("Status: Protected").length).toBe(2); // Privacy and Safety
      expect(screen.getByText("Consent Granted for Research")).toBeInTheDocument();
    });

    it("accurately reports Opted Out when consent is disabled and never misrepresents as Active", () => {
      window.localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          explanationEnabled: true,
          personalizedContext: true,
          dataSharingConsent: false,
        })
      );

      render(<ResponsibleAIScreen />);

      expect(screen.getByText("Status: Opted Out")).toBeInTheDocument();
      expect(screen.getByText("Consent Withdrawn — Telemetry Disabled")).toBeInTheDocument();
      expect(screen.queryByText("Consent Granted for Research")).not.toBeInTheDocument();
      // Essential safety remains protected
      expect(screen.getByText("Essential Safety Enforced")).toBeInTheDocument();
    });

    it("accurately reports Suppressed when AI explanation is disabled", () => {
      window.localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          explanationEnabled: false,
          personalizedContext: true,
          dataSharingConsent: true,
        })
      );

      render(<ResponsibleAIScreen />);

      expect(screen.getByText("Status: Suppressed")).toBeInTheDocument();
      expect(screen.getByText("Explanations Suppressed by Preference")).toBeInTheDocument();
    });

    it("accurately reports Restricted when personalized context is disabled", () => {
      window.localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          explanationEnabled: true,
          personalizedContext: false,
          dataSharingConsent: true,
        })
      );

      render(<ResponsibleAIScreen />);

      expect(screen.getByText("Status: Restricted")).toBeInTheDocument();
      expect(screen.getByText("Personalized Context Restricted")).toBeInTheDocument();
    });

    it("updates storage and displayed state upon toggle interaction", () => {
      render(<ResponsibleAIScreen />);

      // Find toggles (buttons)
      const buttons = screen.getAllByRole("button");
      expect(buttons.length).toBe(3);

      // Toggle Data sharing consent (3rd toggle)
      fireEvent.click(buttons[2]);

      expect(screen.getByText("Status: Opted Out")).toBeInTheDocument();
      expect(screen.getByText("Consent Withdrawn — Telemetry Disabled")).toBeInTheDocument();

      const saved = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "{}");
      expect(saved.dataSharingConsent).toBe(false);
    });

    it("saves personalizedContext false when toggled off and validated preferences reflect it", () => {
      render(<ResponsibleAIScreen />);

      const buttons = screen.getAllByRole("button");
      // Toggle Personalized financial context (2nd toggle)
      fireEvent.click(buttons[1]);

      expect(screen.getByText("Status: Restricted")).toBeInTheDocument();
      expect(screen.getByText("Personalized Context Restricted")).toBeInTheDocument();

      const prefs = getValidatedResponsibleAIPreferences();
      expect(prefs.personalizedContext).toBe(false);
    });
  });

  describe("Portfolio Evidence Gating in Explanation Parser", () => {
    it("does not activate portfolio mode or return 1,000,000 when evidence indicates personalization disabled", () => {
      const { parseSessionEvidence, parsePortfolioResult } = require("../explanationParser");

      const disabledResult =
        "PORTFOLIO ANALYSIS — PERSONALIZED CONTEXT DISABLED\n" +
        "Personalized financial context is disabled by user preference.\n" +
        "Account-specific portfolio lookup was not performed.";

      const parsedPortfolio = parsePortfolioResult(disabledResult);
      expect(parsedPortfolio.hasEvidence).toBe(false);
      expect(parsedPortfolio.totalValue).toBe("Disabled");
      expect(parsedPortfolio.totalValue).not.toBe("1,000,000");

      const session = parseSessionEvidence([
        { tool: "get_portfolio_analysis", result: disabledResult },
      ]);
      expect(session.mode).not.toBe("portfolio");
    });
  });

  describe("ChatScreen Request Payload Serialization Regression", () => {
    it("submits personalizedContext false to assistant endpoint when personalized context is disabled in localStorage", async () => {
      // Set localStorage preference with personalizedContext = false
      window.localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          explanationEnabled: true,
          personalizedContext: false,
          dataSharingConsent: true,
        })
      );

      // Mock client request
      (request as jest.Mock).mockResolvedValueOnce({
        answer:
          "Notice: Personalized financial context is disabled by user preference. Account-specific portfolio lookup and asset holdings analysis are blocked for this request, and no personalized portfolio data has been retrieved.",
        agent_execution: { tools_used: [] },
        evidence: [],
      });

      render(<ChatScreen onNavigateToExplanation={jest.fn()} />);

      const input = screen.getByPlaceholderText("Ask the financial assistant...");
      const sendButton = screen.getByRole("button", { name: "Send" });

      fireEvent.change(input, {
        target: {
          value:
            "Analyze my investment portfolio. Show my current holdings and total valuation.",
        },
      });

      await act(async () => {
        fireEvent.click(sendButton);
      });

      expect(request).toHaveBeenCalledWith(
        "assistant",
        "/assistant/chat",
        expect.objectContaining({
          method: "POST",
          body: expect.objectContaining({
            message:
              "Analyze my investment portfolio. Show my current holdings and total valuation.",
            preferences: {
              explanationEnabled: true,
              personalizedContext: false,
              dataSharingConsent: true,
            },
          }),
        })
      );
    });
  });

  describe("Sequential Turn Evidence and Provenance Regression", () => {
    const { sanitizeAssistantResponse } = require("../ChatScreen");

    it("sanitizes leaked portfolio notices when fraud evidence is present", () => {
      const leakedAnswer =
        "Notice: No live portfolio was retrieved for this account from the platform service. " +
        "The user has zero verified live holdings on record. Baseline Valuation: LKR 1,000,000.\n\n" +
        "For TX1001, recorded fraud score is 0.92, status BLOCKED. Stated factors: new device, unusual location.";

      const fraudEvidence = [
        {
          tool: "get_fraud_analysis",
          result:
            "FRAUD ANALYSIS — FALLBACK MOCK DATA\nTransaction ID: TX1001\nRecorded Fraud Score: 0.92\nRecorded Status: BLOCKED\nRECORDED CONTRIBUTING FACTORS:\n- New device\n- Unusual location\n- High transfer amount",
        },
      ];

      const cleaned = sanitizeAssistantResponse(leakedAnswer, fraudEvidence);
      expect(cleaned).not.toContain("Notice: No live portfolio");
      expect(cleaned).not.toContain("1,000,000");
      expect(cleaned).toContain("0.92");
      expect(cleaned).toContain("BLOCKED");
    });

    it("derives complete grounded fraud explanation when response only contained leaked portfolio text", () => {
      const leakedOnlyAnswer =
        "Notice: No live portfolio was retrieved for this account from the platform service. " +
        "The user has zero verified live holdings on record. Baseline Valuation: LKR 1,000,000.";

      const fraudEvidence = [
        {
          tool: "get_fraud_analysis",
          result:
            "FRAUD ANALYSIS — FALLBACK MOCK DATA\nTransaction ID: TX1001\nRecorded Fraud Score: 0.92\nRecorded Risk Level: HIGH\nRecorded Status: BLOCKED\nRECORDED CONTRIBUTING FACTORS:\n- New device\n- Unusual location\n- High transfer amount",
        },
      ];

      const derived = sanitizeAssistantResponse(leakedOnlyAnswer, fraudEvidence);
      expect(derived).toContain("TX1001");
      expect(derived).toContain("0.92");
      expect(derived).toContain("BLOCKED");
      expect(derived).toContain("HIGH");
      expect(derived.toLowerCase()).toContain("new device, unusual location, high transfer amount");
      expect(derived).toContain("mock transaction record");
      expect(derived).toContain("calibrated probability");
      expect(derived).not.toContain("portfolio");
      expect(derived).not.toContain("1,000,000");
    });

    it("preserves portfolio response when get_portfolio_analysis is in evidence", () => {
      const portfolioAnswer =
        "Notice: No live portfolio was retrieved for this account from the platform service. " +
        "Baseline Valuation: LKR 1,000,000 (Mock Baseline Demonstration Only).";

      const portfolioEvidence = [
        {
          tool: "get_portfolio_analysis",
          result: "PORTFOLIO ANALYSIS — FALLBACK DATA\nUSER / PORTFOLIO IDENTIFIER: USER001",
        },
      ];

      const preserved = sanitizeAssistantResponse(portfolioAnswer, portfolioEvidence);
      expect(preserved).toBe(portfolioAnswer);
    });

    it("reproduces sequential conversation: Turn 1 portfolio fallback followed by Turn 2 TX1001 fraud query", async () => {
      const onNavigateToExplanation = jest.fn();
      const onEvidenceReceived = jest.fn();

      // Mock Turn 1: Portfolio query
      (request as jest.Mock).mockResolvedValueOnce({
        answer:
          "Notice: No live portfolio was retrieved for this account from the platform service. " +
          "The user has zero verified live holdings on record. Baseline Valuation: LKR 1,000,000 (Mock Baseline Demonstration Only).",
        agent_execution: { tools_used: ["get_portfolio_analysis"] },
        evidence: [
          {
            tool: "get_portfolio_analysis",
            result: "PORTFOLIO ANALYSIS — FALLBACK DATA\nUSER / PORTFOLIO IDENTIFIER: USER001",
          },
        ],
      });

      render(
        <ChatScreen
          onNavigateToExplanation={onNavigateToExplanation}
          onEvidenceReceived={onEvidenceReceived}
        />
      );

      const input = screen.getByPlaceholderText("Ask the financial assistant...");
      const sendButton = screen.getByRole("button", { name: "Send" });

      // Send Turn 1
      fireEvent.change(input, { target: { value: "What is my portfolio risk profile?" } });
      await act(async () => {
        fireEvent.click(sendButton);
      });

      // Verify Turn 1 visible response contains portfolio notice and valuation
      expect(screen.getByText(/Baseline Valuation: LKR 1,000,000/i)).toBeInTheDocument();
      expect(onEvidenceReceived).toHaveBeenCalledWith(
        expect.arrayContaining([
          expect.objectContaining({ tool: "get_portfolio_analysis" }),
        ])
      );

      // Mock Turn 2: Fraud query (even if backend leaked portfolio text)
      (request as jest.Mock).mockResolvedValueOnce({
        answer:
          "Notice: No live portfolio was retrieved for this account from the platform service. " +
          "The user has zero verified live holdings on record. Baseline Valuation: LKR 1,000,000.\n\n" +
          "For TX1001, recorded fraud score is 0.92, status BLOCKED.",
        agent_execution: { tools_used: ["get_fraud_analysis"] },
        evidence: [
          {
            tool: "get_fraud_analysis",
            result:
              "FRAUD ANALYSIS — FALLBACK MOCK DATA\nTransaction ID: TX1001\nRecorded Fraud Score: 0.92\nRecorded Risk Level: HIGH\nRecorded Status: BLOCKED\nRECORDED CONTRIBUTING FACTORS:\n- New device\n- Unusual location\n- High transfer amount",
          },
        ],
      });

      // Send Turn 2
      fireEvent.change(input, { target: { value: "Why was TX1001 blocked?" } });
      await act(async () => {
        fireEvent.click(sendButton);
      });

      // Verify Turn 2 evidence replaced Turn 1 evidence in onEvidenceReceived callback
      expect(onEvidenceReceived).toHaveBeenLastCalledWith([
        expect.objectContaining({
          tool: "get_fraud_analysis",
          result: expect.stringContaining("TX1001"),
        }),
      ]);

      // Turn 2 visible message must display fraud evidence only and not retain portfolio notice or valuation
      const messagesContainer = screen.getAllByText(/FinTrust Agent/i);
      expect(messagesContainer.length).toBeGreaterThanOrEqual(2);

      // Raw evidence details is rendered and accessible
      expect(screen.getAllByText(/View raw tool evidence/i).length).toBeGreaterThanOrEqual(1);

      // Clicking View Why passes Turn 2's specific evidence
      const whyButtons = screen.getAllByRole("button", { name: /View Why\?/i });
      fireEvent.click(whyButtons[whyButtons.length - 1]);
      expect(onNavigateToExplanation).toHaveBeenCalledWith(
        expect.arrayContaining([
          expect.objectContaining({ tool: "get_fraud_analysis" }),
        ])
      );
    });
  });
});
