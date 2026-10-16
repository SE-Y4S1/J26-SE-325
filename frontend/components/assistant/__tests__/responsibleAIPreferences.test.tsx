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
          body: {
            message:
              "Analyze my investment portfolio. Show my current holdings and total valuation.",
            preferences: {
              explanationEnabled: true,
              personalizedContext: false,
              dataSharingConsent: true,
            },
          },
        })
      );
    });
  });
});
