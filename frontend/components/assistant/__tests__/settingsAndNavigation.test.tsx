import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { SettingsScreen, clearAssistantSessionCache, ASSISTANT_SESSION_CACHE_KEYS } from "../SettingsScreen";
import { ResponsibleAIScreen } from "../ResponsibleAIScreen";
import { AssistantSidebar } from "../AssistantSidebar";
import AssistantPage from "@/app/(platform)/assistant/page";

const mockPush = jest.fn();
let mockSearchParams = new URLSearchParams();

jest.mock("next/navigation", () => ({
  useRouter: () => ({
    push: mockPush,
    replace: jest.fn(),
  }),
  useSearchParams: () => mockSearchParams,
}));

jest.mock("@/lib/auth/context", () => ({
  useAuth: () => ({
    user: {
      id: 42,
      display_name: "Test Research User",
      email: "researcher@university.edu",
      created_at: "2026-09-01T00:00:00Z",
    },
    logout: jest.fn(),
    loading: false,
    login: jest.fn(),
    register: jest.fn(),
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
  request: jest.fn(),
}));

describe("Settings Screen and Assistant Navigation", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockSearchParams = new URLSearchParams();
  });

  describe("SettingsScreen isolation", () => {
    it("renders Settings header, session info, connected microservices, and diagnostics", () => {
      render(<SettingsScreen />);

      expect(screen.getByRole("heading", { name: "Settings", level: 2 })).toBeInTheDocument();
      expect(screen.getByText("System & Account Configuration")).toBeInTheDocument();
      expect(screen.getByText("Authenticated Session")).toBeInTheDocument();
      expect(screen.getByText("Test Research User")).toBeInTheDocument();
      expect(screen.getByText("researcher@university.edu")).toBeInTheDocument();
      expect(screen.getByText("#42")).toBeInTheDocument();
      expect(screen.getByText("Active Token")).toBeInTheDocument();

      // Connected microservices endpoints
      expect(screen.getByText("Platform Service")).toBeInTheDocument();
      expect(screen.getByText("Portfolio Optimization")).toBeInTheDocument();
      expect(screen.getByText("Fraud Detection")).toBeInTheDocument();
      expect(screen.getByText("Blockchain Audit")).toBeInTheDocument();
      expect(screen.getByText("Agentic Assistant")).toBeInTheDocument();
      expect(screen.getByText("http://localhost:8100")).toBeInTheDocument();
      expect(screen.getByText("http://localhost:8003")).toBeInTheDocument();

      // Diagnostics & Cache
      expect(screen.getByText("Assistant Session Cache")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Clear Session Cache" })).toBeInTheDocument();
    });

    it("does NOT duplicate the 3 Responsible AI controls on SettingsScreen", () => {
      render(<SettingsScreen />);

      // The 3 Responsible AI preference controls must not exist on Settings
      expect(screen.queryByText("AI explanation enabled")).not.toBeInTheDocument();
      expect(screen.queryByText("Personalized financial context")).not.toBeInTheDocument();
      expect(screen.queryByText("Data sharing consent")).not.toBeInTheDocument();

      // Governance framework title must not be on Settings
      expect(screen.queryByText("Governance & Ethics Framework")).not.toBeInTheDocument();
      expect(screen.queryByText("Responsible AI & Safety")).not.toBeInTheDocument();
    });

    it("verifies the 3 Responsible AI preference controls remain on ResponsibleAIScreen", () => {
      render(<ResponsibleAIScreen />);

      expect(screen.getByText("Responsible AI & Safety")).toBeInTheDocument();
      expect(screen.getByText("AI explanation enabled")).toBeInTheDocument();
      expect(screen.getByText("Personalized financial context")).toBeInTheDocument();
      expect(screen.getByText("Data sharing consent")).toBeInTheDocument();
      expect(screen.getByText("User Privacy & AI Preference Controls")).toBeInTheDocument();
    });
  });

  describe("AssistantSidebar highlighting", () => {
    it("highlights Settings when activeTab is settings", () => {
      const onSelectTab = jest.fn();
      render(<AssistantSidebar activeTab="settings" onSelectTab={onSelectTab} />);

      const settingsBtn = screen.getByRole("button", { name: /Settings/i });
      const responsibleAIBtn = screen.getByRole("button", { name: /Responsible AI/i });

      expect(settingsBtn.className).toContain("bg-blue-600 text-white");
      expect(responsibleAIBtn.className).not.toContain("bg-blue-600 text-white");

      fireEvent.click(responsibleAIBtn);
      expect(onSelectTab).toHaveBeenCalledWith("responsible-ai");
    });

    it("highlights Responsible AI when activeTab is responsible-ai", () => {
      const onSelectTab = jest.fn();
      render(<AssistantSidebar activeTab="responsible-ai" onSelectTab={onSelectTab} />);

      const settingsBtn = screen.getByRole("button", { name: /Settings/i });
      const responsibleAIBtn = screen.getByRole("button", { name: /Responsible AI/i });

      expect(responsibleAIBtn.className).toContain("bg-blue-600 text-white");
      expect(settingsBtn.className).not.toContain("bg-blue-600 text-white");

      fireEvent.click(settingsBtn);
      expect(onSelectTab).toHaveBeenCalledWith("settings");
    });
  });

  describe("AssistantPage Routing, Direct Navigation & Refresh", () => {
    it("renders AI Assistant screen by default when no tab is in searchParams", () => {
      mockSearchParams = new URLSearchParams();
      render(<AssistantPage />);

      expect(screen.getByRole("button", { name: /AI Assistant/i }).className).toContain("bg-blue-600 text-white");
      expect(screen.getByPlaceholderText(/Ask the financial assistant/i)).toBeInTheDocument();
      expect(screen.queryByRole("heading", { name: "Settings" })).not.toBeInTheDocument();
      expect(screen.queryByText("Responsible AI & Safety")).not.toBeInTheDocument();
    });

    it("directly navigates to Settings screen when ?tab=settings is in URL", () => {
      mockSearchParams = new URLSearchParams("tab=settings");
      render(<AssistantPage />);

      expect(screen.getByRole("heading", { name: "Settings", level: 2 })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /Settings/i }).className).toContain("bg-blue-600 text-white");
      expect(screen.getByRole("button", { name: /Responsible AI/i }).className).not.toContain("bg-blue-600 text-white");
      expect(screen.queryByText("Responsible AI & Safety")).not.toBeInTheDocument();
    });

    it("directly navigates to Responsible AI screen when ?tab=responsible-ai is in URL", () => {
      mockSearchParams = new URLSearchParams("tab=responsible-ai");
      render(<AssistantPage />);

      expect(screen.getByText("Responsible AI & Safety")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /Responsible AI/i }).className).toContain("bg-blue-600 text-white");
      expect(screen.getByRole("button", { name: /Settings/i }).className).not.toContain("bg-blue-600 text-white");
      expect(screen.queryByRole("heading", { name: "Settings" })).not.toBeInTheDocument();
    });

    it("updates router on tab switch", () => {
      mockSearchParams = new URLSearchParams();
      render(<AssistantPage />);

      const settingsBtn = screen.getByRole("button", { name: /Settings/i });
      fireEvent.click(settingsBtn);
      expect(mockPush).toHaveBeenCalledWith("/assistant?tab=settings");

      const responsibleAIBtn = screen.getByRole("button", { name: /Responsible AI/i });
      fireEvent.click(responsibleAIBtn);
      expect(mockPush).toHaveBeenCalledWith("/assistant?tab=responsible-ai");

      const assistantBtn = screen.getByRole("button", { name: /AI Assistant/i });
      fireEvent.click(assistantBtn);
      expect(mockPush).toHaveBeenCalledWith("/assistant");
    });
  });

  describe("Clear Session Cache and storage isolation", () => {
    const AUTH_KEY = "j26_access_token";
    const RAI_PREFS_KEY = "finagent_responsible_ai_preferences";
    const RAI_FALLBACK_KEY = "finagent_responsible_ai_prefs";

    beforeEach(() => {
      window.localStorage.clear();
      window.sessionStorage.clear();
    });

    it("clears explicitly defined assistant session cache and scratch keys while preserving non-assistant session keys", () => {
      // Seed assistant session keys
      window.sessionStorage.setItem("finagent_session_cache", JSON.stringify({ query: "tx audit", res: "ok" }));
      window.sessionStorage.setItem("finagent_evidence_scratch", JSON.stringify([{ tool: "get_fraud_analysis" }]));
      window.sessionStorage.setItem("finagent_temp_query", "Explain transaction TX99");
      window.sessionStorage.setItem("finagent_scratch_workflow", "step_2");

      // Seed unrelated platform session keys
      window.sessionStorage.setItem("platform_active_tab", "portfolio");
      window.sessionStorage.setItem("oauth_state_nonce", "session_random_nonce_123");

      render(<SettingsScreen />);

      const clearBtn = screen.getByRole("button", { name: "Clear Session Cache" });
      fireEvent.click(clearBtn);

      // Verify assistant session keys are wiped
      expect(window.sessionStorage.getItem("finagent_session_cache")).toBeNull();
      expect(window.sessionStorage.getItem("finagent_evidence_scratch")).toBeNull();
      expect(window.sessionStorage.getItem("finagent_temp_query")).toBeNull();
      expect(window.sessionStorage.getItem("finagent_scratch_workflow")).toBeNull();

      // Verify unrelated session keys remain completely intact
      expect(window.sessionStorage.getItem("platform_active_tab")).toBe("portfolio");
      expect(window.sessionStorage.getItem("oauth_state_nonce")).toBe("session_random_nonce_123");
    });

    it("preserves authentication tokens and Responsible AI preferences in localStorage", () => {
      const mockAuthToken = "header.payload.signature-j26-jwt";
      const mockRaiPrefs = JSON.stringify({
        explanationEnabled: true,
        personalizedContext: false,
        dataSharingConsent: true,
      });

      window.localStorage.setItem(AUTH_KEY, mockAuthToken);
      window.localStorage.setItem(RAI_PREFS_KEY, mockRaiPrefs);
      window.localStorage.setItem(RAI_FALLBACK_KEY, JSON.stringify({ personalizedContext: false }));
      window.sessionStorage.setItem("finagent_session_cache", "some-cache");

      render(<SettingsScreen />);

      const clearBtn = screen.getByRole("button", { name: "Clear Session Cache" });
      fireEvent.click(clearBtn);

      // Verify localStorage is completely untouched
      expect(window.localStorage.getItem(AUTH_KEY)).toBe(mockAuthToken);
      expect(window.localStorage.getItem(RAI_PREFS_KEY)).toBe(mockRaiPrefs);
      expect(window.localStorage.getItem(RAI_FALLBACK_KEY)).toBe(JSON.stringify({ personalizedContext: false }));
    });

    it("does not call backend services, keeping persistent governance records and audit logs intact", () => {
      const { request } = require("@/lib/api/client");
      render(<SettingsScreen />);

      const clearBtn = screen.getByRole("button", { name: "Clear Session Cache" });
      fireEvent.click(clearBtn);

      // Verify zero network calls were made
      expect(request).not.toHaveBeenCalled();
    });

    it("displays success feedback when clearing succeeds", () => {
      render(<SettingsScreen />);

      const clearBtn = screen.getByRole("button", { name: "Clear Session Cache" });
      fireEvent.click(clearBtn);

      expect(screen.getByText("Cache Cleared!")).toBeInTheDocument();
      expect(screen.getByRole("status")).toHaveTextContent("Assistant session cache cleared successfully.");
    });

    it("displays failure feedback when sessionStorage throws an error", () => {
      const removeItemSpy = jest.spyOn(Storage.prototype, "removeItem").mockImplementation(() => {
        throw new Error("SecurityError: Access to Storage is denied");
      });

      window.sessionStorage.setItem("finagent_session_cache", "test");

      render(<SettingsScreen />);

      const clearBtn = screen.getByRole("button", { name: "Clear Session Cache" });
      fireEvent.click(clearBtn);

      expect(screen.getByRole("alert")).toHaveTextContent("SecurityError: Access to Storage is denied");
      expect(screen.queryByRole("status")).not.toBeInTheDocument();

      removeItemSpy.mockRestore();
    });

    it("clearAssistantSessionCache helper accurately reports cleared keys and handles errors", () => {
      window.sessionStorage.setItem("finagent_session_cache", "cached");
      window.sessionStorage.setItem("finagent_evidence_scratch", "evidence");
      window.sessionStorage.setItem("unrelated_key", "keep");

      const result = clearAssistantSessionCache();
      expect(result.success).toBe(true);
      expect(result.clearedKeys).toContain("finagent_session_cache");
      expect(result.clearedKeys).toContain("finagent_evidence_scratch");
      expect(result.clearedKeys).not.toContain("unrelated_key");
      expect(window.sessionStorage.getItem("unrelated_key")).toBe("keep");
    });
  });
});

