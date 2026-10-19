import React from "react";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { ChatScreen } from "../ChatScreen";
import { AgentWorkflowPanel } from "../AgentWorkflowPanel";
import { AssistantHeader } from "../AssistantHeader";
import { ResearchBanner } from "../ResearchBanner";
import AssistantPage from "@/app/(platform)/assistant/page";
import { request } from "@/lib/api/client";

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
      display_name: "Research Evaluator",
      email: "evaluator@university.edu",
    },
    logout: jest.fn(),
    loading: false,
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

describe("FinTrust AI Redesign - Professional Enterprise UI & Pipeline Verification", () => {
  beforeEach(() => {
    window.localStorage.clear();
    window.sessionStorage.clear();
    jest.clearAllMocks();
    mockSearchParams = new URLSearchParams();
  });

  describe("1. Agent Activity & Pipeline Trace (Dynamic & Honest)", () => {
    it("is collapsed by default and shows Ready status when no query has been run", () => {
      render(<AgentWorkflowPanel executionTrace={null} />);

      expect(screen.getByText("Agent Activity & Pipeline Trace")).toBeInTheDocument();
      expect(screen.getByText("Ready · Awaiting query")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /Agent Activity & Pipeline Trace/i })).toHaveAttribute(
        "aria-expanded",
        "false"
      );
      // Detailed steps should not be rendered when collapsed
      expect(screen.queryByText("Live Observable Agent Execution Loop")).not.toBeInTheDocument();
    });

    it("dynamically displays actual tool calls, counts, and latencies from real execution data", () => {
      render(
        <AgentWorkflowPanel
          executionTrace={{
            toolsCalled: ["get_fraud_analysis", "get_blockchain_audit"],
            evidenceUsed: ["get_fraud_analysis", "get_blockchain_audit"],
            timing: {
              total_duration_ms: 142.5,
              total_tool_duration_ms: 18.4,
              request_id: "req-live-99",
            },
            evidenceDetails: [
              {
                tool: "get_fraud_analysis",
                result: "Transaction ID: TX1001\nRecorded Fraud Score: 0.92\nSHAP EXPLANATION:\n+0.193",
              },
            ],
          }}
          lastUserQuery="Why was TX1001 blocked?"
        />
      );

      // Collapsed badge shows exact tool count and latency
      expect(screen.getByText("2 tools executed")).toBeInTheDocument();
      expect(screen.getByText(/142.5ms latency/i)).toBeInTheDocument();

      // Click to expand
      fireEvent.click(screen.getByRole("button", { name: /Agent Activity & Pipeline Trace/i }));

      // Expanded details
      expect(screen.getByText("Live Observable Agent Execution Loop")).toBeInTheDocument();
      expect(screen.getByText("get_fraud_analysis, get_blockchain_audit")).toBeInTheDocument();
      expect(screen.getByText(/ID: req-live-99/i)).toBeInTheDocument();

      // Tool step must show completed with latency
      expect(screen.getByText(/Called: get_fraud_analysis, get_blockchain_audit \(18.4ms\)/i)).toBeInTheDocument();

      // SHAP step is completed because SHAP was in evidence
      expect(screen.getByText("Tree SHAP contributions extracted from Component 2")).toBeInTheDocument();
    });

    it("marks tool execution as Skipped (not completed) for conversational fast-path greetings", () => {
      render(
        <AgentWorkflowPanel
          executionTrace={{
            toolsCalled: [],
            evidenceUsed: [],
            evidenceDetails: [],
            timing: {
              total_duration_ms: 0.12,
            },
          }}
          lastUserQuery="Hello"
        />
      );

      expect(screen.getByText("Fast-Path Reply · 0 tools")).toBeInTheDocument();

      // Expand panel
      fireEvent.click(screen.getByRole("button", { name: /Agent Activity & Pipeline Trace/i }));

      // Tool step must explicitly say Skipped, not Completed
      expect(screen.getByText(/Skipped: General query requires no external tools/i)).toBeInTheDocument();
      expect(screen.getByText(/Skipped: No external evidence required/i)).toBeInTheDocument();
    });

    it("marks status as Failed when backend API is unreachable", () => {
      render(
        <AgentWorkflowPanel
          executionTrace={{
            isError: true,
          }}
          lastUserQuery="Check status"
        />
      );

      expect(screen.getByText("Connection Failed · Port 8003")).toBeInTheDocument();

      fireEvent.click(screen.getByRole("button", { name: /Agent Activity & Pipeline Trace/i }));
      expect(screen.getByText(/Failed: Backend API unavailable/i)).toBeInTheDocument();
    });
  });

  describe("2. System Architecture & Evaluator Diagnostics Panel", () => {
    it("renders honestly labeled pipeline stages including DistilBERT as PP2 Planned", () => {
      render(<ResearchBanner isOpen={true} />);

      expect(screen.getByText("FinTrust AI Observable Agentic Architecture")).toBeInTheDocument();
      expect(screen.getByText("LLM Orchestration")).toBeInTheDocument();
      expect(screen.getByText("LangGraph Agent")).toBeInTheDocument();
      expect(screen.getByText("Tool Calling")).toBeInTheDocument();
      expect(screen.getByText("Feature Attribution")).toBeInTheDocument();
      expect(screen.getByText("Natural Language SLM")).toBeInTheDocument();

      // Explicitly verifies PP2 Planned label for DistilBERT
      expect(screen.getByText("PP2 Planned")).toBeInTheDocument();
      expect(
        screen.getByText(/DistilBERT fine-tuning is reserved for PP2; PP1 uses evidence-grounded Ollama LLM/i)
      ).toBeInTheDocument();

      // Relocated Research Focus card is present in diagnostics
      expect(screen.getByText("Academic & Research Focus (PP1 Evaluation)")).toBeInTheDocument();
      expect(
        screen.getByText(/heuristic anomaly detections, not verified criminal convictions/i)
      ).toBeInTheDocument();
    });

    it("can be closed cleanly using the close button", () => {
      const onClose = jest.fn();
      render(<ResearchBanner isOpen={true} onClose={onClose} />);

      const closeBtn = screen.getByRole("button", { name: /Hide Diagnostics/i });
      fireEvent.click(closeBtn);
      expect(onClose).toHaveBeenCalled();
    });

    it("renders nothing when isOpen is false", () => {
      const { container } = render(<ResearchBanner isOpen={false} />);
      expect(container.firstChild).toBeNull();
    });
  });

  describe("3. ChatScreen Layout, Readability, and Human Oversight Links", () => {
    it("renders FinTrust AI conversation workspace with suggestions and send controls", () => {
      render(
        <ChatScreen
          onNavigateToExplanation={jest.fn()}
          onNavigateToTrustPanel={jest.fn()}
        />
      );

      expect(screen.getByText("Financial Decision Dialogue")).toBeInTheDocument();
      expect(screen.getByText("Grounding: Zero-Hallucination Policy")).toBeInTheDocument();
      expect(screen.getByPlaceholderText("Ask the financial assistant...")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Send" })).toBeInTheDocument();

      // Suggested questions
      expect(screen.getByText("Why was my transaction blocked?")).toBeInTheDocument();
      expect(screen.getByText("Explain my portfolio risk")).toBeInTheDocument();
    });

    it("renders structured response with limitation notice, View Why, and Trust Panel buttons", async () => {
      const onNavigateToExplanation = jest.fn();
      const onNavigateToTrustPanel = jest.fn();

      (request as jest.Mock).mockResolvedValueOnce({
        answer:
          "For TX1001, the recorded fraud score is 0.92, risk level HIGH, and status BLOCKED in a Component 2 live gateway record. " +
          "The recorded contributing factors are High transfer amount, Unusual location, New device. " +
          "The meaning of the score as a calibrated probability is not established by the available evidence.",
        agent_execution: {
          tools_used: ["get_fraud_analysis"],
          timing: { total_duration_ms: 125.4 },
        },
        evidence: [
          {
            tool: "get_fraud_analysis",
            result:
              "LIVE COMPONENT 2 AUDIT RECORD\nTransaction ID: TX1001\nRecorded Fraud Score: 0.92\nRecorded Status: BLOCKED",
          },
        ],
      });

      render(
        <ChatScreen
          onNavigateToExplanation={onNavigateToExplanation}
          onNavigateToTrustPanel={onNavigateToTrustPanel}
        />
      );

      const input = screen.getByPlaceholderText("Ask the financial assistant...");
      const sendBtn = screen.getByRole("button", { name: "Send" });

      fireEvent.change(input, { target: { value: "Why was TX1001 blocked?" } });
      await act(async () => {
        fireEvent.click(sendBtn);
      });

      // Model limitation / calibration notice rendered
      expect(screen.getByText(/Model Limitation & Calibration Notice:/i)).toBeInTheDocument();
      expect(screen.getByText(/not verified proof of fraud/i)).toBeInTheDocument();

      // Compact live badge rendered
      expect(screen.getByText("Fraud Gateway (Live Port 8001)")).toBeInTheDocument();

      // Trust Panel button rendered and clickable
      const trustPanelBtn = screen.getByRole("button", { name: /Trust Panel/i });
      expect(trustPanelBtn).toBeInTheDocument();
      fireEvent.click(trustPanelBtn);
      expect(onNavigateToTrustPanel).toHaveBeenCalled();

      // View Why button rendered and clickable
      const whyButtons = screen.getAllByRole("button", { name: /View Why\?/i });
      expect(whyButtons.length).toBeGreaterThanOrEqual(2);
      fireEvent.click(whyButtons[whyButtons.length - 1]);
      expect(onNavigateToExplanation).toHaveBeenCalled();
    });
  });

  describe("4. AssistantHeader & Full AssistantPage Integration", () => {
    it("renders FinTrust AI branding, port status, and toggles architecture diagnostics cleanly", () => {
      render(<AssistantPage />);

      expect(screen.getByRole("heading", { name: "FinTrust AI", level: 1 })).toBeInTheDocument();
      expect(screen.getByText("Online (Port 8003)")).toBeInTheDocument();

      // Initially architecture panel is closed (no vertical clutter)
      expect(screen.queryByText("FinTrust AI Observable Agentic Architecture")).not.toBeInTheDocument();

      // Click Architecture toggle button
      const archToggleBtn = screen.getByTitle("Toggle System Architecture & Evaluator Diagnostics");
      fireEvent.click(archToggleBtn);

      // Now architecture diagnostics panel is visible
      expect(screen.getByText("FinTrust AI Observable Agentic Architecture")).toBeInTheDocument();

      // Click again or close button to hide
      const hideBtn = screen.getByRole("button", { name: /Hide Diagnostics/i });
      fireEvent.click(hideBtn);
      expect(screen.queryByText("FinTrust AI Observable Agentic Architecture")).not.toBeInTheDocument();
    });
  });
});
