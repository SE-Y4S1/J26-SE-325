import React from "react";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import { ChatScreen } from "../ChatScreen";
import { TrustPanelScreen } from "../TrustPanelScreen";
import FraudPage from "@/app/(platform)/fraud/page";
import * as clientModule from "@/lib/api/client";
import * as fraudApiModule from "@/lib/api/fraud";

jest.mock("@/lib/api/client", () => ({
  request: jest.fn(),
  ApiError: class ApiError extends Error {
    isUnavailable = false;
    status = 500;
  },
}));

jest.mock("@/lib/api/fraud", () => {
  const actual = jest.requireActual("@/lib/api/fraud");
  return {
    ...actual,
    scoreTransaction: jest.fn(),
    fraudHealth: jest.fn().mockResolvedValue({
      status: "ok",
      model_version: "trained-v1",
      benchmark_model: true,
      kafka_enabled: false,
    }),
  };
});

describe("Fraud Page -> FinTrust Assistant Transaction ID Handoff Regression", () => {
  const mockRequest = clientModule.request as jest.Mock;
  const mockScoreTransaction = fraudApiModule.scoreTransaction as jest.Mock;

  beforeEach(() => {
    jest.clearAllMocks();
    if (typeof window !== "undefined" && window.localStorage) {
      window.localStorage.clear();
    }
  });

  it("1 & 2. Fraud page scores transaction, returns backend ID, renders 'Explain in Assistant' link with ?tx=", async () => {
    mockScoreTransaction.mockResolvedValueOnce({
      transaction_id: "ui-test-1500k-block",
      fraud_score: 1.0,
      decision: "BLOCK",
      model_version: "benchmark-v1",
      benchmark_model: true,
      explanation: {
        rules: ["score_above_block_threshold", "amount_hard_block"],
        streams: { amount: 1.0, heuristic: 1.0 },
        top_factors: [{ name: "amount", contribution: 1.0 }],
        notes: ["Hard block limit triggered for USD 1,500,000"],
      },
    });

    render(<FraudPage />);

    // Click score transaction
    const scoreBtn = screen.getByRole("button", { name: /Score transaction/i });
    await act(async () => {
      fireEvent.click(scoreBtn);
    });

    // Verify Explain in Assistant link rendered with exact transaction ID returned by backend
    await waitFor(() => {
      const explainBtn = screen.getByTestId("explain-in-assistant-btn");
      expect(explainBtn).toBeInTheDocument();
      expect(explainBtn.getAttribute("href")).toBe("/assistant?tx=ui-test-1500k-block");
    });

    // Verify localStorage persistence of most recently scored transaction
    expect(window.localStorage.getItem("fintrust_last_scored_tx_id")).toBe("ui-test-1500k-block");
  });

  it("3 & 4. ChatScreen receives selectedTransactionId and sends it in Component 4 request", async () => {
    mockRequest.mockResolvedValueOnce({
      answer:
        "For ui-test-1500k-block, the available tool returned a recorded fraud score of 1.0000, and status BLOCK in a Component 2 live gateway record. The recorded contributing factors are score_above_block_threshold, amount_hard_block.",
      agent_execution: {
        tools_used: ["get_fraud_analysis"],
        timing: { total_duration_ms: 85 },
      },
      evidence: [
        {
          tool: "get_fraud_analysis",
          result:
            "FRAUD ANALYSIS — LIVE COMPONENT 2 AUDIT RECORD\nTransaction ID:\nui-test-1500k-block\nRecorded Fraud Score:\n1.0000\nRecorded Risk Level:\nHIGH\nRecorded Status:\nBLOCK\nRECORDED CONTRIBUTING FACTORS:\n- score_above_block_threshold, amount_hard_block",
        },
      ],
    });

    const onEvidenceReceived = jest.fn();
    render(
      <ChatScreen
        selectedTransactionId="ui-test-1500k-block"
        onEvidenceReceived={onEvidenceReceived}
        onNavigateToExplanation={jest.fn()}
      />
    );

    // Verify transaction badge is visible in header
    expect(screen.getByText("ui-test-1500k-block")).toBeInTheDocument();

    // User asks "Why was my $1,500,000 transaction blocked?"
    const input = screen.getByPlaceholderText("Ask the financial assistant...");
    const sendBtn = screen.getByRole("button", { name: "Send" });

    fireEvent.change(input, { target: { value: "Why was my $1,500,000 transaction blocked?" } });
    await act(async () => {
      fireEvent.click(sendBtn);
    });

    // Verify request payload sent exact selectedTransactionId
    expect(mockRequest).toHaveBeenCalledWith(
      "assistant",
      "/assistant/chat",
      expect.objectContaining({
        method: "POST",
        body: expect.objectContaining({
          transaction_id: "ui-test-1500k-block",
          message: "Why was my $1,500,000 transaction blocked?",
        }),
      })
    );

    // Verify onEvidenceReceived called with live evidence containing exact ID
    expect(onEvidenceReceived).toHaveBeenCalledWith(
      expect.arrayContaining([
        expect.objectContaining({
          tool: "get_fraud_analysis",
          result: expect.stringContaining("ui-test-1500k-block"),
        }),
      ])
    );
  });

  it("5. TrustPanelScreen shows the same transaction ID as the raw tool evidence", () => {
    const liveEvidence = [
      {
        tool: "get_fraud_analysis",
        result:
          "FRAUD ANALYSIS — LIVE COMPONENT 2 AUDIT RECORD\nTransaction ID:\nui-test-1500k-block\nRecorded Fraud Score:\n1.0000\nRecorded Risk Level:\nHIGH\nRecorded Status:\nBLOCK\nRECORDED CONTRIBUTING FACTORS:\n- amount_hard_block",
      },
    ];

    render(
      <TrustPanelScreen
        latestEvidence={liveEvidence}
        selectedTransactionId="ui-test-1500k-block"
      />
    );

    // Target Transaction in Trust Panel must match exact ID from evidence, not TX1001
    expect(screen.getByText(/Transaction ui-test-1500k-block/i)).toBeInTheDocument();
    expect(screen.queryByText(/Transaction TX1001/i)).not.toBeInTheDocument();
  });

  it("6. Unknown transaction ID remains RECORD_NOT_FOUND and never falls back to TX1001", () => {
    const notFoundEvidence = [
      {
        tool: "get_fraud_analysis",
        result:
          "FRAUD ANALYSIS — RECORD NOT FOUND / SERVICE UNAVAILABLE\nTransaction ID:\nui-unknown-999\nLookup Status:\nRECORD_NOT_FOUND\nLive Service Detail:\nTransaction 'ui-unknown-999' not found in Component 2 live decision store",
      },
    ];

    render(
      <TrustPanelScreen
        latestEvidence={notFoundEvidence}
        selectedTransactionId="ui-unknown-999"
      />
    );

    expect(screen.getAllByText(/Transaction ui-unknown-999/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.queryByText(/Transaction TX1001/i)).not.toBeInTheDocument();
    expect(screen.getAllByText(/Record Not Found/i).length).toBeGreaterThanOrEqual(1);
  });
});
