import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { TrustPanelScreen } from "../TrustPanelScreen";
import * as assistantApi from "@/lib/api/assistant";

jest.mock("@/lib/api/assistant", () => ({
  recordGovernanceDecision: jest.fn(),
  getGovernanceDecision: jest.fn(),
}));

const mockRecordGovernance = assistantApi.recordGovernanceDecision as jest.MockedFunction<
  typeof assistantApi.recordGovernanceDecision
>;
const mockGetGovernance = assistantApi.getGovernanceDecision as jest.MockedFunction<
  typeof assistantApi.getGovernanceDecision
>;

describe("TrustPanelScreen - Human Oversight Decision Persistence", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockGetGovernance.mockResolvedValue({ found: false, decision: null });
  });

  it("loads and displays saved decision state on mount (page refresh simulation)", async () => {
    mockGetGovernance.mockResolvedValueOnce({
      found: true,
      decision: {
        decision_id: "gov-saved-12345",
        decision_type: "confirmed",
        reference_id: "TX1001",
        reference_type: "transaction",
        user_id: "99",
        user_email: "auditor@university.edu",
        timestamp: "2026-10-19T13:30:00Z",
        evidence_reference: {},
        status: "recorded",
      },
    });

    render(<TrustPanelScreen latestEvidence={[]} />);

    await waitFor(() => {
      expect(screen.getByText(/CONFIRMED BY HUMAN REVIEW/i)).toBeInTheDocument();
      expect(screen.getByText(/Record ID: gov-saved-12345/i)).toBeInTheDocument();
      expect(screen.getByText(/Reviewer: #99/i)).toBeInTheDocument();
    });
  });

  it("successfully persists Confirm Decision and updates UI with record metadata", async () => {
    mockRecordGovernance.mockResolvedValueOnce({
      status: "recorded",
      is_duplicate: false,
      decision: {
        decision_id: "gov-new-8888",
        decision_type: "confirmed",
        reference_id: "TX1001",
        reference_type: "transaction",
        user_id: "101",
        user_email: "evaluator@university.edu",
        timestamp: "2026-10-19T14:00:00Z",
        evidence_reference: { fraudScore: 0.92 },
        status: "recorded",
      },
    });

    render(<TrustPanelScreen latestEvidence={[]} />);

    const confirmBtn = screen.getByRole("button", { name: "Confirm Decision" });
    fireEvent.click(confirmBtn);

    await waitFor(() => {
      expect(mockRecordGovernance).toHaveBeenCalledWith({
        decision_type: "confirmed",
        reference_id: "TX1001",
        reference_type: "transaction",
        evidence_reference: expect.objectContaining({
          status: "BLOCKED",
        }),
        rationale: "Human operator confirmed decision",
      });
    });

    await waitFor(() => {
      expect(
        screen.getByText(/Decision confirmed and permanently anchored in Component 4 governance log/i)
      ).toBeInTheDocument();
      expect(screen.getByText(/Record ID: gov-new-8888/i)).toBeInTheDocument();
      expect(screen.getByText(/CONFIRMED BY HUMAN REVIEW/i)).toBeInTheDocument();
    });
  });

  it("successfully persists Reject Action and enforces execution blocking", async () => {
    mockRecordGovernance.mockResolvedValueOnce({
      status: "recorded",
      is_duplicate: false,
      decision: {
        decision_id: "gov-rej-9999",
        decision_type: "rejected",
        reference_id: "TX1001",
        reference_type: "transaction",
        user_id: "101",
        timestamp: "2026-10-19T14:05:00Z",
        evidence_reference: { fraudScore: 0.92 },
        status: "recorded",
      },
    });

    render(<TrustPanelScreen latestEvidence={[]} />);

    const rejectBtn = screen.getByRole("button", { name: "Reject Action" });
    fireEvent.click(rejectBtn);

    await waitFor(() => {
      expect(mockRecordGovernance).toHaveBeenCalledWith({
        decision_type: "rejected",
        reference_id: "TX1001",
        reference_type: "transaction",
        evidence_reference: expect.anything(),
        rationale: "Human operator rejected action",
      });
    });

    await waitFor(() => {
      expect(screen.getByText(/Action REJECTED by human operator and anchored in governance log/i)).toBeInTheDocument();
      expect(screen.getByText(/Execution Prohibited/i)).toBeInTheDocument();
      expect(screen.getByText(/REJECTED BY HUMAN OVERSIGHT/i)).toBeInTheDocument();
      expect(screen.getByText(/Execution Halted \(Human Rejected\)/i)).toBeInTheDocument();
      expect(screen.getByText(/Record ID: gov-rej-9999/i)).toBeInTheDocument();
    });
  });

  it("handles persistence failure gracefully and does NOT claim decision was recorded", async () => {
    mockRecordGovernance.mockRejectedValueOnce(new Error("Network timeout: Component 4 unreachable"));

    render(<TrustPanelScreen latestEvidence={[]} />);

    const confirmBtn = screen.getByRole("button", { name: "Confirm Decision" });
    fireEvent.click(confirmBtn);

    await waitFor(() => {
      expect(screen.getByText(/Persistence Failed/i)).toBeInTheDocument();
      expect(screen.getByText(/Network timeout: Component 4 unreachable/i)).toBeInTheDocument();
      expect(screen.getByText(/The decision was NOT recorded in the governance log/i)).toBeInTheDocument();
    });

    // Success text must NOT appear
    expect(
      screen.queryByText(/Decision confirmed and permanently anchored in Component 4 governance log/i)
    ).not.toBeInTheDocument();
    expect(screen.queryByText(/CONFIRMED BY HUMAN REVIEW/i)).not.toBeInTheDocument();
  });

  it("prevents redundant duplicate submissions when already confirmed", async () => {
    mockRecordGovernance.mockResolvedValueOnce({
      status: "recorded",
      is_duplicate: false,
      decision: {
        decision_id: "gov-single-111",
        decision_type: "confirmed",
        reference_id: "TX1001",
        reference_type: "transaction",
        timestamp: "2026-10-19T14:10:00Z",
        evidence_reference: {},
        status: "recorded",
      },
    });

    render(<TrustPanelScreen latestEvidence={[]} />);

    const confirmBtn = screen.getByRole("button", { name: "Confirm Decision" });
    fireEvent.click(confirmBtn);

    await waitFor(() => {
      expect(screen.getByText(/Record ID: gov-single-111/i)).toBeInTheDocument();
    });

    // Button should now show "Decision Confirmed" and be disabled
    const confirmedBtn = screen.getByRole("button", { name: "Decision Confirmed" });
    expect(confirmedBtn).toBeDisabled();

    // Clicking again should not trigger another call
    fireEvent.click(confirmedBtn);
    expect(mockRecordGovernance).toHaveBeenCalledTimes(1);
  });

  it("verifies Rollback button remains disabled", () => {
    render(<TrustPanelScreen latestEvidence={[]} />);
    const rollbackBtn = screen.getByRole("button", { name: /Rollback \/ Undo/i });
    expect(rollbackBtn).toBeDisabled();
  });
});
