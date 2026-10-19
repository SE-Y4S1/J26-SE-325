/**
 * Component 4 — Localized Explainable Agentic LLM Assistant (W V A D K Chamara).
 *
 * Shapes mirror the return of `backend/Agentic-Assistance/main.py::chat`.
 */

import { request } from "./client";

export interface AssistantEvidence {
  tool: string;
  result: string;
}

export interface ResponsibleAIPreferences {
  explanationEnabled?: boolean;
  personalizedContext?: boolean;
  dataSharingConsent?: boolean;
}

export interface AssistantReply {
  answer: string;
  agent_execution: {
    tools_used: string[];
    number_of_tools: number;
    status: string;
  };
  evidence: AssistantEvidence[];
  responsible_ai: Record<string, unknown>;
}

export interface ChatHistoryItem {
  role: "user" | "assistant" | "system";
  content: string;
}

export function askAssistant(
  message: string,
  preferences?: ResponsibleAIPreferences,
  history?: ChatHistoryItem[]
): Promise<AssistantReply> {
  return request<AssistantReply>("assistant", "/assistant/chat", {
    method: "POST",
    body: { message, preferences, history: history || [] },
    // The user's own token, forwarded so get_execution_explanation can read THEIR execution
    // trace from the platform service. The assistant never mints one for another user.
    auth: true,
  });
}

export function assistantHealth(): Promise<{ status: string }> {
  return request<{ status: string }>("assistant", "/health", { auth: false });
}

export interface GovernanceDecisionRecord {
  decision_id: string;
  decision_type: "confirmed" | "rejected";
  reference_id: string;
  reference_type: string;
  user_id?: string | null;
  user_email?: string | null;
  timestamp: string;
  evidence_reference: Record<string, unknown>;
  rationale?: string | null;
  status: string;
  is_duplicate?: boolean;
}

export interface GovernanceDecisionResponse {
  status: string;
  is_duplicate: boolean;
  decision: GovernanceDecisionRecord;
}

export interface GovernanceDecisionQueryResponse {
  found: boolean;
  decision: GovernanceDecisionRecord | null;
}

export function recordGovernanceDecision(payload: {
  decision_type: "confirmed" | "rejected";
  reference_id: string;
  reference_type?: string;
  evidence_reference?: Record<string, unknown>;
  rationale?: string;
}): Promise<GovernanceDecisionResponse> {
  return request<GovernanceDecisionResponse>("assistant", "/assistant/governance/decision", {
    method: "POST",
    body: payload,
    auth: true,
  });
}

export function getGovernanceDecision(referenceId: string): Promise<GovernanceDecisionQueryResponse> {
  return request<GovernanceDecisionQueryResponse>(
    "assistant",
    `/assistant/governance/decision/${encodeURIComponent(referenceId)}`,
    {
      method: "GET",
      auth: true,
    }
  );
}

