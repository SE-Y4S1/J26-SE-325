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

export function askAssistant(
  message: string,
  preferences?: ResponsibleAIPreferences
): Promise<AssistantReply> {
  return request<AssistantReply>("assistant", "/assistant/chat", {
    method: "POST",
    body: { message, preferences },
    auth: false,
  });
}

export function assistantHealth(): Promise<{ status: string }> {
  return request<{ status: string }>("assistant", "/health", { auth: false });
}
