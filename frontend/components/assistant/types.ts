export type TabType = "assistant" | "explanation" | "trust-panel" | "responsible-ai" | "settings";

export interface ShapContribution {
  feature: string;
  label: string;
  shapValue: number; // e.g. +0.1933
  impact: "Increased fraud risk" | "Decreased fraud risk";
  strength: "High" | "Medium" | "Low";
}

export interface BlockchainAudit {
  auditId: string;
  decision: string;
  recordStatus: "VERIFIED" | "PENDING" | "FAILED";
  blockNumber: number;
  timestamp: string;
}

export interface TransactionDetails {
  id: string;
  amount: number;
  currency: string;
  status: "BLOCKED" | "APPROVED" | "FLAGGED";
  riskLevel: "HIGH" | "MEDIUM" | "LOW";
  fraudScore: number; // e.g. 0.92
  factors: string[];
  shapContributions: ShapContribution[];
  blockchainAudit: BlockchainAudit;
}

export interface WorkflowStep {
  id: string;
  name: string;
  description: string;
  status: "completed" | "in_progress" | "pending";
}

export interface ChatMessage {
  id: string;
  sender: "user" | "assistant";
  text: string;
  timestamp: string;
  toolsCalled?: string[];
  evidenceUsed?: string[];
  hasWhyButton?: boolean;
}

export type TrustActionState = "pending" | "confirmed" | "rejected" | "more_explanation_requested";

export interface ResponsibleMetric {
  id: string;
  title: string;
  status: "Protected" | "Active" | "Monitoring" | "Enabled";
  description: string;
}
