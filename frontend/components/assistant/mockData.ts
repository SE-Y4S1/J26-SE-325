import { TransactionDetails, WorkflowStep, ChatMessage, ResponsibleMetric } from "./types";

export const MOCK_TRANSACTION: TransactionDetails = {
  id: "TX1001",
  amount: 250000,
  currency: "LKR",
  status: "BLOCKED",
  riskLevel: "HIGH",
  fraudScore: 0.92,
  factors: [
    "High transfer amount",
    "Unusual location",
    "New device"
  ],
  shapContributions: [
    {
      feature: "amount_thousands",
      label: "High transaction amount",
      shapValue: 0.1933,
      impact: "Increased fraud risk",
      strength: "High"
    },
    {
      feature: "unusual_location",
      label: "Unusual location",
      shapValue: 0.1535,
      impact: "Increased fraud risk",
      strength: "Medium"
    },
    {
      feature: "new_device",
      label: "New device",
      shapValue: 0.1308,
      impact: "Increased fraud risk",
      strength: "Medium"
    }
  ],
  blockchainAudit: {
    auditId: "AUDIT-2026-001",
    decision: "TRANSACTION_BLOCKED",
    recordStatus: "VERIFIED",
    blockNumber: 1847291,
    timestamp: "2026-08-30 14:22:09 UTC"
  }
};

export const MOCK_AGENT_TOOLS = [
  "get_fraud_analysis",
  "get_blockchain_audit"
];

export const MOCK_WORKFLOW_STEPS: WorkflowStep[] = [
  { id: "1", name: "User Request", description: "Query received: Why was transaction TX1001 blocked?", status: "completed" },
  { id: "2", name: "Intent Understanding", description: "LangGraph LLM parsed intent: Transaction Risk Query", status: "completed" },
  { id: "3", name: "Tool Selection", description: "Agent selected get_fraud_analysis & get_blockchain_audit", status: "completed" },
  { id: "4", name: "Fraud Analysis Tool", description: "Evaluated anomaly parameters & high-risk score (0.92)", status: "completed" },
  { id: "5", name: "Blockchain Audit Tool", description: "Retrieved immutable log AUDIT-2026-001 (Block 1847291)", status: "completed" },
  { id: "6", name: "Evidence Collection", description: "Aggregated model outputs & transactional telemetry", status: "completed" },
  { id: "7", name: "SHAP Explanation", description: "Computed feature attributions (+0.193, +0.153, +0.131)", status: "completed" },
  { id: "8", name: "Responsible AI Checks", description: "Passed Privacy, Safety & Evidence Grounding filters", status: "completed" },
  { id: "9", name: "Natural Language Generation", description: "SLM summarized evidence into natural explanation", status: "completed" },
  { id: "10", name: "Trust Panel", description: "Dispatched to human-in-the-loop oversight interface", status: "completed" },
];

export const MOCK_INITIAL_MESSAGES: ChatMessage[] = [
  {
    id: "m1",
    sender: "user",
    text: "Why was transaction TX1001 blocked?",
    timestamp: "14:22 PM"
  },
  {
    id: "m2",
    sender: "assistant",
    text: "TX1001 was blocked because the transaction was classified as high risk. The strongest contributing factors were the unusually high transfer amount, unusual location, and use of a new device.",
    timestamp: "14:22 PM",
    toolsCalled: ["get_fraud_analysis", "get_blockchain_audit"],
    evidenceUsed: ["Fraud Analysis", "Blockchain Audit", "SHAP Explanation"],
    hasWhyButton: true
  }
];

export const MOCK_RESPONSIBLE_METRICS: ResponsibleMetric[] = [
  {
    id: "privacy",
    title: "PRIVACY",
    status: "Protected",
    description: "Personal financial information is handled according to the platform's privacy controls."
  },
  {
    id: "consent",
    title: "CONSENT",
    status: "Active",
    description: "User consent is required for applicable AI-assisted processing."
  },
  {
    id: "safety",
    title: "SAFETY",
    status: "Protected",
    description: "The assistant avoids presenting uncertain financial outcomes as guaranteed results."
  },
  {
    id: "fairness",
    title: "FAIRNESS",
    status: "Monitoring",
    description: "Fairness and bias monitoring is enabled."
  },
  {
    id: "transparency",
    title: "TRANSPARENCY",
    status: "Enabled",
    description: "AI-generated explanations are grounded in available evidence."
  },
  {
    id: "user_control",
    title: "USER CONTROL",
    status: "Enabled",
    description: "Users can review and control AI-assisted actions."
  }
];

export const SUGGESTED_QUESTIONS = [
  "Why was my transaction blocked?",
  "Explain my portfolio risk",
  "Show the evidence for this decision"
];
