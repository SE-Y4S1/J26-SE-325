import { ComponentPlaceholder } from "@/components/ComponentPlaceholder";

export default function FraudPage() {
  return (
    <ComponentPlaceholder
      component="2"
      title="Fraud Detection"
      owner="Dushanthini R."
      port={8001}
      objective="Develop a real-time fraud and behavioral anomaly detection engine tightly coupled with a deterministic security enforcement gateway, robust to adversarial evasion."
      capabilities={[
        "Streaming feature ingestion and dynamic transaction graph construction",
        "Hybrid LSTM autoencoder + GNN dual-stream anomaly classifier",
        "Behavioral biometrics and device fingerprinting",
        "Security gateway: rate limiting, IP reputation scoring, step-up authentication",
        "Concept-drift monitoring and adversarial robustness testing (camouflage, slow-drift, structuring)",
        "Analyst feedback dashboard and online learning pipeline",
      ]}
    />
  );
}
