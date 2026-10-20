import { parseFraudResult } from "../explanationParser";
import { sanitizeAssistantResponse } from "../ChatScreen";

describe("Frontend Fraud Evidence Regression Suite", () => {
  it("1. Unknown transaction ID / Not Found never returns TX1001 or 0.92 mock data", () => {
    const notFoundEvidence = `
FRAUD ANALYSIS — RECORD NOT FOUND / SERVICE UNAVAILABLE
Transaction ID: TX-UNKNOWN-8888
Lookup Status: RECORD_NOT_FOUND
Live Service Detail: No decision is held for this transaction.
Recorded Fraud Score: Unavailable (no verified live decision)
Recorded Risk Level: Unavailable
Recorded Status: RECORD_NOT_FOUND
`;

    const parsed = parseFraudResult(notFoundEvidence);
    expect(parsed).not.toBeNull();
    expect(parsed?.transactionId).toBe("TX-UNKNOWN-8888");
    expect(parsed?.provenanceType).toBe("not_found");
    expect(parsed?.fraudScore).toBeUndefined();
    expect(parsed?.status).toBe("RECORD_NOT_FOUND");
    expect(parsed?.factors).toEqual([]);

    const sanitized = sanitizeAssistantResponse(
      "Notice: No live portfolio was retrieved for this account from the platform service.",
      [{ tool: "get_fraud_analysis", result: notFoundEvidence }]
    );
    expect(sanitized).toContain("TX-UNKNOWN-8888");
    expect(sanitized).toContain("No verified live fraud decision was found");
    expect(sanitized).not.toContain("TX1001");
    expect(sanitized).not.toContain("0.92");
    expect(sanitized).not.toContain("BLOCKED");
  });

  it("2. Live lookup failure never produces fabricated score or mock explanation", () => {
    const errorEvidence = `
FRAUD ANALYSIS — RECORD NOT FOUND / SERVICE UNAVAILABLE
Transaction ID: TX-ERROR-1234
Lookup Status: SERVICE_UNAVAILABLE
Live Service Detail: Connection refused
Recorded Fraud Score: Unavailable (no verified live decision)
Recorded Risk Level: Unavailable
Recorded Status: SERVICE_UNAVAILABLE
`;

    const parsed = parseFraudResult(errorEvidence);
    expect(parsed).not.toBeNull();
    expect(parsed?.transactionId).toBe("TX-ERROR-1234");
    expect(parsed?.provenanceType).toBe("unavailable");
    expect(parsed?.fraudScore).toBeUndefined();
    expect(parsed?.status).toBe("SERVICE_UNAVAILABLE");

    const sanitized = sanitizeAssistantResponse(
      "Notice: No live portfolio was retrieved for this account from the platform service.",
      [{ tool: "get_fraud_analysis", result: errorEvidence }]
    );
    expect(sanitized).toContain("TX-ERROR-1234");
    expect(sanitized).not.toContain("0.92");
    expect(sanitized).not.toContain("TX1001");
    expect(sanitized).not.toContain("new device, unusual location");
  });

  it("3. Real transaction returns own live decision and marks uncollected signals as unavailable", () => {
    const liveEvidence = `
FRAUD ANALYSIS — LIVE COMPONENT 2 AUDIT RECORD
Transaction ID: TX-LIVE-9999
Recorded Fraud Score: 0.3500
Recorded Risk Level: LOW
Recorded Status: ALLOW
RECORDED CONTRIBUTING FACTORS:
- Transaction velocity normal
COMPONENT 2 DUAL-STREAM MODEL SIGNALS:
- Behavioral Stream Anomaly Score (LSTM Autoencoder): 0.1200
- Relational Stream Risk Score (GNN): 0.0800
- Fused Final Risk Score: 0.3500
- Model Version: fraud-detector-v2.1 (trained bundle validated on a public benchmark)
GENUINE SIGNALS & TELEMETRY STATUS:
- Device Signals: Unavailable (device fingerprint and new-device telemetry not collected or retained in gateway audit trail)
- Location Signals: Unavailable (geographic location and IP coordinates not collected or retained in gateway audit trail)
`;

    const parsed = parseFraudResult(liveEvidence);
    expect(parsed).not.toBeNull();
    expect(parsed?.transactionId).toBe("TX-LIVE-9999");
    expect(parsed?.provenanceType).toBe("live");
    expect(parsed?.fraudScore).toBe(0.35);
    expect(parsed?.status).toBe("ALLOW");
    expect(parsed?.riskLevel).toBe("LOW");
    expect(parsed?.modelVersion).toBe("fraud-detector-v2.1");
    expect(parsed?.benchmarkModel).toBe(true);
    expect(parsed?.streamSignals?.behavioralScore).toBe(0.12);
    expect(parsed?.streamSignals?.graphScore).toBe(0.08);
    expect(parsed?.unavailableSignals).toContain("Device Telemetry (Not retained in audit trail)");
    expect(parsed?.unavailableSignals).toContain("Location Telemetry (Not retained in audit trail)");
  });

  it("4. Explicit demo mode is clearly labeled as synthetic baseline", () => {
    const demoEvidence = `
FRAUD ANALYSIS — DEMO / SYNTHETIC SPECIFICATION (EXPLICIT DEMO MODE)
Transaction ID: TX1001 [DEMO]
Recorded Fraud Score: 0.92 (Demo Baseline)
Recorded Risk Level: HIGH (Demo Baseline)
Recorded Status: BLOCKED (Demo Baseline)
RECORDED CONTRIBUTING FACTORS:
- New device
- Unusual location
- High transfer amount
LOCAL SHAP PROTOTYPE — NOT THE PRODUCTION MODEL
Prototype fraud probability: 1.0000
`;

    const parsed = parseFraudResult(demoEvidence);
    expect(parsed).not.toBeNull();
    expect(parsed?.transactionId).toBe("TX1001");
    expect(parsed?.provenanceType).toBe("demo");
    expect(parsed?.fraudScore).toBe(0.92);
    expect(parsed?.status).toBe("BLOCKED");
    expect(parsed?.factors).toContain("New device");
    expect(parsed?.prototypeProbability).toBe(1.0);
  });
});
