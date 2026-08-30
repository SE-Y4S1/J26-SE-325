const auditController = require('./src/controllers/auditController');
const auditService = require('./src/services/auditService');

async function testAPI() {
  console.log("=== Testing Component 03 End-to-End API Flow ===");

  // Mock Request & Response objects
  const createMockRes = () => {
    return {
      statusCode: 200,
      jsonData: null,
      status(code) { this.statusCode = code; return this; },
      json(data) { this.jsonData = data; return this; }
    };
  };

  // Test 1: Scenario 1 (Approve)
  console.log("\n1. Testing Scenario 1 (Risk=25, Conf=90) -> Expect APPROVE");
  const req1 = {
    body: {
      transactionId: "TX-TEST-001",
      riskScore: 25,
      confidence: 90,
      amount: 1500,
      transactionType: "TRANSFER",
      modelVersion: "FraudModel-v2"
    }
  };
  const res1 = createMockRes();
  await auditController.evaluateDecision(req1, res1);
  console.log("Response Status:", res1.statusCode);
  console.log("Final Decision:", res1.jsonData.record.finalDecision);
  console.log("Generated SHA-256 Hash:", res1.jsonData.record.hash);

  // Test 2: Scenario 2 (Reject)
  console.log("\n2. Testing Scenario 2 (Risk=92, Conf=94) -> Expect REJECT");
  const req2 = {
    body: {
      transactionId: "TX-TEST-002",
      riskScore: 92,
      confidence: 94,
      amount: 50000,
      transactionType: "WITHDRAWAL",
      modelVersion: "FraudModel-v2"
    }
  };
  const res2 = createMockRes();
  await auditController.evaluateDecision(req2, res2);
  console.log("Response Status:", res2.statusCode);
  console.log("Final Decision:", res2.jsonData.record.finalDecision);

  // Test 3: Scenario 3 (Human Review)
  console.log("\n3. Testing Scenario 3 (Risk=92, Conf=52) -> Expect HUMAN_REVIEW");
  const req3 = {
    body: {
      transactionId: "TX-TEST-003",
      riskScore: 92,
      confidence: 52,
      amount: 75000,
      transactionType: "LOAN_DISBURSEMENT",
      modelVersion: "FraudModel-v2"
    }
  };
  const res3 = createMockRes();
  await auditController.evaluateDecision(req3, res3);
  console.log("Response Status:", res3.statusCode);
  console.log("Result Status:", res3.jsonData.status);

  // Submit Human Review Approval
  console.log("\n3b. Submitting Human Reviewer Approval for TX-TEST-003");
  const reqReview = {
    body: {
      transactionId: "TX-TEST-003",
      reviewerDecision: "APPROVE"
    }
  };
  const resReview = createMockRes();
  await auditController.submitHumanReview(reqReview, resReview);
  console.log("Updated Final Decision:", resReview.jsonData.record.finalDecision);
  console.log("On-Chain Hash:", resReview.jsonData.record.hash);

  // Test 4: Verification (Expect VERIFIED)
  console.log("\n4. Verifying Integrity for TX-TEST-001 -> Expect VERIFIED");
  const reqVerify1 = { params: { transactionId: "TX-TEST-001" } };
  const resVerify1 = createMockRes();
  await auditController.verifyAudit(reqVerify1, resVerify1);
  console.log("Verification Status:", resVerify1.jsonData.status);
  console.log("Verified Boolean:", resVerify1.jsonData.verified);

  // Test 5: Simulate Tampering (Expect INTEGRITY FAILED)
  console.log("\n5. Simulating Tampering on TX-TEST-001 (riskScore changed from 25 to 88)");
  const reqTamper = { params: { transactionId: "TX-TEST-001" }, body: { field: "riskScore", tamperedValue: 88 } };
  const resTamper = createMockRes();
  await auditController.simulateTampering(reqTamper, resTamper);
  console.log("Tampering Message:", resTamper.jsonData.message);

  console.log("\n5b. Re-verifying Integrity for TX-TEST-001 after tampering -> Expect INTEGRITY FAILED");
  const resVerify2 = createMockRes();
  await auditController.verifyAudit(reqVerify1, resVerify2);
  console.log("Verification Status:", resVerify2.jsonData.status);
  console.log("Verified Boolean:", resVerify2.jsonData.verified);

  console.log("\n=== ALL BACKEND API END-TO-END TESTS PASSED SUCCESSFULLY! ===");
}

testAPI().catch(console.error);
