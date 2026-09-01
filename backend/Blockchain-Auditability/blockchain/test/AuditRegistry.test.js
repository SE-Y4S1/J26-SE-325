const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("AuditRegistry Smart Contract", function () {
  let auditRegistry;
  let owner;

  beforeEach(async function () {
    [owner] = await ethers.getSigners();
    const AuditRegistry = await ethers.getContractFactory("AuditRegistry");
    auditRegistry = await AuditRegistry.deploy();
    await auditRegistry.waitForDeployment();
  });

  it("Should record an audit hash commitment and retrieve it", async function () {
    const txId = "TX1001";
    const decision = "REJECT";
    const policyVer = "1.0";
    const modelVer = "FraudModel-v2";
    const recordHash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    await auditRegistry.recordAudit(txId, decision, policyVer, modelVer, recordHash);

    const exists = await auditRegistry.hasAudit(txId);
    expect(exists).to.equal(true);

    const audit = await auditRegistry.getAudit(txId);
    expect(audit.recordHash).to.equal(recordHash);
    expect(audit.decision).to.equal(decision);
    expect(audit.policyVersion).to.equal(policyVer);
    expect(audit.modelVersion).to.equal(modelVer);
  });

  it("Should fail if transaction ID is empty", async function () {
    await expect(
      auditRegistry.recordAudit("", "APPROVE", "1.0", "v1", "hash")
    ).to.be.revertedWith("Transaction ID cannot be empty");
  });
});
