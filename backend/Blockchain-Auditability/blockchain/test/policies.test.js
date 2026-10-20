const { expect } = require("chai");
const { ethers } = require("hardhat");
const { registerPolicies } = require("../scripts/policies");
const { policyRegistry } = require("../../src/services/policyRegistry");

describe("Policy registration (backend registry -> AuditRegistry)", function () {
  let auditRegistry;

  beforeEach(async function () {
    const AuditRegistryFactory = await ethers.getContractFactory("AuditRegistry");
    auditRegistry = await AuditRegistryFactory.deploy();
    await auditRegistry.waitForDeployment();
  });

  it("registers every policy the backend can commit to, with the backend's own hashes", async function () {
    const registered = await registerPolicies(auditRegistry, () => {});
    expect(registered.map((p) => p.id).sort()).to.deep.equal(["P001", "P002"]);
    for (const policy of policyRegistry.list()) {
      expect(await auditRegistry.isPolicyActive(policy.policyHash)).to.equal(true);
    }
  });

  it("is idempotent: a second run sends no transactions", async function () {
    await registerPolicies(auditRegistry, () => {});
    const before = await ethers.provider.getTransactionCount((await ethers.getSigners())[0].address);
    await registerPolicies(auditRegistry, () => {});
    const after = await ethers.provider.getTransactionCount((await ethers.getSigners())[0].address);
    expect(after).to.equal(before);
  });

  it("a record committed under the assurance policy is accepted only once P002 is registered", async function () {
    const [admin] = await ethers.getSigners();
    const p002 = policyRegistry.get("P002").policyHash;
    const auditId = ethers.keccak256(ethers.toUtf8Bytes("TX-P002:salt"));
    const recordHash = ethers.keccak256(ethers.toUtf8Bytes("record"));
    await expect(auditRegistry.connect(admin).recordAudit(auditId, recordHash, p002))
      .to.be.revertedWithCustomError(auditRegistry, "PolicyNotActive");
    await registerPolicies(auditRegistry, () => {});
    await expect(auditRegistry.connect(admin).recordAudit(auditId, recordHash, p002)).to.emit(auditRegistry, "AuditRecorded");
  });
});
