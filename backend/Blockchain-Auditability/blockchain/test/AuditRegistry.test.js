const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("AuditRegistry Smart Contract", function () {
  let auditRegistry;
  let admin, submitter, unauthorizedUser;
  let sampleAuditId, sampleRecordHash, samplePolicyHash;

  beforeEach(async function () {
    [admin, submitter, unauthorizedUser] = await ethers.getSigners();

    const AuditRegistryFactory = await ethers.getContractFactory("AuditRegistry");
    auditRegistry = await AuditRegistryFactory.deploy();
    await auditRegistry.waitForDeployment();

    sampleAuditId = ethers.keccak256(ethers.toUtf8Bytes("TX1001:salt_value_12345"));
    sampleRecordHash = ethers.keccak256(ethers.toUtf8Bytes("canonical_record_json_hash"));
    samplePolicyHash = ethers.keccak256(ethers.toUtf8Bytes("P001_v1.0_rules"));

    // Register active policy
    await auditRegistry.connect(admin).registerPolicy(samplePolicyHash);

    // Grant SUBMITTER_ROLE to submitter account
    const SUBMITTER_ROLE = await auditRegistry.SUBMITTER_ROLE();
    await auditRegistry.connect(admin).grantRole(SUBMITTER_ROLE, submitter.address);
  });

  it("1. Happy path: authorized submitter can record and retrieve commitment", async function () {
    const tx = await auditRegistry.connect(submitter).recordAudit(
      sampleAuditId,
      sampleRecordHash,
      samplePolicyHash
    );
    await tx.wait();

    const has = await auditRegistry.hasAudit(sampleAuditId);
    expect(has).to.equal(true);

    const audit = await auditRegistry.getAudit(sampleAuditId);
    expect(audit.exists).to.equal(true);
    expect(audit.recordHash).to.equal(sampleRecordHash);
    expect(audit.policyHash).to.equal(samplePolicyHash);
    expect(audit.submitter).to.equal(submitter.address);
  });

  it("2. Event verification: emits AuditRecorded with correct indexed parameters", async function () {
    await expect(
      auditRegistry.connect(submitter).recordAudit(
        sampleAuditId,
        sampleRecordHash,
        samplePolicyHash
      )
    )
      .to.emit(auditRegistry, "AuditRecorded")
      .withArgs(sampleAuditId, samplePolicyHash, sampleRecordHash, (val) => val > 0n, submitter.address);
  });

  it("3. Write-once: rejects duplicate auditId with AuditAlreadyExists", async function () {
    await auditRegistry.connect(submitter).recordAudit(
      sampleAuditId,
      sampleRecordHash,
      samplePolicyHash
    );

    // Second write with identical auditId must revert
    await expect(
      auditRegistry.connect(submitter).recordAudit(
        sampleAuditId,
        sampleRecordHash,
        samplePolicyHash
      )
    ).to.be.revertedWithCustomError(auditRegistry, "AuditAlreadyExists");
  });

  it("4. Validation: rejects zero auditId with InvalidAuditId", async function () {
    await expect(
      auditRegistry.connect(submitter).recordAudit(
        ethers.ZeroHash,
        sampleRecordHash,
        samplePolicyHash
      )
    ).to.be.revertedWithCustomError(auditRegistry, "InvalidAuditId");
  });

  it("5. Validation: rejects zero recordHash with InvalidRecordHash", async function () {
    await expect(
      auditRegistry.connect(submitter).recordAudit(
        sampleAuditId,
        ethers.ZeroHash,
        samplePolicyHash
      )
    ).to.be.revertedWithCustomError(auditRegistry, "InvalidRecordHash");
  });

  it("6. Policy registry: rejects record if policy is not active with PolicyNotActive", async function () {
    const unregisteredPolicy = ethers.keccak256(ethers.toUtf8Bytes("P999_unknown"));

    await expect(
      auditRegistry.connect(submitter).recordAudit(
        sampleAuditId,
        sampleRecordHash,
        unregisteredPolicy
      )
    ).to.be.revertedWithCustomError(auditRegistry, "PolicyNotActive");
  });

  it("7. Role security: unauthorized account cannot record audit", async function () {
    const SUBMITTER_ROLE = await auditRegistry.SUBMITTER_ROLE();
    await expect(
      auditRegistry.connect(unauthorizedUser).recordAudit(
        sampleAuditId,
        sampleRecordHash,
        samplePolicyHash
      )
    ).to.be.revertedWithCustomError(auditRegistry, "AccessControlUnauthorizedAccount");
  });

  it("8. Policy administration: non-admin cannot register policy", async function () {
    const newPolicy = ethers.keccak256(ethers.toUtf8Bytes("P002_v2.0"));
    await expect(
      auditRegistry.connect(unauthorizedUser).registerPolicy(newPolicy)
    ).to.be.revertedWithCustomError(auditRegistry, "AccessControlUnauthorizedAccount");
  });

  it("9. Policy lifecycle: admin can deactivate policy and prevent further records", async function () {
    await auditRegistry.connect(admin).deactivatePolicy(samplePolicyHash);
    const isActive = await auditRegistry.isPolicyActive(samplePolicyHash);
    expect(isActive).to.equal(false);

    await expect(
      auditRegistry.connect(submitter).recordAudit(
        sampleAuditId,
        sampleRecordHash,
        samplePolicyHash
      )
    ).to.be.revertedWithCustomError(auditRegistry, "PolicyNotActive");
  });

  it("10. Role management: admin can revoke SUBMITTER_ROLE", async function () {
    const SUBMITTER_ROLE = await auditRegistry.SUBMITTER_ROLE();
    await auditRegistry.connect(admin).revokeRole(SUBMITTER_ROLE, submitter.address);

    await expect(
      auditRegistry.connect(submitter).recordAudit(
        sampleAuditId,
        sampleRecordHash,
        samplePolicyHash
      )
    ).to.be.revertedWithCustomError(auditRegistry, "AccessControlUnauthorizedAccount");
  });

  it("11. Query missing audit: getAudit returns exists == false for non-existent auditId", async function () {
    const nonExistentId = ethers.keccak256(ethers.toUtf8Bytes("TX9999_missing"));
    const audit = await auditRegistry.getAudit(nonExistentId);
    expect(audit.exists).to.equal(false);
    expect(audit.recordHash).to.equal(ethers.ZeroHash);
    expect(audit.timestamp).to.equal(0n);
  });

  it("12. Timestamp fidelity: recorded timestamp matches the mined block timestamp", async function () {
    const tx = await auditRegistry.connect(submitter).recordAudit(
      sampleAuditId,
      sampleRecordHash,
      samplePolicyHash
    );
    const receipt = await tx.wait();
    const block = await ethers.provider.getBlock(receipt.blockNumber);

    const audit = await auditRegistry.getAudit(sampleAuditId);
    expect(audit.timestamp).to.equal(BigInt(block.timestamp));
  });
});
