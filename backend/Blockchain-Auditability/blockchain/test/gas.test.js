const { expect } = require("chai");
const { ethers } = require("hardhat");

/**
 * Gas cost of the ledger operations, measured on the Hardhat network (optimizer on, 200 runs).
 * The numbers are printed for the evaluation's cost table; the assertions only guard against a
 * regression that would make anchoring materially more expensive.
 */
describe("Gas cost of anchoring", function () {
  const gas = {};

  after(function () {
    console.log("\n      Gas used (Hardhat, solc 0.8.20, optimizer 200 runs):");
    for (const [op, used] of Object.entries(gas)) console.log(`        ${op.padEnd(44)} ${used.toString().padStart(8)}`);
  });

  it("measures registerPolicy, recordAudit (decision and review) and the reads", async function () {
    const [admin] = await ethers.getSigners();
    const registry = await (await ethers.getContractFactory("AuditRegistry")).deploy();
    const deployReceipt = await registry.deploymentTransaction().wait();
    gas["deploy AuditRegistry"] = deployReceipt.gasUsed;

    const policyHash = ethers.keccak256(ethers.toUtf8Bytes("P002"));
    gas["registerPolicy"] = (await (await registry.registerPolicy(policyHash)).wait()).gasUsed;

    const h = (s) => ethers.keccak256(ethers.toUtf8Bytes(s));
    gas["recordAudit (decision commitment)"] = (await (await registry.connect(admin).recordAudit(h("tx1:salt"), h("decision"), policyHash)).wait()).gasUsed;
    gas["recordAudit (review commitment)"] = (await (await registry.connect(admin).recordAudit(h("tx1:salt:review"), h("review"), policyHash)).wait()).gasUsed;
    gas["getAudit (eth_call, estimated)"] = await registry.getAudit.estimateGas(h("tx1:salt"));

    expect(gas["recordAudit (decision commitment)"]).to.be.lessThan(150000n);
    expect(gas["recordAudit (review commitment)"]).to.be.lessThan(150000n);
    expect(gas["registerPolicy"]).to.be.lessThan(80000n);
  });
});
