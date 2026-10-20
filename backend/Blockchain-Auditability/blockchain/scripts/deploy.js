const hre = require("hardhat");
const fs = require("fs");
const path = require("path");
const { computePolicyHash } = require("../../src/services/policyDefinition");

async function main() {
  console.log("=============================================================");
  console.log("Deploying AuditRegistry Smart Contract to local Hardhat node...");
  console.log("=============================================================");

  const [deployer] = await hre.ethers.getSigners();
  console.log(`Deployer Account: ${deployer.address}`);

  const AuditRegistry = await hre.ethers.getContractFactory("AuditRegistry");
  const auditRegistry = await AuditRegistry.deploy();
  await auditRegistry.waitForDeployment();

  const contractAddress = await auditRegistry.getAddress();
  console.log(`AuditRegistry deployed successfully at: ${contractAddress}`);

  // 1. Grant SUBMITTER_ROLE to the deployer/backend signer
  const SUBMITTER_ROLE = await auditRegistry.SUBMITTER_ROLE();
  const grantTx = await auditRegistry.grantRole(SUBMITTER_ROLE, deployer.address);
  await grantTx.wait();
  console.log(`Granted SUBMITTER_ROLE to: ${deployer.address}`);

  // 2. Register initial canonical policy hash (P001)
  const policyHash = computePolicyHash();
  const policyTx = await auditRegistry.registerPolicy(policyHash);
  await policyTx.wait();
  console.log(`Registered canonical policy hash on-chain: ${policyHash}`);

  // 3. Export contract configuration and ABI
  const artifactPath = path.join(__dirname, "../artifacts/contracts/AuditRegistry.sol/AuditRegistry.json");
  let artifact = {};
  if (fs.existsSync(artifactPath)) {
    artifact = JSON.parse(fs.readFileSync(artifactPath, "utf8"));
  }

  const backendConfigDir = path.join(__dirname, "../../src/config");
  if (!fs.existsSync(backendConfigDir)) {
    fs.mkdirSync(backendConfigDir, { recursive: true });
  }

  const configData = {
    address: contractAddress,
    networkUrl: process.env.BLOCKCHAIN_RPC_URL || "http://127.0.0.1:8545",
    chainId: (await hre.ethers.provider.getNetwork()).chainId.toString(),
    submitterAddress: deployer.address,
    policyHash: policyHash,
    deployedAt: new Date().toISOString(),
    abi: artifact.abi || []
  };

  const targetPath = process.env.CONTRACT_CONFIG_PATH || path.join(backendConfigDir, "contractConfig.json");
  const tempPath = `${targetPath}.tmp.${Date.now()}`;
  fs.writeFileSync(tempPath, JSON.stringify(configData, null, 2), "utf8");
  fs.renameSync(tempPath, targetPath);

  console.log(`Contract configuration written atomically to: ${targetPath}`);
  console.log("=============================================================");
}

main().catch((error) => {
  console.error("Deployment failed:", error);
  process.exitCode = 1;
});
