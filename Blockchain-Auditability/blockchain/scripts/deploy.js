const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  console.log("Deploying AuditRegistry Smart Contract to local network...");

  const AuditRegistry = await hre.ethers.getContractFactory("AuditRegistry");
  const auditRegistry = await AuditRegistry.deploy();
  await auditRegistry.waitForDeployment();

  const contractAddress = await auditRegistry.getAddress();
  console.log(`AuditRegistry deployed successfully to address: ${contractAddress}`);

  // Export contract address and ABI for backend consumption
  const artifactPath = path.join(__dirname, "../artifacts/contracts/AuditRegistry.sol/AuditRegistry.json");
  let artifact = {};
  if (fs.existsSync(artifactPath)) {
    artifact = JSON.parse(fs.readFileSync(artifactPath, "utf8"));
  }

  const backendConfigDir = path.join(__dirname, "../../backend/src/config");
  if (!fs.existsSync(backendConfigDir)) {
    fs.mkdirSync(backendConfigDir, { recursive: true });
  }

  const configData = {
    address: contractAddress,
    networkUrl: "http://127.0.0.1:8545",
    chainId: 31337,
    abi: artifact.abi || []
  };

  const configFilePath = path.join(backendConfigDir, "contractConfig.json");
  fs.writeFileSync(configFilePath, JSON.stringify(configData, null, 2));
  console.log(`Saved contract configuration to: ${configFilePath}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
