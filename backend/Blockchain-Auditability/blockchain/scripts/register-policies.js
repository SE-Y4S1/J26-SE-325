/**
 * Registers any new policy documents on an AuditRegistry that is already deployed, e.g. after
 * adding or tuning a policy in src/policies. Uses the address in src/config/contractConfig.json
 * (or CONTRACT_CONFIG_PATH) and must be run by an account holding DEFAULT_ADMIN_ROLE.
 *
 *   npx hardhat run scripts/register-policies.js --network localhost
 */
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");
const { registerPolicies } = require("./policies");

async function main() {
  const configPath = process.env.CONTRACT_CONFIG_PATH || path.join(__dirname, "../../src/config/contractConfig.json");
  const config = JSON.parse(fs.readFileSync(configPath, "utf8"));
  const auditRegistry = await hre.ethers.getContractAt("AuditRegistry", config.address);
  console.log(`AuditRegistry at ${config.address}`);
  const policies = await registerPolicies(auditRegistry);

  config.policies = policies;
  const tempPath = `${configPath}.tmp.${Date.now()}`;
  fs.writeFileSync(tempPath, JSON.stringify(config, null, 2), "utf8");
  fs.renameSync(tempPath, configPath);
  console.log(`Updated ${configPath}`);
}

main().catch((error) => {
  console.error("Policy registration failed:", error);
  process.exitCode = 1;
});
