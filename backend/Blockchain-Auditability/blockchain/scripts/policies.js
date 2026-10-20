const { PolicyRegistry } = require("../../src/services/policyRegistry");

/**
 * Registers every policy in src/policies on the AuditRegistry contract, skipping hashes that
 * are already active. Idempotent, so it is safe to run against an existing deployment.
 * The hashes come from the same registry the backend uses, so what is registered on-chain is
 * exactly what the backend will commit to in each record.
 */
async function registerPolicies(auditRegistry, log = console.log) {
  const registered = [];
  for (const policy of new PolicyRegistry().list()) {
    const active = await auditRegistry.isPolicyActive(policy.policyHash);
    if (active) {
      log(`Policy ${policy.id} v${policy.version} already active: ${policy.policyHash}`);
    } else {
      const tx = await auditRegistry.registerPolicy(policy.policyHash);
      await tx.wait();
      log(`Registered policy ${policy.id} v${policy.version} (${policy.hashScheme}): ${policy.policyHash}`);
    }
    registered.push({ id: policy.id, version: policy.version, hashScheme: policy.hashScheme, policyHash: policy.policyHash });
  }
  return registered;
}

module.exports = { registerPolicies };
