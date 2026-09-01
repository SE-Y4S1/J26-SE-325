import nextJest from "next/jest.js";

// next/jest wires up SWC, the tsconfig paths, CSS stubs and next.config for us. Hand-rolling
// a babel or ts-jest transform is the usual way this setup rots against a Next upgrade.
const createJestConfig = nextJest({ dir: "./" });

/** @type {import('jest').Config} */
const config = {
  testEnvironment: "jest-environment-jsdom",
  // next/jest reads tsconfig paths, but not reliably without a baseUrl, and tsconfig.json
  // declares "@/*": ["./*"] with none. Mapping it explicitly is one line and removes the
  // dependency on that behaviour.
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/$1",
  },
  setupFilesAfterEnv: ["<rootDir>/jest.setup.ts"],
  // e2e/ holds Playwright specs. Without this Jest collects them, and a Playwright spec run
  // under Jest fails in a way that looks like a broken test rather than a misrouted one.
  testPathIgnorePatterns: ["<rootDir>/node_modules/", "<rootDir>/.next/", "<rootDir>/e2e/"],
  collectCoverageFrom: [
    "lib/**/*.{ts,tsx}",
    "components/**/*.{ts,tsx}",
    "app/**/*.{ts,tsx}",
    // Generated from the services' OpenAPI; testing it would only assert that
    // openapi-typescript works.
    "!lib/api/generated/**",
  ],
};

export default createJestConfig(config);
