import path from "node:path";

import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end against the REAL backends, not mocks.
 *
 * The services are started from their existing virtualenvs rather than from
 * docker-compose. That is the one deviation from the plan and it is deliberate: building
 * the component1 image installs torch and the full ML stack, which on this machine's very
 * slow link takes hours. These are the same uvicorn processes serving the same FastAPI
 * apps, so the contract under test is identical -- docker-compose only adds isolation.
 * `docker compose up platform component1` works too; point PW_SKIP_WEBSERVER at it.
 *
 * AUTH_REQUIRED is left off for component1, matching how the backend suite runs it. The
 * platform service always enforces auth, so the login and isolation specs are real.
 */

const JWT_SECRET = "playwright-e2e-secret-not-for-any-real-deployment-0123456789";
// Absolute, built from this file's location. A relative command is resolved against the
// webServer's own `cwd`, not the config directory, which is why "../backend/..." failed
// with "'..' is not recognized as an internal or external command".
const BACKEND = path.resolve(__dirname, "..", "backend");
const py = (service: string) => path.join(BACKEND, service, ".venv", "Scripts", "python.exe");

// A backend importing torch takes a while to answer; Playwright's 60s default is not enough.
const BOOT_TIMEOUT = 180_000;

export default defineConfig({
  testDir: "./e2e",
  // Serial: the specs share one platform database, and a parallel run would have them
  // registering and deleting each other's portfolios.
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "list" : [["list"], ["html", { open: "never" }]],

  timeout: 60_000,
  expect: {
    // MOEA/D runs 100 generations over 45 reference directions and takes ~7s server-side.
    timeout: 20_000,
  },

  use: {
    baseURL: "http://localhost:3000",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },

  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        // The full chromium build rather than the headless shell. Playwright 1.61 prefers
        // the shell for headless runs, but only the full build is in this machine's cache
        // and fetching the shell over a ~24 KB/s link is a long wait for no behavioural
        // difference. `npx playwright install chromium` on any other machine covers both.
        channel: "chromium",
      },
    },
  ],

  webServer: process.env.PW_SKIP_WEBSERVER
    ? undefined
    : [
        {
          command: `"${py("Platform")}" -m uvicorn service.api:app --port 8100`,
          cwd: path.join(BACKEND, "Platform"),
          url: "http://localhost:8100/health",
          reuseExistingServer: !process.env.CI,
          timeout: BOOT_TIMEOUT,
          env: { JWT_SECRET, ALLOWED_ORIGINS: "http://localhost:3000" },
        },
        {
          command: `"${py("Portfolio-Optimization")}" -m uvicorn service.api:app --port 8000`,
          cwd: path.join(BACKEND, "Portfolio-Optimization"),
          url: "http://localhost:8000/health",
          reuseExistingServer: !process.env.CI,
          timeout: BOOT_TIMEOUT,
          env: { JWT_SECRET, ALLOWED_ORIGINS: "http://localhost:3000" },
        },
        {
          command: "npm run dev",
          url: "http://localhost:3000",
          reuseExistingServer: !process.env.CI,
          timeout: BOOT_TIMEOUT,
        },
      ],
});
