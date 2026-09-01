/**
 * The three teammate components, end to end through the shared shell.
 *
 * Playwright's webServer starts only the platform and Component 1, so Components 2-4 are
 * normally DOWN during a test run. That is deliberate rather than a gap: the thing most
 * worth guaranteeing is that one teammate's service being absent never breaks the platform
 * for everyone else.
 *
 * So each spec accepts either outcome — a real response, or the explanatory unavailable
 * state — and fails only if the page errors, crashes, or renders neither. Bring the services
 * up (`docker compose up component2 component3 component4`) and the same specs still pass,
 * exercising the live path instead.
 */

import { expect, test, type Page } from "@playwright/test";

import { newUser, register, seedPortfolio } from "./fixtures";

/** Neither a crash nor a blank screen: the page resolved to one of its two real states. */
async function resolvedToSomething(page: Page, resultText: RegExp, downText: RegExp) {
  await expect(
    page.getByText(resultText).first().or(page.getByText(downText).first()),
  ).toBeVisible({ timeout: 30_000 });
}

test.beforeEach(async ({ page }) => {
  await register(page, newUser());
  await seedPortfolio(page);
});

test("all four components are reachable from one session", async ({ page }) => {
  // The integration claim in one assertion: a single login, and every component has a live
  // nav entry rather than a "pending" placeholder.
  for (const label of ["Portfolio", "Withdraw", "Optimize", "Forecast", "Fraud", "Audit", "Assistant"]) {
    await expect(page.getByRole("link", { name: label })).toBeVisible();
  }
});

test("fraud screen scores a transaction or explains the service is down", async ({ page }) => {
  await page.getByRole("link", { name: "Fraud" }).click();
  await expect(page.getByRole("heading", { name: "Fraud detection" })).toBeVisible();

  // The form is prefilled, so this is one click.
  await page.getByRole("button", { name: "Score transaction" }).click();

  await resolvedToSomething(page, /ALLOW|STEP-UP|BLOCK/, /Fraud service is not running/);
});

test("audit screen lists records or explains the bridge is down", async ({ page }) => {
  await page.getByRole("link", { name: "Audit" }).click();
  await expect(page.getByRole("heading", { name: "Audit trail" })).toBeVisible();

  await resolvedToSomething(
    page,
    /Anchored decisions|No decisions anchored yet/,
    /Audit bridge is not running/,
  );
});

test("assistant screen accepts a question and reaches a real state", async ({ page }) => {
  // Deliberately does NOT wait for the answer. The agent is a local Ollama model on CPU:
  // measured at 82s idle and over 240s under load, so any fixed budget here is a coin toss.
  // Waiting on it would be testing Ollama's throughput, not this platform's wiring.
  //
  // Three outcomes all prove the integration: the request is in flight, an answer came back,
  // or the service is down and the page says so. The full round trip is asserted separately
  // in research-features.spec.ts, which skips unless Component 4 is actually running.
  await page.getByRole("link", { name: "Assistant" }).click();
  await expect(page.getByRole("heading", { name: "Assistant" })).toBeVisible();

  await page.getByRole("button", { name: "Ask" }).click();

  await expect(
    page
      .getByText(/The agent is working/)
      .first()
      .or(page.getByText(/Answer|Evidence/).first())
      .or(page.getByText(/Assistant is not running/).first()),
  ).toBeVisible({ timeout: 30_000 });
});

test("a component being down never breaks Component 1", async ({ page }) => {
  // The point of the whole arrangement. Visit every teammate screen, then confirm the
  // withdrawal path still works — no shared state has been poisoned by a failed fetch.
  for (const label of ["Fraud", "Audit", "Assistant"]) {
    await page.getByRole("link", { name: label }).click();
    await page.waitForLoadState("domcontentloaded");
  }

  await page.getByRole("link", { name: "Withdraw" }).click();
  await expect(page.getByRole("heading", { name: "Instant withdrawal" })).toBeVisible();

  await page.getByLabel("Amount to raise").fill("5000");
  await page.getByLabel("Deadline (trading days)").fill("5");
  await page.getByRole("button", { name: "Plan withdrawal" }).click();

  await expect(page.getByText("Raised", { exact: true })).toBeVisible({ timeout: 30_000 });
});
