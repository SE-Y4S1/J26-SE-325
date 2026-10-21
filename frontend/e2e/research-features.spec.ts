/**
 * The cross-component execution flow, and Component 3's tamper-evidence demonstration.
 *
 * Unlike components.spec.ts, these cannot be satisfied by the service being down: an
 * execution through the fraud gateway and a tamper-evidence walkthrough only mean anything against a live
 * backend. Playwright's webServer starts neither, so each spec checks first and SKIPS rather
 * than failing -- a default `npx playwright test` stays green, and bringing the services up
 * turns these on.
 *
 *   docker compose up component2 component3
 */

import { expect, test } from "@playwright/test";

import { newUser, register, seedPortfolio } from "./fixtures";

async function reachable(url: string): Promise<boolean> {
  try {
    const response = await fetch(url, { signal: AbortSignal.timeout(3000) });
    return response.ok;
  } catch {
    return false;
  }
}

test("an executed withdrawal is screened by Component 2 and anchored by Component 3", async ({ page }) => {
  test.skip(
    !(await reachable("http://localhost:8001/health")) || !(await reachable("http://localhost:8002/api/health")),
    "Components 2 and 3 are not both running",
  );
  test.setTimeout(180_000);

  await register(page, newUser());
  await seedPortfolio(page);
  await page.getByRole("link", { name: "Withdraw" }).click();
  await page.getByRole("button", { name: "Plan withdrawal" }).click();
  await expect(page.getByText("Raised", { exact: true })).toBeVisible({ timeout: 60_000 });

  await page.getByRole("button", { name: "Execute withdrawal" }).click();

  // All three stages render from the platform's stored trace, whatever the verdict.
  const trace = page.getByTestId("execution-trace");
  await expect(trace).toBeVisible({ timeout: 60_000 });
  await expect(trace.getByText("1 · Plan · Component 1")).toBeVisible();
  await expect(trace.getByText("2 · Fraud gateway · Component 2")).toBeVisible();
  await expect(trace.getByText("3 · Audit anchor · Component 3")).toBeVisible();
  await expect(trace.getByText(/ALLOW|STEP_UP|BLOCK/).first()).toBeVisible();
  await expect(trace.getByText(/Execution EXE-[0-9a-f]+/)).toBeVisible();
});

test("tamper-evidence: anchor, verify, tamper, detect", async ({ page }) => {
  test.skip(!(await reachable("http://localhost:8002/api/health")), "Component 3 is not running");
  test.setTimeout(180_000);

  await register(page, newUser());
  await seedPortfolio(page);
  await page.getByRole("link", { name: "Audit" }).click();
  // The walkthrough lives on the audit section's Verify sub-page.
  await page.getByRole("navigation", { name: "Audit sections" }).getByRole("link", { name: "Verify", exact: true }).click();

  await page.getByRole("button", { name: "1 · Anchor a decision" }).click();
  await expect(page.getByText("Policy action")).toBeVisible({ timeout: 60_000 });

  await page.getByRole("button", { name: "2 · Verify" }).click();
  await expect(page.getByText("Integrity verified")).toBeVisible({ timeout: 60_000 });

  await page.getByRole("button", { name: "3 · Tamper with the record" }).click();
  await expect(page.getByText("Record altered")).toBeVisible({ timeout: 60_000 });

  // The whole point of an audit bridge: the record still looks plausible, and the hash
  // comparison catches it anyway.
  await page.getByRole("button", { name: "4 · Verify again" }).click();
  await expect(page.getByText("Tampering detected")).toBeVisible({ timeout: 60_000 });
});

test("assurance engine: a decision is routed, held for review, reviewed and verified", async ({ page }) => {
  test.skip(!(await reachable("http://localhost:8002/api/health")), "Component 3 is not running");
  test.setTimeout(180_000);

  await register(page, newUser());
  await seedPortfolio(page);
  await page.getByRole("link", { name: "Audit" }).click();
  const sections = page.getByRole("navigation", { name: "Audit sections" });

  // Two models that disagree: the engine vetoes automation and holds it for an analyst.
  await sections.getByRole("link", { name: "Decisions", exact: true }).click();
  await page.getByRole("button", { name: "Try a decision" }).click();
  await page.getByRole("button", { name: "Models disagree" }).click();
  await page.getByRole("button", { name: "Route and anchor" }).click();
  await expect(page.getByText("Human review").first()).toBeVisible({ timeout: 60_000 });

  await page.getByRole("link", { name: "Why?" }).click();
  await expect(page.getByText("MODEL_DISAGREEMENT", { exact: true })).toBeVisible({ timeout: 60_000 });
  const transactionId = (await page.locator("h2.font-mono").textContent())!.trim();

  await sections.getByRole("link", { name: "Review queue", exact: true }).click();
  await page.getByLabel(`Reason for ${transactionId}`).fill("e2e: disagreement checked");
  const item = page.getByRole("listitem").filter({ hasText: transactionId });
  await item.getByRole("button", { name: "Approve" }).click();
  await expect(page.getByText("Review anchored")).toBeVisible({ timeout: 60_000 });

  await sections.getByRole("link", { name: "Verify", exact: true }).click();
  await page.getByLabel("Transaction id or audit id").fill(transactionId);
  await page.getByRole("button", { name: "Check" }).click();
  await expect(page.getByText("review commitment · matches the chain")).toBeVisible({ timeout: 60_000 });
});


test("assistant answers a real question with its evidence trail", async ({ page, request }) => {
  test.skip(!(await reachable("http://localhost:8003/health")), "Component 4 is not running");
  // A CPU round trip measured between 82s and past 240s depending on machine load. This is
  // the slow, honest version -- kept out of the default suite by the skip above.
  test.setTimeout(900_000);

  // ChatScreen catches a backend failure and renders canned text from mockData instead, so
  // an assistant bubble appearing proves nothing about the agent being alive. The only
  // check the fallback cannot satisfy is asking the service itself, so do that first.
  const direct = await request.post("http://localhost:8003/assistant/chat", {
    data: { message: "What is a liquidity-aware withdrawal?" },
    timeout: 600_000,
  });
  expect(direct.ok()).toBeTruthy();
  const answer: string = (await direct.json()).answer ?? "";
  expect(answer.length).toBeGreaterThan(80);

  await register(page, newUser());
  await seedPortfolio(page);
  await page.getByRole("link", { name: "Assistant" }).click();

  const question = "What is a liquidity-aware withdrawal?";
  const input = page.getByPlaceholder("Ask the financial assistant...");
  await expect(input).toBeVisible({ timeout: 30_000 });

  // The transcript opens with a seeded exchange, so this label is already on the page once.
  // Counting it is what distinguishes a new turn from the furniture that was always there --
  // matching /Assistant/ would also match the nav link and the page heading.
  const agentLabel = page.getByText("FinTrust Agent");
  const before = await agentLabel.count();

  await input.fill(question);
  await page.getByRole("button", { name: "Send" }).click();

  await expect(page.getByText(question).first()).toBeVisible({ timeout: 30_000 });
  await expect(async () => {
    expect(await agentLabel.count()).toBeGreaterThan(before);
  }).toPass({ timeout: 720_000 });
});
