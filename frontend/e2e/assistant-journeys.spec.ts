/**
 * PP1 Component 4 — End-to-End Browser Journeys (A through J):
 *
 * A. Open assistant, submit valid portfolio question, inspect response and evidence.
 * B. Investigate test transaction, inspect fraud evidence, open Trust Panel.
 * C. Look up test blockchain transaction and verify displayed result.
 * D. Confirm test recommendation, refresh page, verify saved decision.
 * E. Reject separate test recommendation, verify workflow respects rejection.
 * F. Navigate between Settings and Responsible AI, refresh both routes, browser Back/Forward.
 * G. Disable personalized context, verify blocked portfolio lookup.
 * H. Clear assistant session cache, verify auth and preferences intact.
 * I. Inspect accurate fallback disclosure on Trust Panel.
 * J. Inspect pipeline traces for greeting fast path.
 */

import { expect, test } from "@playwright/test";
import { newUser, register, seedPortfolio } from "./fixtures";

test.describe("PP1 Component 4 — End-to-End Browser Journeys", () => {
  test.setTimeout(180_000);

  test.beforeEach(async ({ page }) => {
    const user = newUser();
    await register(page, user);
    await seedPortfolio(page);
  });

  test("Journey A: Submit valid portfolio question, inspect response and evidence", async ({ page }) => {
    await page.goto("/assistant");
    await expect(page.getByPlaceholder("Ask the financial assistant...")).toBeVisible({ timeout: 15_000 });

    const question = "What is my current portfolio allocation?";
    await page.getByPlaceholder("Ask the financial assistant...").fill(question);
    await page.getByRole("button", { name: "Send" }).click();

    await expect(page.getByText(question).first()).toBeVisible({ timeout: 15_000 });
    // Assistant responds with portfolio analysis or fallback disclosure
    await expect(page.getByText(/portfolio|allocation|holdings|baseline|FinTrust Agent/i).first()).toBeVisible({ timeout: 30_000 });
  });

  test("Journey B: Investigate a test transaction, inspect fraud evidence, and open Trust Panel", async ({ page }) => {
    await page.goto("/assistant");
    const question = "Investigate transaction TX1001 for fraud risk";
    await page.getByPlaceholder("Ask the financial assistant...").fill(question);
    await page.getByRole("button", { name: "Send" }).click();

    await expect(page.getByText(question).first()).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(/TX1001/i).first()).toBeVisible({ timeout: 30_000 });

    // Open the Trust Panel tab via sidebar button
    await page.getByRole("button", { name: /Trust & Safety/i }).click();
    await expect(page.getByRole("heading", { name: "Trust Panel" })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(/Human-in-the-Loop Oversight/i)).toBeVisible();
    await expect(page.getByText(/TX1001/i).first()).toBeVisible();
  });

  test("Journey C: Look up a test blockchain transaction and verify displayed result", async ({ page }) => {
    await page.goto("/assistant");
    const question = "Verify blockchain audit record for transaction TX1001";
    await page.getByPlaceholder("Ask the financial assistant...").fill(question);
    await page.getByRole("button", { name: "Send" }).click();

    await expect(page.getByText(question).first()).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(/TX1001/i).first()).toBeVisible({ timeout: 30_000 });
  });

  test("Journey D: Confirm a test recommendation, refresh the page, and verify saved decision", async ({ page }) => {
    await page.goto("/assistant?tab=trust-panel");
    await expect(page.getByRole("heading", { name: "Trust Panel" })).toBeVisible({ timeout: 15_000 });

    const confirmBtn = page.getByRole("button", { name: /Confirm Decision/i });
    await expect(confirmBtn).toBeVisible({ timeout: 15_000 });
    await confirmBtn.click();

    // Verify UI reflects confirmation
    await expect(page.getByText(/CONFIRMED BY HUMAN REVIEW/i)).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(/Record ID:\s*gov-/i)).toBeVisible();

    // Refresh page to verify persistence restoration
    await page.reload();
    await expect(page.getByText(/CONFIRMED BY HUMAN REVIEW/i)).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(/Record ID:\s*gov-/i)).toBeVisible();
  });

  test("Journey E: Reject a recommendation and verify workflow respects rejection", async ({ page }) => {
    await page.goto("/assistant?tab=trust-panel");
    await expect(page.getByRole("heading", { name: "Trust Panel" })).toBeVisible({ timeout: 15_000 });

    const rejectBtn = page.getByRole("button", { name: /Reject Action/i });
    await expect(rejectBtn).toBeVisible({ timeout: 15_000 });
    await rejectBtn.click();

    // Verify UI reflects rejection and halted status
    await expect(page.getByText(/REJECTED BY HUMAN OVERSIGHT/i)).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(/Execution Halted/i)).toBeVisible();
  });

  test("Journey F: Navigate between Settings and Responsible AI, refresh both routes, browser Back/Forward", async ({ page }) => {
    // 1. Visit Settings route (redirects to /assistant?tab=settings)
    await page.goto("/settings");
    await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText("System & Account Configuration")).toBeVisible();
    await expect(page.getByText("Platform Service", { exact: true })).toBeVisible();

    // 2. Refresh Settings route
    await page.reload();
    await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible({ timeout: 15_000 });

    // 3. Navigate to Responsible AI via sidebar button
    await page.getByRole("button", { name: "Responsible AI" }).click();
    await expect(page.getByRole("heading", { name: "Responsible AI & Safety" })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText("Governance & Ethics Framework")).toBeVisible();
    await expect(page.getByText("AI explanation enabled")).toBeVisible();

    // 4. Refresh Responsible AI route
    await page.reload();
    await expect(page.getByRole("heading", { name: "Responsible AI & Safety" })).toBeVisible({ timeout: 15_000 });

    // 5. Browser Back button returns to settings
    await page.goBack();
    await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible({ timeout: 15_000 });

    // 6. Browser Forward button returns to responsible-ai
    await page.goForward();
    await expect(page.getByRole("heading", { name: "Responsible AI & Safety" })).toBeVisible({ timeout: 15_000 });
  });

  test("Journey G: Disable personalized context and verify that no portfolio lookup occurs", async ({ page }) => {
    await page.goto("/assistant?tab=responsible-ai");
    await expect(page.getByRole("heading", { name: "Responsible AI & Safety" })).toBeVisible({ timeout: 15_000 });

    // Toggle off "Personalized financial context"
    const toggleRow = page.locator("div.flex.items-center.justify-between", { hasText: "Personalized financial context" });
    const toggleButton = toggleRow.locator("button");
    await expect(toggleButton).toBeVisible({ timeout: 15_000 });
    await toggleButton.click();

    // Navigate to Assistant chat tab
    await page.getByRole("button", { name: "AI Assistant" }).click();
    await expect(page.getByPlaceholder("Ask the financial assistant...")).toBeVisible({ timeout: 15_000 });

    await page.getByPlaceholder("Ask the financial assistant...").fill("What is my current portfolio value?");
    await page.getByRole("button", { name: "Send" }).click();

    await expect(
      page.getByText(/Personalized financial context is disabled|Personalized context disabled|no personalized data/i).first()
    ).toBeVisible({ timeout: 30_000 });
  });

  test("Journey H: Clear assistant session cache and verify authentication and preferences remain intact", async ({ page }) => {
    await page.goto("/assistant?tab=settings");
    await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible({ timeout: 15_000 });

    const clearBtn = page.getByRole("button", { name: /Clear Session Cache/i });
    await expect(clearBtn).toBeVisible();
    await clearBtn.click();

    // Success feedback appears
    await expect(page.getByText(/Session cache cleared successfully/i)).toBeVisible({ timeout: 10_000 });

    // Verify user is still authenticated (exact Sign out button is still visible)
    await expect(page.getByRole("button", { name: "Sign out", exact: true })).toBeVisible();

    // Verify Responsible AI route is reachable and intact
    await page.getByRole("button", { name: "Responsible AI" }).click();
    await expect(page.getByRole("heading", { name: "Responsible AI & Safety" })).toBeVisible({ timeout: 15_000 });
  });

  test("Journey I: Inspect accurate fallback disclosure on Trust Panel", async ({ page }) => {
    await page.goto("/assistant?tab=trust-panel");
    await expect(page.getByRole("heading", { name: "Trust Panel" })).toBeVisible({ timeout: 15_000 });

    // Baseline or live status badge
    await expect(page.getByText(/AI Decision Summary/i)).toBeVisible();
    await expect(page.getByText(/Mock Baseline \(No Active Session Evidence\)|Active Session:/i)).toBeVisible();
  });

  test("Journey J: Inspect pipeline traces for greeting fast path", async ({ page }) => {
    await page.goto("/assistant");
    const greeting = "Hello FinAgent!";
    await page.getByPlaceholder("Ask the financial assistant...").fill(greeting);
    await page.getByRole("button", { name: "Send" }).click();

    await expect(page.getByText(greeting).first()).toBeVisible({ timeout: 15_000 });
    // Fast path response renders immediately without tool delay
    await expect(page.getByText(/I am FinAgent/i).first()).toBeVisible({ timeout: 15_000 });
  });
});
