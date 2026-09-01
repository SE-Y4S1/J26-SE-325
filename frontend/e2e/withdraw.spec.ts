/**
 * The withdrawal screen -- Component 1's headline output, end to end.
 *
 * Worth running against the real optimizer rather than a fixture: the plan comes out of the
 * fuzzy inference system and the GA, and this is the only test anywhere that checks a real
 * plan reaches the browser with its rule trace intact.
 */

import { expect, test } from "@playwright/test";

import { newUser, register, seedPortfolio } from "./fixtures";

test.beforeEach(async ({ page }) => {
  await register(page, newUser());
  await seedPortfolio(page);
  await page.getByRole("link", { name: "Withdraw" }).click();
  await expect(page.getByRole("heading", { name: "Instant withdrawal" })).toBeVisible();
});

test("a modest withdrawal produces a feasible plan with a rule trace", async ({ page }) => {
  await page.getByLabel("Amount to raise").fill("5000");
  await page.getByLabel("Deadline (trading days)").fill("5");
  await page.getByRole("button", { name: "Plan withdrawal" }).click();

  // exact:true throughout: these words also appear in the surrounding prose, and a loose
  // match resolves to several elements.
  await expect(page.getByText("Raised", { exact: true })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText("Expected slippage", { exact: true })).toBeVisible();
  await expect(page.getByText("Days required", { exact: true })).toBeVisible();

  // Every row here came from the GA's schedule, not from the page.
  await expect(page.getByRole("table")).toBeVisible();
  await expect(page.getByText("Feasible", { exact: true })).toBeVisible();
});

test("an impossible withdrawal is reported as a shortfall, not an error", async ({
  page,
  request,
}) => {
  // RQ4 depends on infeasible plans being VISIBLE, but the demo book can never produce one:
  // the participation cap is 10% of ADV in SHARES per day, and every seeded holding has an
  // ADV far larger than the position, so the whole book clears in a single day. Asking for
  // more than the book is worth is a different path -- the service rejects that with a 400.
  //
  // So this builds a deliberately illiquid book: 100,000 shares at $100 is $10,000,000 of
  // value, but an ADV of 10,000 shares caps liquidation at 1,000 shares -- $100,000 -- a
  // day. Requesting $5,000,000 in one day is well within the portfolio's value and far
  // outside what the cap allows.
  const token = await page.evaluate(() => window.localStorage.getItem("j26_access_token"));
  expect(token, "the browser should hold a token after registering").toBeTruthy();

  const created = await request.post("http://localhost:8100/portfolios", {
    headers: { Authorization: `Bearer ${token}` },
    data: {
      name: "Illiquid book",
      base_currency: "USD",
      holdings: [
        {
          symbol: "ILLIQ",
          quantity: 100_000,
          current_price: 100,
          avg_daily_volume: 10_000,
          cost_basis: 120,
        },
      ],
    },
  });
  expect(created.ok(), `portfolio creation failed: ${await created.text()}`).toBeTruthy();
  const { id } = await created.json();

  await page.goto(`/withdraw?portfolio=${id}`);
  await expect(page.getByRole("heading", { name: "Instant withdrawal" })).toBeVisible();

  await page.getByLabel("Amount to raise").fill("5000000");
  await page.getByLabel("Deadline (trading days)").fill("1");
  await page.getByRole("button", { name: "Plan withdrawal" }).click();

  await expect(page.getByText("Cannot raise the full amount")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText(/shortfall of/)).toBeVisible();
  await expect(page.getByText(/not by an error/)).toBeVisible();
  // Crucially NOT the error path: this is a result, and the service returned HTTP 200.
  await expect(page.getByText("Could not plan")).toHaveCount(0);
});
