/**
 * MOEA/D allocation, and the forecast screen's expected-unavailable state.
 */

import { expect, test } from "@playwright/test";

import { newUser, register, seedPortfolio } from "./fixtures";

test.beforeEach(async ({ page }) => {
  await register(page, newUser());
  await seedPortfolio(page);
});

test("MOEA/D returns an allocation with its objectives", async ({ page }) => {
  await page.getByRole("link", { name: /Optimize/i }).click();
  await expect(page.getByRole("heading", { name: "Long-term allocation" })).toBeVisible();

  await page.getByRole("button", { name: "Run MOEA/D" }).click();

  // ~7s server-side: 100 generations over 45 reference directions.
  // exact:true: each of these words also appears in the explanatory prose around the
  // stats, and a loose match resolves to several elements.
  await expect(page.getByText("Expected return", { exact: true })).toBeVisible({
    timeout: 45_000,
  });
  await expect(page.getByText("CVaR (95%)", { exact: true })).toBeVisible();
  await expect(page.getByText("Liquidity cost", { exact: true })).toBeVisible();
  await expect(page.getByText("Pareto front", { exact: true })).toBeVisible();
});

test("the forecast screen explains a 503 instead of showing an error", async ({ page }) => {
  // /forecast returns 503 until a model is registered, which is the expected state until
  // the Colab fine-tune has run. Rendering that as a failure would send someone debugging
  // a working system.
  await page.getByRole("link", { name: /Forecast/i }).click();
  await expect(page.getByRole("heading", { name: "Forecasts" })).toBeVisible();

  await page.getByRole("button", { name: /Forecast \d+ symbols/ }).click();

  await expect(page.getByText(/This is expected/i)).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText(/colab_finetune\.ipynb/)).toBeVisible();
});
