/**
 * Registration and login against the real platform service.
 *
 * This is the one flow where mocking would prove nothing: the whole point is that argon2
 * hashing, JWT issuing and the CORS preflight all line up between two separately-deployed
 * services.
 */

import { expect, test } from "@playwright/test";

import { newUser, register } from "./fixtures";

test("a new user can register and lands in the platform", async ({ page }) => {
  await register(page, newUser());

  // The nav is the proof of a real session: it only renders behind the auth guard.
  await expect(page.getByRole("link", { name: "Withdraw" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Optimize" })).toBeVisible();
  // A brand-new account owns nothing, and the empty state says so rather than erroring.
  await expect(page.getByText("No portfolios yet")).toBeVisible();
});

test("a wrong password is rejected and does not let the user in", async ({ page }) => {
  const user = newUser();
  await register(page, user);

  await page.goto("/login");
  await page.getByLabel("Email").fill(user.email);
  await page.getByLabel("Password").fill("definitely-not-the-password");
  await page.getByRole("button", { name: "Sign in" }).click();

  // Still on login, with something said about it -- not silently swallowed.
  await expect(page).toHaveURL(/\/login/);
  await expect(page.getByText(/incorrect|invalid|failed/i)).toBeVisible();
});

test("an unauthenticated visitor cannot reach the platform shell", async ({ page }) => {
  await page.context().clearCookies();
  await page.goto("/");
  await page.evaluate(() => window.localStorage.clear());
  await page.goto("/withdraw");

  await expect(page).toHaveURL(/\/login/, { timeout: 15_000 });
});

test("the session survives a reload", async ({ page }) => {
  const user = newUser();
  await register(page, user);
  await page.reload();

  await expect(page).not.toHaveURL(/\/login/);
});
