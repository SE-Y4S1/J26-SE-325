/**
 * Portfolio CRUD against the real platform service, including persistence.
 *
 * Holdings are stored in exactly the shape Component 1's `Holding` contract expects, so
 * this also exercises that the two services agree on the field names with no translation
 * layer between them.
 */

import { expect, test } from "@playwright/test";

import { newUser, register, seedPortfolio } from "./fixtures";

test("a fresh account starts empty and can create a portfolio", async ({ page }) => {
  await register(page, newUser());
  await page.goto("/portfolio");

  await expect(page.getByText("No portfolios yet")).toBeVisible();
  await page.getByRole("button", { name: "Create a demo portfolio" }).click();

  // A real total means the holdings round-tripped through the service rather than sitting
  // in React state.
  await expect(page.getByText("Total value")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText(/^\$[\d,]+\.\d\d$/).first()).toBeVisible();
});

test("edits persist across a reload", async ({ page }) => {
  await register(page, newUser());
  await seedPortfolio(page);

  await page.getByRole("button", { name: "Reset to demo book" }).click();
  await page.getByRole("button", { name: "Save" }).click();
  await expect(page.getByText("Total value")).toBeVisible();

  await page.reload();
  await expect(page.getByText("Total value")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText(/^\$[\d,]+\.\d\d$/).first()).toBeVisible();
});

test("one user cannot read another user's portfolio", async ({ request }) => {
  // Driven through the API rather than the UI. The property under test is a server-side
  // authorization rule, and the browser cannot show what the service refuses to send --
  // scraping the URL for an id only tested that the app puts ids in URLs, which it does not
  // always do.
  const PLATFORM = "http://localhost:8100";

  async function accountWithPortfolio() {
    const user = newUser();
    // display_name, not name: RegisterRequest declares display_name, and sending the
    // wrong key produced a 422 that only surfaced as a failed login one step later.
    const created = await request.post(`${PLATFORM}/auth/register`, {
      data: { display_name: user.name, email: user.email, password: user.password },
    });
    expect(created.ok(), `register failed: ${await created.text()}`).toBeTruthy();
    const login = await request.post(`${PLATFORM}/auth/login`, {
      data: { email: user.email, password: user.password },
    });
    expect(login.ok(), "login should succeed").toBeTruthy();
    const { access_token } = await login.json();

    const portfolio = await request.post(`${PLATFORM}/portfolios`, {
      headers: { Authorization: `Bearer ${access_token}` },
      data: { name: "Private book", base_currency: "USD", holdings: [] },
    });
    expect(portfolio.ok(), `portfolio creation failed: ${await portfolio.text()}`).toBeTruthy();
    return { token: access_token, portfolio: await portfolio.json() };
  }

  const owner = await accountWithPortfolio();
  const intruder = await accountWithPortfolio();

  // The owner can read their own.
  const mine = await request.get(`${PLATFORM}/portfolios/${owner.portfolio.id}`, {
    headers: { Authorization: `Bearer ${owner.token}` },
  });
  expect(mine.status()).toBe(200);

  // The intruder gets 404, NOT 403: a 403 would confirm the portfolio exists, which leaks
  // the very thing the check is protecting.
  const theirs = await request.get(`${PLATFORM}/portfolios/${owner.portfolio.id}`, {
    headers: { Authorization: `Bearer ${intruder.token}` },
  });
  expect(theirs.status()).toBe(404);

  // And with no token at all.
  const anonymous = await request.get(`${PLATFORM}/portfolios/${owner.portfolio.id}`);
  expect(anonymous.status()).toBe(401);
});
