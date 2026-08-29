import { expect, type Page } from "@playwright/test";

/** A fresh account per spec: the specs share one platform database. */
export function newUser() {
  const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  // example.com, not example.test: pydantic's EmailStr rejects the reserved special-use
  // TLDs (.test, .invalid, .localhost) outright, and the first version of this fixture
  // failed every registration for that reason -- correctly.
  return { name: `E2E ${id}`, email: `e2e-${id}@example.com`, password: "e2e-password-12345" };
}

export async function register(page: Page, user: ReturnType<typeof newUser>) {
  await page.goto("/register");
  await page.getByLabel("Name").fill(user.name);
  await page.getByLabel("Email").fill(user.email);
  await page.getByLabel("Password").fill(user.password);
  await page.getByRole("button", { name: "Create account" }).click();

  // The signed-in shell, not a page heading: a new account has no portfolio yet, so the
  // main area shows an empty state rather than the Portfolio screen.
  await expect(page.getByRole("button", { name: "Sign out" })).toBeVisible({ timeout: 30_000 });
}

export async function login(page: Page, user: ReturnType<typeof newUser>) {
  await page.goto("/login");
  await page.getByLabel("Email").fill(user.email);
  await page.getByLabel("Password").fill(user.password);
  await page.getByRole("button", { name: "Sign in" }).click();
}

/**
 * Give the account a portfolio to work on.
 *
 * A fresh account owns nothing, so every screen that operates on "the current portfolio"
 * shows an empty state until this runs. The demo book is the app's own seed, which keeps
 * the withdrawal and optimize specs deterministic without inventing holdings here.
 */
export async function seedPortfolio(page: Page) {
  await page.goto("/portfolio");

  const create = page.getByRole("button", { name: "Create a demo portfolio" });
  const total = page.getByText("Total value");

  // Wait for the screen to RESOLVE before branching on it. isVisible() answers immediately,
  // so checking it while the portfolio list is still loading returns false, the click is
  // skipped, and the wait for "Total value" then times out against an empty state that
  // nobody ever dismissed -- which is exactly how this failed the first time.
  await expect(create.or(total).first()).toBeVisible({ timeout: 30_000 });

  if (await create.isVisible()) {
    await create.click();
  }

  await expect(total).toBeVisible({ timeout: 30_000 });
}
