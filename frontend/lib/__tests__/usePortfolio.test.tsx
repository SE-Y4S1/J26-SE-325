/**
 * Portfolio selection, which the Withdraw, Optimize and Forecast screens all sit on.
 *
 * The selection lives in the URL, and the awkward case is an EMPTY `?portfolio=`: written
 * the obvious way (`selectedId && list.find(...)?.id) ?? list[0]?.id`) that yields "",
 * which `??` does not treat as missing, so the empty string reaches getPortfolio and the
 * request 404s. These pin the behaviour that avoids it.
 */

import { renderHook, waitFor } from "@testing-library/react";

import * as platform from "@/lib/api/platform";
import { usePortfolio } from "../usePortfolio";

const searchParams = new URLSearchParams();
const push = jest.fn();

jest.mock("next/navigation", () => ({
  useRouter: () => ({ push, replace: jest.fn() }),
  useSearchParams: () => searchParams,
}));

jest.mock("@/lib/api/platform", () => ({
  listPortfolios: jest.fn(),
  getPortfolio: jest.fn(),
}));

const listPortfolios = platform.listPortfolios as jest.MockedFunction<
  typeof platform.listPortfolios
>;
const getPortfolio = platform.getPortfolio as jest.MockedFunction<typeof platform.getPortfolio>;

const SUMMARIES = [
  { id: 7, name: "Main", holdings_count: 3 },
  { id: 9, name: "Other", holdings_count: 1 },
] as unknown as Awaited<ReturnType<typeof platform.listPortfolios>>;

beforeEach(() => {
  jest.clearAllMocks();
  [...searchParams.keys()].forEach((k) => searchParams.delete(k));
  listPortfolios.mockResolvedValue(SUMMARIES);
  getPortfolio.mockImplementation(
    async (id: number) =>
      ({ id, name: `P${id}`, holdings: [] }) as unknown as Awaited<
        ReturnType<typeof platform.getPortfolio>
      >,
  );
});

it("falls back to the first portfolio when the URL names none", async () => {
  const { result } = renderHook(() => usePortfolio());

  await waitFor(() => expect(result.current.loading).toBe(false));
  expect(getPortfolio).toHaveBeenCalledWith(7);
  expect(result.current.portfolio?.id).toBe(7);
});

it("honours an explicit ?portfolio= id", async () => {
  searchParams.set("portfolio", "9");
  const { result } = renderHook(() => usePortfolio());

  await waitFor(() => expect(result.current.loading).toBe(false));
  expect(getPortfolio).toHaveBeenCalledWith(9);
});

it("falls back rather than requesting an empty id", async () => {
  // THE edge case. `?portfolio=` with no value must not reach getPortfolio as "".
  searchParams.set("portfolio", "");
  const { result } = renderHook(() => usePortfolio());

  await waitFor(() => expect(result.current.loading).toBe(false));
  expect(getPortfolio).toHaveBeenCalledWith(7);
  expect(getPortfolio).not.toHaveBeenCalledWith("");
});

it("falls back when the URL names a portfolio that no longer exists", async () => {
  // A deleted id in a bookmarked URL must not strand the user on an empty screen.
  searchParams.set("portfolio", "404");
  const { result } = renderHook(() => usePortfolio());

  await waitFor(() => expect(result.current.loading).toBe(false));
  expect(getPortfolio).toHaveBeenCalledWith(7);
});

it("reports an empty account without calling getPortfolio", async () => {
  listPortfolios.mockResolvedValue([] as unknown as Awaited<
    ReturnType<typeof platform.listPortfolios>
  >);
  const { result } = renderHook(() => usePortfolio());

  await waitFor(() => expect(result.current.loading).toBe(false));
  expect(getPortfolio).not.toHaveBeenCalled();
  expect(result.current.portfolio).toBeNull();
  expect(result.current.error).toBeNull();
});

it("surfaces a load failure as an error rather than an empty screen", async () => {
  listPortfolios.mockRejectedValue(new Error("Cannot reach the platform service"));
  const { result } = renderHook(() => usePortfolio());

  await waitFor(() => expect(result.current.loading).toBe(false));
  expect(result.current.error).toMatch(/Cannot reach the platform service/);
  expect(result.current.portfolio).toBeNull();
});
