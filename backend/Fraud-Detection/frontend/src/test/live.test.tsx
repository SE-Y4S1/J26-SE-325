import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Live } from "../pages/Live";

describe("live scoring", () => {
  it("does not show a decision when the API call fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("offline")));
    render(<Live />);
    fireEvent.change(screen.getByLabelText(/transaction id/i), { target: { value: "demo-1" } });
    fireEvent.change(screen.getByLabelText(/^amount$/i), { target: { value: "25" } });
    fireEvent.click(screen.getByRole("button", { name: /score transaction/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/no decision was produced/i);
    expect(screen.queryByText("ALLOW")).not.toBeInTheDocument();
    expect(screen.queryByText("STEP_UP")).not.toBeInTheDocument();
    expect(screen.queryByText("BLOCK")).not.toBeInTheDocument();
  });
});
