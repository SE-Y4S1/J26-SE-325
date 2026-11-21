/**
 * The optimizer answers with warnings when its recommendation rests on an assumption the user
 * must see: no forecasts (so expected return is 0 by assumption), an assumed correlation, or a
 * trade that is a large share of a name's volume. A warning the page does not render protects
 * nobody, so the rendering is tested.
 */

import { render, screen } from "@testing-library/react";

import { OptimizeNotices } from "../OptimizeNotices";

describe("OptimizeNotices", () => {
  it("lists every warning the optimizer returned", () => {
    render(
      <OptimizeNotices
        warnings={[
          "No holding has a forecast, so the return objective is inactive.",
          "THIN: the recommended trade is 200% of its average daily volume.",
        ]}
        scenarioSource="historical"
      />,
    );

    expect(screen.getByText(/return objective is inactive/)).toBeInTheDocument();
    expect(screen.getByText(/THIN: the recommended trade is 200%/)).toBeInTheDocument();
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
  });

  it("says when risk was estimated with an assumed correlation rather than real history", () => {
    render(<OptimizeNotices warnings={[]} scenarioSource="assumed_correlation" />);

    expect(screen.getByText(/assumed correlation/i)).toBeInTheDocument();
  });

  it("stays silent when there is nothing to say", () => {
    const { container } = render(<OptimizeNotices warnings={[]} scenarioSource="historical" />);

    expect(container).toBeEmptyDOMElement();
  });

  it("tolerates a response from an older service that has neither field", () => {
    const { container } = render(<OptimizeNotices />);

    expect(container).toBeEmptyDOMElement();
  });
});
