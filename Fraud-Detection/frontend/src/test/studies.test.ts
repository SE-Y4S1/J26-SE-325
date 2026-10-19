import { describe, expect, it } from "vitest";
import { PIPELINE_CHECK, RESEARCH_STUDIES } from "../studies";

describe("saved experiment files", () => {
  it("keeps the synthetic check out of the research list", () => {
    expect(RESEARCH_STUDIES.every((study) => study.research)).toBe(true);
    expect(PIPELINE_CHECK.research).toBe(false);
    expect(PIPELINE_CHECK.id).toBe("synthetic-fixture");
  });

  it("reads the executed IEEE fusion weight instead of inventing one", () => {
    const ieee = RESEARCH_STUDIES.find((study) => study.id === "ieee-cis");
    const wide = RESEARCH_STUDIES.find((study) => study.id === "ieee-cis-100000");
    expect(ieee?.fusionAlpha).toBe(0);
    expect(wide?.fusionAlpha).toBe(0);
    const fused = ieee?.models.find((model) => model.model === "fused_lstm_gnn");
    const gnn = ieee?.models.find((model) => model.model === "gnn_only");
    expect(fused?.auroc).toBe(gnn?.auroc);
    expect(ieee?.bestModel).toBe("Random forest");
  });

  it("records that Elliptic skipped the LSTM and the fused model", () => {
    const elliptic = RESEARCH_STUDIES.find((study) => study.id === "elliptic");
    expect(elliptic?.models.find((model) => model.model === "lstm_only")?.skipped).toBe(true);
    expect(elliptic?.models.find((model) => model.model === "fused_lstm_gnn")?.skipped).toBe(true);
    expect(elliptic?.attacks).toHaveLength(0);
    expect(elliptic?.bestModel).toBe("Static GCN");
  });
});
