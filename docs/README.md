# Research Documents

| File | What it is |
|---|---|
| [`TAF-J26-SE-325.pdf`](TAF-J26-SE-325.pdf) | Topic Assessment Form — the approved proposal |

## Working with the TAF

The PDF is a **CamScanner scan with no text layer**, so `pdftotext` and similar tools return
only the watermark. Reading it requires OCR, and pages 8–10 (the objectives-and-novelty
tables) are landscape and need rotating first.

Two things worth knowing when citing it:

- It contains **no written-out research questions**. RQ1–RQ4 as used in Component 1's
  evaluation originate in the build brief, not the form, and are stated explicitly in
  [`../backend/Portfolio-Optimization/README.md`](../backend/Portfolio-Optimization/README.md)
  so they become the documented evaluation frame.
- It references a *"Section 13 (Risks, Ethics, and Mitigation)"* and an *"RQ4"* that do not
  exist in the twelve pages — dangling references worth fixing in the next revision.

Each component README records where its implementation departs from the form, so those
deviations can be defended rather than discovered during marking.
