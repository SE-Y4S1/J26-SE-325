import { isNavActive } from "../nav";

describe("isNavActive", () => {
  it("matches the entry itself and anything beneath it", () => {
    expect(isNavActive("/audit", "/audit")).toBe(true);
    expect(isNavActive("/audit/decisions", "/audit")).toBe(true);
    expect(isNavActive("/audit/decisions/EXE-1", "/audit")).toBe(true);
  });

  it("does not match a sibling that only shares a prefix, or another section", () => {
    expect(isNavActive("/auditor", "/audit")).toBe(false);
    expect(isNavActive("/withdraw", "/audit")).toBe(false);
    expect(isNavActive(null, "/audit")).toBe(false);
  });
});
