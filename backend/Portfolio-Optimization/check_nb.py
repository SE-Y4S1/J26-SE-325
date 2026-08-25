"""Validate notebook code cells parse as Python.

IPython magics (`!cmd`, `%magic`) are not Python, so they are replaced with `pass` rather
than deleted -- deleting them leaves empty `if`/`for` bodies that ast rejects even though
IPython runs the cell fine. Backslash continuations of a magic are swallowed too.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

BACKSLASH = "\\"


def cell_to_python(source: str) -> str:
    lines: list[str] = []
    swallowing = False

    for line in source.splitlines():
        stripped = line.lstrip()

        if swallowing:
            swallowing = line.rstrip().endswith(BACKSLASH)
            continue

        if stripped.startswith(("!", "%")):
            swallowing = line.rstrip().endswith(BACKSLASH)
            lines.append(" " * (len(line) - len(stripped)) + "pass")
            continue

        lines.append(line)

    return "\n".join(lines)


def main(path: str) -> int:
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]

    failures = 0
    for index, cell in enumerate(cells):
        source = "".join(cell["source"])

        # U+FFFD means text was corrupted on write, not merely un-renderable in a console.
        if "�" in source:
            print(f"cell {index}: contains U+FFFD replacement characters")
            failures += 1

        try:
            ast.parse(cell_to_python(source))
        except SyntaxError as exc:
            print(f"cell {index}: {exc}")
            failures += 1

    print(f"{Path(path).name}: {len(cells)} code cells, {failures} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
