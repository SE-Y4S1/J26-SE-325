"""Ensures the flat module tree (data/, features/, ...) is importable from the repo root.

pyproject sets `pythonpath = ["."]`, but this makes `uv run pytest` work identically from a
subdirectory or an IDE runner that does not read that setting.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
