"""Start the fraud gateway."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn

from fraud_engine.api import create_app
from fraud_engine.config import load_settings


def main() -> None:
    settings = load_settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings["api"]["host"], port=int(settings["api"]["port"]))


if __name__ == "__main__":
    main()
