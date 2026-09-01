"""Capability report for the optional foundation forecasters.

`forecasting/base.py` decides which adapters to register by reading
`artifacts/env_report.json` rather than by importing the packages, so one broken optional
dependency cannot take down the whole registry at import time.

That report used to be written only by `tests/test_env.py`, which made a passing test a
hidden prerequisite for the library working at all. On a fresh checkout -- a Colab session,
a CI runner, a teammate's clone -- `artifacts/` is gitignored, so no report existed and
every foundation model reported itself "not installed" no matter what was actually
installed. The generator lives here now so it can be run without pytest; the test still
calls it, so there is one implementation rather than two that can drift.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
ENV_REPORT_PATH = ARTIFACTS / "env_report.json"

# module name -> the pip distribution that provides it.
OPTIONAL_FORECASTERS = {
    "timesfm": "timesfm[torch]==2.0.2",
    "chronos": "chronos-forecasting",
}


def probe_forecasters() -> dict[str, dict[str, object]]:
    """Import each optional forecaster and record what happened.

    Any exception means unavailable -- not just ImportError. A package whose import raises
    because of a numpy ABI mismatch is exactly as unusable as one that is absent, and the
    difference matters only to whoever has to fix it, so the message is kept.
    """
    results: dict[str, dict[str, object]] = {}
    for module, dist in OPTIONAL_FORECASTERS.items():
        entry: dict[str, object] = {"distribution": dist}
        try:
            mod = importlib.import_module(module)
            entry["available"] = True
            entry["version"] = getattr(mod, "__version__", "unknown")
            if module == "timesfm":
                # PyPI tops out at 2.0.2; "2.5" is the MODEL generation, exposed as this class.
                entry["has_timesfm_2p5_torch"] = hasattr(mod, "TimesFM_2p5_200M_torch")
        except Exception as exc:  # noqa: BLE001 - any failure means "unavailable"
            entry["available"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
        results[module] = entry
    return results


def _version_of(module: str) -> str:
    """Version string, or a note about why the module is not there.

    Deliberately non-fatal: the report describes the environment, and "torch is absent" is
    a fact about it rather than a reason to refuse to write it. The optimization path
    (RQ2-RQ4) needs no torch at all, so a missing one must not block report generation.
    """
    try:
        return str(getattr(importlib.import_module(module), "__version__", "unknown"))
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({type(exc).__name__})"


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def build_env_report() -> dict[str, object]:
    """The full environment snapshot, including which optional forecasters imported."""
    return {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "numpy": _version_of("numpy"),
        "pandas": _version_of("pandas"),
        "torch": _version_of("torch"),
        "torch_cuda_available": _cuda_available(),
        "optional_forecasters": probe_forecasters(),
    }


def write_env_report() -> dict[str, object]:
    """Write the report to its canonical path and return it."""
    report = build_env_report()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ENV_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    """`python -m forecasting.env_report` -- regenerate the report from a shell."""
    report = write_env_report()
    forecasters = report["optional_forecasters"]
    assert isinstance(forecasters, dict)

    print(f"wrote {ENV_REPORT_PATH}")
    for name, meta in sorted(forecasters.items()):
        if meta.get("available"):
            print(f"  {name:10} available (version {meta.get('version')})")
        else:
            print(f"  {name:10} UNAVAILABLE -- {meta.get('error', 'no reason recorded')}")

    if not any(m.get("available") for m in forecasters.values()):
        print("WARNING: RQ1 will have no foundation-model rows until one of these imports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
