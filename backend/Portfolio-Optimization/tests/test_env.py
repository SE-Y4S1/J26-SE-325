"""Phase 0 dependency gate.

Runs BEFORE any business logic is written. Two jobs:

1. Fail loudly if a hard requirement is missing or broken, so we find out now rather than
   three phases in.
2. Record which *optional* foundation forecasters actually import. TimesFM and Chronos-Bolt
   are both optional extras; RQ1 reports on whichever survive this gate. The result is
   written to artifacts/env_report.json and consumed by forecasting/base.py.

Several of these are not idle import checks -- they exercise the specific code path we
depend on, because the known risks here are runtime, not install-time:

  * scikit-fuzzy 0.5.0 has had no release since Aug 2024 and declares no numpy upper bound,
    so pip will happily install it against numpy 2.x. It is pure-Python (no ABI risk), but
    removed aliases like np.float_ would only surface when a control system is actually
    evaluated. skfuzzy is a proposal-named method, so a failure here is a real problem --
    see README "Known risks".
  * pandas-ta-openbb replaces upstream pandas-ta; we verify the indicator API we rely on.
  * confluent-kafka needs a bundled librdkafka; constructing a Producer proves the native
    extension loaded (it does not connect to a broker).
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

# Hard requirements: module name -> pip distribution it comes from.
HARD_REQUIREMENTS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "pydantic": "pydantic",
    "yaml": "PyYAML",
    "yfinance": "yfinance",
    "requests": "requests",
    "transformers": "transformers",
    "torch": "torch",
    "peft": "peft",
    "pymoo": "pymoo",
    "skfuzzy": "scikit-fuzzy",
    "deap": "deap",
    "mlflow": "mlflow",
    "fastapi": "fastapi",
    "confluent_kafka": "confluent-kafka",
    "openai": "openai",
}

# Optional: recorded, never fatal. RQ1 adapts to whatever is available.
OPTIONAL_FORECASTERS = {
    "timesfm": "timesfm[torch]==2.0.2",
    "chronos": "chronos-forecasting",
}


@pytest.mark.parametrize("module,dist", sorted(HARD_REQUIREMENTS.items()))
def test_hard_requirement_imports(module: str, dist: str) -> None:
    try:
        importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - failure path is the point
        pytest.fail(f"Hard requirement '{module}' (from {dist}) failed to import: {exc}")


def test_python_is_312() -> None:
    """3.12 is the only version satisfying pandas-ta (>=3.12), jax (>=3.12), and the
    numba win_amd64 wheel ceiling (cp313). See pyproject.toml."""
    assert sys.version_info[:2] == (3, 12), f"expected Python 3.12, got {sys.version.split()[0]}"


def test_numpy_is_2x() -> None:
    """Several dependencies need numpy >= 2.1.

    No upper cap: the <2.3 ceiling would come from UPSTREAM pandas-ta hard-pinning
    numba==0.61.2, but we use pandas-ta-openbb, which carries no numba pin. Verified working
    on numpy 2.5.2 -- see test_skfuzzy_control_system_evaluates_on_numpy2 and
    test_pandas_ta_computes_the_indicators_the_taf_names, which exercise the two libraries
    most likely to break on a numpy major bump.
    """
    import numpy as np

    major, minor = (int(p) for p in np.__version__.split(".")[:2])
    assert (major, minor) >= (2, 1), f"numpy {np.__version__} too old"


def test_skfuzzy_control_system_evaluates_on_numpy2() -> None:
    """The single highest-risk dependency: build and evaluate a real Mamdani system.

    This mirrors what optimization/fuzzy_withdrawal.py will do -- antecedent, consequent,
    a rule, and a defuzzified crisp output. An import-only check would not catch a removed
    numpy alias inside the inference path.
    """
    import numpy as np
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl

    urgency = ctrl.Antecedent(np.arange(0, 101, 1), "urgency")
    priority = ctrl.Consequent(np.arange(0, 101, 1), "sell_priority")

    urgency["low"] = fuzz.trimf(urgency.universe, [0, 0, 50])
    urgency["high"] = fuzz.trimf(urgency.universe, [50, 100, 100])
    priority["low"] = fuzz.trimf(priority.universe, [0, 0, 50])
    priority["high"] = fuzz.trimf(priority.universe, [50, 100, 100])

    system = ctrl.ControlSystem(
        [
            ctrl.Rule(urgency["low"], priority["low"]),
            ctrl.Rule(urgency["high"], priority["high"]),
        ]
    )
    sim = ctrl.ControlSystemSimulation(system)

    sim.input["urgency"] = 90.0
    sim.compute()
    high_out = sim.output["sell_priority"]

    sim.input["urgency"] = 10.0
    sim.compute()
    low_out = sim.output["sell_priority"]

    assert 0 <= low_out <= 100 and 0 <= high_out <= 100
    # The monotonicity property Phase 5b tests in full: more urgency => higher priority.
    assert high_out > low_out, f"fuzzy inference not monotonic: low={low_out} high={high_out}"


def test_pymoo_moead_is_importable_and_constructs() -> None:
    """Confirms the decomposition algorithm ships with pymoo -- we must not hand-roll it."""
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.util.ref_dirs import get_reference_directions

    ref_dirs = get_reference_directions("uniform", 3, n_partitions=6)
    algorithm = MOEAD(ref_dirs=ref_dirs, n_neighbors=5)
    assert algorithm is not None
    assert len(ref_dirs) > 0


def test_deap_toolbox_constructs() -> None:
    """Phase 5b's GA needs custom (permutation, fraction-vector) chromosomes."""
    from deap import base, creator, tools

    if not hasattr(creator, "_GateFitnessMin"):
        creator.create("_GateFitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "_GateIndividual"):
        creator.create("_GateIndividual", list, fitness=creator._GateFitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    ind = creator._GateIndividual([0, 1, 2, 3])
    ind.fitness.values = (1.0,)
    assert ind.fitness.valid


def test_pandas_ta_computes_the_indicators_the_taf_names() -> None:
    """MACD, RSI, MFI and ATR are named explicitly in the TAF task list -- all four must work."""
    import numpy as np
    import pandas as pd
    import pandas_ta as ta

    n = 250
    rng = np.random.default_rng(42)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    volume = pd.Series(rng.uniform(1e6, 5e6, n))

    rsi = ta.rsi(close, length=14)
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    atr = ta.atr(high=high, low=low, close=close, length=14)
    mfi = ta.mfi(high=high, low=low, close=close, volume=volume, length=14)

    for name, result in [("rsi", rsi), ("macd", macd), ("atr", atr), ("mfi", mfi)]:
        assert result is not None, f"pandas_ta.{name} returned None"
        assert len(result) == n, f"pandas_ta.{name} length mismatch"

    assert rsi.dropna().between(0, 100).all(), "RSI outside [0, 100]"


def test_confluent_kafka_native_extension_loads() -> None:
    """Constructing a Producer proves the bundled librdkafka loaded. No broker contacted."""
    from confluent_kafka import Producer

    producer = Producer({"bootstrap.servers": "localhost:9092", "socket.timeout.ms": 100})
    assert producer is not None
    assert producer.__class__.__name__ == "Producer"


def test_fastapi_testclient_available() -> None:
    """Phase 6 smoke tests depend on this (needs httpx installed)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_write_capability_report() -> None:
    """Record which optional forecasters are usable. Not an assertion about them -- the
    report is the deliverable. forecasting/base.py reads this to decide which adapters to
    register, and Phase 7 reads it to decide which RQ1 rows it can populate.
    """
    import numpy as np
    import pandas as pd
    import torch

    forecasters: dict[str, dict[str, object]] = {}
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
        forecasters[module] = entry

    report = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "optional_forecasters": forecasters,
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "env_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    available = [name for name, meta in forecasters.items() if meta.get("available")]
    print(f"\n[gate] foundation forecasters available: {available or 'NONE'}")
    if not available:
        print("[gate] WARNING: RQ1 will have no foundation-model rows until one installs.")
