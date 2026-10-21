"""Every fixed choice of the experiment, in one place, decided before looking at test data."""

from __future__ import annotations

import os
from pathlib import Path

RESEARCH_DIR = Path(__file__).resolve().parent.parent
COMPONENT_DIR = RESEARCH_DIR.parent
REPO_DIR = COMPONENT_DIR.parent.parent

# Raw data stays where Component 2's downloader puts it (git-ignored); nothing is copied.
DEFAULT_DATA_DIR = REPO_DIR / "backend" / "Fraud-Detection" / "data" / "raw" / "ieee-cis"
DATA_DIR = Path(os.getenv("IEEE_CIS_DIR", str(DEFAULT_DATA_DIR)))

RESULTS_DIR = RESEARCH_DIR / "results" / "ieee-cis"
POLICY_DIR = COMPONENT_DIR / "src" / "policies"
POLICY_FILE = POLICY_DIR / "P002-assurance-v1.json"
GOLDEN_FILE = COMPONENT_DIR / "tests" / "fixtures" / "golden-vectors.json"

SEED = 20261021

# Time-ordered split on TransactionDT; shares of rows (boundaries moved so no timestamp is
# shared across two splits).
SPLITS = {"train": 0.50, "calibration": 0.15, "tuning": 0.15, "test": 0.20}

# Features fixed a priori, following Component 2's documented set (docs/DATASETS.md). No
# data-driven feature selection.
NUMERIC = (
    ["TransactionAmt", "dist1", "dist2"]
    + [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in range(1, 16)]
    + [f"V{i}" for i in range(1, 21)]
)
BINARY = [f"M{i}" for i in range(1, 10) if i != 4]          # M4 is categorical (M0/M1/M2)
CATEGORICAL = ["ProductCD", "card4", "card6", "M4", "P_emaildomain", "R_emaildomain", "card1"]
IDENTITY_NUMERIC = [f"id_{i:02d}" for i in range(1, 12)]
IDENTITY_CATEGORICAL = ["DeviceType", "DeviceInfo"]

# The AI decision: reject when the calibrated fraud probability reaches the Bayes threshold
# for these error costs (a missed fraud costs ten false alarms). Fixed, not tuned.
COST_FALSE_NEGATIVE = 10.0
COST_FALSE_POSITIVE = 1.0
# 1/11 rounded to the 4 decimals the exported policy carries, so the threshold evaluated here
# is exactly the one the service applies.
DECISION_THRESHOLD = round(COST_FALSE_POSITIVE / (COST_FALSE_POSITIVE + COST_FALSE_NEGATIVE), 4)

# Human-review budgets every method is compared at (share of all decisions).
BUDGETS = [0.05, 0.10, 0.15, 0.20]
# The budget the exported, enforced P002 is tuned for.
POLICY_BUDGET = 0.10
# Share of decisions the exported policy may send to VERIFY (customer step-up friction).
VERIFY_BUDGET = 0.10

# Assumptions (not measured) -- reported as sensitivity sweeps.
VERIFY_EFFECTIVENESS = [0.0, 0.25, 0.5, 0.75]
ANALYST_ACCURACY = [1.0, 0.95, 0.9]

BOOTSTRAP_RESAMPLES = 1000
