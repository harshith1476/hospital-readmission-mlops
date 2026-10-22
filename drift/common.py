"""Shared paths and feature lists for the Practical 15 drift scripts."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIFT_DIR = PROJECT_ROOT / "drift"
HISTORICAL = PROJECT_ROOT / "data" / "hospital_data_analysis.csv"   # original labeled training data
APPROVED_MODEL = PROJECT_ROOT / "readmission_model.pkl"             # currently approved model
APPROVED_METRICS = PROJECT_ROOT / "metrics.json"                     # read by quality_gate.py

NUMERIC_FEATURES = ["Age", "Cost", "Length_of_Stay"]
CATEGORICAL_FEATURES = ["Gender", "Condition"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Readmission"            # Yes / No

P_VALUE_THRESHOLD = 0.05          # per-feature drift evidence
MIN_DRIFTED_FEATURES = 2          # classroom rule: overall drift when >= 2 features are flagged
MIN_ACCURACY = 0.90               # same quality gate as Practical 3
