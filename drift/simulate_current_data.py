"""Step 2: simulate a recent batch of patients (classroom only).

Writes two files from the SAME batch:
  current_data.csv          features only      -> drift detection
  current_labeled_data.csv  features + label   -> retraining (labels carried from the sampled historical rows)

stable: uniform resample of the historical patients with tiny noise.
drift : more cardiac / cancer / stroke patients, older patients, longer stays (synthetic shift, not real behaviour).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drift.common import DRIFT_DIR, FEATURES, HISTORICAL, TARGET  # noqa: E402

HEAVY_CONDITIONS = {"Heart Attack", "Heart Disease", "Cancer", "Stroke"}


def simulate(mode="stable", rows=200, seed=42, historical=HISTORICAL, out_dir=DRIFT_DIR):
    rng = np.random.default_rng(seed)
    hist = pd.read_csv(historical).dropna(subset=FEATURES + [TARGET])
    weights = np.where(hist["Condition"].isin(HEAVY_CONDITIONS), 8.0 if mode == "drift" else 1.0, 1.0)
    weights = weights / weights.sum()
    idx = rng.choice(len(hist), size=rows, replace=True, p=weights)
    batch = hist.iloc[idx][FEATURES + [TARGET]].reset_index(drop=True)

    batch["Age"] = batch["Age"] + np.rint(rng.normal(0, 1.0, size=rows))
    batch["Length_of_Stay"] = batch["Length_of_Stay"] + np.rint(rng.normal(0, 1.0, size=rows))
    if mode == "drift":
        batch["Age"] += 10
        batch["Length_of_Stay"] += 12
    batch["Age"] = batch["Age"].clip(18, 100).astype(int)
    batch["Length_of_Stay"] = batch["Length_of_Stay"].clip(1, 120).astype(int)

    out_dir = Path(out_dir)
    batch[FEATURES].to_csv(out_dir / "current_data.csv", index=False)
    batch.to_csv(out_dir / "current_labeled_data.csv", index=False)
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["stable", "drift"], default="stable")
    parser.add_argument("--rows", type=int, default=200)
    args = parser.parse_args()
    b = simulate(args.mode, args.rows)
    print("Mode:", args.mode)
    print("Rows:", len(b))
    print("Files:", DRIFT_DIR / "current_data.csv", "and", DRIFT_DIR / "current_labeled_data.csv")
