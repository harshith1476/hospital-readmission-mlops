"""Step 1: create the fixed reference dataset (features only) tied to the approved model."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drift.common import DRIFT_DIR, FEATURES, HISTORICAL  # noqa: E402


def prepare(source=HISTORICAL, out_dir=DRIFT_DIR):
    df = pd.read_csv(source)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    reference = df[FEATURES].dropna().copy()
    out = Path(out_dir) / "reference_data.csv"
    reference.to_csv(out, index=False)
    return reference, out


if __name__ == "__main__":
    ref, path = prepare()
    print("Reference dataset created.")
    print("Rows:", len(ref))
    print("File:", path)
