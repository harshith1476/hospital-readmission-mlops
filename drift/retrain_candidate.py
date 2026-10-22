"""Step 4: retrain a CANDIDATE model only when drift exists and recent labels are available.
The approved model (readmission_model.pkl) is never touched here."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drift.common import DRIFT_DIR, FEATURES, HISTORICAL, TARGET  # noqa: E402
from train_model import build_model  # noqa: E402


class NoLabelsError(Exception):
    pass


def retrain(drift_dir=DRIFT_DIR, historical=HISTORICAL):
    drift_dir = Path(drift_dir)
    with open(drift_dir / "drift_report.json", encoding="utf-8") as f:
        report = json.load(f)
    if not report["overall_drift"]:
        return None                                   # no drift -> no retraining

    recent_path = drift_dir / "current_labeled_data.csv"
    if not recent_path.exists():
        raise NoLabelsError("Drift was detected, but current_labeled_data.csv is not available. Wait for ground truth.")

    hist = pd.read_csv(historical)
    recent = pd.read_csv(recent_path)
    required = FEATURES + [TARGET]
    for name, frame in (("historical", hist), ("recent", recent)):
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} data missing columns: {missing}")

    combined = pd.concat([hist[required], recent[required]], ignore_index=True).dropna(subset=[TARGET])
    y = (combined[TARGET] == "Yes").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        combined[FEATURES], y, test_size=0.20, random_state=42, stratify=y)

    candidate = build_model()
    candidate.fit(X_train, y_train)
    pred = candidate.predict(X_test)
    metrics = {
        "candidate_accuracy": round(float(accuracy_score(y_test, pred)), 6),
        "candidate_recall": round(float(recall_score(y_test, pred)), 6),
        "candidate_f1": round(float(f1_score(y_test, pred)), 6),
        "training_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "historical_rows": int(len(hist)), "recent_labeled_rows": int(len(recent)),
        "drifted_features": report["drifted_features"],
    }
    joblib.dump(candidate, drift_dir / "readmission_model_candidate.pkl")
    with open(drift_dir / "retraining_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    try:
        m = retrain()
    except NoLabelsError as e:
        print(e)
        sys.exit(2)
    if m is None:
        print("No overall drift detected.")
        print("Retraining skipped.")
        sys.exit(0)
    print(json.dumps(m, indent=2))
    print("Candidate model:", DRIFT_DIR / "readmission_model_candidate.pkl")
