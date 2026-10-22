"""Step 5: quality gate. Promote the candidate only if its accuracy meets the minimum."""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drift.common import APPROVED_METRICS, APPROVED_MODEL, DRIFT_DIR, MIN_ACCURACY  # noqa: E402


def approve(drift_dir=DRIFT_DIR, approved_model=APPROVED_MODEL, approved_metrics=APPROVED_METRICS,
            min_accuracy=MIN_ACCURACY):
    drift_dir = Path(drift_dir)
    with open(drift_dir / "retraining_metrics.json", encoding="utf-8") as f:
        m = json.load(f)
    accuracy = m["candidate_accuracy"]
    print("Candidate accuracy:", accuracy)
    print("Required accuracy :", min_accuracy)
    if accuracy < min_accuracy:
        print("QUALITY GATE: FAIL")
        print("Current approved model is unchanged.")
        return False

    approved_model = Path(approved_model)
    if approved_model.exists():                       # keep the previous model for traceability / rollback
        shutil.copy2(approved_model, drift_dir / "readmission_model_previous.pkl")
    shutil.copy2(drift_dir / "readmission_model_candidate.pkl", approved_model)
    with open(approved_metrics, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy, "recall": m["candidate_recall"], "f1_score": m["candidate_f1"],
            "training_records": m["training_rows"], "testing_records": m["test_rows"],
            "source": "drift-retraining",
        }, f, indent=4)
    print("QUALITY GATE: PASS")
    print("Candidate promoted to:", approved_model)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-accuracy", type=float, default=MIN_ACCURACY)
    args = parser.parse_args()
    sys.exit(0 if approve(min_accuracy=args.min_accuracy) else 1)
