"""Practical 15: tests for drift detection, conditional retraining and the quality gate.
Everything runs in a temporary folder, so the real model and metrics are never touched."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from drift.approve_candidate import approve
from drift.detect_drift import detect
from drift.prepare_reference import prepare
from drift.retrain_candidate import NoLabelsError, retrain
from drift.simulate_current_data import simulate

DATA = Path("data/hospital_data_analysis.csv")


def make_workspace(mode):
    tmp = Path(tempfile.mkdtemp())
    prepare(DATA, tmp)
    simulate(mode, 200, 42, DATA, tmp)
    report = detect(tmp / "reference_data.csv", tmp / "current_data.csv", tmp / "drift_report.json")
    return tmp, report


class TestDrift(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stable_dir, cls.stable = make_workspace("stable")
        cls.drift_dir, cls.drift = make_workspace("drift")
        cls.metrics = retrain(cls.drift_dir, DATA)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.stable_dir, ignore_errors=True)
        shutil.rmtree(cls.drift_dir, ignore_errors=True)

    def test_reference_has_features_only(self):
        header = (self.stable_dir / "reference_data.csv").read_text().splitlines()[0]
        self.assertEqual(header, "Age,Cost,Length_of_Stay,Gender,Condition")

    def test_stable_batch_is_not_overall_drift(self):
        self.assertFalse(self.stable["overall_drift"])

    def test_drifted_batch_is_detected(self):
        self.assertTrue(self.drift["overall_drift"])
        self.assertGreaterEqual(self.drift["drifted_feature_count"], 2)

    def test_report_json_written(self):
        saved = json.loads((self.drift_dir / "drift_report.json").read_text())
        self.assertIn("features", saved)
        self.assertIn("Age", saved["features"])

    def test_no_retraining_without_drift(self):
        self.assertIsNone(retrain(self.stable_dir, DATA))
        self.assertFalse((self.stable_dir / "readmission_model_candidate.pkl").exists())

    def test_drift_without_labels_blocks_retraining(self):
        tmp, report = make_workspace("drift")
        try:
            (tmp / "current_labeled_data.csv").unlink()
            with self.assertRaises(NoLabelsError):
                retrain(tmp, DATA)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_candidate_is_trained_and_scored(self):
        self.assertTrue((self.drift_dir / "readmission_model_candidate.pkl").exists())
        self.assertTrue(0.0 <= self.metrics["candidate_accuracy"] <= 1.0)
        self.assertEqual(self.metrics["recent_labeled_rows"], 200)

    def test_failed_quality_gate_keeps_current_model(self):
        approved = self.drift_dir / "approved.pkl"
        approved.write_bytes(b"current-model")
        ok = approve(self.drift_dir, approved, self.drift_dir / "approved_metrics.json", min_accuracy=1.01)
        self.assertFalse(ok)
        self.assertEqual(approved.read_bytes(), b"current-model")

    def test_passing_candidate_is_promoted(self):
        approved = self.drift_dir / "approved2.pkl"
        approved.write_bytes(b"old-model")
        metrics_out = self.drift_dir / "approved_metrics2.json"
        ok = approve(self.drift_dir, approved, metrics_out, min_accuracy=0.5)
        self.assertTrue(ok)
        self.assertNotEqual(approved.read_bytes(), b"old-model")
        self.assertTrue((self.drift_dir / "readmission_model_previous.pkl").exists())
        self.assertEqual(json.loads(metrics_out.read_text())["source"], "drift-retraining")


if __name__ == "__main__":
    unittest.main()
