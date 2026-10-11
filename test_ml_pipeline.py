import json
import os
import unittest

import joblib
import pandas as pd


def sample(**overrides):
    row = {"Age": 60, "Gender": "Male", "Condition": "Diabetes", "Cost": 2000, "Length_of_Stay": 30}
    row.update(overrides)
    return pd.DataFrame([row])


class TestMLPipeline(unittest.TestCase):
    def test_dataset_created(self):
        self.assertTrue(os.path.exists("hospital_clean.csv"))

    def test_model_created(self):
        self.assertTrue(os.path.exists("readmission_model.pkl"))

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_model_prediction(self):
        model = joblib.load("readmission_model.pkl")
        prediction = model.predict(sample())[0]
        self.assertIn(int(prediction), [0, 1])

    def test_high_risk_patient(self):
        # Heart Attack patients are readmitted in the dataset
        model = joblib.load("readmission_model.pkl")
        patient = sample(Age=75, Gender="Female", Condition="Heart Attack", Cost=18000, Length_of_Stay=41)
        self.assertEqual(int(model.predict(patient)[0]), 0)

    def test_low_risk_patient(self):
        # Hypertension patients are not readmitted in the dataset
        model = joblib.load("readmission_model.pkl")
        patient = sample(Age=70, Gender="Male", Condition="Hypertension", Cost=1000, Length_of_Stay=35)
        self.assertEqual(int(model.predict(patient)[0]), 0)


if __name__ == "__main__":
    unittest.main()
