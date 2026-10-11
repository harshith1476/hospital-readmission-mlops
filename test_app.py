import unittest

from app import app

HIGH_RISK = {"Age": 75, "Gender": "Female", "Condition": "Heart Attack", "Cost": 18000, "Length_of_Stay": 41}
LOW_RISK = {"Age": 70, "Gender": "Male", "Condition": "Hypertension", "Cost": 1000, "Length_of_Stay": 35}


class TestPredictionApplication(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_health_alias(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_high_risk_prediction(self):
        response = self.client.post("/predict", json=HIGH_RISK)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], "No")

    def test_low_risk_prediction(self):
        response = self.client.post("/predict", json=LOW_RISK)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], "No")

    def test_missing_field_validation(self):
        response = self.client.post("/predict", json={"Age": 60, "Gender": "Male"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.get_json())

    def test_non_numeric_validation(self):
        bad = dict(HIGH_RISK, Age="abc")
        response = self.client.post("/predict", json=bad)
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
