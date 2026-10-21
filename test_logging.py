"""Practical 11: tests for logging and error tracking."""
import unittest
from unittest import mock

from app import app

HIGH_RISK = {"Age": 75, "Gender": "Female", "Condition": "Heart Attack", "Cost": 18000, "Length_of_Stay": 41}


class TestLoggingAndErrorTracking(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_request_id_header_present(self):
        response = self.client.get("/health")
        self.assertTrue(response.headers.get("X-Request-ID"))

    def test_request_id_is_echoed(self):
        response = self.client.get("/health", headers={"X-Request-ID": "abc-123"})
        self.assertEqual(response.headers["X-Request-ID"], "abc-123")

    def test_success_logged_as_info(self):
        with self.assertLogs("hospital-readmission-api", level="INFO") as cm:
            self.client.post("/predict", json=HIGH_RISK)
        text = "\n".join(cm.output)
        self.assertIn("event=prediction_request", text)
        self.assertIn("event=prediction_success", text)
        self.assertIn("status=200", text)

    def test_input_values_are_not_logged(self):
        with self.assertLogs("hospital-readmission-api", level="INFO") as cm:
            self.client.post("/predict", json=HIGH_RISK)
        self.assertNotIn("Heart Attack", "\n".join(cm.output))
        self.assertNotIn("18000", "\n".join(cm.output))

    def test_validation_error_logged_as_warning(self):
        with self.assertLogs("hospital-readmission-api", level="WARNING") as cm:
            response = self.client.post("/predict", json={"Age": 60})
        self.assertEqual(response.status_code, 400)
        text = "\n".join(cm.output)
        self.assertIn("WARNING", text)
        self.assertIn("event=validation_error", text)
        self.assertIn("status=400", text)

    def test_unknown_route_returns_404_and_warning(self):
        with self.assertLogs("hospital-readmission-api", level="WARNING") as cm:
            response = self.client.get("/not-a-real-route")
        self.assertEqual(response.status_code, 404)
        self.assertIn("status=404", "\n".join(cm.output))

    def test_prediction_failure_returns_500_and_error_log(self):
        with mock.patch("app.load_model", side_effect=RuntimeError("boom")):
            with self.assertLogs("hospital-readmission-api", level="ERROR") as cm:
                response = self.client.post("/predict", json=HIGH_RISK)
        self.assertEqual(response.status_code, 500)
        self.assertIn("request_id", response.get_json())
        self.assertIn("event=prediction_failed", "\n".join(cm.output))


if __name__ == "__main__":
    unittest.main()
