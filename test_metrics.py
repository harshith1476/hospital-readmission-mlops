"""Practical 12: tests for the Prometheus metrics."""
import unittest
from unittest import mock

from prometheus_client import REGISTRY

from app import app

HIGH_RISK = {"Age": 75, "Gender": "Female", "Condition": "Heart Attack", "Cost": 18000, "Length_of_Stay": 41}


def value(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_metrics_endpoint_is_prometheus_text(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/plain", response.content_type)
        body = response.get_data(as_text=True)
        for name in ("hospital_api_http_requests_total", "hospital_api_predictions_total",
                     "hospital_api_model_loaded", "hospital_api_http_request_duration_seconds"):
            self.assertIn(name, body)

    def test_model_loaded_gauge_is_one(self):
        self.assertEqual(value("hospital_api_model_loaded"), 1.0)

    def test_health_request_is_counted(self):
        before = value("hospital_api_http_requests_total", method="GET", path="/health", status="200")
        self.client.get("/health")
        after = value("hospital_api_http_requests_total", method="GET", path="/health", status="200")
        self.assertEqual(after, before + 1)

    def test_metrics_scrape_is_not_counted(self):
        self.client.get("/metrics")
        self.assertEqual(value("hospital_api_http_requests_total", method="GET", path="/metrics", status="200"), 0.0)

    def test_prediction_success_counter(self):
        before = value("hospital_api_predictions_total", result="success")
        self.client.post("/predict", json=HIGH_RISK)
        self.assertEqual(value("hospital_api_predictions_total", result="success"), before + 1)

    def test_validation_error_counts_as_4xx_not_prediction(self):
        before_400 = value("hospital_api_http_requests_total", method="POST", path="/predict", status="400")
        before_pred = value("hospital_api_predictions_total", result="success")
        self.client.post("/predict", json={"Age": "abc"})
        self.assertEqual(value("hospital_api_http_requests_total", method="POST", path="/predict", status="400"), before_400 + 1)
        self.assertEqual(value("hospital_api_predictions_total", result="success"), before_pred)

    def test_unknown_route_uses_unmatched_label(self):
        before = value("hospital_api_http_requests_total", method="GET", path="unmatched", status="404")
        self.client.get("/not-a-real-route")
        self.assertEqual(value("hospital_api_http_requests_total", method="GET", path="unmatched", status="404"), before + 1)

    def test_prediction_failure_counts_error_and_500(self):
        before = value("hospital_api_predictions_total", result="error")
        with mock.patch("app.load_model", side_effect=RuntimeError("boom")):
            response = self.client.post("/predict", json=HIGH_RISK)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(value("hospital_api_predictions_total", result="error"), before + 1)

    def test_latency_histogram_records_requests(self):
        self.client.get("/health")
        self.assertGreater(value("hospital_api_http_request_duration_seconds_count", method="GET", path="/health"), 0.0)


if __name__ == "__main__":
    unittest.main()
