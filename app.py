"""Practicals 8-15: Flask API for the hospital readmission model.
P11 added logging and error tracking. P12 adds Prometheus metrics (/metrics)."""
import logging
import os
import sys
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, Response, g, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from werkzeug.exceptions import HTTPException

# ---- Practical 11: logging to stdout (visible locally, in docker logs, Render logs and kubectl logs) ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("hospital-readmission-api")

app = Flask(__name__)
APP_VERSION = "1.4.1-auto-deploy"

MODEL_PATH = Path("readmission_model.pkl")
FEATURES = ["Age", "Gender", "Condition", "Cost", "Length_of_Stay"]
NUMERIC_FEATURES = ["Age", "Cost", "Length_of_Stay"]

# Privacy: only feature NAMES are logged. Add non-sensitive feature names here to also log their values.
SAFE_INPUT_FIELDS = set()

# ---- Practical 12: Prometheus metrics (defined once at module level) ----
HTTP_REQUESTS = Counter(
    "hospital_api_http_requests_total",
    "Total HTTP requests received by the Hospital Readmission API",
    ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "hospital_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
PREDICTIONS = Counter(
    "hospital_api_predictions_total",
    "Total Hospital Readmission prediction attempts",
    ["result"],
)
MODEL_LOADED = Gauge(
    "hospital_api_model_loaded",
    "Whether the Hospital Readmission model is loaded (1 = loaded, 0 = not loaded)",
)
MODEL_LOADED.set(0)

_model = None


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            MODEL_LOADED.set(0)
            logger.error("event=model_load_failed model_file=%s reason=file_not_found", MODEL_PATH)
            raise FileNotFoundError(
                "readmission_model.pkl was not found. Run the training pipeline first."
            )
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception:
            MODEL_LOADED.set(0)
            logger.exception("event=model_load_failed model_file=%s", MODEL_PATH)
            raise
        MODEL_LOADED.set(1)
        logger.info("event=model_loaded model_file=%s", MODEL_PATH)
    return _model


# Load at start-up so the model_loaded event and gauge are set before the first request.
if MODEL_PATH.exists():
    load_model()
else:
    logger.warning("event=model_missing_at_startup model_file=%s", MODEL_PATH)


# ---- request / response logging + metrics with request id and duration ----
@app.before_request
def start_timer():
    g.start = time.perf_counter()
    g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))


@app.after_request
def log_request(response):
    duration = time.perf_counter() - g.start if "start" in g else 0.0
    request_id = g.get("request_id", "-")
    level = logging.WARNING if response.status_code >= 400 else logging.INFO
    logger.log(
        level,
        "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id, request.method, request.path, response.status_code, duration * 1000,
    )
    # Do not count Prometheus' own scrapes in the user-traffic metrics.
    if request.path != "/metrics":
        # Route template label keeps cardinality low: unknown URLs all become "unmatched".
        path = request.url_rule.rule if request.url_rule else "unmatched"
        HTTP_REQUESTS.labels(method=request.method, path=path, status=str(response.status_code)).inc()
        HTTP_LATENCY.labels(method=request.method, path=path).observe(duration)
    response.headers["X-Request-ID"] = request_id
    return response


@app.errorhandler(Exception)
def handle_exception(exc):
    # 404 / 405 etc. keep their normal response (and are logged as WARNING by after_request)
    if isinstance(exc, HTTPException):
        return jsonify({"error": exc.name, "status": exc.code}), exc.code
    logger.exception("event=unhandled_exception request_id=%s path=%s", g.get("request_id", "-"), request.path)
    return jsonify({"error": "Internal server error", "request_id": g.get("request_id", "-")}), 500


def health():
    return jsonify({"status": "ok", "service": "hospital-readmission-prediction", "version": APP_VERSION})


# Practical 5 checks "/", Practical 7 and later check "/health" - both are served.
app.add_url_rule("/", "health_root", health, methods=["GET"])
app.add_url_rule("/health", "health_check", health, methods=["GET"])


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), content_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not data:
        logger.warning("event=validation_error path=/predict reason=json_body_required")
        return jsonify({"error": "JSON request body is required"}), 400

    missing_fields = [f for f in FEATURES if f not in data]
    if missing_fields:
        logger.warning("event=validation_error path=/predict reason=missing_fields field_locations=%s error_count=%s",
                       missing_fields, len(missing_fields))
        return jsonify({"error": "Missing required fields", "missing_fields": missing_fields}), 400

    row = {f: data[f] for f in FEATURES}
    for field in NUMERIC_FEATURES:
        try:
            row[field] = float(row[field])
        except (TypeError, ValueError):
            logger.warning("event=validation_error path=/predict reason=not_numeric field_locations=%s error_count=1", [field])
            return jsonify({"error": f"{field} must be a number"}), 400

    safe_input = {k: v for k, v in row.items() if k in SAFE_INPUT_FIELDS}
    logger.info("event=prediction_request feature_names=%s safe_input=%s", sorted(row.keys()), safe_input)
    try:
        sample = pd.DataFrame([row])
        model = load_model()
        prediction_code = int(model.predict(sample)[0])
        probability = float(model.predict_proba(sample)[0][1])
    except Exception:
        PREDICTIONS.labels(result="error").inc()
        logger.exception("event=prediction_failed request_id=%s", g.get("request_id", "-"))
        return jsonify({"error": "Prediction failed", "request_id": g.get("request_id", "-")}), 500

    PREDICTIONS.labels(result="success").inc()
    prediction = "Yes" if prediction_code == 1 else "No"
    logger.info("event=prediction_success prediction=%s probability=%.4f", prediction, probability)
    return jsonify({
        "prediction": prediction,
        "prediction_code": prediction_code,
        "readmission_probability": round(probability, 4),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
