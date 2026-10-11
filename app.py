"""Practical 5: Flask API exposing the trained readmission model."""
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path("readmission_model.pkl")
FEATURES = ["Age", "Gender", "Condition", "Cost", "Length_of_Stay"]
NUMERIC_FEATURES = ["Age", "Cost", "Length_of_Stay"]

_model = None


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "readmission_model.pkl was not found. Run the training pipeline first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def health():
    return jsonify({"status": "ok", "service": "hospital-readmission-prediction"})


# Practical 5 checks "/", Practical 7 checks "/health" - both are served.
app.add_url_rule("/", "health_root", health, methods=["GET"])
app.add_url_rule("/health", "health_check", health, methods=["GET"])


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON request body is required"}), 400

    missing_fields = [f for f in FEATURES if f not in data]
    if missing_fields:
        return jsonify({"error": "Missing required fields", "missing_fields": missing_fields}), 400

    row = {f: data[f] for f in FEATURES}
    for field in NUMERIC_FEATURES:
        try:
            row[field] = float(row[field])
        except (TypeError, ValueError):
            return jsonify({"error": f"{field} must be a number"}), 400

    sample = pd.DataFrame([row])
    model = load_model()
    prediction_code = int(model.predict(sample)[0])
    probability = float(model.predict_proba(sample)[0][1])
    return jsonify({
        "prediction": "Yes" if prediction_code == 1 else "No",
        "prediction_code": prediction_code,
        "readmission_probability": round(probability, 4),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
