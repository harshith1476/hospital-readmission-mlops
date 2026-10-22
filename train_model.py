"""Practical 2: train a Hospital Readmission model from data/hospital_data_analysis.csv."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path("data/hospital_data_analysis.csv")
MODEL_PATH = "readmission_model.pkl"
METRICS_PATH = "metrics.json"
CLEAN_DATA_PATH = "hospital_clean.csv"

# Features known at discharge. Excluded on purpose:
#   Patient_ID   -> identifier, no predictive meaning
#   Procedure    -> 1:1 duplicate of Condition in this dataset
#   Outcome, Satisfaction -> recorded after treatment, would leak the future
NUMERIC_FEATURES = ["Age", "Cost", "Length_of_Stay"]
CATEGORICAL_FEATURES = ["Gender", "Condition"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Readmission"  # Yes / No  -> 1 / 0


RECENT_BATCH_PATH = Path("data/recent_labeled_batch.csv")  # Practical 15: optional approved recent labels


def load_dataset():
    data = pd.read_csv(DATA_PATH)
    if RECENT_BATCH_PATH.exists():
        recent = pd.read_csv(RECENT_BATCH_PATH)
        print("Including recent labeled batch:", len(recent), "rows")
        data = pd.concat([data, recent[FEATURES + [TARGET]]], ignore_index=True)
    data = data.drop_duplicates().dropna(subset=FEATURES + [TARGET])
    data["readmitted"] = (data[TARGET] == "Yes").astype(int)
    return data


def build_model():
    preprocess = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("numeric", "passthrough", NUMERIC_FEATURES),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])


def train_model():
    print("Loading dataset...")
    data = load_dataset()
    data.to_csv(CLEAN_DATA_PATH, index=False)
    print("Number of records:", len(data))
    print("Readmission rate :", round(data["readmitted"].mean(), 4))

    X = data[FEATURES]
    y = data["readmitted"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print("Training records:", len(X_train))
    print("Testing records :", len(X_test))

    print("Training model...")
    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))
    print("Recall  :", round(recall, 4), "(readmitted patients correctly found)")
    print("F1 score:", round(f1, 4))
    print("\nConfusion Matrix:")
    print(matrix)

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved as {MODEL_PATH}")

    metrics = {
        "accuracy": float(accuracy),
        "recall": float(recall),
        "f1_score": float(f1),
        "training_records": len(X_train),
        "testing_records": len(X_test),
    }
    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)
    print(f"Metrics saved as {METRICS_PATH}")
    return accuracy


if __name__ == "__main__":
    train_model()
