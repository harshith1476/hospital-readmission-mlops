# Hospital Readmission MLOps (CI/CD with GitHub Actions + Docker)

Predicts whether a patient will be readmitted, trained on `data/hospital_data_analysis.csv`.
Pipeline: train -> quality gate (accuracy >= 0.90) -> tests -> artifacts -> Flask API -> Docker build and container test.

See `RUN_GUIDE.md` for the step-by-step commands (Practicals 1-7). Per-practical workflows are in `stages/`.
