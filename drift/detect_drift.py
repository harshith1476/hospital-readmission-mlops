"""Step 3: compare recent data with the fixed reference.

Numeric features   -> two-sample Kolmogorov-Smirnov test
Categorical features -> chi-square test of the category counts
A feature is flagged when p < 0.05. Overall drift = at least 2 flagged features (classroom rule).
"""
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drift.common import (CATEGORICAL_FEATURES, DRIFT_DIR, MIN_DRIFTED_FEATURES,  # noqa: E402
                          NUMERIC_FEATURES, P_VALUE_THRESHOLD)


def detect(reference_csv, current_csv, report_path):
    reference = pd.read_csv(reference_csv)
    current = pd.read_csv(current_csv)
    results, drifted = {}, []

    for feature in NUMERIC_FEATURES:
        ref_v, cur_v = reference[feature].dropna(), current[feature].dropna()
        stat, p = ks_2samp(ref_v, cur_v)
        flag = bool(p < P_VALUE_THRESHOLD)
        results[feature] = {
            "test": "ks_2samp", "statistic": round(float(stat), 6), "p_value": round(float(p), 6),
            "drift_detected": flag,
            "reference_mean": round(float(ref_v.mean()), 4), "current_mean": round(float(cur_v.mean()), 4),
        }
        if flag:
            drifted.append(feature)

    for feature in CATEGORICAL_FEATURES:
        ref_c = reference[feature].dropna().value_counts()
        cur_c = current[feature].dropna().value_counts()
        cats = sorted(set(ref_c.index) | set(cur_c.index))
        table = [[int(ref_c.get(c, 0)) for c in cats], [int(cur_c.get(c, 0)) for c in cats]]
        stat, p, _, _ = chi2_contingency(table)
        flag = bool(p < P_VALUE_THRESHOLD)
        ref_share, cur_share = ref_c / ref_c.sum(), cur_c / cur_c.sum()
        results[feature] = {
            "test": "chi2_contingency", "statistic": round(float(stat), 6), "p_value": round(float(p), 6),
            "drift_detected": flag,
            "reference_top": f"{ref_share.idxmax()} ({ref_share.max():.1%})",
            "current_top": f"{cur_share.idxmax()} ({cur_share.max():.1%})",
        }
        if flag:
            drifted.append(feature)

    report = {
        "method": "KS test (numeric) + chi-square test (categorical)",
        "p_value_threshold": P_VALUE_THRESHOLD,
        "min_drifted_features": MIN_DRIFTED_FEATURES,
        "reference_rows": int(len(reference)), "current_rows": int(len(current)),
        "drifted_feature_count": len(drifted), "drifted_features": drifted,
        "overall_drift": len(drifted) >= MIN_DRIFTED_FEATURES,
        "features": results,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    rep = detect(DRIFT_DIR / "reference_data.csv", DRIFT_DIR / "current_data.csv", DRIFT_DIR / "drift_report.json")
    print(json.dumps(rep, indent=2))
    print("\nDrift report:", DRIFT_DIR / "drift_report.json")
    print("RESULT: DRIFT DETECTED - investigate / retrain" if rep["overall_drift"]
          else "RESULT: no overall drift - continue monitoring")
