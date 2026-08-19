"""
scripts/generate_baseline_metrics.py
------------------------------------
Regenerate artifacts/production/<model>/baseline_metrics.json from the
current held-out test split.

These baselines are the reference distribution the drift monitor
(api/drift.py) compares live predictions against, so they must be
regenerated whenever the models or the dataset change — stale baselines
make drift detection meaningless.

Usage:
    python scripts/generate_baseline_metrics.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "production"
TEST_PATH = REPO_ROOT / "data" / "processed" / "test_processed.csv"
TARGET = "placement_status"

MODEL_VERSIONS = {
    "logistic_regression": "2026.08.19-lr.2",
    "random_forest": "2026.08.19-rf.2",
    "xgboost": "2026.08.19-xgb.2",
}


def main() -> int:
    if not TEST_PATH.exists():
        print(f"ERROR: {TEST_PATH} not found. Run preprocessing.py first.")
        return 1

    test_df = pd.read_csv(TEST_PATH)
    X_test = test_df.drop(columns=[TARGET]).values
    y_test = test_df[TARGET].values

    print(f"Evaluating on {len(y_test):,} held-out rows\n")

    for model_key, version in MODEL_VERSIONS.items():
        model_path = ARTIFACT_ROOT / model_key / "model.joblib"
        if not model_path.exists():
            print(f"  [SKIP] {model_key}: {model_path} not found")
            continue

        model = joblib.load(model_path)
        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        metrics = {
            "model_name": model_key,
            "model_version": version,
            "evaluation_dataset": "test_processed_v2",
            "evaluation_sample_size": int(len(y_test)),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "f1_score": round(float(f1_score(y_test, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
            "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
            "mean_probability_placed": round(float(np.mean(proba)), 4),
            "probability_standard_deviation": round(float(np.std(proba)), 4),
        }

        out_path = ARTIFACT_ROOT / model_key / "baseline_metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
            f.write("\n")

        print(
            f"  [OK] {model_key:22s} ROC-AUC={metrics['roc_auc']:.4f}  "
            f"F1={metrics['f1_score']:.4f}  "
            f"mean_p={metrics['mean_probability_placed']:.4f}"
        )
        print(f"       -> {out_path.relative_to(REPO_ROOT)}")

    print("\nBaseline metrics regenerated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
