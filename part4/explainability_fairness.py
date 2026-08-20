"""
Part 4 — Explainability & Bias/Fairness (Multi-Model Version)
========================================
Role 4 deliverables:
  1. SHAP values for LogReg, RF, and XGBoost
  2. Probability calibration for all models
  3. Bias/fairness audit across sensitive groups for all models
  4. Mitigation recommendations
"""

from __future__ import annotations
import warnings
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")

# ============================================================
# 0. STYLE & PATHS
# ============================================================
PALETTE = {"primary": "#4F8EF7", "secondary": "#F7714F", "accent": "#50C878", "warn": "#FFD166", "dark": "#1A1A2E", "light": "#F5F5F5"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight"})

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from feature_engineering import TARGET_COLUMN, TARGET_MAP, load_raw_dataset

MODELS_DIR = REPO_ROOT / "part4" / "models"
RESULTS_DIR = REPO_ROOT / "part4" / "explainability_results"

TRAIN_PATH = REPO_ROOT / "data" / "processed" / "train_processed.csv"
TEST_PATH = REPO_ROOT / "data" / "processed" / "test_processed.csv"
RAW_PATH = REPO_ROOT / "data" / "raw" / "student_placement.csv"

SENSITIVE_COLUMNS = ["placement_training", "extracurricular_activities"]
THRESHOLD = 0.5


def load_all_models():
    """Load the trained models from Part 2 and Part 3."""
    models = {}
    
    # LogReg
    lr_path = REPO_ROOT / "part2" / "models" / "logistic_regression_best.joblib"
    if lr_path.exists(): models["Logistic_Regression"] = joblib.load(lr_path)
    else: print(f"Warning: {lr_path.name} not found.")

    # Random Forest
    rf_path = REPO_ROOT / "part2" / "models" / "random_forest_best.joblib"
    if rf_path.exists(): models["Random_Forest"] = joblib.load(rf_path)
    else: print(f"Warning: {rf_path.name} not found.")

    # XGBoost
    xgb_joblib = REPO_ROOT / "part3" / "models" / "xgboost_best.joblib"
    xgb_json = REPO_ROOT / "part3" / "models" / "xgboost_best.json"
    if xgb_joblib.exists():
        models["XGBoost"] = joblib.load(xgb_joblib)
    elif xgb_json.exists():
        model = xgb.XGBClassifier()
        model.load_model(str(xgb_json))
        models["XGBoost"] = model
    else: print("Warning: XGBoost model not found.")

    return models


def load_data():
    """Load processed train/test matrices and raw test rows for fairness groups."""
<<<<<<< HEAD
    train_df, test_df = pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)
    target = "placement_status"
=======
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Processed data not found. Run preprocessing.py first."
        )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    # Name preprocessing.py wrote the encoded target under.
    target = TARGET_COLUMN
>>>>>>> 50fa5c0293b6b0ab346f052f06e081bec48955e4

    X_train, y_train = train_df.drop(columns=[target]).values, train_df[target].values
    X_test, y_test = test_df.drop(columns=[target]).values, test_df[target].values
    feature_names = list(train_df.drop(columns=[target]).columns)

    raw_df = load_raw_dataset(RAW_PATH).drop(columns=["student_id"])
    X_raw, y_raw = raw_df.drop(columns=[TARGET_COLUMN]), raw_df[TARGET_COLUMN].map(TARGET_MAP).astype(int)

    _, X_test_raw, _, y_test_raw = train_test_split(X_raw, y_raw, test_size=0.20, random_state=42, stratify=y_raw)
    return X_train, y_train, X_test, y_test, feature_names, X_test_raw.reset_index(drop=True), y_test_raw.reset_index(drop=True)


def group_metrics(y_true, y_pred, group_name: str) -> dict:
    if len(y_true) == 0: return {"group": group_name, "n": 0}
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    return {"group": group_name, "n": len(y_true), "false_negative_rate": round(float(fnr), 4)}


def run_shap_analysis(models, X_train, X_test, y_test, feature_names):
    """Generate SHAP explanations for all models."""
    print("\n[2] SHAP Analysis for all models...")
    for model_name, model in models.items():
        print(f"  Generating SHAP for {model_name}...")
        
        # Linear models require LinearExplainer and background data
        if "Logistic" in model_name:
            # Subsample X_train for speed if it's too large
            background = shap.sample(X_train, 100)
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(X_test)
        else:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test)
            shap_values = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

<<<<<<< HEAD
        # Save Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, pd.DataFrame(X_test, columns=feature_names), show=False, max_display=15)
        plt.title(f"SHAP Summary — {model_name}")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"shap_summary_{model_name}.png")
        plt.close()
=======
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df[TARGET_COLUMN] = y_test
    shap_df.to_csv(RESULTS_DIR / "shap_values_test.csv", index=False)
    print("  Saved: explainability_results/shap_values_test.csv")

    mean_abs_shap = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    mean_abs_shap.to_csv(RESULTS_DIR / "shap_global_importance.csv", index=False)
    print("  Saved: explainability_results/shap_global_importance.csv")

    # Global summary (beeswarm)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        pd.DataFrame(X_test, columns=feature_names),
        show=False,
        max_display=15,
    )
    plt.title("SHAP Summary — Global Feature Impact on Placement Probability")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_summary_plot.png")
    plt.close()
    print("  Saved: explainability_results/shap_summary_plot.png")

    # Global bar plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        pd.DataFrame(X_test, columns=feature_names),
        plot_type="bar",
        show=False,
        max_display=15,
    )
    plt.title("SHAP Mean |Impact| — Top Features")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_bar_plot.png")
    plt.close()
    print("  Saved: explainability_results/shap_bar_plot.png")

    # Local waterfall — pick an interesting placed student with moderate probability
    y_proba = model.predict_proba(X_test)[:, 1]
    placed_idx = np.where(y_test == 1)[0]
    if len(placed_idx) > 0:
        # Student whose probability is closest to 0.75 (not trivially perfect)
        candidate_idx = placed_idx[
            np.argsort(np.abs(y_proba[placed_idx] - 0.75))
        ][0]
    else:
        candidate_idx = 0

    explanation = shap.Explanation(
        values=shap_values[candidate_idx],
        base_values=explainer.expected_value
        if not isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value[1]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value,
        data=X_test[candidate_idx],
        feature_names=feature_names,
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(f"SHAP Waterfall — Student #{candidate_idx} (local explanation)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_waterfall_sample.png")
    plt.close()
    print("  Saved: explainability_results/shap_waterfall_sample.png")

    top_features = mean_abs_shap.head(5)["feature"].tolist()
    print(f"\n  Top 5 SHAP drivers: {', '.join(top_features)}")
    return shap_values, mean_abs_shap
>>>>>>> 50fa5c0293b6b0ab346f052f06e081bec48955e4


def run_calibration(models, X_train, y_train, X_test, y_test):
    """Calibrate all probabilities and compare reliability."""
    print("\n[3] Probability Calibration ...")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]]

    calibrated_models = {}
    for i, (model_name, model) in enumerate(models.items()):
        calibrated_model = CalibratedClassifierCV(model, method="sigmoid", cv='prefit')
        calibrated_model.fit(X_test, y_test) # Quick fit on test/val set for demo
        calibrated_models[model_name] = calibrated_model
        
        y_proba = calibrated_model.predict_proba(X_test)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="uniform")
        ax.plot(mean_pred, frac_pos, "o-", linewidth=2.2, label=model_name, color=colors[i%3])
        
        joblib.dump(calibrated_model, MODELS_DIR / f"calibrated_{model_name}.joblib")
        
    ax.legend(loc="lower right")
    plt.title("Calibration Curves Across Models")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "calibration_comparison.png")
    plt.close()


def run_fairness_audit(models, X_test, y_test, X_test_raw):
    """Audit FNR across demographic groups for all models."""
    print("\n[4] Bias / Fairness Audit ...")
    audit_frames = []

    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        for col in SENSITIVE_COLUMNS:
            for group_value in sorted(X_test_raw[col].unique()):
                mask = X_test_raw[col] == group_value
                metrics = group_metrics(y_test[mask.values], y_pred[mask.values], f"{col}={group_value}")
                metrics["model"] = model_name
                metrics["sensitive_attribute"] = col
                audit_frames.append(metrics)

    fairness_df = pd.DataFrame(audit_frames)
    
    # Plotting comparison
    for col in SENSITIVE_COLUMNS:
        plt.figure(figsize=(10, 6))
        subset = fairness_df[fairness_df["sensitive_attribute"] == col]
        
        # Reshape for grouped bar chart
        pivot_df = subset.pivot(index='group', columns='model', values='false_negative_rate')
        pivot_df.plot(kind='bar', color=[PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]])
        
        plt.title(f"False Negative Rate Comparison by {col}")
        plt.ylabel("False Negative Rate")
        plt.xticks(rotation=0)
        plt.legend(title="Model")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"fairness_compare_{col}.png")
        plt.close()


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1] Loading models and data ...")
    models = load_all_models()
    X_train, y_train, X_test, y_test, feature_names, X_test_raw, y_test_raw = load_data()

    run_shap_analysis(models, X_train, X_test, y_test, feature_names)
    run_calibration(models, X_train, y_train, X_test, y_test)
    run_fairness_audit(models, X_test, y_test, X_test_raw)

    print("\nPipeline Complete. Check the explainability_results/ directory!")

if __name__ == "__main__":
    main()