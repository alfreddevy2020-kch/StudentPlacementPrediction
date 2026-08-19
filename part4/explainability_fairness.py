"""
Part 4 — Explainability & Bias/Fairness
========================================
Role 4 deliverables:
  1. SHAP values — global summary + local waterfall for individual students
  2. Probability calibration — Platt (sigmoid) and isotonic vs raw XGBoost scores
  3. Bias/fairness audit — group-wise metrics across placement training and
     extracurricular activity (this dataset has no demographic attributes)
  4. Mitigation recommendations in a text report

Run after:
    python download_dataset.py
    python preprocessing.py
    python part3/xgboost_model.py   (or ensure part3/models/xgboost_best.json exists)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")

# ============================================================
# 0. STYLE & PATHS
# ============================================================

PALETTE = {
    "primary": "#4F8EF7",
    "secondary": "#F7714F",
    "accent": "#50C878",
    "warn": "#FFD166",
    "dark": "#1A1A2E",
    "light": "#F5F5F5",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from feature_engineering import load_raw_dataset, TARGET_COLUMN, TARGET_MAP

MODELS_DIR = REPO_ROOT / "part4" / "models"
RESULTS_DIR = REPO_ROOT / "part4" / "explainability_results"

TRAIN_PATH = REPO_ROOT / "data" / "processed" / "train_processed.csv"
TEST_PATH = REPO_ROOT / "data" / "processed" / "test_processed.csv"
RAW_PATH = REPO_ROOT / "data" / "raw" / "student_placement.csv"

XGB_JOBLIB = REPO_ROOT / "part3" / "models" / "xgboost_best.joblib"
XGB_JSON = REPO_ROOT / "part3" / "models" / "xgboost_best.json"

# This dataset carries no protected demographic attributes (no gender,
# branch or college tier), so the audit runs on the equity axes it does
# have: whether the student received institutional placement training, and
# whether they participated in extracurriculars. Access to training is a
# legitimate fairness concern in its own right - a model that penalises
# students the institution never trained would entrench that gap.
SENSITIVE_COLUMNS = ["placement_training", "extracurricular_activities"]
THRESHOLD = 0.5


def load_xgboost_model():
    """Load the Part 3 XGBoost model from joblib or native JSON."""
    if XGB_JOBLIB.exists():
        print(f"  Loading model from {XGB_JOBLIB.name}")
        return joblib.load(XGB_JOBLIB)

    if XGB_JSON.exists():
        print(f"  Loading model from {XGB_JSON.name}")
        model = xgb.XGBClassifier()
        model.load_model(str(XGB_JSON))
        return model

    raise FileNotFoundError(
        "XGBoost model not found. Run part3/xgboost_model.py first "
        "or ensure part3/models/xgboost_best.json is present."
    )


def load_data():
    """Load processed train/test matrices and raw test rows for fairness groups."""
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Processed data not found. Run preprocessing.py first."
        )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    target = "placement_status"

    X_train = train_df.drop(columns=[target]).values
    y_train = train_df[target].values
    X_test = test_df.drop(columns=[target]).values
    y_test = test_df[target].values
    feature_names = list(train_df.drop(columns=[target]).columns)

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            "Raw dataset not found. Run download_dataset.py first."
        )

    # load_raw_dataset() applies the canonical snake_case renaming, matching
    # what preprocessing.py fit the model on.
    raw_df = load_raw_dataset(RAW_PATH).drop(columns=["student_id"])
    X_raw = raw_df.drop(columns=[TARGET_COLUMN])
    y_raw = raw_df[TARGET_COLUMN].map(TARGET_MAP).astype(int)

    _, X_test_raw, _, y_test_raw = train_test_split(
        X_raw,
        y_raw,
        test_size=0.20,
        random_state=42,
        stratify=y_raw,
    )
    X_test_raw = X_test_raw.reset_index(drop=True)
    y_test_raw = y_test_raw.reset_index(drop=True)

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        feature_names,
        X_test_raw,
        y_test_raw,
    )


def group_metrics(y_true, y_pred, y_proba, group_name: str) -> dict:
    """Compute classification metrics for one demographic group."""
    if len(y_true) == 0:
        return {"group": group_name, "n": 0}

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan

    return {
        "group": group_name,
        "n": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "false_negative_rate": round(float(fnr), 4),
        "false_positive_rate": round(float(fpr), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4)
        if len(np.unique(y_true)) > 1
        else np.nan,
        "placed_count": int((y_true == 1).sum()),
    }


def run_shap_analysis(model, X_test, y_test, feature_names):
    """Generate global and local SHAP explanations."""
    print("\n" + "-" * 50)
    print("[2] SHAP Analysis ...")
    print("-" * 50)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df["placement_status"] = y_test
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


def run_calibration(model, X_train, y_train, X_test, y_test):
    """Calibrate XGBoost probabilities and compare reliability."""
    print("\n" + "-" * 50)
    print("[3] Probability Calibration ...")
    print("-" * 50)

    y_proba_raw = model.predict_proba(X_test)[:, 1]
    brier_raw = brier_score_loss(y_test, y_proba_raw)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    calibrated_sigmoid = CalibratedClassifierCV(model, method="sigmoid", cv=cv)
    calibrated_sigmoid.fit(X_train, y_train)
    y_proba_sigmoid = calibrated_sigmoid.predict_proba(X_test)[:, 1]
    brier_sigmoid = brier_score_loss(y_test, y_proba_sigmoid)

    calibrated_isotonic = CalibratedClassifierCV(model, method="isotonic", cv=cv)
    calibrated_isotonic.fit(X_train, y_train)
    y_proba_isotonic = calibrated_isotonic.predict_proba(X_test)[:, 1]
    brier_isotonic = brier_score_loss(y_test, y_proba_isotonic)

    best_method = min(
        [("raw", brier_raw), ("sigmoid", brier_sigmoid), ("isotonic", brier_isotonic)],
        key=lambda x: x[1],
    )[0]

    if best_method == "sigmoid":
        best_calibrated = calibrated_sigmoid
    elif best_method == "isotonic":
        best_calibrated = calibrated_isotonic
    else:
        best_calibrated = model

    joblib.dump(best_calibrated, MODELS_DIR / "calibrated_xgboost.joblib")
    print(f"  Saved calibrated model ({best_method}): part4/models/calibrated_xgboost.joblib")

    cal_metrics = pd.DataFrame(
        [
            {"method": "raw_xgboost", "brier_score": round(brier_raw, 6)},
            {"method": "platt_sigmoid", "brier_score": round(brier_sigmoid, 6)},
            {"method": "isotonic", "brier_score": round(brier_isotonic, 6)},
        ]
    )
    cal_metrics["selected"] = cal_metrics["method"].map(
        {
            "raw_xgboost": best_method == "raw",
            "platt_sigmoid": best_method == "sigmoid",
            "isotonic": best_method == "isotonic",
        }
    )
    cal_metrics.to_csv(RESULTS_DIR / "calibration_metrics.csv", index=False)
    print("  Saved: explainability_results/calibration_metrics.csv")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")

    for label, proba, color in [
        ("Raw XGBoost", y_proba_raw, PALETTE["secondary"]),
        ("Platt (sigmoid)", y_proba_sigmoid, PALETTE["primary"]),
        ("Isotonic", y_proba_isotonic, PALETTE["accent"]),
    ]:
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="uniform")
        ax.plot(
            mean_pred,
            frac_pos,
            "o-",
            linewidth=2.2,
            markersize=6,
            label=label,
            color=color,
        )

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Actually Placed)")
    ax.set_title("Calibration Curves — Before vs After\n(Lower Brier score = more trustworthy probabilities)")
    ax.legend(loc="lower right")
    ax.set_facecolor(PALETTE["light"])
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "calibration_before_after.png")
    plt.close()
    print("  Saved: explainability_results/calibration_before_after.png")

    print(f"\n  Brier scores — Raw: {brier_raw:.6f} | Platt: {brier_sigmoid:.6f} | Isotonic: {brier_isotonic:.6f}")
    print(f"  Best method selected: {best_method}")
    return best_calibrated, cal_metrics


def run_fairness_audit(model, X_test, y_test, X_test_raw):
    """Audit model performance across sensitive demographic groups."""
    print("\n" + "-" * 50)
    print("[4] Bias / Fairness Audit ...")
    print("-" * 50)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)

    audit_frames = []

    for col in SENSITIVE_COLUMNS:
        print(f"\n  Auditing by: {col}")
        for group_value in sorted(X_test_raw[col].unique()):
            mask = X_test_raw[col] == group_value
            metrics = group_metrics(
                y_test[mask.values],
                y_pred[mask.values],
                y_proba[mask.values],
                f"{col}={group_value}",
            )
            metrics["sensitive_attribute"] = col
            audit_frames.append(metrics)
            if metrics["n"] > 0:
                print(
                    f"    {group_value:8s} n={metrics['n']:4d}  "
                    f"FNR={metrics['false_negative_rate']:.4f}  "
                    f"Recall={metrics['recall']:.4f}  "
                    f"F1={metrics['f1']:.4f}"
                )

    fairness_df = pd.DataFrame(audit_frames)
    fairness_df.to_csv(RESULTS_DIR / "fairness_group_metrics.csv", index=False)
    print("\n  Saved: explainability_results/fairness_group_metrics.csv")

    # Plot FNR by group
    plot_df = fairness_df[fairness_df["n"] > 0].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col in zip(axes, SENSITIVE_COLUMNS):
        subset = plot_df[plot_df["sensitive_attribute"] == col]
        bars = ax.bar(
            subset["group"].str.replace(f"{col}=", "", regex=False),
            subset["false_negative_rate"],
            color=[PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]][: len(subset)],
            alpha=0.85,
            edgecolor="white",
        )
        ax.set_title(f"False Negative Rate by {col.replace('_', ' ').title()}")
        ax.set_ylabel("FNR (missed placement-ready students)")
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylim(0, max(0.1, subset["false_negative_rate"].max() * 1.2))
        for bar, val in zip(bars, subset["false_negative_rate"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(
        "Fairness Audit — False Negative Rate by Group\n"
        "(Lower is better: fewer placement-ready students incorrectly flagged as not ready)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fairness_fnr_by_group.png")
    plt.close()
    print("  Saved: explainability_results/fairness_fnr_by_group.png")

    return fairness_df


def write_fairness_report(fairness_df, mean_abs_shap, cal_metrics):
    """Write a human-readable fairness + explainability report with mitigations."""
    def fnr_gap(attribute: str) -> float:
        """Max-minus-min false-negative-rate spread across one attribute's groups."""
        subset = fairness_df[fairness_df["sensitive_attribute"] == attribute]
        if len(subset) < 2:
            return 0.0
        return (
            subset["false_negative_rate"].max() - subset["false_negative_rate"].min()
        )

    training_fnr_gap = fnr_gap("placement_training")
    extra_fnr_gap = fnr_gap("extracurricular_activities")

    best_cal = cal_metrics.loc[cal_metrics["selected"], "method"].iloc[0]
    top_shap = mean_abs_shap.head(5)

    lines = [
        "=" * 60,
        "PART 4 — EXPLAINABILITY & FAIRNESS REPORT",
        "Role 4: Student Placement Prediction System",
        "=" * 60,
        "",
        "1. SHAP EXPLAINABILITY",
        "-" * 40,
        "SHAP (SHapley Additive exPlanations) assigns each feature a contribution",
        "to a specific prediction. Unlike global feature importance, SHAP explains",
        "WHY a particular student received their placement probability.",
        "",
        "Top 5 global SHAP drivers:",
    ]

    for _, row in top_shap.iterrows():
        lines.append(f"  • {row['feature']}: mean |SHAP| = {row['mean_abs_shap']:.4f}")

    lines.extend(
        [
            "",
            "Key talking point:",
            "  Random Forest 'importance' is global; SHAP tells a placement officer",
            "  exactly which factors pushed THIS student's score up or down.",
            "",
            "2. PROBABILITY CALIBRATION",
            "-" * 40,
        ]
    )

    for _, row in cal_metrics.iterrows():
        selected = " ← SELECTED" if row["selected"] else ""
        lines.append(f"  {row['method']:16s} Brier = {row['brier_score']:.6f}{selected}")

    lines.extend(
        [
            "",
            "Key talking point:",
            "  When the dashboard shows '72% placement likelihood', calibration ensures",
            "  that among similar students scored ~0.72, roughly 72% are actually placed.",
            "",
            "3. BIAS / FAIRNESS AUDIT",
            "-" * 40,
            "Sensitive attributes audited: placement_training, extracurricular_activities",
            "",
            "Note: this dataset contains no protected demographic attributes",
            "(gender, branch, college tier). The audit therefore measures",
            "equity of access to institutional support rather than demographic",
            "parity. A production deployment on real student records should",
            "re-run this audit against actual demographic fields.",
            "",
            "Group-wise false negative rates (FNR):",
        ]
    )

    for _, row in fairness_df.iterrows():
        if row["n"] > 0:
            lines.append(
                f"  {row['group']:35s} n={int(row['n']):4d}  FNR={row['false_negative_rate']:.4f}  "
                f"Recall={row['recall']:.4f}"
            )

    lines.extend(
        [
            "",
            f"  Placement-training FNR gap (max − min): {training_fnr_gap:.4f}",
            f"  Extracurricular FNR gap:                {extra_fnr_gap:.4f}",
            "",
            "4. PROPOSED MITIGATIONS",
            "-" * 40,
        ]
    )

    if training_fnr_gap > 0.05 or extra_fnr_gap > 0.05:
        lines.extend(
            [
                "  Detected meaningful FNR disparity across groups. Proposed mitigations:",
                "  (a) Group-specific decision thresholds — lower threshold for groups",
                "      with higher FNR so fewer placement-ready students are missed.",
                "  (b) Re-sample / re-weight training data per group.",
                "  (c) Monitor FNR by group in production (Role 7 drift logging).",
                "  (d) Treat a high FNR among untrained students as an access problem:",
                "      widen placement-training enrolment rather than only retuning",
                "      the model, since the disparity reflects who the institution",
                "      supported, not who is capable.",
            ]
        )
    else:
        lines.extend(
            [
                "  No large FNR disparity detected between audited groups on this test set.",
                "  Recommended production safeguards anyway:",
                "  (a) Log predictions with group metadata for ongoing monitoring.",
                "  (b) Set alert if any group's FNR exceeds overall FNR by > 5 pp.",
                "  (c) Re-audit after each model retrain (Role 7 trigger policy).",
            ]
        )

    lines.extend(
        [
            "",
            "5. INTEGRATION NOTES FOR TEAM",
            "-" * 40,
            "  • Role 3 (Dashboard): use shap_global_importance.csv + waterfall plot",
            "  • Role 6 (API): serve calibrated_xgboost.joblib for trustworthy probs",
            "  • Role 8 (Evaluation): use fairness_group_metrics.csv in final report",
            "",
            "=" * 60,
        ]
    )

    report_path = RESULTS_DIR / "fairness_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Saved: explainability_results/fairness_report.txt")


def main():
    print("\n" + "=" * 60)
    print("Part 4 — Explainability & Bias/Fairness Pipeline")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1] Loading model and data ...")
    model = load_xgboost_model()
    (
        X_train,
        y_train,
        X_test,
        y_test,
        feature_names,
        X_test_raw,
        y_test_raw,
    ) = load_data()

    assert len(y_test) == len(y_test_raw), "Test set size mismatch between processed and raw splits"

    print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"  Features: {len(feature_names)}")

    _, mean_abs_shap = run_shap_analysis(model, X_test, y_test, feature_names)
    _, cal_metrics = run_calibration(model, X_train, y_train, X_test, y_test)
    fairness_df = run_fairness_audit(model, X_test, y_test, X_test_raw)
    write_fairness_report(fairness_df, mean_abs_shap, cal_metrics)

    print("\n" + "=" * 60)
    print("PART 4 PIPELINE COMPLETE")
    print("=" * 60)
    print("\nGenerated artifacts:")
    for path in sorted(RESULTS_DIR.glob("*")):
        print(f"  explainability_results/{path.name}")
    print(f"  models/calibrated_xgboost.joblib")


if __name__ == "__main__":
    main()
