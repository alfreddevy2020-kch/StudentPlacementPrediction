"""
Role 2 — Model Lead 1: Model Comparison
=========================================
Head-to-head comparison of Logistic Regression vs Random Forest with:
  - Overlaid ROC + Precision-Recall curves (PR is more informative for imbalanced data)
  - Calibration plots (reliability diagrams — are probabilities trustworthy?)
  - Threshold-sweep analysis (precision, recall, F1 vs threshold)
  - Full metrics table
  - McNemar's statistical test (proves difference is significant, not noise)
  - Prediction disagreement analysis (where do models diverge?)

Run after:
    python logistic_regression_model.py
    python random_forest_model.py
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba

from scipy.stats import chi2

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

# ============================================================
# 0. STYLE
# ============================================================

PALETTE = {
    "logreg":  "#4F8EF7",
    "rf":      "#F7714F",
    "neutral": "#50C878",
    "dark":    "#1A1A2E",
    "light":   "#F5F5F5",
    "grid":    "#E0E0E0",
}

MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "rf":     "Random Forest",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
})

# ============================================================
# 1. LOAD MODELS AND DATA
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON — Logistic Regression vs Random Forest")
print("=" * 60)

os.makedirs("model_results", exist_ok=True)

print("\n[1] Loading models and test data ...")

logreg = joblib.load("part2/models/logistic_regression_best.joblib")
rf     = joblib.load("part2/models/random_forest_best.joblib")

test_df = pd.read_csv("data/processed/test_processed.csv")
TARGET  = "placement_status"

X_test = test_df.drop(columns=[TARGET]).values
y_test = test_df[TARGET].values

# Load optimal thresholds from metadata
logreg_meta = pd.read_csv("part2/model_results/logreg_metadata.csv").iloc[0]
rf_meta     = pd.read_csv("part2/model_results/rf_metadata.csv").iloc[0]

thresh_logreg = float(logreg_meta["optimal_threshold"])
thresh_rf     = float(rf_meta["optimal_threshold"])

# Probabilities
y_proba_logreg = logreg.predict_proba(X_test)[:, 1]
y_proba_rf     = rf.predict_proba(X_test)[:, 1]

# Predictions at optimal thresholds
y_pred_logreg = (y_proba_logreg >= thresh_logreg).astype(int)
y_pred_rf     = (y_proba_rf     >= thresh_rf    ).astype(int)

print(f"  Logistic Regression optimal threshold : {thresh_logreg:.3f}")
print(f"  Random Forest optimal threshold       : {thresh_rf:.3f}")

# ============================================================
# 2. FULL METRICS TABLE
# ============================================================

print("\n" + "-" * 50)
print("[2] Full Metrics Comparison")
print("-" * 50)

def compute_metrics(y_true, y_pred, y_proba, name):
    return {
        "Model":              name,
        "Accuracy":           round(accuracy_score(y_true, y_pred), 4),
        "Precision":          round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":             round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1 Score":           round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC":            round(roc_auc_score(y_true, y_proba), 4),
        "Avg Precision (PR)": round(average_precision_score(y_true, y_proba), 4),
        "Brier Score":        round(brier_score_loss(y_true, y_proba), 4),
    }

metrics_logreg = compute_metrics(y_test, y_pred_logreg, y_proba_logreg, "Logistic Regression")
metrics_rf     = compute_metrics(y_test, y_pred_rf,     y_proba_rf,     "Random Forest")

metrics_df = pd.DataFrame([metrics_logreg, metrics_rf]).set_index("Model")
print("\n" + metrics_df.to_string())
metrics_df.to_csv("part2/model_results/comparison_metrics.csv")

# ============================================================
# 3. McNEMAR'S TEST  (statistical significance of difference)
# ============================================================

print("\n" + "-" * 50)
print("[3] McNemar's Test — Statistical Significance")
print("-" * 50)

# Contingency table:
#   b = LogReg correct, RF wrong
#   c = LogReg wrong,   RF correct
correct_logreg = (y_pred_logreg == y_test)
correct_rf     = (y_pred_rf     == y_test)

b = np.sum( correct_logreg & ~correct_rf)   # LogReg right, RF wrong
c = np.sum(~correct_logreg &  correct_rf)   # LogReg wrong, RF right
n = b + c                                    # disagreements only

print(f"\n  Disagreement cells: b={b} (LogReg=right RF=wrong), c={c} (LogReg=wrong RF=right)")
print(f"  Total test samples: {len(y_test)}")
print(f"  Samples where models disagree: {n} ({100*n/len(y_test):.1f}%)")

if n == 0:
    print("  Models agree on all samples — McNemar's test not applicable.")
    mcnemar_stat = 0.0
    mcnemar_p    = 1.0
else:
    # Use continuity correction (recommended for small samples)
    mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c)
    mcnemar_p    = 1 - chi2.cdf(mcnemar_stat, df=1)

print(f"\n  McNemar's chi2 statistic : {mcnemar_stat:.4f}")
print(f"  p-value                  : {mcnemar_p:.4f}")

if mcnemar_p < 0.05:
    print("  -> Difference is STATISTICALLY SIGNIFICANT (p < 0.05)")
    print("     The models make meaningfully different errors.")
else:
    print("  -> Difference is NOT statistically significant (p >= 0.05)")
    print("     The models make similar errors; choose based on interpretability.")

# ============================================================
# 4. PREDICTION DISAGREEMENT ANALYSIS
# ============================================================

print("\n" + "-" * 50)
print("[4] Prediction Disagreement Analysis")
print("-" * 50)

agree_mask      = (y_pred_logreg == y_pred_rf)
disagree_mask   = ~agree_mask
n_agree         = agree_mask.sum()
n_disagree      = disagree_mask.sum()

print(f"\n  Agree    : {n_agree}  ({100*n_agree/len(y_test):.1f}%)")
print(f"  Disagree : {n_disagree} ({100*n_disagree/len(y_test):.1f}%)")

if n_disagree > 0:
    # Among disagreements, who is more accurate?
    logreg_wins = np.sum((y_pred_logreg[disagree_mask] == y_test[disagree_mask]))
    rf_wins     = np.sum((y_pred_rf[disagree_mask]     == y_test[disagree_mask]))
    print(f"\n  On disagreements:")
    print(f"    LogReg correct : {logreg_wins}/{n_disagree}")
    print(f"    RF correct     : {rf_wins}/{n_disagree}")

# ============================================================
# 5. PLOT 1 — ROC Curves (overlaid)
# ============================================================

print("\n[5] Generating comparison plots ...")

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_logreg)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
auc_lr = roc_auc_score(y_test, y_proba_logreg)
auc_rf = roc_auc_score(y_test, y_proba_rf)

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(fpr_lr, tpr_lr, color=PALETTE["logreg"], linewidth=2.5,
        label=f"Logistic Regression  (AUC = {auc_lr:.4f})")
ax.plot(fpr_rf, tpr_rf, color=PALETTE["rf"],     linewidth=2.5,
        label=f"Random Forest        (AUC = {auc_rf:.4f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Random baseline (AUC = 0.5)")
ax.fill_between(fpr_lr, tpr_lr, alpha=0.06, color=PALETTE["logreg"])
ax.fill_between(fpr_rf, tpr_rf, alpha=0.06, color=PALETTE["rf"])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Logistic Regression vs Random Forest")
ax.legend(loc="lower right")
ax.set_facecolor(PALETTE["light"])
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("part2/model_results/comparison_roc_curves.png")
plt.close(fig)
print("  Saved: model_results/comparison_roc_curves.png")

# ============================================================
# 6. PLOT 2 — Precision-Recall Curves
# ============================================================

prec_lr, rec_lr, _ = precision_recall_curve(y_test, y_proba_logreg)
prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_proba_rf)
ap_lr = average_precision_score(y_test, y_proba_logreg)
ap_rf = average_precision_score(y_test, y_proba_rf)

baseline_pr = y_test.mean()   # minority class fraction

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(rec_lr, prec_lr, color=PALETTE["logreg"], linewidth=2.5,
        label=f"Logistic Regression  (AP = {ap_lr:.4f})")
ax.plot(rec_rf, prec_rf, color=PALETTE["rf"],     linewidth=2.5,
        label=f"Random Forest        (AP = {ap_rf:.4f})")
ax.axhline(baseline_pr, color="#AAAAAA", linestyle="--", linewidth=1.2,
           label=f"Random baseline (AP ~= {baseline_pr:.4f})")
ax.fill_between(rec_lr, prec_lr, alpha=0.06, color=PALETTE["logreg"])
ax.fill_between(rec_rf, prec_rf, alpha=0.06, color=PALETTE["rf"])
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title(
    "Precision-Recall Curves — Logistic Regression vs Random Forest\n"
    "(Better metric than ROC for imbalanced datasets — 82/18 split)"
)
ax.legend(loc="upper right")
ax.set_facecolor(PALETTE["light"])
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("part2/model_results/comparison_pr_curves.png")
plt.close(fig)
print("  Saved: model_results/comparison_pr_curves.png")

# ============================================================
# 7. PLOT 3 — Calibration Plot (Reliability Diagram)
# ============================================================

fraction_pos_lr, mean_pred_lr = calibration_curve(y_test, y_proba_logreg, n_bins=10, strategy="uniform")
fraction_pos_rf, mean_pred_rf = calibration_curve(y_test, y_proba_rf,     n_bins=10, strategy="uniform")

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration", zorder=0)
ax.plot(mean_pred_lr, fraction_pos_lr, "o-", color=PALETTE["logreg"], linewidth=2.5,
        label="Logistic Regression", markersize=7)
ax.plot(mean_pred_rf, fraction_pos_rf, "s-", color=PALETTE["rf"], linewidth=2.5,
        label="Random Forest", markersize=7)
ax.fill_between([0, 1], [0, 1], [0, 1], alpha=0.05, color="black")
ax.set_xlabel("Mean Predicted Probability (Confidence)")
ax.set_ylabel("Fraction of Positives (Actual Rate)")
ax.set_title(
    "Calibration Plot (Reliability Diagram)\n"
    "A well-calibrated model hugs the diagonal — '0.7' should mean 70% of cases are positive"
)
ax.legend()
ax.set_facecolor(PALETTE["light"])
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("part2/model_results/comparison_calibration.png")
plt.close(fig)
print("  Saved: model_results/comparison_calibration.png")

# ============================================================
# 8. PLOT 4 — Threshold Sweep (Precision, Recall, F1)
# ============================================================

thresholds = np.linspace(0.01, 0.99, 200)

def compute_threshold_metrics(y_true, y_proba):
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        y_hat = (y_proba >= t).astype(int)
        precisions.append(precision_score(y_true, y_hat, zero_division=0))
        recalls.append(recall_score(y_true, y_hat, zero_division=0))
        f1s.append(f1_score(y_true, y_hat, zero_division=0))
    return np.array(precisions), np.array(recalls), np.array(f1s)

prec_sweep_lr, rec_sweep_lr, f1_sweep_lr = compute_threshold_metrics(y_test, y_proba_logreg)
prec_sweep_rf, rec_sweep_rf, f1_sweep_rf = compute_threshold_metrics(y_test, y_proba_rf)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for ax, prec, rec, f1, opt_thresh, color, label in zip(
    axes,
    [prec_sweep_lr, prec_sweep_rf],
    [rec_sweep_lr,  rec_sweep_rf],
    [f1_sweep_lr,   f1_sweep_rf],
    [thresh_logreg, thresh_rf],
    [PALETTE["logreg"], PALETTE["rf"]],
    ["Logistic Regression", "Random Forest"],
):
    ax.plot(thresholds, prec, linestyle="-",  linewidth=2, color=color,         label="Precision", alpha=0.9)
    ax.plot(thresholds, rec,  linestyle="--", linewidth=2, color=color,         label="Recall",    alpha=0.6)
    ax.plot(thresholds, f1,   linestyle="-.", linewidth=2.5, color="#333333",   label="F1",        alpha=1.0)
    ax.axvline(opt_thresh, color="#FF5555", linestyle=":", linewidth=2,
               label=f"Optimal threshold = {opt_thresh:.3f}")
    ax.axvline(0.5, color="#AAAAAA", linestyle=":", linewidth=1.2, label="Default = 0.5")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"{label}\nThreshold vs Precision / Recall / F1")
    ax.legend(fontsize=8)
    ax.set_facecolor(PALETTE["light"])

fig.suptitle("Threshold Analysis — Why 0.5 is Not the Optimal Cut-off\n(Imbalanced classes require threshold tuning based on business cost)", fontsize=12, fontweight="bold")
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("part2/model_results/comparison_threshold_analysis.png")
plt.close(fig)
print("  Saved: model_results/comparison_threshold_analysis.png")

# ============================================================
# 9. PLOT 5 — Metrics Comparison Table (as image for slides)
# ============================================================

fig, ax = plt.subplots(figsize=(12, 3))
ax.axis("off")

col_labels = [c for c in metrics_df.columns]
row_labels  = list(metrics_df.index)
cell_data   = [[f"{v:.4f}" for v in metrics_df.loc[r]] for r in row_labels]

colors_header = ["#1A1A2E"] * len(col_labels)
row_colors = [
    [PALETTE["logreg"] + "44"] * len(col_labels),
    [PALETTE["rf"]     + "44"] * len(col_labels),
]

tbl = ax.table(
    cellText=cell_data,
    rowLabels=row_labels,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    rowLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.1, 2.0)

# Style header
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#1A1A2E")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Style rows
for i, rcolor in enumerate(row_colors):
    for j in range(len(col_labels)):
        tbl[i + 1, j].set_facecolor(rcolor[j])
    tbl[i + 1, -1].set_text_props(fontweight="bold")

ax.set_title("Model Performance Comparison — Full Metric Suite", fontsize=13, fontweight="bold", pad=20)
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("part2/model_results/comparison_metrics_table.png")
plt.close(fig)
print("  Saved: model_results/comparison_metrics_table.png")

# ============================================================
# 10. SAVE MCNEMAR RESULT
# ============================================================

mcnemar_result = {
    "b_logreg_right_rf_wrong": int(b),
    "c_logreg_wrong_rf_right": int(c),
    "chi2_stat":               round(float(mcnemar_stat), 4),
    "p_value":                 round(float(mcnemar_p), 4),
    "significant_at_0.05":    bool(mcnemar_p < 0.05),
    "n_disagree":              int(n_disagree),
    "n_agree":                 int(n_agree),
    "pct_agree":               round(100 * n_agree / len(y_test), 2),
}
pd.DataFrame([mcnemar_result]).to_csv("part2/model_results/mcnemar_test.csv", index=False)

# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON — SUMMARY")
print("=" * 60)
print(f"\n  {'Metric':<20} {'LogReg':>12} {'RF':>12}")
print("  " + "-" * 45)
for col in metrics_df.columns:
    lr_val = metrics_df.loc["Logistic Regression", col]
    rf_val = metrics_df.loc["Random Forest",       col]
    # Lower is better for Brier Score
    if col == "Brier Score":
        winner = "← LR" if lr_val < rf_val else "← RF"
    else:
        winner = "← LR" if lr_val > rf_val else "← RF"
    print(f"  {col:<20} {lr_val:>12.4f} {rf_val:>12.4f}  {winner}")

print(f"\n  McNemar's test p-value : {mcnemar_p:.4f}")
if mcnemar_p < 0.05:
    print("  -> Models differ SIGNIFICANTLY in their error patterns")
else:
    print("  -> No significant difference in error patterns")

print("\n  Generated artifacts:")
print("  model_results/comparison_roc_curves.png")
print("  model_results/comparison_pr_curves.png")
print("  model_results/comparison_calibration.png")
print("  model_results/comparison_threshold_analysis.png")
print("  model_results/comparison_metrics_table.png")
print("  model_results/comparison_metrics.csv")
print("  model_results/mcnemar_test.csv")
