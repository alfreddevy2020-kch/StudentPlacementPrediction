"""
Role 2 — Model Lead 1: Summary Report Generator
=================================================
Generates:
  1. A clean text report (model_results/model_report.txt)
  2. A 4-panel executive summary figure (model_results/executive_summary.png)
     designed to be dropped directly into a presentation slide

Run after all 3 model scripts have completed:
    python logistic_regression_model.py
    python random_forest_model.py
    python model_comparison.py
"""

import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    f1_score, confusion_matrix,
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
    "light":   "#F7F7FA",
    "bg":      "#FFFFFF",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       220,
    "savefig.bbox":      "tight",
})

# ============================================================
# 1. LOAD EVERYTHING
# ============================================================

print("\n" + "=" * 60)
print("MODEL SUMMARY REPORT GENERATOR")
print("=" * 60)

os.makedirs("part2/model_results", exist_ok=True)

logreg = joblib.load("part2/models/logistic_regression_best.joblib")
rf     = joblib.load("part2/models/random_forest_best.joblib")

test_df  = pd.read_csv("data/processed/test_processed.csv")
train_df = pd.read_csv("data/processed/train_processed.csv")

TARGET = "placement_status"
X_test  = test_df.drop(columns=[TARGET]).values
y_test  = test_df[TARGET].values
X_train = train_df.drop(columns=[TARGET]).values
y_train = train_df[TARGET].values

feature_names = list(test_df.drop(columns=[TARGET]).columns)

logreg_meta  = pd.read_csv("part2/model_results/logreg_metadata.csv").iloc[0]
rf_meta      = pd.read_csv("part2/model_results/rf_metadata.csv").iloc[0]
metrics_df   = pd.read_csv("part2/model_results/comparison_metrics.csv", index_col=0)
mcnemar_df   = pd.read_csv("part2/model_results/mcnemar_test.csv").iloc[0]
perm_imp_df  = pd.read_csv("part2/model_results/rf_importance_permutation.csv")

thresh_lr = float(logreg_meta["optimal_threshold"])
thresh_rf = float(rf_meta["optimal_threshold"])

y_proba_lr = logreg.predict_proba(X_test)[:, 1]
y_proba_rf = rf.predict_proba(X_test)[:, 1]

y_pred_lr = (y_proba_lr >= thresh_lr).astype(int)
y_pred_rf = (y_proba_rf >= thresh_rf).astype(int)

# ============================================================
# 2. TEXT REPORT
# ============================================================

print("\n[1] Generating text report ...")

def sep(char="=", n=65):
    return char * n

now = datetime.now().strftime("%Y-%m-%d %H:%M")

lines = [
    sep(),
    "STUDENT PLACEMENT PREDICTION — MODEL REPORT",
    f"Role 2: Logistic Regression + Random Forest",
    f"Generated: {now}",
    sep(),
    "",
    sep("-"),
    "1. DATASET SUMMARY",
    sep("-"),
    f"  Training samples : {len(X_train)}",
    f"  Testing  samples : {len(X_test)}",
    f"  Features         : {len(feature_names)}",
    f"  Train split      : 80/20 stratified",
    f"  Class balance    : {(y_train == 1).sum()} placed ({100*(y_train==1).mean():.1f}%) "
    f"vs {(y_train == 0).sum()} not placed ({100*(y_train==0).mean():.1f}%)",
    f"  Class imbalance handled via: computed class weights (not SMOTE)",
    "",
    sep("-"),
    "2. LOGISTIC REGRESSION — RESULTS",
    sep("-"),
    f"  Regularization   : L2 (Ridge) — selected for interpretability",
    f"  Best C           : {logreg_meta['best_C_l2']} (found via GridSearchCV, 5-fold CV, F1 scoring)",
    f"  L1 Best C        : {logreg_meta['best_C_l1']} (for comparison — Lasso variant)",
    f"  L1 zeroed features: {logreg_meta['zeroed_features_l1']} of {len(feature_names)} (automatic feature selection)",
    f"  CV F1 (L2)       : {logreg_meta['cv_f1_l2']}",
    f"  Test ROC-AUC     : {logreg_meta['test_roc_auc']}",
    f"  Test Avg Prec    : {logreg_meta['test_avg_precision']}",
    f"  Brier Score      : {logreg_meta['test_brier']} (lower = better calibration)",
    f"  Optimal threshold: {logreg_meta['optimal_threshold']} (tuned for F1, not default 0.5)",
    f"  Optimal F1       : {logreg_meta['optimal_f1']}",
    f"  Robust features  : {logreg_meta['significant_features']} (bootstrap CI doesn't cross zero)",
    "",
    sep("-"),
    "3. RANDOM FOREST — RESULTS",
    sep("-"),
    f"  Tuning method    : RandomizedSearchCV (80 iters) → GridSearchCV (fine-tune)",
    f"  Best n_estimators: {rf_meta['best_n_estimators']}",
    f"  Best max_depth   : {rf_meta['best_max_depth']}",
    f"  Best min_samples_split : {rf_meta['best_min_samples_split']}",
    f"  Best min_samples_leaf  : {rf_meta['best_min_samples_leaf']}",
    f"  Best max_features      : {rf_meta['best_max_features']}",
    f"  CV F1            : {rf_meta['cv_f1']}",
    f"  Test ROC-AUC     : {rf_meta['test_roc_auc']}",
    f"  Test Avg Prec    : {rf_meta['test_avg_precision']}",
    f"  Brier Score      : {rf_meta['test_brier']}",
    f"  Optimal threshold: {rf_meta['optimal_threshold']}",
    f"  Optimal F1       : {rf_meta['optimal_f1']}",
    f"  Top feature (MDI)        : {rf_meta['top_mdi_feature']}",
    f"  Top feature (Permutation): {rf_meta['top_perm_feature']}",
    f"  Top feature (Drop-Col)   : {rf_meta['top_drop_feature']}",
    "",
    sep("-"),
    "4. HEAD-TO-HEAD COMPARISON",
    sep("-"),
]

for col in metrics_df.columns:
    lr_val = metrics_df.loc["Logistic Regression", col]
    rf_val = metrics_df.loc["Random Forest",       col]
    better = "LR" if (lr_val > rf_val if col != "Brier Score" else lr_val < rf_val) else "RF"
    lines.append(f"  {col:<22} LR={lr_val:.4f}  RF={rf_val:.4f}  [Better: {better}]")

lines += [
    "",
    sep("-"),
    "5. STATISTICAL SIGNIFICANCE — McNEMAR'S TEST",
    sep("-"),
    f"  Samples where models agree    : {mcnemar_df['n_agree']} ({mcnemar_df['pct_agree']:.1f}%)",
    f"  Samples where models disagree : {mcnemar_df['n_disagree']}",
    f"  LogReg right, RF wrong (b)    : {mcnemar_df['b_logreg_right_rf_wrong']}",
    f"  LogReg wrong, RF right (c)    : {mcnemar_df['c_logreg_wrong_rf_right']}",
    f"  McNemar χ² statistic          : {mcnemar_df['chi2_stat']:.4f}",
    f"  p-value                       : {mcnemar_df['p_value']:.4f}",
    f"  Significant (p<0.05)          : {mcnemar_df['significant_at_0.05']}",
    "",
    sep("-"),
    "6. TOP 10 MOST PREDICTIVE FEATURES (Permutation Importance)",
    sep("-"),
]

for i, row in perm_imp_df.head(10).iterrows():
    lines.append(f"  {i+1:2d}. {row['feature']:<40} importance={row['importance']:.4f} (±{row['std']:.4f})")

cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_rf = confusion_matrix(y_test, y_pred_rf)

lines += [
    "",
    sep("-"),
    "7. CONFUSION MATRICES (at optimal thresholds)",
    sep("-"),
    "  Logistic Regression:",
    f"    TN={cm_lr[0,0]}  FP={cm_lr[0,1]}",
    f"    FN={cm_lr[1,0]}  TP={cm_lr[1,1]}",
    f"    False Negatives (missed placements): {cm_lr[1,0]} — costly errors",
    "",
    "  Random Forest:",
    f"    TN={cm_rf[0,0]}  FP={cm_rf[0,1]}",
    f"    FN={cm_rf[1,0]}  TP={cm_rf[1,1]}",
    f"    False Negatives (missed placements): {cm_rf[1,0]} — costly errors",
    "",
    sep("-"),
    "8. KEY FINDINGS & TALKING POINTS",
    sep("-"),
    "  [a] Why F1 was used as the CV scoring metric:",
    "      The dataset has an 82/18 class imbalance. Optimizing for accuracy",
    "      would trivially predict 'Not Placed' always. F1 balances precision/recall.",
    "",
    "  [b] Why Precision-Recall curves > ROC for this problem:",
    "      With heavy class imbalance, a model can have high ROC-AUC while",
    "      performing poorly on the minority class (Placed). PR curves directly",
    "      show this and are a more honest evaluation.",
    "",
    "  [c] Why L1 vs L2 comparison matters:",
    f"      L1 (Lasso) zeroed out {int(logreg_meta['zeroed_features_l1'])} features, effectively performing",
    "      automatic feature selection. L2 kept all features but shrunk their",
    "      magnitudes. Comparing both reveals which features are truly essential.",
    "",
    "  [d] Why 3 feature importance methods for Random Forest:",
    "      MDI (built-in) is biased toward high-cardinality continuous features.",
    "      Permutation importance measures actual predictive impact on unseen data.",
    "      Drop-column is the most honest but expensive. When MDI and Permutation",
    "      disagree, Permutation is the more trustworthy signal.",
    "",
    "  [e] Why bootstrap CIs for Logistic Regression coefficients:",
    "      Point estimates of coefficients don't tell you if the effect is",
    "      statistically robust. Bootstrap CIs identify features where the",
    "      coefficient reliably pushes predictions in one direction.",
    "",
    "  [f] Why OOB convergence plot matters:",
    f"      We chose n_estimators={rf_meta['best_n_estimators']} because the OOB error curve",
    "      plateaus — not because it's a round number. This is empirical evidence",
    "      that adding more trees won't help.",
    "",
    sep(),
    "END OF REPORT",
    sep(),
]

report_text = "\n".join(lines)
with open("part2/model_results/model_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

print(report_text)
print("\n  Saved: part2/model_results/model_report.txt")

# ============================================================
# 3. EXECUTIVE SUMMARY FIGURE  (4-panel, slide-ready)
# ============================================================

print("\n[2] Generating executive summary figure ...")

fig = plt.figure(figsize=(18, 13), facecolor=PALETTE["bg"])
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

ax_roc  = fig.add_subplot(gs[0, 0])
ax_pr   = fig.add_subplot(gs[0, 1])
ax_feat = fig.add_subplot(gs[1, 0])
ax_cal  = fig.add_subplot(gs[1, 1])

for ax in [ax_roc, ax_pr, ax_feat, ax_cal]:
    ax.set_facecolor(PALETTE["light"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Panel 1 — ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
auc_lr = roc_auc_score(y_test, y_proba_lr)
auc_rf = roc_auc_score(y_test, y_proba_rf)

ax_roc.plot(fpr_lr, tpr_lr, color=PALETTE["logreg"], linewidth=2.5, label=f"LogReg  (AUC={auc_lr:.3f})")
ax_roc.plot(fpr_rf, tpr_rf, color=PALETTE["rf"],     linewidth=2.5, label=f"RF      (AUC={auc_rf:.3f})")
ax_roc.plot([0,1],[0,1],"k--",linewidth=1.0,label="Random")
ax_roc.fill_between(fpr_lr, tpr_lr, alpha=0.06, color=PALETTE["logreg"])
ax_roc.fill_between(fpr_rf, tpr_rf, alpha=0.06, color=PALETTE["rf"])
ax_roc.set_xlabel("False Positive Rate", fontsize=10)
ax_roc.set_ylabel("True Positive Rate", fontsize=10)
ax_roc.set_title("ROC Curves", fontsize=12)
ax_roc.legend(fontsize=9, loc="lower right")

# Panel 2 — Precision-Recall
prec_lr, rec_lr, _ = precision_recall_curve(y_test, y_proba_lr)
prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_proba_rf)
ap_lr = average_precision_score(y_test, y_proba_lr)
ap_rf = average_precision_score(y_test, y_proba_rf)

ax_pr.plot(rec_lr, prec_lr, color=PALETTE["logreg"], linewidth=2.5, label=f"LogReg  (AP={ap_lr:.3f})")
ax_pr.plot(rec_rf, prec_rf, color=PALETTE["rf"],     linewidth=2.5, label=f"RF      (AP={ap_rf:.3f})")
ax_pr.axhline(y_test.mean(), color="#AAAAAA", linestyle="--", linewidth=1.2, label=f"Random (AP≈{y_test.mean():.3f})")
ax_pr.set_xlabel("Recall", fontsize=10)
ax_pr.set_ylabel("Precision", fontsize=10)
ax_pr.set_title("Precision-Recall Curves\n(More informative than ROC for 82/18 imbalance)", fontsize=11)
ax_pr.legend(fontsize=9, loc="upper right")

# Panel 3 — Top 10 Feature Importance (Permutation)
top10 = perm_imp_df.head(10).iloc[::-1].reset_index(drop=True)
colors_feat = [PALETTE["rf"] if i < 3 else PALETTE["logreg"] for i in range(len(top10))]
ax_feat.barh(top10["feature"], top10["importance"],
             xerr=top10["std"], color=colors_feat[::-1],
             alpha=0.85, edgecolor="white", ecolor="#888888", capsize=3)
ax_feat.set_xlabel("Permutation Importance (F1 drop)", fontsize=10)
ax_feat.set_title("Top 10 Predictive Features\n(Permutation Importance on test set — most honest method)", fontsize=11)
ax_feat.tick_params(axis="y", labelsize=8)

# Panel 4 — Calibration
frac_lr, mean_lr = calibration_curve(y_test, y_proba_lr, n_bins=10, strategy="uniform")
frac_rf, mean_rf = calibration_curve(y_test, y_proba_rf, n_bins=10, strategy="uniform")

ax_cal.plot([0,1],[0,1],"k--",linewidth=1.5,label="Perfect calibration", zorder=0)
ax_cal.plot(mean_lr, frac_lr, "o-", color=PALETTE["logreg"], linewidth=2.5, label="Logistic Regression", markersize=6)
ax_cal.plot(mean_rf, frac_rf, "s-", color=PALETTE["rf"],     linewidth=2.5, label="Random Forest",       markersize=6)
ax_cal.set_xlabel("Mean Predicted Probability", fontsize=10)
ax_cal.set_ylabel("Fraction of Positives", fontsize=10)
ax_cal.set_title("Calibration Plot\nAre predicted probabilities trustworthy?", fontsize=11)
ax_cal.legend(fontsize=9)

# Supertitle with key stats
rf_f1_val = metrics_df.loc["Random Forest",       "F1 Score"]
lr_f1_val = metrics_df.loc["Logistic Regression", "F1 Score"]

fig.suptitle(
    f"Student Placement Prediction — Role 2 Summary\n"
    f"LogReg F1={lr_f1_val:.4f} | Random Forest F1={rf_f1_val:.4f} | "
    f"McNemar p={float(mcnemar_df['p_value']):.4f}",
    fontsize=14, fontweight="bold", y=0.98,
)

plt.savefig("part2/model_results/executive_summary.png")
plt.close(fig)
print("  Saved: part2/model_results/executive_summary.png")

print("\n" + "=" * 60)
print("SUMMARY REPORT — COMPLETE")
print("=" * 60)
print("\nGenerated artifacts:")
print("  part2/model_results/model_report.txt")
print("  part2/model_results/executive_summary.png")
