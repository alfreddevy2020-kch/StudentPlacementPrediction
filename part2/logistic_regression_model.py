"""
Role 2 — Model Lead 1: Logistic Regression
===========================================
Builds and evaluates Logistic Regression as the interpretable baseline model.

Key innovations beyond a basic implementation:
  - L1 (Lasso) vs L2 (Ridge) regularization comparison
  - GridSearchCV with stratified 5-fold CV, scoring on F1 (correct for imbalanced data)
  - Bootstrap confidence intervals for coefficients (research-grade rigor)
  - Precision-Recall curve analysis for imbalanced evaluation
  - Optimal threshold selection based on F1, not just default 0.5

Run after:
    python download_dataset.py
    python preprocessing.py
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    brier_score_loss,
)

warnings.filterwarnings("ignore")

# ============================================================
# 0. STYLE
# ============================================================

PALETTE = {
    "primary":   "#4F8EF7",
    "secondary": "#F7714F",
    "accent":    "#50C878",
    "dark":      "#1A1A2E",
    "light":     "#F5F5F5",
    "grid":      "#E0E0E0",
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
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION — Student Placement Prediction")
print("=" * 60)

TRAIN_PATH  = "data/processed/train_processed.csv"
TEST_PATH   = "data/processed/test_processed.csv"
WEIGHTS_PATH = "data/processed/class_weights.csv"

print("\n[1] Loading preprocessed data ...")

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
weights_df = pd.read_csv(WEIGHTS_PATH)

TARGET = "placement_status"

X_train = train_df.drop(columns=[TARGET]).values
y_train = train_df[TARGET].values
X_test  = test_df.drop(columns=[TARGET]).values
y_test  = test_df[TARGET].values

feature_names = list(train_df.drop(columns=[TARGET]).columns)

# Reconstruct class-weight dict
class_weights = dict(zip(weights_df["class"].astype(int), weights_df["weight"]))

print(f"  Training  : {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"  Testing   : {X_test.shape[0]} samples")
print(f"  Class weights: {class_weights}")

# ============================================================
# 2. OUTPUT DIRECTORIES
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("model_results", exist_ok=True)

# ============================================================
# 3. CROSS-VALIDATION SETUP
# ============================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# 4. L2 LOGISTIC REGRESSION  (Ridge — default, good for correlated features)
# ============================================================

print("\n" + "-" * 50)
print("[2] Tuning L2 Logistic Regression (Ridge) ...")
print("-" * 50)

C_values = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

param_grid_l2 = {"C": C_values}

base_l2 = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    max_iter=2000,
    class_weight=class_weights,
    random_state=42,
)

grid_l2 = GridSearchCV(
    base_l2,
    param_grid_l2,
    cv=cv,
    scoring="f1",           # F1 — correct metric for imbalanced data
    n_jobs=-1,
    verbose=0,
    return_train_score=True,
)
grid_l2.fit(X_train, y_train)

best_l2 = grid_l2.best_estimator_
best_C_l2 = grid_l2.best_params_["C"]
print(f"  Best C (L2): {best_C_l2}")
print(f"  Best CV F1 : {grid_l2.best_score_:.4f}")

# ============================================================
# 5. L1 LOGISTIC REGRESSION  (Lasso — drives unimportant features to zero)
# ============================================================

print("\n" + "-" * 50)
print("[3] Tuning L1 Logistic Regression (Lasso) ...")
print("-" * 50)

base_l1 = LogisticRegression(
    penalty="l1",
    solver="saga",    # saga supports L1 for multi-class; also works for binary
    max_iter=3000,
    class_weight=class_weights,
    random_state=42,
)

grid_l1 = GridSearchCV(
    base_l1,
    param_grid_l2,    # same C grid
    cv=cv,
    scoring="f1",
    n_jobs=-1,
    verbose=0,
    return_train_score=True,
)
grid_l1.fit(X_train, y_train)

best_l1 = grid_l1.best_estimator_
best_C_l1 = grid_l1.best_params_["C"]
print(f"  Best C (L1): {best_C_l1}")
print(f"  Best CV F1 : {grid_l1.best_score_:.4f}")

# ============================================================
# 6. EVALUATE BEST MODEL ON TEST SET
# ============================================================

print("\n" + "=" * 60)
print("[4] TEST SET EVALUATION — L2 (Best Model)")
print("=" * 60)

y_pred_l2   = best_l2.predict(X_test)
y_proba_l2  = best_l2.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred_l2, target_names=["Not Placed", "Placed"]))

cm_l2 = confusion_matrix(y_test, y_pred_l2)
print("Confusion Matrix:")
print(cm_l2)
print(f"\nROC-AUC           : {roc_auc_score(y_test, y_proba_l2):.4f}")
print(f"Average Precision : {average_precision_score(y_test, y_proba_l2):.4f}")
print(f"Brier Score       : {brier_score_loss(y_test, y_proba_l2):.4f}")

print("\n" + "=" * 60)
print("[5] TEST SET EVALUATION — L1")
print("=" * 60)

y_pred_l1   = best_l1.predict(X_test)
y_proba_l1  = best_l1.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred_l1, target_names=["Not Placed", "Placed"]))
print(f"\nROC-AUC           : {roc_auc_score(y_test, y_proba_l1):.4f}")
print(f"Average Precision : {average_precision_score(y_test, y_proba_l1):.4f}")

# ============================================================
# 7. OPTIMAL THRESHOLD SELECTION  (not just default 0.5)
# ============================================================

print("\n" + "-" * 50)
print("[6] Finding optimal classification threshold ...")
print("-" * 50)

thresholds = np.linspace(0.01, 0.99, 200)
f1_scores  = [
    f1_score(y_test, (y_proba_l2 >= t).astype(int), zero_division=0)
    for t in thresholds
]
best_threshold = thresholds[np.argmax(f1_scores)]
best_f1        = max(f1_scores)

print(f"  Default threshold (0.5) F1 : {f1_score(y_test, y_pred_l2):.4f}")
print(f"  Optimal threshold           : {best_threshold:.3f}")
print(f"  Optimal threshold F1        : {best_f1:.4f}")

y_pred_optimal = (y_proba_l2 >= best_threshold).astype(int)
print("\nClassification Report at Optimal Threshold:")
print(classification_report(y_test, y_pred_optimal, target_names=["Not Placed", "Placed"]))

# ============================================================
# 8. BOOTSTRAP CONFIDENCE INTERVALS FOR COEFFICIENTS
# ============================================================

print("\n" + "-" * 50)
print("[7] Bootstrap confidence intervals for L2 coefficients (500 iterations) ...")
print("-" * 50)

N_BOOTSTRAP = 500
rng = np.random.default_rng(42)
bootstrap_coefs = np.zeros((N_BOOTSTRAP, len(feature_names)))

model_for_bootstrap = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    C=best_C_l2,
    max_iter=2000,
    class_weight=class_weights,
    random_state=42,
)

for i in range(N_BOOTSTRAP):
    idx = rng.integers(0, len(X_train), size=len(X_train))   # resample with replacement
    X_boot = X_train[idx]
    y_boot = y_train[idx]
    # Ensure both classes are present
    if len(np.unique(y_boot)) < 2:
        bootstrap_coefs[i] = bootstrap_coefs[i - 1]
        continue
    model_for_bootstrap.fit(X_boot, y_boot)
    bootstrap_coefs[i] = model_for_bootstrap.coef_[0]
    if (i + 1) % 100 == 0:
        print(f"    ... {i + 1}/{N_BOOTSTRAP} done")

ci_low  = np.percentile(bootstrap_coefs, 2.5,  axis=0)
ci_high = np.percentile(bootstrap_coefs, 97.5, axis=0)
coef_mean = np.mean(bootstrap_coefs, axis=0)

# Features where 95% CI does NOT cross zero → statistically "significant"
significant_mask = ~((ci_low < 0) & (ci_high > 0))
print(f"\n  Features with CI not crossing zero (robust): {significant_mask.sum()}/{len(feature_names)}")

# ============================================================
# 9. COEFFICIENT ANALYSIS
# ============================================================

coefs = best_l2.coef_[0]
coef_df = pd.DataFrame({
    "feature":   feature_names,
    "coef":      coefs,
    "boot_mean": coef_mean,
    "ci_low":    ci_low,
    "ci_high":   ci_high,
    "significant": significant_mask,
}).sort_values("coef", ascending=True).reset_index(drop=True)

print("\n  Top features pushing toward PLACED (positive coef):")
top_pos = coef_df[coef_df["coef"] > 0].tail(5)[["feature", "coef"]]
print(top_pos.to_string(index=False))

print("\n  Top features pushing toward NOT PLACED (negative coef):")
top_neg = coef_df[coef_df["coef"] < 0].head(5)[["feature", "coef"]]
print(top_neg.to_string(index=False))

l1_coefs = best_l1.coef_[0]
zeroed_out = np.sum(np.abs(l1_coefs) < 1e-6)
print(f"\n  L1 zeroed out {zeroed_out}/{len(feature_names)} features (feature selection effect)")

# ============================================================
# 10. PLOT 1 — L2 Coefficient Importance
# ============================================================

print("\n[8] Generating plots ...")

fig, ax = plt.subplots(figsize=(10, 7))
colors = [PALETTE["primary"] if c > 0 else PALETTE["secondary"] for c in coef_df["coef"]]
ax.barh(coef_df["feature"], coef_df["coef"], color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
ax.axvline(0, color="#333333", linewidth=1.2, linestyle="--")
ax.set_xlabel("Coefficient Value  (positive → Placed, negative → Not Placed)", fontsize=11)
ax.set_title("Logistic Regression (L2) — Feature Coefficients\nBlue = pushes toward Placed  |  Red = pushes toward Not Placed", fontsize=12)
ax.set_facecolor(PALETTE["light"])
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("part2/model_results/logreg_coefficient_importance.png")
plt.close(fig)
print("  Saved: model_results/logreg_coefficient_importance.png")

# ============================================================
# 11. PLOT 2 — L1 vs L2 Coefficient Comparison
# ============================================================

compare_df = pd.DataFrame({
    "feature": feature_names,
    "L2 (Ridge)": coefs,
    "L1 (Lasso)": l1_coefs,
}).set_index("feature")

# sort by absolute L2 coefficient
compare_df = compare_df.reindex(compare_df["L2 (Ridge)"].abs().sort_values(ascending=True).index)

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

for ax, col, color, title in zip(
    axes,
    ["L2 (Ridge)", "L1 (Lasso)"],
    [PALETTE["primary"], PALETTE["secondary"]],
    [
        "L2 (Ridge) Regression\nAll features kept; magnitudes shrunk",
        "L1 (Lasso) Regression\nUnimportant features driven to zero",
    ],
):
    bars = compare_df[col].values
    bar_colors = [color if v > 0 else "#AAAAAA" for v in bars]
    ax.barh(compare_df.index, bars, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="#333333", linewidth=1.0, linestyle="--")
    ax.set_title(title, fontsize=11)
    ax.set_facecolor(PALETTE["light"])

axes[0].set_xlabel("Coefficient Value")
axes[1].set_xlabel("Coefficient Value")
fig.suptitle("L1 vs L2 Regularization — Coefficient Comparison\nL1 drives unimportant coefficients to exactly zero (automatic feature selection)", fontsize=12, fontweight="bold")
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("part2/model_results/logreg_l1_vs_l2_coefficients.png")
plt.close(fig)
print("  Saved: model_results/logreg_l1_vs_l2_coefficients.png")

# ============================================================
# 12. PLOT 3 — Bootstrap Confidence Intervals
# ============================================================

# Show top 15 features by absolute mean bootstrap coefficient
top15_idx = np.argsort(np.abs(coef_mean))[-15:]
plot_features = [feature_names[i] for i in top15_idx]
plot_mean     = coef_mean[top15_idx]
plot_low      = ci_low[top15_idx]
plot_high     = ci_high[top15_idx]
plot_sig      = significant_mask[top15_idx]

fig, ax = plt.subplots(figsize=(10, 7))
y_pos = np.arange(len(plot_features))

for i, (m, lo, hi, sig) in enumerate(zip(plot_mean, plot_low, plot_high, plot_sig)):
    color = PALETTE["primary"] if m > 0 else PALETTE["secondary"]
    alpha = 1.0 if sig else 0.45
    ax.plot([lo, hi], [i, i], color=color, linewidth=2.5, alpha=alpha)
    ax.plot(m, i, "o", color=color, markersize=7, alpha=alpha)

ax.axvline(0, color="#333333", linewidth=1.2, linestyle="--", label="Zero (no effect)")
ax.set_yticks(y_pos)
ax.set_yticklabels(plot_features, fontsize=9)
ax.set_xlabel("Coefficient Value (95% Bootstrap CI)")
ax.set_title(
    "Bootstrap Confidence Intervals — Top 15 Features\n"
    "Opaque = CI doesn't cross zero (statistically robust)  |  Faded = crosses zero",
    fontsize=11,
)
ax.legend(fontsize=9)
ax.set_facecolor(PALETTE["light"])
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("part2/model_results/logreg_bootstrap_ci.png")
plt.close(fig)
print("  Saved: model_results/logreg_bootstrap_ci.png")

# ============================================================
# 13. PLOT 4 — GridSearchCV: C vs CV F1
# ============================================================

cv_results = pd.DataFrame(grid_l2.cv_results_)
mean_test  = cv_results["mean_test_score"].values
mean_train = cv_results["mean_train_score"].values

fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogx(C_values, mean_train, marker="o", color=PALETTE["primary"],  label="Train F1", linewidth=2)
ax.semilogx(C_values, mean_test,  marker="s", color=PALETTE["secondary"], label="CV F1 (5-fold)", linewidth=2)
ax.axvline(best_C_l2, color=PALETTE["accent"], linestyle="--", linewidth=1.8, label=f"Best C = {best_C_l2}")
ax.set_xlabel("Regularization Strength C (log scale)")
ax.set_ylabel("F1 Score")
ax.set_title("Logistic Regression (L2) — Regularization Tuning\nHigher C = weaker regularization")
ax.legend()
ax.set_facecolor(PALETTE["light"])
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("part2/model_results/logreg_regularization_tuning.png")
plt.close(fig)
print("  Saved: model_results/logreg_regularization_tuning.png")

# ============================================================
# 14. PLOT 5 — Threshold Optimization
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(thresholds, f1_scores, color=PALETTE["primary"], linewidth=2, label="F1 Score")
ax.axvline(best_threshold, color=PALETTE["secondary"], linestyle="--", linewidth=1.8,
           label=f"Optimal threshold = {best_threshold:.3f}\nF1 = {best_f1:.4f}")
ax.axvline(0.5, color="#AAAAAA", linestyle=":", linewidth=1.5, label="Default threshold = 0.5")
ax.set_xlabel("Classification Threshold")
ax.set_ylabel("F1 Score")
ax.set_title("Threshold Optimization — F1 vs Decision Threshold\n(Imbalanced data requires threshold tuning; 0.5 is rarely optimal)")
ax.legend()
ax.set_facecolor(PALETTE["light"])
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("part2/model_results/logreg_threshold_optimization.png")
plt.close(fig)
print("  Saved: model_results/logreg_threshold_optimization.png")

# ============================================================
# 15. SAVE MODEL + METADATA
# ============================================================

joblib.dump(best_l2, "part2/models/logistic_regression_best.joblib")
print("\n  Saved model: models/logistic_regression_best.joblib")

# Save metadata for use in model_comparison.py
logreg_meta = {
    "best_C_l2":        best_C_l2,
    "best_C_l1":        best_C_l1,
    "cv_f1_l2":         round(grid_l2.best_score_, 4),
    "cv_f1_l1":         round(grid_l1.best_score_, 4),
    "test_roc_auc":     round(roc_auc_score(y_test, y_proba_l2), 4),
    "test_avg_precision": round(average_precision_score(y_test, y_proba_l2), 4),
    "test_brier":       round(brier_score_loss(y_test, y_proba_l2), 4),
    "optimal_threshold": round(float(best_threshold), 4),
    "optimal_f1":       round(float(best_f1), 4),
    "zeroed_features_l1": int(zeroed_out),
    "significant_features": int(significant_mask.sum()),
}

pd.DataFrame([logreg_meta]).to_csv("part2/model_results/logreg_metadata.csv", index=False)
print("  Saved metadata: model_results/logreg_metadata.csv")

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION — COMPLETE")
print("=" * 60)
print("\nGenerated artifacts:")
print("  models/logistic_regression_best.joblib")
print("  model_results/logreg_coefficient_importance.png")
print("  model_results/logreg_l1_vs_l2_coefficients.png")
print("  model_results/logreg_bootstrap_ci.png")
print("  model_results/logreg_regularization_tuning.png")
print("  model_results/logreg_threshold_optimization.png")
print("  model_results/logreg_metadata.csv")
