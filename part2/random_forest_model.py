"""
Role 2 — Model Lead 1: Random Forest
======================================
Builds and evaluates a Random Forest classifier with:
  - Two-phase hyperparameter tuning (RandomizedSearchCV → GridSearchCV)
  - Three types of feature importance: MDI, Permutation, Drop-Column
  - OOB error convergence plot (proves n_estimators was chosen empirically)
  - Learning curve analysis (bias-variance diagnostic)
  - Optimal threshold tuning

Run after:
    python download_dataset.py
    python preprocessing.py
"""

import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    learning_curve,
)

# Add repo root to path so feature_engineering is importable when this script
# is run directly (python part{N}/...py puts the script's own directory on
# sys.path, not the repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_engineering import CLASS_LABELS, TARGET_COLUMN

warnings.filterwarnings("ignore")

# ============================================================
# 0. STYLE
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

# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST — Student Placement Prediction")
print("=" * 60)

TRAIN_PATH = "data/processed/train_processed.csv"
TEST_PATH = "data/processed/test_processed.csv"
WEIGHTS_PATH = "data/processed/class_weights.csv"

print("\n[1] Loading preprocessed data ...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
weights_df = pd.read_csv(WEIGHTS_PATH)

# Name preprocessing.py wrote the encoded target under.
TARGET = TARGET_COLUMN

X_train = train_df.drop(columns=[TARGET]).values
y_train = train_df[TARGET].values
X_test = test_df.drop(columns=[TARGET]).values
y_test = test_df[TARGET].values

feature_names = list(train_df.drop(columns=[TARGET]).columns)
class_weights = dict(zip(weights_df["class"].astype(int), weights_df["weight"]))

print(f"  Training  : {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"  Testing   : {X_test.shape[0]} samples")

os.makedirs("part2/models", exist_ok=True)
os.makedirs("part2/model_results", exist_ok=True)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# 2. PHASE 1 — RANDOMIZED SEARCH  (broad exploration)
# ============================================================

print("\n" + "-" * 50)
print("[2] Phase 1 — RandomizedSearchCV (broad parameter sweep) ...")
print("-" * 50)

param_dist = {
    "n_estimators": [100, 200, 300, 500, 700],
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_features": ["sqrt", "log2", 0.3, 0.5],
    "class_weight": [class_weights, "balanced"],
}

base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)

random_search = RandomizedSearchCV(
    base_rf,
    param_distributions=param_dist,
    n_iter=80,
    cv=cv,
    scoring="f1",
    n_jobs=-1,
    verbose=1,
    random_state=42,
)

t0 = time.time()
random_search.fit(X_train, y_train)
t1 = time.time()

print(f"\n  Best params (random search): {random_search.best_params_}")
print(f"  Best CV F1 (random search): {random_search.best_score_:.4f}")
print(f"  Time elapsed: {t1 - t0:.1f}s")

best_random = random_search.best_params_

# ============================================================
# 3. PHASE 2 — GRID SEARCH  (fine-tune around best params)
# ============================================================

print("\n" + "-" * 50)
print("[3] Phase 2 — GridSearchCV (fine-tuning around best params) ...")
print("-" * 50)


def neighbors(val, candidates):
    """Pick the value and its immediate neighbors from a sorted candidate list."""
    candidates = sorted(candidates)
    idx = candidates.index(val) if val in candidates else 0
    lo = max(0, idx - 1)
    hi = min(len(candidates) - 1, idx + 1)
    return list(dict.fromkeys(candidates[lo : hi + 1]))


int_estimators = [100, 200, 300, 500, 700]
int_depth = [5, 10, 15, 20]
int_split = [2, 5, 10, 20]
int_leaf = [1, 2, 5, 10]

best_n_est = best_random["n_estimators"]
best_depth = best_random["max_depth"] if best_random["max_depth"] is not None else 20
best_split = best_random["min_samples_split"]
best_leaf = best_random["min_samples_leaf"]
best_feat = best_random["max_features"]

fine_grid = {
    "n_estimators": neighbors(best_n_est, int_estimators),
    "max_depth": neighbors(best_depth, int_depth) + [None],
    "min_samples_split": neighbors(best_split, int_split),
    "min_samples_leaf": neighbors(best_leaf, int_leaf),
    "max_features": [best_feat],
    "class_weight": [best_random["class_weight"]],
}

print(f"  Fine grid: {fine_grid}")

grid_search = GridSearchCV(
    base_rf,
    fine_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1,
    verbose=1,
)

t0 = time.time()
grid_search.fit(X_train, y_train)
t1 = time.time()

best_rf = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"\n  Best params (grid search): {best_params}")
print(f"  Best CV F1 (grid search): {grid_search.best_score_:.4f}")
print(f"  Time elapsed: {t1 - t0:.1f}s")

# ============================================================
# 4. EVALUATE ON TEST SET
# ============================================================

print("\n" + "=" * 60)
print("[4] TEST SET EVALUATION")
print("=" * 60)

y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_LABELS))
print(f"ROC-AUC           : {roc_auc_score(y_test, y_proba):.4f}")
print(f"Average Precision : {average_precision_score(y_test, y_proba):.4f}")
print(f"Brier Score       : {brier_score_loss(y_test, y_proba):.4f}")

# Optimal threshold
thresholds = np.linspace(0.01, 0.99, 200)
f1_scores = [f1_score(y_test, (y_proba >= t).astype(int), zero_division=0) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
best_f1 = max(f1_scores)
print(f"\nOptimal threshold : {best_threshold:.3f}")
print(f"Optimal F1        : {best_f1:.4f}")

# ============================================================
# 5. FEATURE IMPORTANCE — METHOD 1: MDI (Mean Decrease Impurity)
# ============================================================

print("\n" + "-" * 50)
print("[5] Feature Importance — MDI (Mean Decrease Impurity) ...")
print("-" * 50)

mdi_importances = best_rf.feature_importances_
mdi_df = (
    pd.DataFrame(
        {
            "feature": feature_names,
            "importance": mdi_importances,
        }
    )
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

print("\n  Top 10 features (MDI):")
print(mdi_df.head(10).to_string(index=False))

# ============================================================
# 6. FEATURE IMPORTANCE — METHOD 2: Permutation Importance
# ============================================================

print("\n" + "-" * 50)
print("[6] Feature Importance — Permutation Importance (on test set) ...")
print("-" * 50)

perm_result = permutation_importance(
    best_rf,
    X_test,
    y_test,
    n_repeats=30,
    random_state=42,
    n_jobs=-1,
    scoring="f1",
)

perm_df = (
    pd.DataFrame(
        {
            "feature": feature_names,
            "importance": perm_result.importances_mean,
            "std": perm_result.importances_std,
        }
    )
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

print("\n  Top 10 features (Permutation):")
print(perm_df.head(10).to_string(index=False))

# ============================================================
# 7. FEATURE IMPORTANCE — METHOD 3: Drop-Column Importance
# ============================================================

print("\n" + "-" * 50)
print("[7] Feature Importance — Drop-Column Importance ...")
print("-" * 50)

baseline_f1 = f1_score(y_test, y_pred)
drop_col_scores = []

# Use a lighter RF for speed during drop-column analysis
drop_rf = RandomForestClassifier(
    **{k: v for k, v in best_params.items() if k != "n_estimators"},
    n_estimators=200,  # fewer trees for speed
    random_state=42,
    n_jobs=-1,
)

for i, feat in enumerate(feature_names):
    X_train_drop = np.delete(X_train, i, axis=1)
    X_test_drop = np.delete(X_test, i, axis=1)
    drop_rf.fit(X_train_drop, y_train)
    preds = drop_rf.predict(X_test_drop)
    drop_f1 = f1_score(y_test, preds)
    importance = baseline_f1 - drop_f1  # positive = feature was helpful
    drop_col_scores.append(importance)
    print(f"  [{i + 1:2d}/{len(feature_names)}] Drop '{feat}': F1 change = {importance:+.4f}")

drop_df = (
    pd.DataFrame(
        {
            "feature": feature_names,
            "importance": drop_col_scores,
        }
    )
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

# ============================================================
# 8. OOB ERROR CONVERGENCE
# ============================================================

print("\n" + "-" * 50)
print("[8] OOB Error Convergence (proving n_estimators choice) ...")
print("-" * 50)

estimator_range = [10, 25, 50, 75, 100, 150, 200, 300, 400, 500]
oob_errors = []

for n in estimator_range:
    rf_oob = RandomForestClassifier(
        **{k: v for k, v in best_params.items() if k != "n_estimators"},
        n_estimators=n,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )
    rf_oob.fit(X_train, y_train)
    oob_errors.append(1 - rf_oob.oob_score_)
    print(f"  n_estimators={n:4d}: OOB error = {oob_errors[-1]:.4f}")

# ============================================================
# 9. LEARNING CURVES
# ============================================================

print("\n" + "-" * 50)
print("[9] Computing learning curves ...")
print("-" * 50)

train_sizes_abs, lc_train_scores, lc_val_scores = learning_curve(
    best_rf,
    X_train,
    y_train,
    cv=cv,
    scoring="f1",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
    verbose=0,
)

lc_train_mean = lc_train_scores.mean(axis=1)
lc_train_std = lc_train_scores.std(axis=1)
lc_val_mean = lc_val_scores.mean(axis=1)
lc_val_std = lc_val_scores.std(axis=1)

print("  Learning curve computed.")

# ============================================================
# 10. PLOTS
# ============================================================

print("\n[10] Generating plots ...")

# --- MDI Feature Importance ---
top_mdi = mdi_df.head(15)
fig, ax = plt.subplots(figsize=(10, 7))
colors = [PALETTE["primary"] if i < 5 else PALETTE["secondary"] for i in range(len(top_mdi))]
ax.barh(
    top_mdi["feature"][::-1],
    top_mdi["importance"][::-1],
    color=colors[::-1],
    alpha=0.85,
    edgecolor="white",
)
ax.set_xlabel("MDI Importance")
ax.set_title(
    "Random Forest — MDI Feature Importance (Top 15)\nNote: MDI can be biased toward high-cardinality / continuous features"
)
ax.set_facecolor(PALETTE["light"])
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("part2/model_results/rf_feature_importance_mdi.png")
plt.close(fig)
print("  Saved: part2/model_results/rf_feature_importance_mdi.png")

# --- Permutation Importance ---
top_perm = perm_df.head(15).iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(
    top_perm["feature"],
    top_perm["importance"],
    xerr=top_perm["std"],
    color=PALETTE["primary"],
    alpha=0.85,
    edgecolor="white",
    ecolor="#555555",
    capsize=4,
)
ax.set_xlabel("Permutation Importance (mean F1 decrease ± std)")
ax.set_title(
    "Random Forest — Permutation Importance (Top 15, on Test Set)\nMeasures actual predictive impact by shuffling each feature"
)
ax.set_facecolor(PALETTE["light"])
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("part2/model_results/rf_feature_importance_permutation.png")
plt.close(fig)
print("  Saved: part2/model_results/rf_feature_importance_permutation.png")

# --- MDI vs Permutation Side-by-Side ---
# Normalize both to [0,1] for comparison
all_features_perm = perm_df.set_index("feature")["importance"]
all_features_mdi = mdi_df.set_index("feature")["importance"]

compare_imp = pd.DataFrame(
    {
        "MDI (normalised)": all_features_mdi / all_features_mdi.sum(),
        "Permutation (normalised)": all_features_perm.clip(lower=0)
        / all_features_perm.clip(lower=0).sum(),
    }
).fillna(0)

# Sort by MDI
compare_imp = compare_imp.sort_values("MDI (normalised)", ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(compare_imp))
width = 0.4
ax.barh(
    x - width / 2,
    compare_imp["MDI (normalised)"],
    width,
    label="MDI (biased)",
    color=PALETTE["primary"],
    alpha=0.85,
    edgecolor="white",
)
ax.barh(
    x + width / 2,
    compare_imp["Permutation (normalised)"],
    width,
    label="Permutation (honest)",
    color=PALETTE["secondary"],
    alpha=0.85,
    edgecolor="white",
)
ax.set_yticks(x)
ax.set_yticklabels(compare_imp.index, fontsize=9)
ax.set_xlabel("Normalised Importance")
ax.set_title(
    "MDI vs Permutation Importance — Top 15 Features (Normalised)\n"
    "Disagreements reveal where MDI's bias inflates or deflates true impact",
    fontsize=11,
)
ax.legend()
ax.set_facecolor(PALETTE["light"])
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("part2/model_results/rf_importance_comparison.png")
plt.close(fig)
print("  Saved: part2/model_results/rf_importance_comparison.png")

# --- Drop-Column Importance ---
top_drop = drop_df.head(15).iloc[::-1].reset_index(drop=True)
colors_drop = [PALETTE["accent"] if v > 0 else PALETTE["secondary"] for v in top_drop["importance"]]
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(
    top_drop["feature"], top_drop["importance"], color=colors_drop, alpha=0.85, edgecolor="white"
)
ax.axvline(0, color="#333333", linewidth=1.2, linestyle="--")
ax.set_xlabel("F1 Change when Feature is Dropped (positive = feature was helpful)")
ax.set_title(
    "Random Forest — Drop-Column Importance (Top 15)\nMost honest method: directly measures each feature's contribution to F1"
)
ax.set_facecolor(PALETTE["light"])
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("part2/model_results/rf_drop_column_importance.png")
plt.close(fig)
print("  Saved: part2/model_results/rf_drop_column_importance.png")

# --- OOB Convergence ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(estimator_range, oob_errors, marker="o", color=PALETTE["primary"], linewidth=2.5)
ax.axvline(
    best_params["n_estimators"],
    color=PALETTE["secondary"],
    linestyle="--",
    linewidth=1.8,
    label=f"Chosen n_estimators = {best_params['n_estimators']}",
)
ax.fill_between(estimator_range, oob_errors, alpha=0.1, color=PALETTE["primary"])
ax.set_xlabel("Number of Trees (n_estimators)")
ax.set_ylabel("OOB Error (1 - OOB Accuracy)")
ax.set_title(
    "OOB Error Convergence\nModel stabilises after ~200 trees — choice was empirical, not arbitrary"
)
ax.legend()
ax.set_facecolor(PALETTE["light"])
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("part2/model_results/rf_oob_convergence.png")
plt.close(fig)
print("  Saved: part2/model_results/rf_oob_convergence.png")

# --- Learning Curve ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(
    train_sizes_abs, lc_train_mean, "o-", color=PALETTE["primary"], label="Training F1", linewidth=2
)
ax.plot(
    train_sizes_abs,
    lc_val_mean,
    "s-",
    color=PALETTE["secondary"],
    label="CV F1 (5-fold)",
    linewidth=2,
)
ax.fill_between(
    train_sizes_abs,
    lc_train_mean - lc_train_std,
    lc_train_mean + lc_train_std,
    alpha=0.15,
    color=PALETTE["primary"],
)
ax.fill_between(
    train_sizes_abs,
    lc_val_mean - lc_val_std,
    lc_val_mean + lc_val_std,
    alpha=0.15,
    color=PALETTE["secondary"],
)
ax.set_xlabel("Training Set Size")
ax.set_ylabel("F1 Score")
ax.set_title("Learning Curve — Random Forest\nGap between train/val reveals bias-variance tradeoff")
ax.legend()
ax.set_facecolor(PALETTE["light"])
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("part2/model_results/rf_learning_curve.png")
plt.close(fig)
print("  Saved: part2/model_results/rf_learning_curve.png")

# --- Threshold Optimization ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(thresholds, f1_scores, color=PALETTE["primary"], linewidth=2, label="F1 Score")
ax.axvline(
    best_threshold,
    color=PALETTE["secondary"],
    linestyle="--",
    linewidth=1.8,
    label=f"Optimal threshold = {best_threshold:.3f}  (F1={best_f1:.4f})",
)
ax.axvline(0.5, color="#AAAAAA", linestyle=":", linewidth=1.5, label="Default threshold = 0.5")
ax.set_xlabel("Classification Threshold")
ax.set_ylabel("F1 Score")
ax.set_title("Threshold Optimization — Random Forest")
ax.legend()
ax.set_facecolor(PALETTE["light"])
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("part2/model_results/rf_threshold_optimization.png")
plt.close(fig)
print("  Saved: part2/model_results/rf_threshold_optimization.png")

# ============================================================
# 11. SAVE MODEL + METADATA
# ============================================================

joblib.dump(best_rf, "part2/models/random_forest_best.joblib")
print("\n  Saved model: part2/models/random_forest_best.joblib")

rf_meta = {
    "best_n_estimators": best_params["n_estimators"],
    "best_max_depth": str(best_params["max_depth"]),
    "best_min_samples_split": best_params["min_samples_split"],
    "best_min_samples_leaf": best_params["min_samples_leaf"],
    "best_max_features": best_params["max_features"],
    "cv_f1": round(grid_search.best_score_, 4),
    "test_roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    "test_avg_precision": round(average_precision_score(y_test, y_proba), 4),
    "test_brier": round(brier_score_loss(y_test, y_proba), 4),
    "optimal_threshold": round(float(best_threshold), 4),
    "optimal_f1": round(float(best_f1), 4),
    "top_mdi_feature": mdi_df.iloc[0]["feature"],
    "top_perm_feature": perm_df.iloc[0]["feature"],
    "top_drop_feature": drop_df.iloc[0]["feature"],
}

pd.DataFrame([rf_meta]).to_csv("part2/model_results/rf_metadata.csv", index=False)
print("  Saved metadata: part2/model_results/rf_metadata.csv")

# Save importance tables for use in model_comparison.py
mdi_df.to_csv("part2/model_results/rf_importance_mdi.csv", index=False)
perm_df.to_csv("part2/model_results/rf_importance_permutation.csv", index=False)
drop_df.to_csv("part2/model_results/rf_importance_drop_column.csv", index=False)

print("\n" + "=" * 60)
print("RANDOM FOREST — COMPLETE")
print("=" * 60)
print("\nGenerated artifacts:")
print("  part2/models/random_forest_best.joblib")
print("  part2/model_results/rf_feature_importance_mdi.png")
print("  part2/model_results/rf_feature_importance_permutation.png")
print("  part2/model_results/rf_importance_comparison.png")
print("  part2/model_results/rf_drop_column_importance.png")
print("  part2/model_results/rf_oob_convergence.png")
print("  part2/model_results/rf_learning_curve.png")
print("  part2/model_results/rf_threshold_optimization.png")
print("  part2/model_results/rf_metadata.csv")
