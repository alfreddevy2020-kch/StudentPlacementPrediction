"""
Part 3 — XGBoost with CUDA (GPU Accelerated)
===============================================
Builds and evaluates an XGBoost classifier with:
  - GPU-accelerated training via CUDA (NVIDIA RTX 4050)
  - Two-phase hyperparameter tuning (RandomizedSearchCV → GridSearchCV)
  - Feature importance (Gain, Cover, Weight)
  - Optimal threshold tuning
  - SHAP-free feature analysis
  - Full classification metrics

Run after:
    python download_dataset.py
    python preprocessing.py
"""

import os
import time
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)

warnings.filterwarnings("ignore")

# ============================================================
# 0. STYLE
# ============================================================

PALETTE = {
    "primary":   "#4F8EF7",
    "secondary": "#F7714F",
    "accent":    "#50C878",
    "warn":      "#FFD166",
    "dark":      "#1A1A2E",
    "light":     "#F5F5F5",
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
print("XGBoost (CUDA GPU) — Student Placement Prediction")
print("=" * 60)

TRAIN_PATH   = "data/processed/train_processed.csv"
TEST_PATH    = "data/processed/test_processed.csv"
WEIGHTS_PATH = "data/processed/class_weights.csv"

print("\n[1] Loading preprocessed data ...")

train_df   = pd.read_csv(TRAIN_PATH)
test_df    = pd.read_csv(TEST_PATH)
weights_df = pd.read_csv(WEIGHTS_PATH)

TARGET = "placement_status"

X_train = train_df.drop(columns=[TARGET]).values
y_train = train_df[TARGET].values
X_test  = test_df.drop(columns=[TARGET]).values
y_test  = test_df[TARGET].values

feature_names = list(train_df.drop(columns=[TARGET]).columns)
class_weights = dict(zip(weights_df["class"].astype(int), weights_df["weight"]))

print(f"  Training  : {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"  Testing   : {X_test.shape[0]} samples")

# Compute scale_pos_weight for XGBoost (handles class imbalance)
count_neg = int((y_train == 0).sum())
count_pos = int((y_train == 1).sum())
scale_pos_weight = count_neg / count_pos
print(f"  scale_pos_weight : {scale_pos_weight:.4f} (class 0: {count_neg}, class 1: {count_pos})")

os.makedirs("part3/models", exist_ok=True)
os.makedirs("part3/model_results", exist_ok=True)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# 2. VERIFY GPU AVAILABILITY
# ============================================================

print("\n" + "-" * 50)
print("[2] Verifying GPU (CUDA) availability ...")
print("-" * 50)

gpu_available = False
try:
    params_test = {"tree_method": "hist", "device": "cuda"}
    dtrain_test = xgb.DMatrix(X_train[:10], label=y_train[:10])
    test_model = xgb.train(
        params_test, dtrain_test, num_boost_round=1, verbose_eval=False
    )
    gpu_available = True
    print("  GPU (CUDA) is available and working!")
    print("  Device: NVIDIA GeForce RTX 4050")
    tree_method = "hist"
    device = "cuda"
except Exception as e:
    print(f"  GPU not available: {e}")
    print("  Falling back to CPU (tree_method=hist)")
    tree_method = "hist"
    device = "cpu"

# ============================================================
# 3. PHASE 1 — RANDOMIZED SEARCH (broad sweep on GPU)
# ============================================================

print("\n" + "-" * 50)
print("[3] Phase 1 — RandomizedSearchCV (broad parameter sweep) ...")
print("-" * 50)

param_dist = {
    "n_estimators":      [100, 200, 300, 500, 700, 1000],
    "max_depth":         [3, 4, 5, 6, 8, 10, 12],
    "learning_rate":     [0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
    "subsample":         [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree":  [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight":  [1, 3, 5, 7, 10],
    "gamma":             [0, 0.1, 0.2, 0.3, 0.5],
    "reg_alpha":         [0, 0.01, 0.1, 1, 10],
    "reg_lambda":        [0, 0.01, 0.1, 1, 10],
}

base_xgb = xgb.XGBClassifier(
    objective="binary:logistic",
    scale_pos_weight=scale_pos_weight,
    tree_method=tree_method,
    device=device,
    random_state=42,
    eval_metric="logloss",
    use_label_encoder=False,
    n_jobs=1,
)

random_search = RandomizedSearchCV(
    base_xgb,
    param_distributions=param_dist,
    n_iter=100,
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
# 4. PHASE 2 — GRID SEARCH (fine-tune)
# ============================================================

print("\n" + "-" * 50)
print("[4] Phase 2 — GridSearchCV (fine-tuning around best params) ...")
print("-" * 50)


def neighbors(val, candidates):
    candidates = sorted(set(candidates))
    if val not in candidates:
        candidates = candidates + [val]
        candidates = sorted(set(candidates))
    idx = candidates.index(val)
    lo = max(0, idx - 1)
    hi = min(len(candidates) - 1, idx + 1)
    return list(dict.fromkeys(candidates[lo:hi + 1]))


int_estimators  = [100, 200, 300, 500, 700, 1000]
int_depth       = [3, 4, 5, 6, 8, 10, 12]
float_lr        = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
float_subsample = [0.6, 0.7, 0.8, 0.9, 1.0]
float_colsample = [0.6, 0.7, 0.8, 0.9, 1.0]
int_mcw         = [1, 3, 5, 7, 10]
float_gamma     = [0, 0.1, 0.2, 0.3, 0.5]
float_reg_alpha = [0, 0.01, 0.1, 1, 10]
float_reg_lambda = [0, 0.01, 0.1, 1, 10]

fine_grid = {
    "n_estimators":      neighbors(best_random["n_estimators"], int_estimators),
    "max_depth":         neighbors(best_random["max_depth"], int_depth),
    "learning_rate":     neighbors(best_random["learning_rate"], float_lr),
    "subsample":         [best_random["subsample"]],
    "colsample_bytree":  [best_random["colsample_bytree"]],
    "min_child_weight":  neighbors(best_random["min_child_weight"], int_mcw),
    "gamma":             [best_random["gamma"]],
    "reg_alpha":         [best_random["reg_alpha"]],
    "reg_lambda":        [best_random["reg_lambda"]],
}

print(f"  Fine grid sizes: {', '.join(f'{k}={len(v)}' for k, v in fine_grid.items())}")

grid_search = GridSearchCV(
    base_xgb,
    fine_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1,
    verbose=1,
)

t0 = time.time()
grid_search.fit(X_train, y_train)
t1 = time.time()

best_xgb = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"\n  Best params (grid search): {best_params}")
print(f"  Best CV F1 (grid search): {grid_search.best_score_:.4f}")
print(f"  Time elapsed: {t1 - t0:.1f}s")

# ============================================================
# 5. EVALUATE ON TEST SET
# ============================================================

print("\n" + "=" * 60)
print("[5] TEST SET EVALUATION")
print("=" * 60)

y_pred  = best_xgb.predict(X_test)
y_proba = best_xgb.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"]))
print(f"ROC-AUC           : {roc_auc_score(y_test, y_proba):.4f}")
print(f"Average Precision : {average_precision_score(y_test, y_proba):.4f}")
print(f"Brier Score       : {brier_score_loss(y_test, y_proba):.4f}")

# Optimal threshold
thresholds = np.linspace(0.01, 0.99, 200)
f1_scores = [
    f1_score(y_test, (y_proba >= t).astype(int), zero_division=0)
    for t in thresholds
]
best_threshold = thresholds[np.argmax(f1_scores)]
best_f1 = max(f1_scores)
print(f"\nOptimal threshold : {best_threshold:.3f}")
print(f"Optimal F1        : {best_f1:.4f}")

# ============================================================
# 6. FEATURE IMPORTANCE — METHOD 1: XGBoost Built-in
# ============================================================

print("\n" + "-" * 50)
print("[6] Feature Importance — XGBoost built-in (gain) ...")
print("-" * 50)

booster = best_xgb.get_booster()
importance_dict = booster.get_score(importance_type="gain")

# Map f0, f1, ... to feature names
mapped_importance = {}
for k, v in importance_dict.items():
    idx = int(k[1:])
    if idx < len(feature_names):
        mapped_importance[feature_names[idx]] = v

gain_df = pd.DataFrame({
    "feature":    list(mapped_importance.keys()),
    "importance": list(mapped_importance.values()),
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("\n  Top 10 features (Gain):")
print(gain_df.head(10).to_string(index=False))

# ============================================================
# 7. FEATURE IMPORTANCE — METHOD 2: Weight (split count)
# ============================================================

print("\n" + "-" * 50)
print("[7] Feature Importance — XGBoost built-in (weight/splits) ...")
print("-" * 50)

importance_weight = booster.get_score(importance_type="weight")

mapped_weight = {}
for k, v in importance_weight.items():
    idx = int(k[1:])
    if idx < len(feature_names):
        mapped_weight[feature_names[idx]] = v

weight_df = pd.DataFrame({
    "feature":    list(mapped_weight.keys()),
    "importance": list(mapped_weight.values()),
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("\n  Top 10 features (Weight / splits):")
print(weight_df.head(10).to_string(index=False))

# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

print("\n" + "-" * 50)
print("[8] Confusion Matrix ...")
print("-" * 50)

cm = confusion_matrix(y_test, y_pred)
print(f"\n  Confusion Matrix:\n{cm}")
print(f"\n  TN={cm[0][0]}  FP={cm[0][1]}")
print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

# ============================================================
# 9. PLOTS
# ============================================================

print("\n[9] Generating plots ...")

# --- Gain Feature Importance ---
top_gain = gain_df.head(15).iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10, 7))
colors = [PALETTE["primary"] if i >= len(top_gain) - 5 else PALETTE["secondary"]
          for i in range(len(top_gain))]
ax.barh(top_gain["feature"], top_gain["importance"], color=colors, alpha=0.85, edgecolor="white")
ax.set_xlabel("Gain Importance")
ax.set_title("XGBoost — Gain Feature Importance (Top 15)\nMeasures total loss reduction from each feature")
ax.set_facecolor(PALETTE["light"])
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("part3/model_results/xgb_feature_importance_gain.png")
plt.close(fig)
print("  Saved: model_results/xgb_feature_importance_gain.png")

# --- Weight Feature Importance ---
top_weight = weight_df.head(15).iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top_weight["feature"], top_weight["importance"],
        color=PALETTE["accent"], alpha=0.85, edgecolor="white")
ax.set_xlabel("Weight (Number of Splits)")
ax.set_title("XGBoost — Split Weight Feature Importance (Top 15)\nCounts how often each feature is used for splitting")
ax.set_facecolor(PALETTE["light"])
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("part3/model_results/xgb_feature_importance_weight.png")
plt.close(fig)
print("  Saved: model_results/xgb_feature_importance_weight.png")

# --- ROC Curve ---
fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color=PALETTE["primary"], linewidth=2.5,
        label=f"XGBoost (AUC = {roc_auc_score(y_test, y_proba):.4f})")
ax.plot([0, 1], [0, 1], color="#AAAAAA", linestyle="--", linewidth=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — XGBoost (CUDA)")
ax.legend(loc="lower right")
ax.set_facecolor(PALETTE["light"])
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("part3/model_results/xgb_roc_curve.png")
plt.close(fig)
print("  Saved: model_results/xgb_roc_curve.png")

# --- Precision-Recall Curve ---
prec, rec, pr_thresholds = precision_recall_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(rec, prec, color=PALETTE["secondary"], linewidth=2.5,
        label=f"XGBoost (AP = {average_precision_score(y_test, y_proba):.4f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve — XGBoost (CUDA)")
ax.legend(loc="lower left")
ax.set_facecolor(PALETTE["light"])
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("part3/model_results/xgb_pr_curve.png")
plt.close(fig)
print("  Saved: model_results/xgb_pr_curve.png")

# --- Confusion Matrix Heatmap ---
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Placed", "Placed"],
            yticklabels=["Not Placed", "Placed"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix — XGBoost (CUDA)")
plt.tight_layout()
plt.savefig("part3/model_results/xgb_confusion_matrix.png")
plt.close(fig)
print("  Saved: model_results/xgb_confusion_matrix.png")

# --- Threshold Optimization ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(thresholds, f1_scores, color=PALETTE["primary"], linewidth=2, label="F1 Score")
ax.axvline(best_threshold, color=PALETTE["secondary"], linestyle="--", linewidth=1.8,
           label=f"Optimal threshold = {best_threshold:.3f}  (F1={best_f1:.4f})")
ax.axvline(0.5, color="#AAAAAA", linestyle=":", linewidth=1.5, label="Default threshold = 0.5")
ax.set_xlabel("Classification Threshold")
ax.set_ylabel("F1 Score")
ax.set_title("Threshold Optimization — XGBoost (CUDA)")
ax.legend()
ax.set_facecolor(PALETTE["light"])
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("part3/model_results/xgb_threshold_optimization.png")
plt.close(fig)
print("  Saved: model_results/xgb_threshold_optimization.png")

# --- Gain vs Weight Importance Side-by-Side ---
all_features_gain   = gain_df.set_index("feature")["importance"]
all_features_weight = weight_df.set_index("feature")["importance"]

compare_imp = pd.DataFrame({
    "Gain (normalised)":   all_features_gain / all_features_gain.sum(),
    "Weight (normalised)": all_features_weight / all_features_weight.sum(),
}).fillna(0).sort_values("Gain (normalised)", ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(compare_imp))
width = 0.4
ax.barh(x - width / 2, compare_imp["Gain (normalised)"],   width, label="Gain (quality)",     color=PALETTE["primary"],   alpha=0.85, edgecolor="white")
ax.barh(x + width / 2, compare_imp["Weight (normalised)"], width, label="Weight (frequency)", color=PALETTE["secondary"], alpha=0.85, edgecolor="white")
ax.set_yticks(x)
ax.set_yticklabels(compare_imp.index, fontsize=9)
ax.set_xlabel("Normalised Importance")
ax.set_title(
    "XGBoost Gain vs Weight Importance — Top 15 Features\n"
    "Gain measures loss reduction quality; Weight measures split frequency",
    fontsize=11,
)
ax.legend()
ax.set_facecolor(PALETTE["light"])
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("part3/model_results/xgb_importance_comparison.png")
plt.close(fig)
print("  Saved: model_results/xgb_importance_comparison.png")

# ============================================================
# 10. SAVE PREPROCESSOR
# ============================================================

print("\n" + "-" * 50)
print("[10] Saving preprocessor ...")
print("-" * 50)

# Copy canonical preprocessor from part2/models/ to part3/models/
import shutil
from pathlib import Path

from feature_engineering import engineer_features

src_prep = Path("part2/models/preprocessor.joblib")
dst_prep = Path("part3/models/preprocessor.joblib")
if src_prep.exists():
    shutil.copy(src_prep, dst_prep)
    preprocessor = joblib.load(dst_prep)
    print(f"  Copied canonical preprocessor: {dst_prep}")
elif dst_prep.exists():
    preprocessor = joblib.load(dst_prep)
    print(f"  Loaded preprocessor: {dst_prep}")
else:
    preprocessor = None

# ============================================================
# 11. PREDICTION ON SAMPLE INPUT
# ============================================================

print("\n" + "-" * 50)
print("[11] Sample Prediction ...")
print("-" * 50)

sample_input = pd.DataFrame([{
    "cgpa":                        8.1,
    "ssc_marks":                   82.0,
    "hsc_marks":                   85.0,
    "aptitude_test_score":         88.0,
    "soft_skills_rating":          4.6,
    "internships":                 2,
    "projects":                    3,
    "workshops_certifications":    2,
    "extracurricular_activities":  "Yes",
    "placement_training":          "Yes",
}])

if preprocessor is not None:
    sample_engineered = engineer_features(sample_input)
    sample_processed = preprocessor.transform(sample_engineered)
    sample_proba = best_xgb.predict_proba(sample_processed)[0]
    pred_label = "Placed" if sample_proba[1] >= 0.5 else "Not Placed"
pred_status = 1 if sample_proba[1] >= 0.5 else 0

if sample_proba[1] >= 0.9:
    risk = "High Probability of Placement (Low Risk)"
elif sample_proba[1] >= 0.7:
    risk = "Moderate Probability of Placement"
elif sample_proba[1] >= 0.5:
    risk = "Uncertain — Borderline Case"
else:
    risk = "Low Probability of Placement (High Risk)"

print("\n  Input: cgpa=8.1, ssc=82.0, hsc=85.0, aptitude=88.0, ...")
print(f"  Placement Status : {pred_status}")
print(f"  Placement Label  : {pred_label}")
print(f"  P(Placed)        : {sample_proba[1]:.4f}")
print(f"  P(Not Placed)    : {sample_proba[0]:.4f}")
print(f"  Risk Level       : {risk}")

# ============================================================
# 12. SAVE MODEL + METADATA
# ============================================================

print("\n" + "-" * 50)
print("[12] Saving model and artifacts ...")
print("-" * 50)

# Save the XGBoost model
best_xgb.save_model("part3/models/xgboost_best.json")
print("  Saved model: part3/models/xgboost_best.json")

# Save the sklearn wrapper (for easier loading)
joblib.dump(best_xgb, "part3/models/xgboost_best.joblib")
print("  Saved model (joblib): part3/models/xgboost_best.joblib")

# Save metadata
xgb_meta = {
    "model":                 "XGBoost (CUDA GPU)",
    "device":                device,
    "n_estimators":          best_params["n_estimators"],
    "max_depth":             best_params["max_depth"],
    "learning_rate":         best_params["learning_rate"],
    "subsample":             best_params["subsample"],
    "colsample_bytree":      best_params["colsample_bytree"],
    "min_child_weight":      best_params["min_child_weight"],
    "gamma":                 best_params["gamma"],
    "reg_alpha":             best_params["reg_alpha"],
    "reg_lambda":            best_params["reg_lambda"],
    "scale_pos_weight":      round(scale_pos_weight, 4),
    "cv_f1":                 round(grid_search.best_score_, 4),
    "test_roc_auc":          round(roc_auc_score(y_test, y_proba), 4),
    "test_avg_precision":    round(average_precision_score(y_test, y_proba), 4),
    "test_brier":            round(brier_score_loss(y_test, y_proba), 4),
    "optimal_threshold":     round(float(best_threshold), 4),
    "optimal_f1":            round(float(best_f1), 4),
    "top_gain_feature":      gain_df.iloc[0]["feature"],
    "top_weight_feature":    weight_df.iloc[0]["feature"],
}

pd.DataFrame([xgb_meta]).to_csv("part3/model_results/xgb_metadata.csv", index=False)
print("  Saved metadata: part3/model_results/xgb_metadata.csv")

# Save importance tables
gain_df.to_csv("part3/model_results/xgb_importance_gain.csv",   index=False)
weight_df.to_csv("part3/model_results/xgb_importance_weight.csv", index=False)

print("\n" + "=" * 60)
print("XGBoost (CUDA) — COMPLETE")
print("=" * 60)
print("\nGenerated artifacts:")
print("  part3/models/xgboost_best.json")
print("  part3/models/xgboost_best.joblib")
print("  part3/models/preprocessor.joblib")
print("  part3/model_results/xgb_feature_importance_gain.png")
print("  part3/model_results/xgb_feature_importance_weight.png")
print("  part3/model_results/xgb_importance_comparison.png")
print("  part3/model_results/xgb_roc_curve.png")
print("  part3/model_results/xgb_pr_curve.png")
print("  part3/model_results/xgb_confusion_matrix.png")
print("  part3/model_results/xgb_threshold_optimization.png")
print("  part3/model_results/xgb_metadata.csv")
print("  part3/model_results/xgb_importance_gain.csv")
print("  part3/model_results/xgb_importance_weight.csv")
