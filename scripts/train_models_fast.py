"""
Quick model trainer for Random Forest and XGBoost.
Reads whatever preprocessing.py last wrote to data/processed/, so it stays
correct across dataset changes.
Produces:
  - part2/models/random_forest_best.joblib
  - part3/models/xgboost_best.joblib
  - part3/models/xgboost_best.json
"""

import os
import shutil
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb

TRAIN_PATH = "data/processed/train_processed.csv"
TEST_PATH = "data/processed/test_processed.csv"
WEIGHTS_PATH = "data/processed/class_weights.csv"

print("\n[1] Loading preprocessed data ...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
weights_df = pd.read_csv(WEIGHTS_PATH)

print("=" * 60)
print(f"Training Models on {len(train_df):,} Samples x {train_df.shape[1] - 1} Features")
print("=" * 60)

TARGET = "placement_status"
X_train = train_df.drop(columns=[TARGET]).values
y_train = train_df[TARGET].values
X_test = test_df.drop(columns=[TARGET]).values
y_test = test_df[TARGET].values

class_weights = dict(zip(weights_df["class"].astype(int), weights_df["weight"]))

print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"  Test : {X_test.shape[0]} samples, {X_test.shape[1]} features")

os.makedirs("part2/models", exist_ok=True)
os.makedirs("part3/models", exist_ok=True)

# ------------------------------------------------------------
# 1. RANDOM FOREST
# ------------------------------------------------------------
print("\n[2] Training Random Forest Classifier ...")
t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=16,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight=class_weights,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
t1 = time.time()

rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probs)
rf_acc = accuracy_score(y_test, rf_preds)

print(f"  RF Training completed in {t1 - t0:.2f}s")
print(f"  RF Test Accuracy : {rf_acc:.4f}")
print(f"  RF Test ROC-AUC  : {rf_auc:.4f}")

rf_path = "part2/models/random_forest_best.joblib"
joblib.dump(rf, rf_path)
print(f"  Saved: {rf_path}")

# ------------------------------------------------------------
# 2. XGBOOST
# ------------------------------------------------------------
print("\n[3] Training XGBoost Classifier ...")
t0 = time.time()
scale_pos_weight = float(np.sum(y_train == 0) / np.sum(y_train == 1))

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)
xgb_model.fit(X_train, y_train)
t1 = time.time()

xgb_preds = xgb_model.predict(X_test)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_probs)
xgb_acc = accuracy_score(y_test, xgb_preds)

print(f"  XGB Training completed in {t1 - t0:.2f}s")
print(f"  XGB Test Accuracy : {xgb_acc:.4f}")
print(f"  XGB Test ROC-AUC  : {xgb_auc:.4f}")

xgb_joblib_path = "part3/models/xgboost_best.joblib"
xgb_json_path = "part3/models/xgboost_best.json"

joblib.dump(xgb_model, xgb_joblib_path)
xgb_model.save_model(xgb_json_path)
print(f"  Saved: {xgb_joblib_path}")
print(f"  Saved: {xgb_json_path}")

# Ensure preprocessor is in part3/models as well
src_prep = Path("part2/models/preprocessor.joblib")
dst_prep = Path("part3/models/preprocessor.joblib")
if src_prep.exists():
    shutil.copy(src_prep, dst_prep)
    print(f"  Synced: {dst_prep}")

print("\n" + "=" * 60)
print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("=" * 60)
