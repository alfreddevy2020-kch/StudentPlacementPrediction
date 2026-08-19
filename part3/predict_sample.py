"""
Part 3 — Sample Prediction Script
===================================
Loads the saved XGBoost model and preprocessor to make a prediction
on the user-specified sample input.
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

# Add repo root to path so feature_engineering is accessible
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from feature_engineering import engineer_features

# ============================================================
# 1. LOAD MODEL AND PREPROCESSOR
# ============================================================

print("\nLoading model and preprocessor ...")
model = joblib.load("part3/models/xgboost_best.joblib")

preprocessor_path = Path("part2/models/preprocessor.joblib")
if not preprocessor_path.exists():
    preprocessor_path = Path("part3/models/preprocessor.joblib")
preprocessor = joblib.load(preprocessor_path)
print(f"  Loaded successfully from {preprocessor_path}.")

# ============================================================
# 2. DEFINE SAMPLE INPUT (19 Numerical + 4 Categorical Raw Features)
# ============================================================

sample_input = pd.DataFrame([{
    # Numerical features (19)
    "age":                         21,
    "cgpa":                        8.1,
    "attendance_percentage":       88.0,
    "backlogs":                    0,
    "coding_skill_score":          82.0,
    "aptitude_score":              78.0,
    "communication_skill_score":   79.0,
    "logical_reasoning_score":     76.0,
    "mock_interview_score":        75.0,
    "internships_count":           2,
    "projects_count":              3,
    "certifications_count":        4,
    "hackathons_participated":     2,
    "github_repos":                12,
    "linkedin_connections":        250,
    "extracurricular_score":       70.0,
    "leadership_score":            65.0,
    "sleep_hours":                 7.0,
    "study_hours_per_day":         5.0,

    # Categorical features (4)
    "gender":                      "Male",
    "branch":                      "CSE",
    "college_tier":                "Tier 1",
    "volunteer_experience":        "Yes",
}])

print("\nSample Input:")
print(sample_input.to_string(index=False))

# ============================================================
# 3. PREPROCESS AND PREDICT
# ============================================================

sample_engineered = engineer_features(sample_input)
sample_processed = preprocessor.transform(sample_engineered)
proba = model.predict_proba(sample_processed)[0]

pred_status = int(proba[1] >= 0.5)
pred_label = "Placed" if pred_status == 1 else "Not Placed"
prob_placed = round(float(proba[1]), 4)
prob_not_placed = round(float(proba[0]), 4)

if prob_placed >= 0.9:
    risk = "High Probability of Placement (Low Risk)"
elif prob_placed >= 0.7:
    risk = "Moderate Probability of Placement"
elif prob_placed >= 0.5:
    risk = "Uncertain — Borderline Case"
else:
    risk = "Low Probability of Placement (High Risk)"

# ============================================================
# 4. OUTPUT
# ============================================================

output = {
    "placement_status":       pred_status,
    "placement_label":        pred_label,
    "probability_placed":     prob_placed,
    "probability_not_placed": prob_not_placed,
    "risk_level":             risk,
}

print("\n========== PREDICTION OUTPUT ==========")
print(output)
