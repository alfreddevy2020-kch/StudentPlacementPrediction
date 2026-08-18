"""
Part 3 — Sample Prediction Script
===================================
Loads the saved XGBoost model and preprocessor to make a prediction
on the user-specified sample input.
"""

import joblib
import pandas as pd

# ============================================================
# 1. LOAD MODEL AND PREPROCESSOR
# ============================================================

print("\nLoading model and preprocessor ...")
model = joblib.load("part3/models/xgboost_best.joblib")
preprocessor = joblib.load("part3/models/preprocessor.joblib")
print("  Loaded successfully.")

# ============================================================
# 2. DEFINE SAMPLE INPUT
# ============================================================

sample_input = pd.DataFrame(
    [
        {
            "ssc_percentage": 82.5,
            "hsc_percentage": 79.2,
            "degree_percentage": 76.8,
            "cgpa": 8.1,
            "attendance_percentage": 88,
            "backlogs": 0,
            "entrance_exam_score": 84,
            "technical_skill_score": 78,
            "soft_skill_score": 81,
            "certifications": 4,
            "live_projects": 3,
            "internship_count": 2,
            "work_experience_months": 0,
            "gender": "Male",
            "extracurricular_activities": "Yes",
        }
    ]
)

print("\nSample Input:")
print(sample_input.to_string(index=False))

# ============================================================
# 3. PREPROCESS AND PREDICT
# ============================================================

sample_processed = preprocessor.transform(sample_input)
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
    "placement_status": pred_status,
    "placement_label": pred_label,
    "probability_placed": prob_placed,
    "probability_not_placed": prob_not_placed,
    "risk_level": risk,
}

print("\n========== PREDICTION OUTPUT ==========")
print(output)
