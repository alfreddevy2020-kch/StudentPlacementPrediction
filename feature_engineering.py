import numpy as np
import pandas as pd

# ============================================================
# LOAD DATASET
# ============================================================

file_path = "data/raw/student_placement.csv"

df = pd.read_csv(file_path)

print("========== ORIGINAL DATASET ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================================
# REMOVE IDENTIFIER
# ============================================================

# student_id is only an identifier.
# It should not be used as a predictive feature.

df = df.drop(columns=["student_id"])


# ============================================================
# CREATE BINARY TARGET
# ============================================================

df["placement_target"] = df["placement_status"].map({"Not Placed": 0, "Placed": 1})

print("\n========== TARGET DISTRIBUTION ==========")
print(df["placement_target"].value_counts())


# ============================================================
# EXPERIENCE FEATURES
# ============================================================

df["experience_score"] = (
    df["internships_count"] + df["projects_count"] + df["hackathons_participated"]
)


# Internship + project interaction

df["internship_project_score"] = df["internships_count"] * df["projects_count"]


# ============================================================
# SKILL FEATURES
# ============================================================

skill_columns = [
    "coding_skill_score",
    "aptitude_score",
    "communication_skill_score",
    "logical_reasoning_score",
    "mock_interview_score",
]

df["overall_skill_score"] = df[skill_columns].mean(axis=1)


# Technical skill

df["technical_skill_score"] = (df["coding_skill_score"] + df["logical_reasoning_score"]) / 2


# Soft skill

df["soft_skill_score"] = (df["communication_skill_score"] + df["mock_interview_score"]) / 2


# ============================================================
# ACADEMIC FEATURES
# ============================================================

# Convert CGPA from 10-point scale to 100-point scale

df["cgpa_percentage"] = df["cgpa"] * 10


# Academic strength

df["academic_score"] = (df["cgpa_percentage"] + df["attendance_percentage"]) / 2


# Backlog penalty

df["backlog_penalty"] = df["backlogs"] * 10


# Adjusted academic score

df["adjusted_academic_score"] = df["academic_score"] - df["backlog_penalty"]


# ============================================================
# PROFESSIONAL PRESENCE
# ============================================================

# Normalize GitHub and LinkedIn values before combining

github_max = df["github_repos"].max()
linkedin_max = df["linkedin_connections"].max()

df["github_normalized"] = df["github_repos"] / github_max

df["linkedin_normalized"] = df["linkedin_connections"] / linkedin_max

df["professional_presence_score"] = (df["github_normalized"] + df["linkedin_normalized"]) / 2


# ============================================================
# CERTIFICATION / EXTRACURRICULAR PROFILE
# ============================================================

df["achievement_score"] = (
    df["certifications_count"] + df["hackathons_participated"] + (df["extracurricular_score"] / 20)
)


# ============================================================
# LEADERSHIP / VOLUNTEER FEATURE
# ============================================================

df["leadership_profile_score"] = (df["leadership_score"] + df["extracurricular_score"]) / 2


# Convert volunteer experience to binary

df["volunteer_binary"] = df["volunteer_experience"].map({"No": 0, "Yes": 1})


# ============================================================
# STUDY / LIFESTYLE FEATURES
# ============================================================

# Study-to-sleep ratio

df["study_sleep_ratio"] = df["study_hours_per_day"] / (df["sleep_hours"] + 0.1)


# ============================================================
# SKILL × EXPERIENCE INTERACTION
# ============================================================

df["skill_experience_score"] = df["overall_skill_score"] * (1 + df["experience_score"])


# ============================================================
# INTERVIEW READINESS
# ============================================================

df["interview_readiness_score"] = (
    df["communication_skill_score"]
    + df["aptitude_score"]
    + df["logical_reasoning_score"]
    + df["mock_interview_score"]
) / 4


# ============================================================
# PLACEMENT READINESS SCORE
# ============================================================

df["placement_readiness_score"] = (
    df["overall_skill_score"]
    + df["academic_score"]
    + (df["experience_score"] * 5)
    + df["leadership_score"]
) / 4


# ============================================================
# REMOVE ORIGINAL TARGET COLUMN
# ============================================================

# placement_status is the original categorical target.
# placement_target is the numerical target used for ML.

df = df.drop(columns=["placement_status"])


# ============================================================
# REMOVE SALARY FROM PLACEMENT MODEL
# ============================================================

# salary_package_lpa is an outcome of placement.
# Using it would cause data leakage.

df = df.drop(columns=["salary_package_lpa"])


# ============================================================
# DISPLAY FEATURE ENGINEERED DATASET
# ============================================================

print("\n========== FEATURE ENGINEERED DATASET ==========")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


print("\n========== FEATURE NAMES ==========")

for column in df.columns:
    print(column)


print("\n========== FIRST 5 ROWS ==========")

print(df.head())


# ============================================================
# SAVE FEATURE ENGINEERED DATASET
# ============================================================

output_file = "student_placement_engineered.csv"

df.to_csv(output_file, index=False)

print("\n========== FILE SAVED ==========")
print(output_file)


# ============================================================
# CORRELATION OF ENGINEERED FEATURES
# ============================================================

print("\n========== CORRELATION WITH PLACEMENT TARGET ==========")

numeric_df = df.select_dtypes(include=np.number)

correlations = (
    numeric_df.corr()["placement_target"].drop("placement_target").sort_values(ascending=False)
)

print(correlations.round(4))


print("\n========== FEATURE ENGINEERING COMPLETE ==========")
