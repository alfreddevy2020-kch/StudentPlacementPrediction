"""
feature_engineering.py
-----------------------
Canonical, importable feature-engineering step. Every consumer that needs
to turn raw student rows into what part2/models/preprocessor.joblib expects
(37 numerical + 4 categorical) must call engineer_features() from here —
never reimplement these formulas locally. That duplication is exactly what
broke on the last schema migration.
"""
import pandas as pd

RAW_NUMERICAL_FEATURES = [
    "age", "cgpa", "attendance_percentage", "backlogs",
    "coding_skill_score", "aptitude_score", "communication_skill_score",
    "logical_reasoning_score", "mock_interview_score",
    "internships_count", "projects_count", "certifications_count",
    "hackathons_participated", "github_repos", "linkedin_connections",
    "extracurricular_score", "leadership_score",
    "sleep_hours", "study_hours_per_day",
]
RAW_CATEGORICAL_FEATURES = ["gender", "branch", "college_tier", "volunteer_experience"]
ALL_RAW_FEATURES = RAW_NUMERICAL_FEATURES + RAW_CATEGORICAL_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the 18 derived columns preprocessing.py's ColumnTransformer expects.
    Input: raw columns (RAW_NUMERICAL_FEATURES + RAW_CATEGORICAL_FEATURES).
    Does not mutate the input."""
    df = df.copy()

    df["experience_score"] = df["internships_count"] + df["projects_count"] + df["hackathons_participated"]
    df["internship_project_score"] = df["internships_count"] * df["projects_count"]

    skill_columns = ["coding_skill_score", "aptitude_score", "communication_skill_score",
                      "logical_reasoning_score", "mock_interview_score"]
    df["overall_skill_score"] = df[skill_columns].mean(axis=1)
    df["technical_skill_score"] = (df["coding_skill_score"] + df["logical_reasoning_score"]) / 2
    df["soft_skill_score"] = (df["communication_skill_score"] + df["mock_interview_score"]) / 2

    df["cgpa_percentage"] = df["cgpa"] * 10
    df["academic_score"] = (df["cgpa_percentage"] + df["attendance_percentage"]) / 2
    df["backlog_penalty"] = df["backlogs"] * 10
    df["adjusted_academic_score"] = df["academic_score"] - df["backlog_penalty"]

    github_max = df["github_repos"].max()
    linkedin_max = df["linkedin_connections"].max()
    df["github_normalized"] = df["github_repos"] / github_max if github_max > 0 else 0
    df["linkedin_normalized"] = df["linkedin_connections"] / linkedin_max if linkedin_max > 0 else 0
    df["professional_presence_score"] = (df["github_normalized"] + df["linkedin_normalized"]) / 2

    df["achievement_score"] = df["certifications_count"] + df["hackathons_participated"] + (df["extracurricular_score"] / 20)
    df["leadership_profile_score"] = (df["leadership_score"] + df["extracurricular_score"]) / 2
    df["volunteer_binary"] = df["volunteer_experience"].map({"No": 0, "Yes": 1})
    df["study_sleep_ratio"] = df["study_hours_per_day"] / (df["sleep_hours"] + 0.1)

    df["interview_readiness_score"] = (df["communication_skill_score"] + df["aptitude_score"]
                                        + df["logical_reasoning_score"] + df["mock_interview_score"]) / 4
    df["placement_readiness_score"] = (df["overall_skill_score"] + df["academic_score"]
                                        + (df["experience_score"] * 5) + df["leadership_score"]) / 4
    return df


ENGINEERED_NUMERICAL_FEATURES = [
    "experience_score", "internship_project_score", "overall_skill_score",
    "technical_skill_score", "soft_skill_score", "cgpa_percentage", "academic_score",
    "backlog_penalty", "adjusted_academic_score", "github_normalized", "linkedin_normalized",
    "professional_presence_score", "achievement_score", "leadership_profile_score",
    "volunteer_binary", "study_sleep_ratio", "interview_readiness_score", "placement_readiness_score",
]
ALL_NUMERICAL_FEATURES = RAW_NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES  # 37, verified