"""
feature_engineering.py
-----------------------
Canonical, importable feature-engineering step. Every consumer that needs
to turn raw student rows into what part2/models/preprocessor.joblib expects
(37 numerical + 4 categorical) must call engineer_features() from here —
never reimplement these formulas locally. That duplication is exactly what
broke on the last schema migration.

Normalization stats (github_repos_max / linkedin_connections_max) are fit
ONCE on the full training set by preprocessing.py and persisted to
NORM_STATS_PATH. engineer_features() always reuses that frozen constant so
a single student, a filtered cohort, and the full dataset all get the same
github_normalized / linkedin_normalized scale the model was trained on.
"""
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent
NORM_STATS_PATH = _REPO_ROOT / "data" / "processed" / "normalization_stats.json"

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


def fit_normalization_stats(df: pd.DataFrame) -> dict:
    """Compute github_repos_max / linkedin_connections_max from the FULL
    training dataset and persist them to NORM_STATS_PATH.

    Call this exactly once, from preprocessing.py, on the full deduplicated
    dataset before the train/test split - never on a filtered cohort or a
    single row. Every other consumer (batch_predictor.py, api/predictor.py,
    simulator.py, predict_sample.py) picks these up transparently through
    engineer_features(); they do not need to call this themselves.
    """
    stats = {
        "github_repos_max": float(df["github_repos"].max()),
        "linkedin_connections_max": float(df["linkedin_connections"].max()),
    }
    NORM_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NORM_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    _load_normalization_stats.cache_clear()
    return stats


@lru_cache(maxsize=1)
def _load_normalization_stats() -> dict | None:
    """Read the persisted training-time stats, cached for the process
    lifetime. Returns None if preprocessing.py has never been run."""
    if NORM_STATS_PATH.exists():
        with open(NORM_STATS_PATH) as f:
            return json.load(f)
    return None


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

    stats = _load_normalization_stats()
    if stats is not None:
        # Frozen training-time constants - identical for a single student,
        # a filtered cohort, or the full dataset.
        github_max = stats["github_repos_max"]
        linkedin_max = stats["linkedin_connections_max"]
    else:
        # preprocessing.py has not been (re)run yet, so there is no frozen
        # constant. Falls back to this batch's own max, which is only
        # correct when df IS the full training dataset. Any filtered
        # cohort or single-row caller hitting this branch gets skewed
        # github_normalized / linkedin_normalized - run preprocessing.py.
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