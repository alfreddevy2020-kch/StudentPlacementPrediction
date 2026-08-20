"""
feature_engineering.py
-----------------------
Canonical, importable feature-engineering step. Every consumer that needs
to turn raw student rows into what part2/models/preprocessor.joblib expects
must call engineer_features() from here — never reimplement these formulas
locally. That duplication is exactly what breaks on schema migrations.

Dataset
~~~~~~~
synthetic_placement_dataset.csv (10,000 rows x 19 cols).
Contains 13 raw numerical features + 4 raw categorical features + student_id + placement_status.

Column naming
~~~~~~~~~~~~~
normalize_columns() maps any legacy or casing variants to canonical snake_case.
Always load through load_raw_dataset() or call normalize_columns() on a freshly read CSV.

Normalization stats
~~~~~~~~~~~~~~~~~~~
The count/experience features have no natural upper bound, so their 0-1 normalized
forms are scaled by maxima fit ONCE on the full training set by preprocessing.py
and persisted to NORM_STATS_PATH. engineer_features() always reuses that frozen
constant so a single student, a filtered cohort, and the full dataset all get
the same scale the model was trained on.
"""

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent
NORM_STATS_FILENAME = "normalization_stats.json"
NORM_STATS_PATH = _REPO_ROOT / "data" / "processed" / NORM_STATS_FILENAME
RAW_DATASET_PATH = (
    _REPO_ROOT / "data" / "raw" / "synthetic_placement_dataset.csv"
    if (_REPO_ROOT / "data" / "raw" / "synthetic_placement_dataset.csv").exists()
    else _REPO_ROOT / "data" / "raw" / "student_placement.csv"
)

# Every key a stats file must carry to be usable.
NORM_STATS_KEYS = (
    "internships_max",
    "projects_max",
    "certifications_max",
    "work_experience_months_max",
)

# Raw CSV header -> canonical snake_case name.
COLUMN_RENAME_MAP = {
    # Legacy Kaggle headers & case aliases -> canonical snake_case
    "StudentID": "student_id",
    "CGPA": "cgpa",
    "Internships": "internships",
    "Projects": "projects",
    "Workshops/Certifications": "certifications",
    "workshops_certifications": "certifications",
    "Certifications": "certifications",
    "AptitudeTestScore": "aptitude_test_score",
    "SoftSkillsRating": "soft_skills_rating",
    "ExtracurricularActivities": "extracurricular_activities",
    "PlacementTraining": "placement_training",
    "SSC_Marks": "ssc_percentage",
    "ssc_marks": "ssc_percentage",
    "HSC_Marks": "hsc_percentage",
    "hsc_marks": "hsc_percentage",
    "PlacementStatus": "placement_status",
    "Gender": "gender",
    "Department": "department",
    "DegreePercentage": "degree_percentage",
    "TechnicalSkillScore": "technical_skill_score",
    "AttendancePercentage": "attendance_percentage",
    "Backlogs": "backlogs",
    "WorkExperienceMonths": "work_experience_months",
}

RAW_NUMERICAL_FEATURES = [
    "cgpa",
    "ssc_percentage",
    "hsc_percentage",
    "degree_percentage",
    "aptitude_test_score",
    "technical_skill_score",
    "soft_skills_rating",
    "attendance_percentage",
    "backlogs",
    "internships",
    "projects",
    "certifications",
    "work_experience_months",
]

RAW_CATEGORICAL_FEATURES = [
    "gender",
    "placement_training",
    "extracurricular_activities",
    "department",
]

ALL_RAW_FEATURES = RAW_NUMERICAL_FEATURES + RAW_CATEGORICAL_FEATURES

# Observed training-data ranges, used for dashboard slider bounds and API validation.
FEATURE_RANGES = {
    "cgpa":                     (5.0, 10.0),
    "ssc_percentage":           (35.0, 100.0),
    "hsc_percentage":           (35.0, 100.0),
    "degree_percentage":        (35.0, 100.0),
    "aptitude_test_score":      (30.0, 100.0),
    "technical_skill_score":    (30.0, 100.0),
    "soft_skills_rating":       (1.0, 5.0),
    "attendance_percentage":    (50.0, 100.0),
    "backlogs":                 (0, 8),
    "internships":              (0, 4),
    "projects":                 (0, 8),
    "certifications":           (0, 10),
    "work_experience_months":   (0, 33),
}

CATEGORICAL_LEVELS = {
    "gender": ["Female", "Male"],
    "placement_training": ["No", "Yes"],
    "extracurricular_activities": ["No", "Yes"],
    "department": ["CE", "CS", "ChemE", "ECE", "EE", "IT", "ME"],
}

TARGET_COLUMN = "placement_status"
TARGET_MAP = {"NotPlaced": 0, "Placed": 1}

# Human-readable class names, ordered so index == encoded target value.
CLASS_LABELS = [
    label for label, _ in sorted(TARGET_MAP.items(), key=lambda kv: kv[1])
]

# Levels carried by binary flags
BINARY_LEVEL_MAP = {"No": 0, "Yes": 1}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw CSV headers to canonical snake_case. Safe to call twice."""
    return df.rename(columns=COLUMN_RENAME_MAP)


def load_raw_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Read the raw placement CSV and normalize its column names."""
    path = Path(path) if path is not None else RAW_DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\nPlease ensure your placement dataset CSV is placed in data/raw/."
        )
    return normalize_columns(pd.read_csv(path))


def fit_normalization_stats(df: pd.DataFrame) -> dict:
    """Compute count/experience maxima from the FULL training dataset and persist them."""
    stats = {
        "internships_max": float(df["internships"].max()),
        "projects_max": float(df["projects"].max()),
        "certifications_max": float(df["certifications"].max()),
        "work_experience_months_max": float(df["work_experience_months"].max()),
    }
    NORM_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NORM_STATS_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")
    _load_normalization_stats.cache_clear()
    return stats


def load_normalization_stats(path: Path | str) -> dict:
    """Read and validate a normalization-stats file from an explicit path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {path}")
    with open(path) as f:
        stats = json.load(f)
    missing = [k for k in NORM_STATS_KEYS if k not in stats]
    if missing:
        raise ValueError(f"Normalization stats at {path} are missing keys: {missing}")
    return stats


@lru_cache(maxsize=1)
def _load_normalization_stats() -> dict | None:
    """Read repo-local stats written by preprocessing.py, cached for process lifetime."""
    if NORM_STATS_PATH.exists():
        return load_normalization_stats(NORM_STATS_PATH)
    return None


def _safe_div(numerator, denominator):
    """Divide by a frozen constant, guarding against degenerate zero maximum."""
    return numerator / denominator if denominator else 0.0


def engineer_features(df: pd.DataFrame, stats: dict | None = None) -> pd.DataFrame:
    """Adds the derived columns preprocessing.py's ColumnTransformer expects.

    Input: raw columns (RAW_NUMERICAL_FEATURES + RAW_CATEGORICAL_FEATURES),
    already snake_cased via normalize_columns(). Does not mutate the input.
    """
    df = normalize_columns(df.copy())

    # Fallbacks for optional/legacy schema inputs
    if "degree_percentage" not in df.columns:
        df["degree_percentage"] = 68.0
    if "technical_skill_score" not in df.columns:
        df["technical_skill_score"] = 65.0
    if "attendance_percentage" not in df.columns:
        df["attendance_percentage"] = 80.0
    if "backlogs" not in df.columns:
        df["backlogs"] = 0
    if "work_experience_months" not in df.columns:
        df["work_experience_months"] = 0
    if "gender" not in df.columns:
        df["gender"] = "Male"
    if "department" not in df.columns:
        df["department"] = "CS"
    if "placement_training" not in df.columns:
        df["placement_training"] = "Yes"
    if "extracurricular_activities" not in df.columns:
        df["extracurricular_activities"] = "Yes"
    if "certifications" not in df.columns:
        df["certifications"] = 1
    if "internships" not in df.columns:
        df["internships"] = 0
    if "projects" not in df.columns:
        df["projects"] = 1

    if stats is None:
        stats = _load_normalization_stats()
    if stats is None:
        raise RuntimeError(
            f"Normalization stats not found at {NORM_STATS_PATH}.\n"
            "engineer_features() will not infer them from the input, because "
            "an inferred maximum silently corrupts every *_normalized feature.\n"
            "Fix: run `python preprocessing.py` to regenerate them, or pass "
            "stats= explicitly (production bundles ship their own copy)."
        )

    # ── 1. Academic Dimension ───────────────────────────────────────────
    df["cgpa_percentage"] = df["cgpa"] * 10.0
    df["overall_academic_score"] = (
        df["ssc_percentage"] + df["hsc_percentage"] + df["degree_percentage"] + df["cgpa_percentage"]
    ) / 4.0
    df["academic_consistency"] = 100.0 - (
        df[["ssc_percentage", "hsc_percentage", "degree_percentage"]].max(axis=1)
        - df[["ssc_percentage", "hsc_percentage", "degree_percentage"]].min(axis=1)
    ).abs()
    df["academic_trend"] = df["degree_percentage"] - df["ssc_percentage"]
    df["backlog_penalty"] = df["backlogs"] * 5.0
    df["academic_risk_adjusted"] = df["overall_academic_score"] - df["backlog_penalty"]

    # ── 2. Skills & Cognitive Dimension ─────────────────────────────────
    df["soft_skills_scaled"] = df["soft_skills_rating"] * 20.0
    df["total_skill_composite"] = (
        df["aptitude_test_score"] + df["technical_skill_score"] + df["soft_skills_scaled"]
    ) / 3.0
    df["technical_aptitude_ratio"] = df["technical_skill_score"] / (df["aptitude_test_score"] + 1e-5)
    df["skill_gap_tech_soft"] = df["technical_skill_score"] - df["soft_skills_scaled"]
    df["skill_gap_apt_soft"] = df["aptitude_test_score"] - df["soft_skills_scaled"]

    # ── 3. Experience & Portfolio Dimension ─────────────────────────────
    df["work_experience_years"] = df["work_experience_months"] / 12.0
    df["experience_score"] = (
        df["internships"] * 2 + df["projects"] + df["certifications"] + (df["work_experience_months"] / 6.0)
    )
    df["practical_exposure_score"] = (
        df["internships"] * 2 + df["projects"] + df["certifications"]
    )
    df["projects_per_internship"] = df["projects"] / (df["internships"] + 1)

    internships_max = stats.get("internships_max", 4.0)
    projects_max = stats.get("projects_max", 8.0)
    certs_max = stats.get("certifications_max") or stats.get("workshops_certifications_max", 10.0)
    work_exp_max = stats.get("work_experience_months_max", 33.0)

    df["internships_normalized"] = _safe_div(df["internships"], internships_max)
    df["projects_normalized"] = _safe_div(df["projects"], projects_max)
    df["certifications_normalized"] = _safe_div(df["certifications"], certs_max)
    df["workshops_normalized"] = df["certifications_normalized"]
    df["work_experience_normalized"] = _safe_div(df["work_experience_months"], work_exp_max)
    df["portfolio_strength"] = (
        df["internships_normalized"]
        + df["projects_normalized"]
        + df["certifications_normalized"]
        + df["work_experience_normalized"]
    ) / 4.0 * 100.0

    # ── 4. Engagement, Attendance & Risk ────────────────────────────────
    df["attendance_risk"] = (75.0 - df["attendance_percentage"]).clip(lower=0)
    df["engagement_index"] = (df["attendance_percentage"] * 0.7) + (
        (100.0 - df["backlogs"] * 12.5).clip(lower=0, upper=100.0) * 0.3
    )

    # ── 5. Institutional Support Flags ──────────────────────────────────
    df["extracurricular_binary"] = (df["extracurricular_activities"] == "Yes").astype(int)
    df["placement_training_binary"] = (df["placement_training"] == "Yes").astype(int)
    df["support_index"] = df["extracurricular_binary"] + df["placement_training_binary"]

    # ── 6. Composite Readiness Indicators ───────────────────────────────
    df["interview_readiness_score"] = (
        df["technical_skill_score"] * 0.4
        + df["aptitude_test_score"] * 0.3
        + df["soft_skills_scaled"] * 0.3
    )
    df["placement_readiness_score"] = (
        df["interview_readiness_score"] * 0.35
        + df["academic_risk_adjusted"] * 0.30
        + df["portfolio_strength"] * 0.20
        + df["attendance_percentage"] * 0.15
    )

    return df


ENGINEERED_NUMERICAL_FEATURES = [
    "cgpa_percentage",
    "overall_academic_score",
    "academic_consistency",
    "academic_trend",
    "backlog_penalty",
    "academic_risk_adjusted",
    "soft_skills_scaled",
    "total_skill_composite",
    "technical_aptitude_ratio",
    "skill_gap_tech_soft",
    "skill_gap_apt_soft",
    "work_experience_years",
    "experience_score",
    "practical_exposure_score",
    "projects_per_internship",
    "internships_normalized",
    "projects_normalized",
    "certifications_normalized",
    "work_experience_normalized",
    "portfolio_strength",
    "attendance_risk",
    "engagement_index",
    "extracurricular_binary",
    "placement_training_binary",
    "support_index",
    "interview_readiness_score",
    "placement_readiness_score",
]

ALL_NUMERICAL_FEATURES = RAW_NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES


def make_sample_student(position: float = 0.75) -> pd.DataFrame:
    """Build one representative raw student row, derived from the schema above."""
    if not 0.0 <= position <= 1.0:
        raise ValueError(
            f"position must be within [0, 1] (the observed training range), got {position}"
        )

    row: dict = {}
    for column in RAW_NUMERICAL_FEATURES:
        low, high = FEATURE_RANGES[column]
        value = low + (high - low) * position
        if isinstance(low, int) and isinstance(high, int):
            row[column] = int(round(value))
        else:
            row[column] = round(float(value), 2)

    # Defaults for categoricals
    row["gender"] = "Male"
    row["placement_training"] = "Yes"
    row["extracurricular_activities"] = "Yes"
    row["department"] = "CS"

    return pd.DataFrame([row])
