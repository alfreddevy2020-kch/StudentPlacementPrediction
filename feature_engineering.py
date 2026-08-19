"""
feature_engineering.py
-----------------------
Canonical, importable feature-engineering step. Every consumer that needs
to turn raw student rows into what part2/models/preprocessor.joblib expects
(29 numerical + 2 categorical) must call engineer_features() from here —
never reimplement these formulas locally. That duplication is exactly what
broke on the last schema migration.

Dataset
~~~~~~~
ruchikakumbhar/placement-prediction-dataset (10,000 rows x 12 cols).
Chosen because it is the only screened Kaggle placement dataset that both
clears the 0.80 ROC-AUC milestone (~0.88 held-out) and carries genuine
graded signal across every feature. The two datasets used previously were
synthetic: one was pure noise (0.577 AUC), the other encoded a
deterministic AND-rule (1.000 AUC). See SCHEMA.md.

Column naming
~~~~~~~~~~~~~
The raw CSV ships mixed-case headers and one ("Workshops/Certifications")
that is not a valid Python identifier. normalize_columns() maps every
header to snake_case ONCE, here, so the API, dashboard and training
scripts all speak the same names. Always load through
load_raw_dataset() or call normalize_columns() on a freshly read CSV.

Normalization stats
~~~~~~~~~~~~~~~~~~~
The count features have no natural upper bound, so their 0-1 normalized
forms are scaled by maxima fit ONCE on the full training set by
preprocessing.py and persisted to NORM_STATS_PATH. engineer_features()
always reuses that frozen constant so a single student, a filtered cohort
and the full dataset all get the same scale the model was trained on.
"""
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent
NORM_STATS_PATH = _REPO_ROOT / "data" / "processed" / "normalization_stats.json"
RAW_DATASET_PATH = _REPO_ROOT / "data" / "raw" / "student_placement.csv"

# Raw CSV header -> canonical snake_case name.
COLUMN_RENAME_MAP = {
    "StudentID": "student_id",
    "CGPA": "cgpa",
    "Internships": "internships",
    "Projects": "projects",
    "Workshops/Certifications": "workshops_certifications",
    "AptitudeTestScore": "aptitude_test_score",
    "SoftSkillsRating": "soft_skills_rating",
    "ExtracurricularActivities": "extracurricular_activities",
    "PlacementTraining": "placement_training",
    "SSC_Marks": "ssc_marks",
    "HSC_Marks": "hsc_marks",
    "PlacementStatus": "placement_status",
}

RAW_NUMERICAL_FEATURES = [
    "cgpa", "ssc_marks", "hsc_marks",
    "aptitude_test_score", "soft_skills_rating",
    "internships", "projects", "workshops_certifications",
]
RAW_CATEGORICAL_FEATURES = ["extracurricular_activities", "placement_training"]
ALL_RAW_FEATURES = RAW_NUMERICAL_FEATURES + RAW_CATEGORICAL_FEATURES

# Observed training-data ranges, used for dashboard slider bounds and API
# validation. Inputs outside these are extrapolation - the model has never
# seen them.
FEATURE_RANGES = {
    "cgpa":                     (6.5, 9.1),
    "ssc_marks":                (55.0, 90.0),
    "hsc_marks":                (57.0, 88.0),
    "aptitude_test_score":      (60.0, 90.0),
    "soft_skills_rating":       (3.0, 4.8),
    "internships":              (0, 2),
    "projects":                 (0, 3),
    "workshops_certifications": (0, 3),
}

TARGET_COLUMN = "placement_status"
TARGET_MAP = {"NotPlaced": 0, "Placed": 1}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw CSV headers to canonical snake_case. Safe to call twice."""
    return df.rename(columns=COLUMN_RENAME_MAP)


def load_raw_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Read the raw placement CSV and normalize its column names."""
    path = Path(path) if path is not None else RAW_DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\nRun download_dataset.py first."
        )
    return normalize_columns(pd.read_csv(path))


def fit_normalization_stats(df: pd.DataFrame) -> dict:
    """Compute the count-feature maxima from the FULL training dataset and
    persist them to NORM_STATS_PATH.

    Call this exactly once, from preprocessing.py, on the full deduplicated
    dataset before the train/test split - never on a filtered cohort or a
    single row. Every other consumer (batch_predictor.py, api/predictor.py,
    simulator.py, predict_sample.py) picks these up transparently through
    engineer_features(); they do not need to call this themselves.
    """
    stats = {
        "internships_max": float(df["internships"].max()),
        "projects_max": float(df["projects"].max()),
        "workshops_certifications_max": float(df["workshops_certifications"].max()),
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


def _safe_div(numerator, denominator):
    """Divide by a frozen constant, guarding the never-yet-preprocessed case."""
    return numerator / denominator if denominator else 0.0


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the 21 derived columns preprocessing.py's ColumnTransformer expects.

    Input: raw columns (RAW_NUMERICAL_FEATURES + RAW_CATEGORICAL_FEATURES),
    already snake_cased via normalize_columns(). Does not mutate the input.

    Note: these derived features do NOT measurably raise ROC-AUC on this
    dataset (0.8780 -> 0.8778) — the raw features are already close to
    sufficient. They earn their place by powering the dashboard's
    skill-gap radar, per-student readiness scores and the what-if
    simulator, and by making SHAP explanations legible.
    """
    df = df.copy()

    # ── Academic ────────────────────────────────────────────────────────
    df["cgpa_percentage"] = df["cgpa"] * 10
    df["academic_score"] = (df["ssc_marks"] + df["hsc_marks"]) / 2
    df["overall_academic_score"] = (
        df["cgpa_percentage"] + df["ssc_marks"] + df["hsc_marks"]
    ) / 3
    # Smaller 10th-vs-12th spread = a steadier academic record.
    df["academic_consistency"] = 100 - (df["ssc_marks"] - df["hsc_marks"]).abs()
    # Positive = improved between 10th and 12th.
    df["academic_trend"] = df["hsc_marks"] - df["ssc_marks"]

    # ── Skills ──────────────────────────────────────────────────────────
    # soft_skills_rating is 0-5; rescale to 0-100 so it is comparable with
    # aptitude_test_score on the radar and in composites.
    df["soft_skills_scaled"] = df["soft_skills_rating"] * 20
    df["skill_composite"] = (df["aptitude_test_score"] + df["soft_skills_scaled"]) / 2
    # Positive = technically ahead of communication, negative = the reverse.
    # This is the headline "skill gap" the dashboard surfaces.
    df["aptitude_soft_gap"] = df["aptitude_test_score"] - df["soft_skills_scaled"]

    # ── Experience / portfolio ──────────────────────────────────────────
    df["experience_score"] = df["internships"] + df["projects"]
    df["practical_exposure_score"] = (
        df["internships"] * 2 + df["projects"] + df["workshops_certifications"]
    )
    df["upskilling_score"] = df["workshops_certifications"] + df["projects"]
    df["projects_per_internship"] = df["projects"] / (df["internships"] + 1)

    stats = _load_normalization_stats()
    if stats is not None:
        # Frozen training-time constants - identical for a single student,
        # a filtered cohort, or the full dataset.
        internships_max = stats["internships_max"]
        projects_max = stats["projects_max"]
        workshops_max = stats["workshops_certifications_max"]
    else:
        # preprocessing.py has not been (re)run yet, so there is no frozen
        # constant. Falls back to this batch's own max, which is only
        # correct when df IS the full training dataset. Any filtered
        # cohort or single-row caller hitting this branch gets skewed
        # *_normalized values - run preprocessing.py.
        internships_max = df["internships"].max()
        projects_max = df["projects"].max()
        workshops_max = df["workshops_certifications"].max()

    df["internships_normalized"] = _safe_div(df["internships"], internships_max)
    df["projects_normalized"] = _safe_div(df["projects"], projects_max)
    df["workshops_normalized"] = _safe_div(df["workshops_certifications"], workshops_max)
    df["portfolio_strength"] = (
        df["internships_normalized"]
        + df["projects_normalized"]
        + df["workshops_normalized"]
    ) / 3 * 100

    # ── Institutional support flags ─────────────────────────────────────
    df["extracurricular_binary"] = df["extracurricular_activities"].map({"No": 0, "Yes": 1})
    df["placement_training_binary"] = df["placement_training"].map({"No": 0, "Yes": 1})
    df["support_index"] = df["extracurricular_binary"] + df["placement_training_binary"]

    # ── Composite readiness ─────────────────────────────────────────────
    df["interview_readiness_score"] = (
        df["aptitude_test_score"] + df["soft_skills_scaled"] + df["cgpa_percentage"]
    ) / 3
    df["placement_readiness_score"] = (
        df["interview_readiness_score"] * 0.5
        + df["overall_academic_score"] * 0.3
        + df["portfolio_strength"] * 0.2
    )
    return df


ENGINEERED_NUMERICAL_FEATURES = [
    "cgpa_percentage", "academic_score", "overall_academic_score",
    "academic_consistency", "academic_trend",
    "soft_skills_scaled", "skill_composite", "aptitude_soft_gap",
    "experience_score", "practical_exposure_score", "upskilling_score",
    "projects_per_internship",
    "internships_normalized", "projects_normalized", "workshops_normalized",
    "portfolio_strength",
    "extracurricular_binary", "placement_training_binary", "support_index",
    "interview_readiness_score", "placement_readiness_score",
]
ALL_NUMERICAL_FEATURES = RAW_NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES  # 29
