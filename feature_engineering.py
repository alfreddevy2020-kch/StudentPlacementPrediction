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

These stats are a fitted parameter of the pipeline, exactly like
preprocessor.joblib, so scripts/package_model.py copies them into each
production bundle and records their checksum in the manifest. The API
loads that per-bundle copy, which keeps a served model pinned to the
scale it was trained on and makes the bundle self-contained on a fresh
clone (data/processed/ is gitignored).

If no stats are available, engineer_features() raises rather than
inferring a maximum from the batch at hand - see the comment at the
raise for why that fallback was removed.
"""
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent
NORM_STATS_FILENAME = "normalization_stats.json"
NORM_STATS_PATH = _REPO_ROOT / "data" / "processed" / NORM_STATS_FILENAME
RAW_DATASET_PATH = _REPO_ROOT / "data" / "raw" / "student_placement.csv"

# Every key a stats file must carry to be usable.
NORM_STATS_KEYS = (
    "internships_max",
    "projects_max",
    "workshops_certifications_max",
)

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

# Human-readable class names, ordered so index == encoded target value.
# Every downstream report (classification_report target_names, confusion-matrix
# tick labels, predicted-label strings) reads this instead of repeating a
# literal ["Not Placed", "Placed"], so relabelling the target is a one-line
# change here.
CLASS_LABELS = [
    label for label, _ in sorted(TARGET_MAP.items(), key=lambda kv: kv[1])
]

# Levels carried by RAW_CATEGORICAL_FEATURES, and their encoding. Used both by
# engineer_features() and by make_sample_student().
BINARY_LEVEL_MAP = {"No": 0, "Yes": 1}


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
    # newline="\n" is required, not cosmetic: this file is SHA-256'd into
    # every production manifest. Python's text mode would emit CRLF on
    # Windows, so the hash recorded at packaging time would not match the
    # same file checked out on Linux CI.
    with open(NORM_STATS_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")
    _load_normalization_stats.cache_clear()
    return stats


def load_normalization_stats(path: Path | str) -> dict:
    """Read and validate a normalization-stats file from an explicit path.

    Used by api/predictor.py to load the copy shipped inside a production
    bundle, so served predictions use the stats that were frozen when that
    exact model was trained.
    """
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
    """Read the repo-local stats written by preprocessing.py, cached for the
    process lifetime. Returns None if preprocessing.py has never been run."""
    if NORM_STATS_PATH.exists():
        return load_normalization_stats(NORM_STATS_PATH)
    return None


def _safe_div(numerator, denominator):
    """Divide by a frozen constant, guarding a degenerate zero maximum."""
    return numerator / denominator if denominator else 0.0


def engineer_features(df: pd.DataFrame, stats: dict | None = None) -> pd.DataFrame:
    """Adds the 21 derived columns preprocessing.py's ColumnTransformer expects.

    Input: raw columns (RAW_NUMERICAL_FEATURES + RAW_CATEGORICAL_FEATURES),
    already snake_cased via normalize_columns(). Does not mutate the input.

    Parameters
    ----------
    stats : dict, optional
        Frozen normalization maxima. Pass the copy from a production bundle
        to guarantee the served model sees the scale it was trained on.
        When omitted, the repo-local file written by preprocessing.py is
        used; if that is absent this raises rather than guessing, because
        an inferred maximum silently corrupts every *_normalized feature.

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

    if stats is None:
        stats = _load_normalization_stats()
    if stats is None:
        # Deriving the maximum from the batch at hand is only correct when
        # that batch IS the full training set. For a cohort or a single row
        # it divides each value by itself, so every *_normalized feature
        # pins to 1.0 and portfolio_strength reads 100 instead of its true
        # value - wrong answers with no error. Refuse instead.
        raise RuntimeError(
            f"Normalization stats not found at {NORM_STATS_PATH}.\n"
            "engineer_features() will not infer them from the input, because "
            "an inferred maximum silently corrupts every *_normalized "
            "feature.\n"
            "Fix: run `python preprocessing.py` to regenerate them, or pass "
            "stats= explicitly (production bundles ship their own copy)."
        )

    # Frozen training-time constants - identical for a single student,
    # a filtered cohort, or the full dataset.
    internships_max = stats["internships_max"]
    projects_max = stats["projects_max"]
    workshops_max = stats["workshops_certifications_max"]

    df["internships_normalized"] = _safe_div(df["internships"], internships_max)
    df["projects_normalized"] = _safe_div(df["projects"], projects_max)
    df["workshops_normalized"] = _safe_div(df["workshops_certifications"], workshops_max)
    df["portfolio_strength"] = (
        df["internships_normalized"]
        + df["projects_normalized"]
        + df["workshops_normalized"]
    ) / 3 * 100

    # ── Institutional support flags ─────────────────────────────────────
    df["extracurricular_binary"] = df["extracurricular_activities"].map(BINARY_LEVEL_MAP)
    df["placement_training_binary"] = df["placement_training"].map(BINARY_LEVEL_MAP)
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


def make_sample_student(position: float = 0.75) -> pd.DataFrame:
    """Build one representative raw student row, derived from the schema above.

    Demo predictions and smoke tests used to carry their own literal dict of
    feature values. Every one of those had to be hand-edited on a schema
    change, and a missed edit surfaced as a confusing KeyError deep inside
    engineer_features(). Building the row from RAW_NUMERICAL_FEATURES,
    RAW_CATEGORICAL_FEATURES and FEATURE_RANGES means those callers follow
    this module automatically.

    Parameters
    ----------
    position : float
        Where in each feature's observed training range the value sits.
        0.0 is the weakest student the model has seen, 1.0 the strongest,
        and the 0.75 default gives a plausibly strong candidate. Values
        outside [0, 1] would be extrapolation, so they are rejected.

    Returns
    -------
    pd.DataFrame
        A single row carrying exactly ALL_RAW_FEATURES, ready for
        engineer_features().
    """
    if not 0.0 <= position <= 1.0:
        raise ValueError(
            f"position must be within [0, 1] (the observed training range), got {position}"
        )

    missing = [c for c in RAW_NUMERICAL_FEATURES if c not in FEATURE_RANGES]
    if missing:
        raise RuntimeError(
            f"FEATURE_RANGES is missing entries for {missing}. Add them when "
            "adding a raw numerical feature - make_sample_student() and the "
            "dashboard sliders both read from it."
        )

    row: dict = {}
    for column in RAW_NUMERICAL_FEATURES:
        low, high = FEATURE_RANGES[column]
        value = low + (high - low) * position
        # Count-style features are declared with integer bounds; keep them
        # integral so the row stays a realistic student rather than 1.5
        # internships.
        if isinstance(low, int) and isinstance(high, int):
            row[column] = int(round(value))
        else:
            row[column] = round(float(value), 2)

    # Highest-encoded level, i.e. "Yes" - matches the strong-candidate default.
    top_level = max(BINARY_LEVEL_MAP, key=lambda k: BINARY_LEVEL_MAP[k])
    for column in RAW_CATEGORICAL_FEATURES:
        row[column] = top_level

    return pd.DataFrame([row])
