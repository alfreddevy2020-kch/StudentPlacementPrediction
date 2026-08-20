"""Validated, treatment-safe feature frames for Role 5 analyses."""

from __future__ import annotations

import pandas as pd

from feature_engineering import (
    NORM_STATS_PATH,
    TARGET_COLUMN,
    TARGET_MAP,
    load_normalization_stats,
)

TREATMENT_COLUMN = "placement_training"

BASELINE_NUMERIC_FEATURES = (
    "cgpa",
    "ssc_marks",
    "hsc_marks",
    "aptitude_test_score",
    "soft_skills_rating",
    "internships",
    "projects",
    "workshops_certifications",
)
BASELINE_CATEGORICAL_FEATURES = ("extracurricular_activities",)
BASELINE_FEATURES = BASELINE_NUMERIC_FEATURES + BASELINE_CATEGORICAL_FEATURES

READINESS_DIMENSIONS = (
    "academic_foundation",
    "academic_consistency",
    "aptitude_readiness",
    "communication_readiness",
    "portfolio_readiness",
)


class Role5DataError(ValueError):
    """Raised when data cannot support a safe Role 5 analysis."""


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise Role5DataError(f"Missing required Role 5 columns: {', '.join(missing)}")


def prepare_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return observed pre-training covariates and reject invalid values.

    The returned frame deliberately excludes placement training, outcome,
    treatment-derived fields, and `support_index`. This makes it safe for both
    clustering and observational adjustment.
    """
    _require_columns(df, BASELINE_FEATURES)
    result = pd.DataFrame(index=df.index)
    for column in BASELINE_NUMERIC_FEATURES:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any():
            raise Role5DataError(f"Baseline feature '{column}' contains missing or invalid values.")
        result[column] = values.astype(float)

    extracurricular = df["extracurricular_activities"].astype(str)
    invalid = sorted(set(extracurricular) - {"Yes", "No"})
    if invalid:
        raise Role5DataError(
            f"extracurricular_activities must contain only Yes/No; found {', '.join(invalid)}."
        )
    result["extracurricular_activities_yes"] = (extracurricular == "Yes").astype(int)
    return result


def prepare_treatment_outcome(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build baseline X, binary treatment T, and binary outcome Y.

    Both treatment arms and valid placement labels are mandatory. This is a
    failure gate: an analysis with a missing arm or an unknown outcome is not
    estimated as a misleading zero-effect result.
    """
    _require_columns(df, (TREATMENT_COLUMN, TARGET_COLUMN))
    X = prepare_baseline_features(df)
    treatment_raw = df[TREATMENT_COLUMN].astype(str)
    invalid_treatment = sorted(set(treatment_raw) - {"Yes", "No"})
    if invalid_treatment:
        raise Role5DataError(
            f"placement_training must contain only Yes/No; found {', '.join(invalid_treatment)}."
        )
    treatment = (treatment_raw == "Yes").astype(int).rename("treatment")
    if treatment.nunique() != 2:
        raise Role5DataError("Both placement-training arms are required for an effect estimate.")

    outcome = df[TARGET_COLUMN].map(TARGET_MAP)
    if outcome.isna().any():
        invalid_outcomes = sorted(df.loc[outcome.isna(), TARGET_COLUMN].astype(str).unique())
        raise Role5DataError(
            f"{TARGET_COLUMN} must contain only {list(TARGET_MAP)}; found {invalid_outcomes}."
        )
    return X, treatment, outcome.astype(int).rename("outcome")


def prepare_readiness_frame(
    df: pd.DataFrame, normalization_stats: dict | None = None
) -> pd.DataFrame:
    """Create the five 0–100 readiness dimensions used for clustering.

    `placement_status`, `placement_training`, its binary encoding, and
    `support_index` are not read or derived here. Outcome and treatment may be
    described *after* archetype assignment, but never influence distance.
    """
    baseline = prepare_baseline_features(df)
    if normalization_stats is None:
        normalization_stats = load_normalization_stats(NORM_STATS_PATH)

    required_stats = (
        "internships_max",
        "projects_max",
        "workshops_certifications_max",
    )
    missing_stats = [key for key in required_stats if key not in normalization_stats]
    if missing_stats:
        raise Role5DataError(f"Normalization stats are missing: {', '.join(missing_stats)}")

    portfolio_components = (
        baseline["internships"] / max(float(normalization_stats["internships_max"]), 1.0)
        + baseline["projects"] / max(float(normalization_stats["projects_max"]), 1.0)
        + baseline["workshops_certifications"]
        / max(float(normalization_stats["workshops_certifications_max"]), 1.0)
    ) / 3

    readiness = pd.DataFrame(index=df.index)
    readiness["academic_foundation"] = (
        baseline["cgpa"] * 10 + baseline["ssc_marks"] + baseline["hsc_marks"]
    ) / 3
    readiness["academic_consistency"] = 100 - (baseline["ssc_marks"] - baseline["hsc_marks"]).abs()
    readiness["aptitude_readiness"] = baseline["aptitude_test_score"]
    readiness["communication_readiness"] = baseline["soft_skills_rating"] * 20
    readiness["portfolio_readiness"] = portfolio_components.clip(0, 1) * 100
    return readiness.loc[:, READINESS_DIMENSIONS]
