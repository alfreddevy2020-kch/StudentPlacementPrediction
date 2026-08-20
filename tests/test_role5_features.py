"""Tests for treatment-safe Role 5 feature construction."""

from __future__ import annotations

import pandas as pd
import pytest

from role5.features import (
    READINESS_DIMENSIONS,
    Role5DataError,
    prepare_baseline_features,
    prepare_readiness_frame,
    prepare_treatment_outcome,
)

STATS = {
    "internships_max": 2.0,
    "projects_max": 3.0,
    "workshops_certifications_max": 3.0,
}


def sample_cohort() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cgpa": [7.0, 8.1, 7.5, 8.4],
            "ssc_marks": [65.0, 80.0, 70.0, 82.0],
            "hsc_marks": [68.0, 82.0, 71.0, 84.0],
            "aptitude_test_score": [66.0, 85.0, 72.0, 88.0],
            "soft_skills_rating": [3.2, 4.5, 3.6, 4.6],
            "internships": [0, 2, 1, 2],
            "projects": [1, 3, 1, 2],
            "workshops_certifications": [0, 3, 1, 2],
            "extracurricular_activities": ["No", "Yes", "No", "Yes"],
            "placement_training": ["No", "Yes", "No", "Yes"],
            "placement_status": ["NotPlaced", "Placed", "NotPlaced", "Placed"],
        }
    )


def test_readiness_frame_has_only_non_treatment_dimensions():
    cohort = sample_cohort()
    baseline = prepare_baseline_features(cohort)
    readiness = prepare_readiness_frame(cohort, STATS)

    assert tuple(readiness.columns) == READINESS_DIMENSIONS
    assert "placement_training" not in readiness
    assert "placement_training_binary" not in readiness
    assert "support_index" not in readiness
    assert "placement_status" not in readiness
    assert "extracurricular_activities_yes" in baseline


def test_changing_treatment_does_not_change_clustering_frame():
    cohort = sample_cohort()
    altered = cohort.copy()
    altered["placement_training"] = ["Yes", "No", "Yes", "No"]

    pd.testing.assert_frame_equal(
        prepare_readiness_frame(cohort, STATS),
        prepare_readiness_frame(altered, STATS),
    )


def test_treatment_pipeline_rejects_missing_treatment_arm():
    cohort = sample_cohort()
    cohort["placement_training"] = "Yes"

    with pytest.raises(Role5DataError, match="Both placement-training arms"):
        prepare_treatment_outcome(cohort)
