"""Tests for observational effect diagnostics and failure gates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from role5.uplift_modeling import run_observational_effect_pipeline


def synthetic_cohort(n_students: int = 240, deterministic_treatment: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(10)
    treatment = np.tile(["No", "Yes"], n_students // 2)
    signal = (
        np.where(treatment == "Yes", 1.0, 0.0)
        if deterministic_treatment
        else rng.normal(0, 1, n_students)
    )
    outcome_probability = 1 / (1 + np.exp(-(signal * 0.5 + (treatment == "Yes") * 0.2)))
    outcome = np.where(rng.random(n_students) < outcome_probability, "Placed", "NotPlaced")
    magnitude = 100.0 if deterministic_treatment else 1.0
    return pd.DataFrame(
        {
            "cgpa": 7.5 + signal * magnitude,
            "ssc_marks": 70 + signal * magnitude,
            "hsc_marks": 72 + signal * magnitude,
            "aptitude_test_score": 74 + signal * magnitude,
            "soft_skills_rating": 3.8 + signal * 0.1,
            "internships": (signal > 0).astype(int),
            "projects": 1 + (signal > 0).astype(int),
            "workshops_certifications": 1 + (signal > 0).astype(int),
            "extracurricular_activities": np.where(signal > 0, "Yes", "No"),
            "placement_training": treatment,
            "placement_status": outcome,
        }
    )


def test_effect_pipeline_returns_cohort_evidence_and_schema():
    cohort = synthetic_cohort()
    archetypes = pd.Series(np.where(cohort.index % 2 == 0, "A", "B"), index=cohort.index)
    result = run_observational_effect_pipeline(
        cohort, archetypes=archetypes, folds=3, bootstrap_iterations=10
    )

    assert len(result.cate_scores) == len(cohort)
    assert {"propensity_score", "observational_cate"}.issubset(result.cate_scores)
    assert {"association_difference"}.issubset(result.association_scores)
    assert set(result.cate_by_archetype["archetype"]) == {"A", "B"}
    assert result.diagnostics.treatment_count == result.diagnostics.control_count


def test_near_deterministic_treatment_emits_overlap_warning_state():
    result = run_observational_effect_pipeline(
        synthetic_cohort(deterministic_treatment=True), folds=3, bootstrap_iterations=5
    )

    assert result.diagnostics.status in {"no_overlap", "insufficient_evidence"}
    assert result.diagnostics.clipped_proportion > 0.10
