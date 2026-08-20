"""Integration tests for end-to-end Role 5 reporting and artifact persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from role5.reporting import run_role5_analysis
from role5.train_role5 import save_role5_analysis


def sample_cohort(n_students: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    training = np.tile(["No", "Yes"], n_students // 2)
    score_base = rng.uniform(6.0, 9.5, n_students)
    prob = 1 / (1 + np.exp(-(score_base - 7.5 + (training == "Yes") * 0.5)))
    outcome = np.where(rng.random(n_students) < prob, "Placed", "NotPlaced")
    return pd.DataFrame(
        {
            "student_id": [f"STU_{i:04d}" for i in range(n_students)],
            "cgpa": score_base,
            "ssc_marks": rng.uniform(55.0, 90.0, n_students),
            "hsc_marks": rng.uniform(55.0, 90.0, n_students),
            "aptitude_test_score": rng.uniform(50.0, 95.0, n_students),
            "soft_skills_rating": rng.uniform(2.5, 5.0, n_students),
            "internships": rng.integers(0, 3, n_students),
            "projects": rng.integers(0, 4, n_students),
            "workshops_certifications": rng.integers(0, 4, n_students),
            "extracurricular_activities": rng.choice(["Yes", "No"], n_students),
            "placement_training": training,
            "placement_status": outcome,
        }
    )


def test_run_role5_analysis_generates_all_components():
    cohort = sample_cohort(100)
    analysis = run_role5_analysis(cohort, bootstrap_iterations=5)

    assert len(analysis.readiness_frame) == len(cohort)
    assert len(analysis.archetype_assignments) == len(cohort)
    assert "archetype" in analysis.archetype_assignments.columns
    assert "student_id" in analysis.archetype_assignments.columns
    assert not analysis.archetype_profile.empty
    assert analysis.observational_effects is not None
    assert analysis.clustering.selected_k >= 2


def test_save_role5_analysis_writes_all_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cohort = sample_cohort(80)
    monkeypatch.setattr("role5.train_role5.load_raw_dataset", lambda: cohort)

    out_dir = tmp_path / "role5_output"
    save_role5_analysis(out_dir, bootstrap_iterations=5)

    expected_files = [
        "archetype_assignments.csv",
        "archetype_profile.csv",
        "cluster_k_search.csv",
        "cate_by_archetype.csv",
        "association_scores.csv",
        "observational_cate_scores.csv",
        "balance_before.csv",
        "balance_after.csv",
        "summary.json",
    ]
    for filename in expected_files:
        assert (out_dir / filename).exists(), f"Missing expected artifact: {filename}"

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "selected_k" in summary
    assert "aggregate_ate" in summary
    assert "diagnostic_status" in summary


def test_role5_pipeline_raises_on_invalid_data():
    invalid_cohort = pd.DataFrame({"student_id": ["STU_1", "STU_2"]})
    with pytest.raises((ValueError, KeyError)):
        run_role5_analysis(invalid_cohort, bootstrap_iterations=2)
