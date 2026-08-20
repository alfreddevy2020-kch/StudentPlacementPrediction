"""Determinism and leakage tests for skill-gap archetypes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from role5.features import READINESS_DIMENSIONS
from role5.skill_gap_clustering import attach_archetypes, fit_skill_gap_clusters


def synthetic_readiness(n_students: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    group = np.repeat([0, 1], n_students // 2)
    return pd.DataFrame(
        {
            "academic_foundation": 72 + group * 14 + rng.normal(0, 2, n_students),
            "academic_consistency": 84 + rng.normal(0, 3, n_students),
            "aptitude_readiness": 68 + group * 16 + rng.normal(0, 2, n_students),
            "communication_readiness": 66 + group * 14 + rng.normal(0, 2, n_students),
            "portfolio_readiness": 30 + group * 45 + rng.normal(0, 4, n_students),
        }
    )


def test_clustering_is_deterministic_and_uses_only_readiness_dimensions():
    readiness = synthetic_readiness()
    first = fit_skill_gap_clusters(readiness, bootstrap_iterations=3)
    second = fit_skill_gap_clusters(readiness, bootstrap_iterations=3)

    assert tuple(readiness.columns) == READINESS_DIMENSIONS
    assert first.selected_k == second.selected_k
    np.testing.assert_array_equal(first.labels, second.labels)
    assert first.k_search["eligible"].any()


def test_every_student_receives_one_archetype():
    readiness = synthetic_readiness()
    result = fit_skill_gap_clusters(readiness, bootstrap_iterations=0)
    cohort = pd.DataFrame({"student_id": range(len(readiness))})
    assigned = attach_archetypes(cohort, result)

    assert len(assigned) == len(cohort)
    assert assigned["archetype"].notna().all()
