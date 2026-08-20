"""Cohort-level Role 5 reporting assembled from safe analysis components."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .features import prepare_readiness_frame
from .skill_gap_clustering import (
    SkillGapClusteringResult,
    attach_archetypes,
    fit_skill_gap_clusters,
    profile_archetypes,
)
from .uplift_modeling import ObservationalEffectResult, run_observational_effect_pipeline


@dataclass(frozen=True)
class Role5Analysis:
    """Cached, cohort-level results used by the Programme insights view."""

    readiness_frame: pd.DataFrame
    clustering: SkillGapClusteringResult
    archetype_assignments: pd.DataFrame
    archetype_profile: pd.DataFrame
    observational_effects: ObservationalEffectResult


def run_role5_analysis(
    raw_cohort: pd.DataFrame,
    normalization_stats: dict | None = None,
    random_state: int = 42,
    bootstrap_iterations: int = 200,
) -> Role5Analysis:
    """Run deterministic clustering and guarded observational evidence.

    `raw_cohort` is copied before any labels are attached. The caller should
    pass only the built-in reference cohort; uploaded cohorts remain separate.
    """
    source = raw_cohort.copy()
    readiness = prepare_readiness_frame(source, normalization_stats)
    clustering = fit_skill_gap_clusters(readiness, random_state=random_state)
    assigned = attach_archetypes(source, clustering)
    profile = profile_archetypes(assigned, readiness)
    effects = run_observational_effect_pipeline(
        source,
        archetypes=assigned["archetype"],
        random_state=random_state,
        bootstrap_iterations=bootstrap_iterations,
    )
    archetype_effects = effects.cate_by_archetype.rename(
        columns={"students": "effect_estimate_students"}
    )
    profile = profile.merge(archetype_effects, on="archetype", how="left")
    columns = [
        column for column in ("student_id", "archetype_id", "archetype") if column in assigned
    ]
    assignments = assigned.loc[:, columns].copy()
    return Role5Analysis(
        readiness_frame=readiness,
        clustering=clustering,
        archetype_assignments=assignments,
        archetype_profile=profile,
        observational_effects=effects,
    )
