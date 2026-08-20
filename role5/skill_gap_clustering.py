"""Deterministic, treatment-safe skill-gap archetype clustering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .features import READINESS_DIMENSIONS, TREATMENT_COLUMN

MIN_CLUSTER_SHARE = 0.05
_DIMENSION_LABELS = {
    "academic_foundation": "academic foundation",
    "academic_consistency": "academic consistency",
    "aptitude_readiness": "aptitude",
    "communication_readiness": "communication",
    "portfolio_readiness": "portfolio",
}


@dataclass(frozen=True)
class SkillGapClusteringResult:
    """Fitted clustering resources and validation evidence."""

    labels: np.ndarray
    model: KMeans
    scaler: StandardScaler
    selected_k: int
    silhouette_score: float
    bootstrap_ari_mean: float
    bootstrap_ari_std: float
    k_search: pd.DataFrame
    archetype_names: dict[int, str]


def _candidate_search(X_scaled: np.ndarray, k_values: range, random_state: int) -> pd.DataFrame:
    rows: list[dict[str, float | int | bool]] = []
    minimum_size = int(np.ceil(len(X_scaled) * MIN_CLUSTER_SHARE))
    for k in k_values:
        if k < 2 or k >= len(X_scaled):
            continue
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20).fit(X_scaled)
        counts = np.bincount(model.labels_, minlength=k)
        rows.append(
            {
                "k": k,
                "silhouette_score": float(silhouette_score(X_scaled, model.labels_)),
                "inertia": float(model.inertia_),
                "minimum_cluster_size": int(counts.min()),
                "minimum_required_size": minimum_size,
                "eligible": bool(counts.min() >= minimum_size),
            }
        )
    return pd.DataFrame(rows)


def _name_archetypes(model: KMeans) -> dict[int, str]:
    """Name each archetype from its largest standardized centroid deficits."""
    names: dict[int, str] = {}
    for cluster_id, centroid in enumerate(model.cluster_centers_):
        deficits = np.argsort(centroid)[:2]
        meaningful = [index for index in deficits if centroid[index] <= -0.25]
        if meaningful:
            gap_names = [_DIMENSION_LABELS[READINESS_DIMENSIONS[index]] for index in meaningful]
            label = " and ".join(f"{name} gap" for name in gap_names)
        else:
            strengths = np.argsort(centroid)[-2:][::-1]
            strength_names = [_DIMENSION_LABELS[READINESS_DIMENSIONS[index]] for index in strengths]
            label = "well-rounded " + " and ".join(strength_names)
        names[cluster_id] = f"Archetype {cluster_id + 1}: {label.capitalize()}"
    return names


def _bootstrap_stability(
    X_scaled: np.ndarray,
    reference_labels: np.ndarray,
    n_clusters: int,
    random_state: int,
    iterations: int,
) -> tuple[float, float]:
    """Measure solution stability with ARI over deterministic bootstrap fits."""
    if iterations <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(random_state)
    sample_size = max(n_clusters, int(np.ceil(len(X_scaled) * 0.80)))
    scores = []
    for _ in range(iterations):
        sample = rng.integers(0, len(X_scaled), size=sample_size)
        boot_model = KMeans(
            n_clusters=n_clusters,
            random_state=int(rng.integers(0, 2**31 - 1)),
            n_init=20,
        ).fit(X_scaled[sample])
        scores.append(adjusted_rand_score(reference_labels, boot_model.predict(X_scaled)))
    return float(np.mean(scores)), float(np.std(scores, ddof=0))


def fit_skill_gap_clusters(
    readiness_frame: pd.DataFrame,
    k_values: range = range(2, 7),
    random_state: int = 42,
    bootstrap_iterations: int = 30,
) -> SkillGapClusteringResult:
    """Fit K-means on five readiness dimensions, selecting a viable k by silhouette."""
    missing = [column for column in READINESS_DIMENSIONS if column not in readiness_frame]
    if missing:
        raise ValueError(f"Readiness frame is missing dimensions: {', '.join(missing)}")
    if readiness_frame.loc[:, READINESS_DIMENSIONS].isna().any().any():
        raise ValueError("Readiness frame contains missing values.")
    if len(readiness_frame) < 40:
        raise ValueError("At least 40 students are required for stable archetype clustering.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(readiness_frame.loc[:, READINESS_DIMENSIONS])
    k_search = _candidate_search(X_scaled, k_values, random_state)
    eligible = k_search[k_search["eligible"]]
    if eligible.empty:
        raise ValueError("No candidate k produced clusters with at least 5% of the cohort.")
    selected_k = int(
        eligible.sort_values(["silhouette_score", "k"], ascending=[False, True]).iloc[0]["k"]
    )
    model = KMeans(n_clusters=selected_k, random_state=random_state, n_init=20).fit(X_scaled)
    labels = model.labels_
    stability_mean, stability_std = _bootstrap_stability(
        X_scaled, labels, selected_k, random_state, bootstrap_iterations
    )
    silhouette = float(k_search.loc[k_search["k"] == selected_k, "silhouette_score"].iloc[0])
    return SkillGapClusteringResult(
        labels=labels,
        model=model,
        scaler=scaler,
        selected_k=selected_k,
        silhouette_score=silhouette,
        bootstrap_ari_mean=stability_mean,
        bootstrap_ari_std=stability_std,
        k_search=k_search,
        archetype_names=_name_archetypes(model),
    )


def attach_archetypes(df: pd.DataFrame, result: SkillGapClusteringResult) -> pd.DataFrame:
    """Attach only post-assignment archetype labels to a copy of the cohort."""
    assigned = df.copy()
    assigned["archetype_id"] = result.labels
    assigned["archetype"] = assigned["archetype_id"].map(result.archetype_names)
    return assigned


def profile_archetypes(assigned_df: pd.DataFrame, readiness_frame: pd.DataFrame) -> pd.DataFrame:
    """Describe archetypes after assignment without influencing their fit."""
    combined = assigned_df.join(readiness_frame)
    rows: list[dict[str, float | int | str]] = []
    for archetype_id, group in combined.groupby("archetype_id", sort=True):
        row: dict[str, float | int | str] = {
            "archetype_id": int(archetype_id),
            "archetype": str(group["archetype"].iloc[0]),
            "students": int(len(group)),
            "cohort_share": float(len(group) / len(combined)),
        }
        for dimension in READINESS_DIMENSIONS:
            row[dimension] = float(group[dimension].mean())
        if "placement_status" in group:
            row["observed_placement_rate"] = float((group["placement_status"] == "Placed").mean())
        if TREATMENT_COLUMN in group:
            row["observed_training_rate"] = float((group[TREATMENT_COLUMN] == "Yes").mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("archetype_id").reset_index(drop=True)
