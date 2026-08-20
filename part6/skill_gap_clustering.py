"""
Role 5 -- Innovation & Research: Skill-Gap Clustering
======================================================
Student Placement Prediction System -- Batch 1

WHAT THIS ANSWERS THAT TAB 2 DOESN'T
--------------------------------------
The dashboard's per-student radar (Tab 2) is a DIAGNOSTIC: it shows one
student's profile against the placed-peer benchmark. It answers "what's
this student's gap?" one student at a time.

This module is a SEGMENTATION: it groups the whole cohort into a small
number of skill-profile archetypes using unsupervised clustering (KMeans),
so a placement cell can ask "how many students look like THIS, and what
does THIS group need?" -- a cohort-level question the per-student view
can't answer by itself.

METHOD: K-Means on standardized skill composites
----------------------------------------------------
Five features, chosen to each represent a genuinely distinct readiness
axis rather than five near-duplicates of "academic strength":

    overall_academic_score  -- CGPA + SSC + HSC combined (one academic
                                number, not three correlated ones)
    aptitude_test_score     -- cognitive/aptitude measure
    soft_skills_scaled      -- communication/soft-skills rating (0-100)
    portfolio_strength      -- internships + projects + certifications,
                                already normalized 0-100 by
                                feature_engineering.py
    support_index           -- institutional engagement (extracurricular
                                + placement training, 0/1/2)

All five come straight out of the canonical engineer_features() -- no
feature reinvented here, per feature_engineering.py's own warning about
what happened last time two pipeline stages defined the same column
differently.

Features are z-scored (StandardScaler) before clustering. K-Means uses
Euclidean distance, so an unscaled feature with a wider numeric range
(e.g. an 0-100 score next to a 0-2 count) would dominate the distance
metric purely because of its units, not because it's more informative --
standardizing first is what makes "distance" mean the same thing on
every axis.

K SELECTION: silhouette score over k = 2..7, elbow (inertia) plotted
alongside it as the second opinion your presentation slide should show,
because "we picked the k with the best silhouette" invites the obvious
follow-up ("what if you'd used the elbow instead?") -- better to show
both curves and let the room see they agree (or explain it if they don't).

ARCHETYPE NAMING: rule-based, not manual. Each cluster's centroid is
compared to the cohort mean on all five axes (in standard-deviation
units); the two axes where a cluster deviates most from the cohort
become its label ("Strong Academics, Low Portfolio", etc.). This keeps
the names reproducible if the pipeline is rerun on new data, rather than
hand-picked once and stale forever -- the same lesson as the README/part3
drift the gap analysis flagged elsewhere in this repo.

INTEGRATION: reports actual placement rate, mean model-predicted
probability, AND mean training uplift (from uplift_modeling.py) per
cluster -- so "which archetype is highest-leverage for a training push"
is answerable directly from this table, not left as a separate exercise.

Run after: preprocessing.py (needs data/processed/normalization_stats.json)
Usage:
    python part5/skill_gap_clustering.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from feature_engineering import TARGET_COLUMN, engineer_features, load_raw_dataset  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"

CLUSTER_FEATURES = [
    "overall_academic_score",
    "aptitude_test_score",
    "soft_skills_scaled",
    "portfolio_strength",
    "support_index",
]

# Short, presentation-friendly names for the archetype labeler.
_AXIS_LABEL = {
    "overall_academic_score": "Academics",
    "aptitude_test_score": "Aptitude",
    "soft_skills_scaled": "Soft Skills",
    "portfolio_strength": "Portfolio",
    "support_index": "Support Engagement",
}


@dataclass
class ClusterResult:
    labels: np.ndarray
    model: KMeans
    scaler: StandardScaler
    k: int
    silhouette: float
    k_search: pd.DataFrame  # silhouette + inertia for every k tried


def select_k(
    X_scaled: np.ndarray, k_range: range = range(2, 8), random_state: int = 42
) -> pd.DataFrame:
    """Fit KMeans for every k in k_range, return silhouette + inertia for
    each so both the elbow and the silhouette curve can go on the same
    slide. Does not pick a winner -- fit_clusters() does that.
    """
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X_scaled)
        sil = silhouette_score(X_scaled, km.labels_)
        rows.append({"k": k, "silhouette": sil, "inertia": km.inertia_})
    return pd.DataFrame(rows)


def fit_clusters(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    k_range: range = range(2, 8),
    random_state: int = 42,
) -> ClusterResult:
    """Standardize the five skill axes, search k by silhouette, fit the
    winner, and return everything needed to label and describe clusters.
    """
    if feature_cols is None:
        feature_cols = CLUSTER_FEATURES

    scaler = StandardScaler().fit(df[feature_cols])
    X_scaled = scaler.transform(df[feature_cols])

    k_search = select_k(X_scaled, k_range=k_range, random_state=random_state)
    best_k = int(k_search.loc[k_search["silhouette"].idxmax(), "k"])

    model = KMeans(n_clusters=best_k, random_state=random_state, n_init=10).fit(X_scaled)
    best_sil = float(k_search.loc[k_search["k"] == best_k, "silhouette"].iloc[0])

    return ClusterResult(
        labels=model.labels_,
        model=model,
        scaler=scaler,
        k=best_k,
        silhouette=best_sil,
        k_search=k_search,
    )


def name_archetypes(
    df: pd.DataFrame,
    result: ClusterResult,
    feature_cols: list[str] | None = None,
    top_n_axes: int = 2,
) -> dict[int, str]:
    """Auto-generate a human-readable name per cluster from how its
    centroid deviates from the cohort mean, in standard-deviation units,
    on the top_n_axes most distinguishing features. Data-driven, so it
    stays correct if this is rerun after a schema or data change -- the
    exact failure mode that hit README/part3 elsewhere in this repo.
    """
    if feature_cols is None:
        feature_cols = CLUSTER_FEATURES

    cohort_mean = df[feature_cols].mean()
    cohort_std = df[feature_cols].std().replace(0, 1.0)

    names: dict[int, str] = {}
    for cluster_id in sorted(set(result.labels)):
        mask = result.labels == cluster_id
        centroid = df.loc[mask, feature_cols].mean()
        z = (centroid - cohort_mean) / cohort_std
        top_axes = z.abs().sort_values(ascending=False).index[:top_n_axes]

        parts = []
        for axis in top_axes:
            direction = "High" if z[axis] >= 0 else "Low"
            parts.append(f"{direction} {_AXIS_LABEL.get(axis, axis)}")
        names[cluster_id] = ", ".join(parts)
    return names


def profile_clusters(
    df: pd.DataFrame,
    result: ClusterResult,
    names: dict[int, str],
    predicted_prob_col: str | None = None,
    uplift_col: str | None = None,
) -> pd.DataFrame:
    """One row per cluster: size, share of cohort, actual placement rate,
    and (if supplied) mean model-predicted probability and mean uplift --
    the join point with batch_predictor.py's scores and
    uplift_modeling.py's CATE estimates.
    """
    work = df.copy()
    work["_cluster"] = result.labels
    placed_binary = (work[TARGET_COLUMN].astype(str) == "Placed").astype(int)

    rows = []
    for cluster_id in sorted(set(result.labels)):
        mask = work["_cluster"] == cluster_id
        row = {
            "cluster": cluster_id,
            "archetype": names.get(cluster_id, f"Cluster {cluster_id}"),
            "n_students": int(mask.sum()),
            "share_of_cohort": round(mask.mean() * 100, 1),
            "actual_placement_rate": round(placed_binary[mask].mean() * 100, 1),
        }
        for axis in CLUSTER_FEATURES:
            row[f"avg_{axis}"] = round(work.loc[mask, axis].mean(), 2)
        if predicted_prob_col and predicted_prob_col in work.columns:
            row["avg_predicted_prob"] = round(work.loc[mask, predicted_prob_col].mean(), 4)
        if uplift_col and uplift_col in work.columns:
            row["avg_training_uplift"] = round(work.loc[mask, uplift_col].mean(), 4)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("actual_placement_rate", ascending=False)


def run_clustering_pipeline(save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end: load -> engineer -> select k -> fit -> name -> profile.
    Also merges in uplift_modeling.py's per-student scores when available,
    so the cluster profile table includes avg_training_uplift out of the
    box. Returns (per_student_df, cluster_profile_df).
    """
    raw = load_raw_dataset()
    df = engineer_features(raw)

    result = fit_clusters(df)
    print(f"\nSelected k={result.k} (silhouette={result.silhouette:.4f})")
    print("\n=== k search (silhouette + inertia) ===")
    print(result.k_search.to_string(index=False))

    names = name_archetypes(df, result)
    print("\n=== Archetypes ===")
    for cid, name in names.items():
        print(f"  Cluster {cid}: {name}")

    per_student = df.copy()
    per_student["cluster"] = result.labels
    per_student["archetype"] = per_student["cluster"].map(names)

    uplift_col = None
    uplift_scores_path = RESULTS_DIR / "uplift_scores.csv"
    if uplift_scores_path.exists():
        uplift_df = pd.read_csv(uplift_scores_path)
        if "uplift" in uplift_df.columns and len(uplift_df) == len(per_student):
            per_student = per_student.reset_index(drop=True)
            per_student["uplift"] = uplift_df["uplift"].reset_index(drop=True)
            uplift_col = "uplift"
        else:
            print(
                "\nNote: found part5/results/uplift_scores.csv but its row count "
                "doesn't match the current dataset -- skipping the merge rather "
                "than risk pairing rows that don't correspond to the same students. "
                "Run uplift_modeling.py again against the current data first."
            )

    profile = profile_clusters(
        per_student, result, names, predicted_prob_col=None, uplift_col=uplift_col
    )
    print("\n=== Cluster profile ===")
    print(profile.to_string(index=False))

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        per_student.to_csv(RESULTS_DIR / "cluster_assignments.csv", index=False)
        profile.to_csv(RESULTS_DIR / "cluster_profile.csv", index=False)
        result.k_search.to_csv(RESULTS_DIR / "k_search.csv", index=False)
        print(f"\nSaved to: {RESULTS_DIR}")

    return per_student, profile


if __name__ == "__main__":
    run_clustering_pipeline()
