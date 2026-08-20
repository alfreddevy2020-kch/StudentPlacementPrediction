"""Observational treatment-effect evidence with explicit diagnostic gates.

This module estimates an observational association under stated assumptions.
It does not prove that placement training causes a placement outcome and it
never produces an individual training-seat recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .features import Role5DataError, prepare_treatment_outcome

PROPENSITY_CLIP = 0.05
MIN_ARM_SIZE = 100
MAX_CLIPPED_PROPORTION = 0.10
MIN_OVERLAP_PROPORTION = 0.50
MAX_WEIGHTED_SMD = 0.10
MIN_EFFECTIVE_SAMPLE_RATIO = 0.25


@dataclass(frozen=True)
class AssociationTLearner:
    """Two-model association baseline, intentionally not a causal model."""

    scaler: StandardScaler
    control_model: LogisticRegression
    treated_model: LogisticRegression
    feature_columns: tuple[str, ...]


@dataclass(frozen=True)
class CrossFittedRLearner:
    """Cross-fitted propensity/outcome nuisances plus a residual CATE model."""

    scaler: StandardScaler
    cate_model: Ridge
    feature_columns: tuple[str, ...]
    propensity_scores: np.ndarray
    outcome_scores: np.ndarray
    cate_scores: np.ndarray


@dataclass(frozen=True)
class ObservationalDiagnostics:
    """Balance, overlap, and sample-size evidence required to interpret estimates."""

    treatment_count: int
    control_count: int
    propensity_min: float
    propensity_max: float
    overlap_proportion: float
    clipped_proportion: float
    effective_sample_size: float
    treatment_effective_sample_size: float
    control_effective_sample_size: float
    balance_before: pd.DataFrame
    balance_after: pd.DataFrame
    status: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ObservationalEffectResult:
    """Cohort-level effect estimates and their diagnostics."""

    association_scores: pd.DataFrame
    cate_scores: pd.DataFrame
    aggregate_ate: float
    ate_ci_lower: float
    ate_ci_upper: float
    cate_by_archetype: pd.DataFrame
    diagnostics: ObservationalDiagnostics


def _validate_outcome_variation(treatment: pd.Series, outcome: pd.Series) -> None:
    for arm, label in ((0, "control"), (1, "treatment")):
        if outcome[treatment == arm].nunique() < 2:
            raise Role5DataError(
                f"The {label} arm contains only one outcome class; it cannot support modelling."
            )


def _safe_logistic_regression(random_state: int) -> LogisticRegression:
    return LogisticRegression(max_iter=2000, random_state=random_state)


def fit_association_t_learner(
    X: pd.DataFrame, treatment: pd.Series, outcome: pd.Series, random_state: int = 42
) -> AssociationTLearner:
    """Fit a transparent T-learner association baseline.

    This remains labelled associational even though it generates differences
    between arm-specific outcome models. It has no cross-fitting, overlap, or
    causal-identification claim.
    """
    _validate_outcome_variation(treatment, outcome)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    control_model = _safe_logistic_regression(random_state)
    treated_model = _safe_logistic_regression(random_state)
    control_model.fit(X_scaled[treatment.to_numpy() == 0], outcome[treatment.to_numpy() == 0])
    treated_model.fit(X_scaled[treatment.to_numpy() == 1], outcome[treatment.to_numpy() == 1])
    return AssociationTLearner(
        scaler=scaler,
        control_model=control_model,
        treated_model=treated_model,
        feature_columns=tuple(X.columns),
    )


def score_association_t_learner(learner: AssociationTLearner, X: pd.DataFrame) -> pd.DataFrame:
    """Return an explicitly non-causal two-model association difference."""
    X_scaled = learner.scaler.transform(X.loc[:, learner.feature_columns])
    control = learner.control_model.predict_proba(X_scaled)[:, 1]
    treated = learner.treated_model.predict_proba(X_scaled)[:, 1]
    return pd.DataFrame(
        {
            "association_control_probability": control,
            "association_treated_probability": treated,
            "association_difference": treated - control,
        },
        index=X.index,
    )


def _weighted_mean_and_variance(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return 0.0, 0.0
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    return mean, variance


def _standardized_mean_difference(
    values: np.ndarray, treatment: np.ndarray, weights: np.ndarray
) -> float:
    treated_values, control_values = values[treatment == 1], values[treatment == 0]
    treated_weights, control_weights = weights[treatment == 1], weights[treatment == 0]
    treated_mean, treated_variance = _weighted_mean_and_variance(treated_values, treated_weights)
    control_mean, control_variance = _weighted_mean_and_variance(control_values, control_weights)
    pooled_std = np.sqrt((treated_variance + control_variance) / 2)
    return 0.0 if pooled_std == 0 else float((treated_mean - control_mean) / pooled_std)


def standardized_mean_differences(
    X: pd.DataFrame, treatment: pd.Series, weights: np.ndarray | None = None
) -> pd.DataFrame:
    """Calculate covariate SMDs before or after propensity weighting."""
    treatment_values = treatment.to_numpy(dtype=int)
    if weights is None:
        weights = np.ones(len(X), dtype=float)
    rows = []
    for feature in X.columns:
        values = X[feature].to_numpy(dtype=float)
        rows.append(
            {
                "feature": feature,
                "smd": _standardized_mean_difference(values, treatment_values, weights),
            }
        )
    return pd.DataFrame(rows).sort_values("smd", key=lambda series: series.abs(), ascending=False)


def _effective_sample_size(weights: np.ndarray) -> float:
    denominator = float(np.square(weights).sum())
    return 0.0 if denominator == 0 else float(np.square(weights.sum()) / denominator)


def _diagnostics(
    X: pd.DataFrame, treatment: pd.Series, propensity_scores: np.ndarray
) -> ObservationalDiagnostics:
    treatment_values = treatment.to_numpy(dtype=int)
    clipped = np.clip(propensity_scores, PROPENSITY_CLIP, 1 - PROPENSITY_CLIP)
    inverse_probability_weights = np.where(treatment_values == 1, 1 / clipped, 1 / (1 - clipped))
    balance_before = standardized_mean_differences(X, treatment)
    balance_after = standardized_mean_differences(X, treatment, inverse_probability_weights)
    treated_weights = inverse_probability_weights[treatment_values == 1]
    control_weights = inverse_probability_weights[treatment_values == 0]
    treatment_count = int(treatment_values.sum())
    control_count = int(len(treatment_values) - treatment_count)
    overlap = (propensity_scores >= PROPENSITY_CLIP) & (propensity_scores <= 1 - PROPENSITY_CLIP)
    clipped_proportion = float((~overlap).mean())
    overlap_proportion = float(overlap.mean())
    max_weighted_smd = float(balance_after["smd"].abs().max())
    treatment_ess = _effective_sample_size(treated_weights)
    control_ess = _effective_sample_size(control_weights)
    warnings: list[str] = []
    if min(treatment_count, control_count) < MIN_ARM_SIZE:
        warnings.append("A treatment arm has fewer than 100 students.")
    if overlap_proportion < MIN_OVERLAP_PROPORTION:
        warnings.append("Less than half of the cohort has adequate propensity-score overlap.")
    if clipped_proportion > MAX_CLIPPED_PROPORTION:
        warnings.append("More than 10% of propensity scores require overlap clipping.")
    if max_weighted_smd > MAX_WEIGHTED_SMD:
        warnings.append(
            "Observed covariate balance remains above the |SMD| 0.10 target after weighting."
        )
    if (
        treatment_ess < treatment_count * MIN_EFFECTIVE_SAMPLE_RATIO
        or control_ess < control_count * MIN_EFFECTIVE_SAMPLE_RATIO
    ):
        warnings.append("Propensity weighting leaves too little effective sample size in an arm.")
    if overlap_proportion == 0:
        status = "no_overlap"
    elif warnings:
        status = "insufficient_evidence"
    else:
        status = "diagnostics_passed"
    return ObservationalDiagnostics(
        treatment_count=treatment_count,
        control_count=control_count,
        propensity_min=float(propensity_scores.min()),
        propensity_max=float(propensity_scores.max()),
        overlap_proportion=overlap_proportion,
        clipped_proportion=clipped_proportion,
        effective_sample_size=_effective_sample_size(inverse_probability_weights),
        treatment_effective_sample_size=treatment_ess,
        control_effective_sample_size=control_ess,
        balance_before=balance_before,
        balance_after=balance_after,
        status=status,
        warnings=tuple(warnings),
    )


def fit_cross_fitted_r_learner(
    X: pd.DataFrame,
    treatment: pd.Series,
    outcome: pd.Series,
    folds: int = 5,
    random_state: int = 42,
) -> CrossFittedRLearner:
    """Fit a residualized R-learner using cross-fitted nuisance models.

    Each held-out fold gets its propensity P(T=1|X) and pooled outcome
    E(Y|X) from models trained without that fold. The final Ridge model learns
    the residualized CATE signal with squared treatment-residual weights.
    """
    _validate_outcome_variation(treatment, outcome)
    treatment_counts = treatment.value_counts()
    n_splits = min(folds, int(treatment_counts.min()))
    if n_splits < 2:
        raise Role5DataError("At least two students in each treatment arm are required.")

    values = X.to_numpy(dtype=float)
    treatment_values = treatment.to_numpy(dtype=int)
    outcome_values = outcome.to_numpy(dtype=int)
    propensity_scores = np.empty(len(X), dtype=float)
    outcome_scores = np.empty(len(X), dtype=float)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_index, holdout_index) in enumerate(splitter.split(values, treatment_values)):
        propensity_model = _safe_logistic_regression(random_state + fold)
        outcome_model = _safe_logistic_regression(random_state + 100 + fold)
        propensity_model.fit(values[train_index], treatment_values[train_index])
        outcome_model.fit(values[train_index], outcome_values[train_index])
        propensity_scores[holdout_index] = propensity_model.predict_proba(values[holdout_index])[
            :, 1
        ]
        outcome_scores[holdout_index] = outcome_model.predict_proba(values[holdout_index])[:, 1]

    feature_scaler = StandardScaler()
    scaled_values = feature_scaler.fit_transform(values)
    treatment_residual = treatment_values - propensity_scores
    outcome_residual = outcome_values - outcome_scores
    valid = np.abs(treatment_residual) > 1e-4
    if valid.sum() < max(20, len(X) // 10):
        raise Role5DataError("Treatment residuals are too small to estimate heterogeneous effects.")
    pseudo_outcome = outcome_residual[valid] / treatment_residual[valid]
    cate_model = Ridge(alpha=10.0)
    cate_model.fit(
        scaled_values[valid],
        pseudo_outcome,
        sample_weight=np.square(treatment_residual[valid]),
    )
    cate_scores = cate_model.predict(scaled_values)
    return CrossFittedRLearner(
        scaler=feature_scaler,
        cate_model=cate_model,
        feature_columns=tuple(X.columns),
        propensity_scores=propensity_scores,
        outcome_scores=outcome_scores,
        cate_scores=cate_scores,
    )


def _bootstrap_ate_ci(
    cate_scores: np.ndarray, iterations: int, random_state: int
) -> tuple[float, float]:
    if iterations < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, len(cate_scores), size=(iterations, len(cate_scores)))
    bootstrap_means = cate_scores[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(lower), float(upper)


def run_observational_effect_pipeline(
    df: pd.DataFrame,
    archetypes: pd.Series | None = None,
    folds: int = 5,
    bootstrap_iterations: int = 200,
    random_state: int = 42,
) -> ObservationalEffectResult:
    """Estimate cohort-level observational evidence and publish diagnostics.

    The aggregate ATE is the average cross-fitted CATE. Its confidence interval
    is a bootstrap interval over the fitted cohort CATE distribution; it does
    not remove bias from unmeasured selection factors.
    """
    X, treatment, outcome = prepare_treatment_outcome(df)
    association = fit_association_t_learner(X, treatment, outcome, random_state)
    association_scores = score_association_t_learner(association, X)
    r_learner = fit_cross_fitted_r_learner(
        X, treatment, outcome, folds=folds, random_state=random_state
    )
    diagnostics = _diagnostics(X, treatment, r_learner.propensity_scores)
    cate_scores = pd.DataFrame(
        {
            "propensity_score": r_learner.propensity_scores,
            "outcome_score": r_learner.outcome_scores,
            "observational_cate": r_learner.cate_scores,
        },
        index=df.index,
    )
    aggregate_ate = float(cate_scores["observational_cate"].mean())
    ci_lower, ci_upper = _bootstrap_ate_ci(
        r_learner.cate_scores, bootstrap_iterations, random_state
    )
    if archetypes is None:
        cate_by_archetype = pd.DataFrame()
    else:
        if not archetypes.index.equals(cate_scores.index):
            raise ValueError("Archetypes must share the source cohort index.")
        cate_by_archetype = (
            cate_scores.assign(archetype=archetypes)
            .groupby("archetype", dropna=False)["observational_cate"]
            .agg(students="count", estimated_ate="mean")
            .reset_index()
            .sort_values("estimated_ate", ascending=False)
            .reset_index(drop=True)
        )
    return ObservationalEffectResult(
        association_scores=association_scores,
        cate_scores=cate_scores,
        aggregate_ate=aggregate_ate,
        ate_ci_lower=ci_lower,
        ate_ci_upper=ci_upper,
        cate_by_archetype=cate_by_archetype,
        diagnostics=diagnostics,
    )
