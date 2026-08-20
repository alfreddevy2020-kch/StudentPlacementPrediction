"""Validation and scoring helpers for the uploaded-cohort dashboard flow."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from feature_engineering import (
    RAW_CATEGORICAL_FEATURES,
    RAW_NUMERICAL_FEATURES,
    normalize_columns,
)


class CohortPredictor(Protocol):
    """Minimal predictor contract used by the uploaded-cohort workflow."""

    def predict_probabilities(
        self, df: pd.DataFrame, model_name: str | None = None
    ) -> np.ndarray: ...

    @staticmethod
    def classify_risk(probability: float) -> str: ...


REQUIRED_UPLOAD_COLUMNS = RAW_NUMERICAL_FEATURES + RAW_CATEGORICAL_FEATURES


def normalize_uploaded_cohort(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Normalize raw or snake-case headers and report missing input fields."""
    normalized = normalize_columns(df).copy()
    missing = [column for column in REQUIRED_UPLOAD_COLUMNS if column not in normalized]
    return normalized, missing


def score_uploaded_cohort(
    df: pd.DataFrame, predictor: CohortPredictor, model_name: str | None = None
) -> pd.DataFrame:
    """Score each valid uploaded row exactly once and attach safe display fields."""
    probabilities = np.asarray(
        predictor.predict_probabilities(df, model_name=model_name), dtype=float
    )
    if probabilities.shape != (len(df),):
        raise ValueError("The prediction pipeline did not return one probability per uploaded row.")
    if (
        not np.isfinite(probabilities).all()
        or not np.logical_and(probabilities >= 0.0, probabilities <= 1.0).all()
    ):
        raise ValueError("The prediction pipeline returned probabilities outside [0, 1].")

    scored = df.copy()
    scored["placement_prob"] = np.round(probabilities * 100, 1)
    scored["predicted_status"] = np.where(probabilities >= 0.50, "Placed", "Not Placed")
    scored["risk_tier"] = [
        predictor.classify_risk(float(probability)) for probability in probabilities
    ]
    return scored
