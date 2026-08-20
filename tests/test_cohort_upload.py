"""Regression tests for the preserved uploaded-cohort scoring workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from frontend.cohort_upload import normalize_uploaded_cohort, score_uploaded_cohort


class RecordingPredictor:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str | None]] = []

    def predict_probabilities(self, df: pd.DataFrame, model_name: str | None = None) -> np.ndarray:
        self.calls.append((len(df), model_name))
        return np.linspace(0.2, 0.8, len(df))

    @staticmethod
    def classify_risk(probability: float) -> str:
        return "High Risk" if probability < 0.5 else "Interview Ready"


def _raw_upload() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "StudentID": [101, 102],
            "CGPA": [7.0, 8.2],
            "SSC_Marks": [70.0, 80.0],
            "HSC_Marks": [72.0, 82.0],
            "AptitudeTestScore": [70.0, 82.0],
            "SoftSkillsRating": [3.5, 4.4],
            "Internships": [0, 1],
            "Projects": [1, 2],
            "Workshops/Certifications": [1, 2],
            "ExtracurricularActivities": ["No", "Yes"],
            "PlacementTraining": ["No", "Yes"],
        }
    )


@pytest.mark.parametrize("headers", ["raw", "snake_case"])
def test_valid_upload_normalizes_and_scores_every_row_once(headers: str):
    upload = _raw_upload()
    if headers == "snake_case":
        upload = upload.rename(
            columns={
                "StudentID": "student_id",
                "CGPA": "cgpa",
                "SSC_Marks": "ssc_marks",
                "HSC_Marks": "hsc_marks",
                "AptitudeTestScore": "aptitude_test_score",
                "SoftSkillsRating": "soft_skills_rating",
                "Internships": "internships",
                "Projects": "projects",
                "Workshops/Certifications": "workshops_certifications",
                "ExtracurricularActivities": "extracurricular_activities",
                "PlacementTraining": "placement_training",
            }
        )

    normalized, missing = normalize_uploaded_cohort(upload)
    predictor = RecordingPredictor()
    scored = score_uploaded_cohort(normalized, predictor, "Logistic Regression")

    assert not missing
    assert predictor.calls == [(len(upload), "Logistic Regression")]
    assert len(scored) == len(upload)
    assert scored["placement_prob"].between(0.0, 100.0).all()
    assert {"predicted_status", "risk_tier"}.issubset(scored.columns)


def test_missing_required_upload_columns_are_reported_without_scoring():
    normalized, missing = normalize_uploaded_cohort(_raw_upload().drop(columns="CGPA"))

    assert "cgpa" in missing
    assert "cgpa" not in normalized
