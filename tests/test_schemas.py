"""
tests/test_schemas.py
---------------------
Unit tests for Pydantic request/response schemas.
Tests run without loading any model artifacts.
"""

import pytest
from pydantic import ValidationError

from api.schemas import ModelName, PredictionResponse, StudentInput

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "model": "random_forest",
    "cgpa": 7.7,
    "ssc_marks": 70.0,
    "hsc_marks": 74.0,
    "aptitude_test_score": 80.0,
    "soft_skills_rating": 4.4,
    "internships": 1,
    "projects": 2,
    "workshops_certifications": 1,
    "extracurricular_activities": "Yes",
    "placement_training": "Yes",
}


# ---------------------------------------------------------------------------
# StudentInput — valid cases
# ---------------------------------------------------------------------------


class TestStudentInputValid:
    def test_default_model_is_logistic_regression(self):
        payload = {key: value for key, value in VALID_PAYLOAD.items() if key != "model"}
        assert StudentInput(**payload).model == ModelName.logistic_regression

    def test_full_valid_payload(self):
        s = StudentInput(**VALID_PAYLOAD)
        assert s.cgpa == 7.7
        assert s.model == ModelName.random_forest

    def test_all_model_names_accepted(self):
        for m in ("logistic_regression", "random_forest", "xgboost"):
            s = StudentInput(**{**VALID_PAYLOAD, "model": m})
            assert s.model.value == m

    def test_boundary_values(self):
        s = StudentInput(
            **{
                **VALID_PAYLOAD,
                "ssc_marks": 0.0,
                "hsc_marks": 100.0,
                "cgpa": 0.0,
                "soft_skills_rating": 5.0,
                "internships": 0,
            }
        )
        assert s.ssc_marks == 0.0
        assert s.soft_skills_rating == 5.0

    def test_placement_training_no(self):
        s = StudentInput(**{**VALID_PAYLOAD, "placement_training": "No"})
        assert s.placement_training == "No"

    def test_extracurricular_no(self):
        s = StudentInput(**{**VALID_PAYLOAD, "extracurricular_activities": "No"})
        assert s.extracurricular_activities == "No"


# ---------------------------------------------------------------------------
# StudentInput — invalid cases
# ---------------------------------------------------------------------------


class TestStudentInputInvalid:
    def test_missing_required_field(self):
        bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "cgpa"}
        with pytest.raises(ValidationError):
            StudentInput(**bad)

    def test_cgpa_above_max(self):
        with pytest.raises(ValidationError):
            StudentInput(**{**VALID_PAYLOAD, "cgpa": 10.1})

    def test_ssc_marks_negative(self):
        with pytest.raises(ValidationError):
            StudentInput(**{**VALID_PAYLOAD, "ssc_marks": -1.0})

    def test_internships_negative(self):
        with pytest.raises(ValidationError):
            StudentInput(**{**VALID_PAYLOAD, "internships": -1})

    def test_soft_skills_rating_above_max(self):
        # Rated on a 5-point scale, not 0-100.
        with pytest.raises(ValidationError):
            StudentInput(**{**VALID_PAYLOAD, "soft_skills_rating": 5.1})

    def test_invalid_placement_training(self):
        with pytest.raises(ValidationError):
            StudentInput(**{**VALID_PAYLOAD, "placement_training": "Maybe"})

    def test_invalid_model_name(self):
        with pytest.raises(ValidationError):
            StudentInput(**{**VALID_PAYLOAD, "model": "neural_network"})

    def test_invalid_extracurricular(self):
        with pytest.raises(ValidationError):
            StudentInput(**{**VALID_PAYLOAD, "extracurricular_activities": "Maybe"})


# ---------------------------------------------------------------------------
# PredictionResponse — valid construction
# ---------------------------------------------------------------------------


class TestPredictionResponse:
    def test_valid_placed(self):
        r = PredictionResponse(
            model_used="Random Forest",
            placement_status=1,
            placement_label="Placed",
            probability_placed=0.85,
            probability_not_placed=0.15,
            risk_level="High Probability of Placement (Low Risk)",
        )
        assert r.placement_status == 1
        assert r.probability_placed == pytest.approx(0.85)

    def test_probability_out_of_range(self):
        with pytest.raises(ValidationError):
            PredictionResponse(
                model_used="XGBoost",
                placement_status=1,
                placement_label="Placed",
                probability_placed=1.1,  # > 1.0
                probability_not_placed=0.0,
                risk_level="High Probability of Placement (Low Risk)",
            )
