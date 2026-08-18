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
    "ssc_percentage": 75.5,
    "hsc_percentage": 78.0,
    "degree_percentage": 72.0,
    "cgpa": 8.2,
    "attendance_percentage": 90.0,
    "backlogs": 0,
    "entrance_exam_score": 85.0,
    "technical_skill_score": 80.0,
    "soft_skill_score": 75.0,
    "certifications": 3,
    "live_projects": 1,
    "internship_count": 2,
    "work_experience_months": 6,
    "gender": "Male",
    "extracurricular_activities": "Yes",
}


# ---------------------------------------------------------------------------
# StudentInput — valid cases
# ---------------------------------------------------------------------------


class TestStudentInputValid:
    def test_full_valid_payload(self):
        s = StudentInput(**VALID_PAYLOAD)
        assert s.cgpa == 8.2
        assert s.model == ModelName.random_forest

    def test_all_model_names_accepted(self):
        for m in ("logistic_regression", "random_forest", "xgboost"):
            s = StudentInput(**{**VALID_PAYLOAD, "model": m})
            assert s.model.value == m

    def test_boundary_values(self):
        s = StudentInput(
            **{
                **VALID_PAYLOAD,
                "ssc_percentage": 0.0,
                "hsc_percentage": 100.0,
                "cgpa": 0.0,
                "backlogs": 0,
            }
        )
        assert s.ssc_percentage == 0.0

    def test_female_gender(self):
        s = StudentInput(**{**VALID_PAYLOAD, "gender": "Female"})
        assert s.gender == "Female"

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

    def test_ssc_percentage_negative(self):
        with pytest.raises(ValidationError):
            StudentInput(**{**VALID_PAYLOAD, "ssc_percentage": -1.0})

    def test_backlogs_negative(self):
        with pytest.raises(ValidationError):
            StudentInput(**{**VALID_PAYLOAD, "backlogs": -1})

    def test_invalid_gender(self):
        with pytest.raises(ValidationError):
            StudentInput(**{**VALID_PAYLOAD, "gender": "Other"})

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
