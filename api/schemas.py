"""
api/schemas.py
--------------
Pydantic models that define the API contract:
  - StudentInput  : validated request body for POST /api/v1/predict
  - PredictionResponse : structured JSON response returned to the client
  - HealthResponse     : payload for GET /health

Validation rules mirror the training-data constraints to prevent out-of-
distribution inputs from reaching the model silently.
"""

from typing import Literal

from pydantic import BaseModel, Field


# ── Request Schema ──────────────────────────────────────────────────────────

class StudentInput(BaseModel):
    """
    15 raw student features expected by the prediction endpoint.

    **Do not include** `student_id` or `salary_package_lpa`.
    """

    # ── Numerical ──────────────────────────────────────────────────────────
    ssc_percentage: float = Field(
        ...,
        ge=0.0, le=100.0,
        description="Secondary School (10th grade) percentage.",
        examples=[75.5],
    )
    hsc_percentage: float = Field(
        ...,
        ge=0.0, le=100.0,
        description="Higher Secondary (12th grade) percentage.",
        examples=[78.0],
    )
    degree_percentage: float = Field(
        ...,
        ge=0.0, le=100.0,
        description="Undergraduate degree percentage.",
        examples=[72.0],
    )
    cgpa: float = Field(
        ...,
        ge=0.0, le=10.0,
        description="College Cumulative GPA on a 10-point scale.",
        examples=[8.2],
    )
    attendance_percentage: float = Field(
        ...,
        ge=0.0, le=100.0,
        description="College attendance percentage.",
        examples=[90.0],
    )
    backlogs: int = Field(
        ...,
        ge=0,
        description="Number of active academic backlogs (failed subjects).",
        examples=[0],
    )
    entrance_exam_score: float = Field(
        ...,
        ge=0.0, le=100.0,
        description="Entrance examination score.",
        examples=[85.0],
    )
    technical_skill_score: float = Field(
        ...,
        ge=0.0, le=100.0,
        description="Technical / coding skills assessment score.",
        examples=[80.0],
    )
    soft_skill_score: float = Field(
        ...,
        ge=0.0, le=100.0,
        description="Soft skills (communication, teamwork) assessment score.",
        examples=[75.0],
    )
    certifications: int = Field(
        ...,
        ge=0,
        description="Number of professional certifications earned.",
        examples=[3],
    )
    live_projects: int = Field(
        ...,
        ge=0,
        description="Number of live / capstone projects completed.",
        examples=[1],
    )
    internship_count: int = Field(
        ...,
        ge=0,
        description="Number of internships completed.",
        examples=[2],
    )
    work_experience_months: int = Field(
        ...,
        ge=0,
        description="Prior professional work experience in months.",
        examples=[6],
    )

    # ── Categorical ────────────────────────────────────────────────────────
    gender: Literal["Male", "Female"] = Field(
        ...,
        description="Student gender. Accepted values: 'Male', 'Female'.",
        examples=["Male"],
    )
    extracurricular_activities: Literal["Yes", "No"] = Field(
        ...,
        description="Participation in extracurricular activities. Accepted values: 'Yes', 'No'.",
        examples=["Yes"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


# ── Response Schemas ────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Structured placement prediction returned by POST /api/v1/predict."""

    placement_status: int = Field(
        ...,
        description="Binary class label: 1 = Placed, 0 = Not Placed.",
        examples=[1],
    )
    placement_label: str = Field(
        ...,
        description="Human-readable label corresponding to placement_status.",
        examples=["Placed"],
    )
    probability_placed: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Model confidence (0–1) that the student will be placed.",
        examples=[0.938],
    )
    probability_not_placed: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Model confidence (0–1) that the student will NOT be placed.",
        examples=[0.062],
    )
    risk_level: str = Field(
        ...,
        description=(
            "Qualitative placement risk derived from probability_placed. "
            "One of: 'High Probability of Placement (Low Risk)', "
            "'Moderate Probability of Placement (Medium Risk)', "
            "'High Risk of Non-Placement'."
        ),
        examples=["High Probability of Placement (Low Risk)"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "placement_status": 1,
                "placement_label": "Placed",
                "probability_placed": 0.938,
                "probability_not_placed": 0.062,
                "risk_level": "High Probability of Placement (Low Risk)",
            }
        }
    }


class HealthResponse(BaseModel):
    """Payload returned by GET /health."""

    status: str = Field(
        ...,
        description="'healthy' when both artifacts are loaded; 'degraded' otherwise.",
        examples=["healthy"],
    )
    preprocessor_loaded: bool = Field(
        ...,
        description="True when preprocessor.joblib was loaded successfully at startup.",
        examples=[True],
    )
    model_loaded: bool = Field(
        ...,
        description="True when random_forest_best.joblib was loaded successfully at startup.",
        examples=[True],
    )
