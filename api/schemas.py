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

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ── Model Selection Enum ────────────────────────────────────────────────────

class ModelName(str, Enum):
    """Allowed prediction model identifiers."""
    logistic_regression = "logistic_regression"
    random_forest = "random_forest"
    xgboost = "xgboost"


# ── Request Schema ──────────────────────────────────────────────────────────

class StudentInput(BaseModel):
    """
    23 raw student features expected by the prediction endpoint.

    **Do not include** `student_id` or `salary_package_lpa`.
    """

    # ── Model Selection ────────────────────────────────────────────────────
    model: ModelName = Field(
        default=ModelName.random_forest,
        description="Prediction model to use.",
        examples=["random_forest"],
    )

    # ── Numerical ──────────────────────────────────────────────────────────
    age: int = Field(..., ge=15, le=60, examples=[21])
    cgpa: float = Field(..., ge=0.0, le=10.0, examples=[8.2])
    attendance_percentage: float = Field(..., ge=0.0, le=100.0, examples=[90.0])
    backlogs: int = Field(..., ge=0, examples=[0])
    coding_skill_score: float = Field(..., ge=0.0, le=100.0, examples=[80.0])
    aptitude_score: float = Field(..., ge=0.0, le=100.0, examples=[75.0])
    communication_skill_score: float = Field(..., ge=0.0, le=100.0, examples=[78.0])
    logical_reasoning_score: float = Field(..., ge=0.0, le=100.0, examples=[72.0])
    mock_interview_score: float = Field(..., ge=0.0, le=100.0, examples=[70.0])
    internships_count: int = Field(..., ge=0, examples=[2])
    projects_count: int = Field(..., ge=0, examples=[1])
    certifications_count: int = Field(..., ge=0, examples=[3])
    hackathons_participated: int = Field(..., ge=0, examples=[1])
    github_repos: int = Field(..., ge=0, examples=[5])
    linkedin_connections: int = Field(..., ge=0, examples=[150])
    extracurricular_score: float = Field(..., ge=0.0, le=100.0, examples=[60.0])
    leadership_score: float = Field(..., ge=0.0, le=100.0, examples=[55.0])
    sleep_hours: float = Field(..., ge=0.0, le=24.0, examples=[7.0])
    study_hours_per_day: float = Field(..., ge=0.0, le=24.0, examples=[4.0])

    # ── Categorical ─────────────────────────────────────────────────────────
    gender: Literal["Male", "Female"]
    branch: str
    college_tier: str
    volunteer_experience: Literal["Yes", "No"]

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": "random_forest",
                "age": 21,
                "cgpa": 8.2,
                "attendance_percentage": 90.0,
                "backlogs": 0,
                "coding_skill_score": 80.0,
                "aptitude_score": 75.0,
                "communication_skill_score": 78.0,
                "logical_reasoning_score": 72.0,
                "mock_interview_score": 70.0,
                "internships_count": 2,
                "projects_count": 1,
                "certifications_count": 3,
                "hackathons_participated": 1,
                "github_repos": 5,
                "linkedin_connections": 150,
                "extracurricular_score": 60.0,
                "leadership_score": 55.0,
                "sleep_hours": 7.0,
                "study_hours_per_day": 4.0,
                "gender": "Male",
                "branch": "CSE",
                "college_tier": "Tier 1",
                "volunteer_experience": "Yes",
            }
        }
    }


# ── Response Schemas ────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Structured placement prediction returned by POST /api/v1/predict."""

    model_used: str = Field(
        ...,
        description="Human-readable name of the model used for prediction.",
        examples=["Random Forest"],
    )
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
                "model_used": "Random Forest",
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
        description="'healthy' when all artifacts are loaded; 'degraded' otherwise.",
        examples=["healthy"],
    )
    preprocessor_loaded: bool = Field(
        ...,
        description="True when preprocessor.joblib was loaded successfully at startup.",
        examples=[True],
    )
    models_loaded: dict[str, bool] = Field(
        ...,
        description="Map of model name to whether it was loaded successfully.",
        examples=[{"logistic_regression": True, "random_forest": True, "xgboost": True}],
    )
