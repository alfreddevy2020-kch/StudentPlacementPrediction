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

from pydantic import BaseModel, Field, model_validator

# ── Model Selection Enum ────────────────────────────────────────────────────

class ModelName(str, Enum):
    """Allowed prediction model identifiers."""
    logistic_regression = "logistic_regression"
    random_forest = "random_forest"
    xgboost = "xgboost"


# ── Request Schema ──────────────────────────────────────────────────────────

class StudentInput(BaseModel):
    """
    17 raw student features expected by the prediction endpoint.

    **Do not include** `student_id` or `placement_status`.
    """

    # ── Model Selection ────────────────────────────────────────────────────
    model: ModelName = Field(
        default=ModelName.random_forest,
        description="Prediction model to use.",
        examples=["random_forest"],
    )

    # ── Numerical ──────────────────────────────────────────────────────────
    cgpa: float = Field(..., ge=0.0, le=10.0, examples=[7.7],
                        description="Cumulative GPA on a 10-point scale.")
    ssc_percentage: float | None = Field(default=None, ge=0.0, le=100.0, examples=[70.0],
                                         description="Class 10 (SSC) percentage.")
    ssc_marks: float | None = Field(default=None, ge=0.0, le=100.0, examples=[70.0],
                                    description="Alias for ssc_percentage.")
    hsc_percentage: float | None = Field(default=None, ge=0.0, le=100.0, examples=[74.0],
                                         description="Class 12 (HSC) percentage.")
    hsc_marks: float | None = Field(default=None, ge=0.0, le=100.0, examples=[74.0],
                                    description="Alias for hsc_percentage.")
    degree_percentage: float = Field(default=68.0, ge=0.0, le=100.0, examples=[72.0],
                                     description="Undergraduate degree percentage.")
    aptitude_test_score: float = Field(..., ge=0.0, le=100.0, examples=[80.0],
                                       description="Aptitude/mock-test score.")
    technical_skill_score: float = Field(default=65.0, ge=0.0, le=100.0, examples=[75.0],
                                         description="Technical skill assessment score (0-100).")
    soft_skills_rating: float = Field(..., ge=0.0, le=5.0, examples=[4.4],
                                      description="Soft-skills rating on a 5-point scale.")
    attendance_percentage: float = Field(default=80.0, ge=0.0, le=100.0, examples=[85.0],
                                         description="College attendance percentage.")
    backlogs: int = Field(default=0, ge=0, le=20, examples=[0],
                          description="Number of backlogs/arrears.")
    internships: int = Field(default=0, ge=0, le=10, examples=[1],
                             description="Completed internships.")
    projects: int = Field(default=1, ge=0, le=20, examples=[2],
                          description="Completed projects.")
    certifications: int | None = Field(default=None, ge=0, le=20, examples=[1],
                                       description="Certifications earned.")
    workshops_certifications: int | None = Field(default=None, ge=0, le=20, examples=[1],
                                                 description="Alias for certifications.")
    work_experience_months: int = Field(default=0, ge=0, le=120, examples=[6],
                                        description="Work experience in months.")

    # ── Categorical ─────────────────────────────────────────────────────────
    gender: Literal["Female", "Male"] = Field(default="Male", description="Gender demographic.")
    placement_training: Literal["Yes", "No"] = Field(default="Yes", description="Institutional placement training.")
    extracurricular_activities: Literal["Yes", "No"] = Field(default="Yes", description="Extracurricular participation.")
    department: Literal["CE", "CS", "ChemE", "ECE", "EE", "IT", "ME"] = Field(default="CS", description="Department/discipline.")

    @model_validator(mode="after")
    def populate_aliases(self) -> "StudentInput":
        if self.ssc_percentage is None:
            self.ssc_percentage = self.ssc_marks if self.ssc_marks is not None else 70.0
        if self.hsc_percentage is None:
            self.hsc_percentage = self.hsc_marks if self.hsc_marks is not None else 70.0
        if self.certifications is None:
            self.certifications = self.workshops_certifications if self.workshops_certifications is not None else 1
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": "random_forest",
                "cgpa": 7.7,
                "ssc_percentage": 70.0,
                "hsc_percentage": 74.0,
                "degree_percentage": 72.0,
                "aptitude_test_score": 80.0,
                "technical_skill_score": 75.0,
                "soft_skills_rating": 4.4,
                "attendance_percentage": 85.0,
                "backlogs": 0,
                "internships": 1,
                "projects": 2,
                "certifications": 1,
                "work_experience_months": 6,
                "gender": "Male",
                "department": "CS",
                "extracurricular_activities": "Yes",
                "placement_training": "Yes",
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
        description=(
            "'healthy' (all models loaded), 'degraded' (1-2 models loaded), "
            "or 'unavailable' (0 loaded)."
        ),
        examples=["healthy"],
    )
    models_loaded: dict[str, bool] = Field(
        ...,
        description="Map of model name to whether it was loaded successfully.",
        examples=[{"logistic_regression": True, "random_forest": True, "xgboost": True}],
    )


class ModelsResponse(BaseModel):
    """Payload returned by GET /api/v1/models."""

    available_models: list[str] = Field(
        ...,
        description="List of model identifiers available for inference.",
        examples=[["logistic_regression", "random_forest", "xgboost"]],
    )
