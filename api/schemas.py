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
    10 raw student features expected by the prediction endpoint.

    **Do not include** `student_id` or `placement_status`.

    Bounds are the widest sensible range for each field. The model's actual
    training range is narrower (see FEATURE_RANGES in feature_engineering.py);
    values outside it are accepted but are extrapolation.
    """

    # ── Model Selection ────────────────────────────────────────────────────
    model: ModelName = Field(
        default=ModelName.logistic_regression,
        description="Prediction model to use.",
        examples=["logistic_regression"],
    )

    # ── Numerical ──────────────────────────────────────────────────────────
    cgpa: float = Field(..., ge=0.0, le=10.0, examples=[7.7],
                        description="Cumulative GPA on a 10-point scale. Trained on 6.5-9.1.")
    ssc_marks: float = Field(..., ge=0.0, le=100.0, examples=[70.0],
                             description="Class 10 (SSC) percentage. Trained on 55-90.")
    hsc_marks: float = Field(..., ge=0.0, le=100.0, examples=[74.0],
                             description="Class 12 (HSC) percentage. Trained on 57-88.")
    aptitude_test_score: float = Field(..., ge=0.0, le=100.0, examples=[80.0],
                                       description="Aptitude/mock-test score. Trained on 60-90.")
    soft_skills_rating: float = Field(..., ge=0.0, le=5.0, examples=[4.4],
                                      description="Soft-skills rating on a 5-point scale. Trained on 3.0-4.8.")
    internships: int = Field(..., ge=0, le=10, examples=[1],
                             description="Completed internships. Trained on 0-2.")
    projects: int = Field(..., ge=0, le=20, examples=[2],
                          description="Completed projects. Trained on 0-3.")
    workshops_certifications: int = Field(..., ge=0, le=20, examples=[1],
                                          description="Workshops/certifications earned. Trained on 0-3.")

    # ── Categorical ─────────────────────────────────────────────────────────
    extracurricular_activities: Literal["Yes", "No"]
    placement_training: Literal["Yes", "No"]

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": "logistic_regression",
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
