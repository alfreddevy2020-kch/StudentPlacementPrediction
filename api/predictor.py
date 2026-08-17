"""
api/predictor.py
----------------
Modular inference layer.

Architecture
~~~~~~~~~~~~
BasePredictor (ABC)
    └── RandomForestPredictor   ← current production model

To replace Random Forest with XGBoost (or any other model):
  1. Create a new class that inherits from BasePredictor.
  2. Override load() and predict().
  3. In main.py, swap RandomForestPredictor for the new class.
  The API contract (StudentInput → PredictionResponse) stays unchanged.

Column ordering note
~~~~~~~~~~~~~~~~~~~~
The ColumnTransformer fitted in preprocessing.py selects columns by name,
so the DataFrame column order does not need to match the transformer order.
What matters is that all expected column names are present.
"""

from __future__ import annotations

import abc
from pathlib import Path

import joblib
import pandas as pd

from api.schemas import PredictionResponse, StudentInput

# ── Feature column names (must match what the ColumnTransformer was fitted on) ──
NUMERICAL_FEATURES: list[str] = [
    "ssc_percentage",
    "hsc_percentage",
    "degree_percentage",
    "cgpa",
    "entrance_exam_score",
    "technical_skill_score",
    "soft_skill_score",
    "internship_count",
    "live_projects",
    "work_experience_months",
    "certifications",
    "attendance_percentage",
    "backlogs",
]

CATEGORICAL_FEATURES: list[str] = [
    "gender",
    "extracurricular_activities",
]

ALL_FEATURES: list[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# ── Helpers ─────────────────────────────────────────────────────────────────

def _input_to_dataframe(data: StudentInput) -> pd.DataFrame:
    """
    Convert a validated StudentInput into a single-row DataFrame.
    Column names match the training dataset so the ColumnTransformer can
    select numerical / categorical columns by name.
    """
    row = {col: [getattr(data, col)] for col in ALL_FEATURES}
    return pd.DataFrame(row)


def _derive_risk_level(prob_placed: float) -> str:
    """Map a placement probability to a qualitative risk label."""
    if prob_placed >= 0.8:
        return "High Probability of Placement (Low Risk)"
    if prob_placed >= 0.5:
        return "Moderate Probability of Placement (Medium Risk)"
    return "High Risk of Non-Placement"


# ── Abstract Base ────────────────────────────────────────────────────────────

class BasePredictor(abc.ABC):
    """
    Abstract predictor interface.

    Any concrete implementation must be able to:
      - load its artifacts from disk (classmethod load)
      - accept a StudentInput and return a PredictionResponse (predict)
      - report whether it is ready to serve (is_ready property)
    """

    @classmethod
    @abc.abstractmethod
    def load(cls, preprocessor_path: Path, model_path: Path) -> "BasePredictor":
        """Load preprocessor and model artifacts from disk."""
        ...

    @abc.abstractmethod
    def predict(self, data: StudentInput) -> PredictionResponse:
        """
        Transform raw input and run model inference.
        Returns a fully populated PredictionResponse.
        """
        ...

    @property
    @abc.abstractmethod
    def is_ready(self) -> bool:
        """True when both preprocessor and model are loaded."""
        ...


# ── Concrete: Random Forest ──────────────────────────────────────────────────

class RandomForestPredictor(BasePredictor):
    """
    Inference back-end backed by:
      - preprocessor.joblib  (sklearn ColumnTransformer)
      - random_forest_best.joblib  (sklearn RandomForestClassifier)
    """

    def __init__(self, preprocessor, model) -> None:
        self._preprocessor = preprocessor
        self._model = model

    # -- Lifecycle ────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        preprocessor_path: Path,
        model_path: Path,
    ) -> "RandomForestPredictor":
        """
        Deserialise both artifacts from disk.
        Called once during application startup via the lifespan handler.
        Raises FileNotFoundError if either artifact is missing.
        """
        if not preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor artifact not found: {preprocessor_path}\n"
                "Run preprocessing.py to regenerate it."
            )
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {model_path}\n"
                "Run part2/random_forest_model.py to regenerate it."
            )

        preprocessor = joblib.load(preprocessor_path)
        model = joblib.load(model_path)
        return cls(preprocessor, model)

    # -- Readiness ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._preprocessor is not None and self._model is not None

    # -- Inference ────────────────────────────────────────────────────────────

    def predict(self, data: StudentInput) -> PredictionResponse:
        """
        Full inference pipeline:
          1. Convert StudentInput → one-row DataFrame (preserves column names).
          2. Apply ColumnTransformer (StandardScaler + OneHotEncoder).
          3. Run RandomForestClassifier.predict + predict_proba.
          4. Build and return PredictionResponse.
        """
        # Step 1: structured tabular input
        df = _input_to_dataframe(data)

        # Step 2: preprocessing — applies the same scaler / encoder used during training
        X_transformed = self._preprocessor.transform(df)

        # Step 3: model inference
        label: int = int(self._model.predict(X_transformed)[0])
        class_probabilities = self._model.predict_proba(X_transformed)[0]
        # predict_proba returns [P(class=0), P(class=1)]
        prob_not_placed: float = round(float(class_probabilities[0]), 4)
        prob_placed:     float = round(float(class_probabilities[1]), 4)

        # Step 4: assemble response
        return PredictionResponse(
            placement_status=label,
            placement_label="Placed" if label == 1 else "Not Placed",
            probability_placed=prob_placed,
            probability_not_placed=prob_not_placed,
            risk_level=_derive_risk_level(prob_placed),
        )
