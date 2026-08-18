"""
api/predictor.py
----------------
Modular inference layer.

Architecture
~~~~~~~~~~~~
BasePredictor (ABC)
    ├── LogisticRegressionPredictor
    ├── RandomForestPredictor
    └── XGBoostPredictor

All concrete predictors:
  1. Receive StudentInput
  2. Convert it into the same DataFrame structure
  3. Apply the shared preprocessor
  4. Run the selected model
  5. Return a common PredictionResponse

Column ordering note
~~~~~~~~~~~~~~~~~~~~~
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

    @property
    @abc.abstractmethod
    def model_display_name(self) -> str:
        """Human-readable model name included in PredictionResponse."""
        ...

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


# ── Concrete: Logistic Regression ───────────────────────────────────────────

class LogisticRegressionPredictor(BasePredictor):
    """
    Inference back-end backed by:
      - preprocessor.joblib  (sklearn ColumnTransformer)
      - logistic_regression_best.joblib  (sklearn LogisticRegression)
    """

    _MODEL_DISPLAY_NAME = "Logistic Regression"

    def __init__(self, preprocessor, model) -> None:
        self._preprocessor = preprocessor
        self._model = model

    @property
    def model_display_name(self) -> str:
        return self._MODEL_DISPLAY_NAME

    @classmethod
    def load(
        cls,
        preprocessor_path: Path,
        model_path: Path,
    ) -> "LogisticRegressionPredictor":
        if not preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor artifact not found: {preprocessor_path}\n"
                "Run preprocessing.py to regenerate it."
            )
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {model_path}\n"
                "Run part2/logistic_regression_model.py to regenerate it."
            )

        preprocessor = joblib.load(preprocessor_path)
        model = joblib.load(model_path)
        return cls(preprocessor, model)

    @property
    def is_ready(self) -> bool:
        return self._preprocessor is not None and self._model is not None

    def predict(self, data: StudentInput) -> PredictionResponse:
        df = _input_to_dataframe(data)
        X_transformed = self._preprocessor.transform(df)
        label: int = int(self._model.predict(X_transformed)[0])
        class_probabilities = self._model.predict_proba(X_transformed)[0]
        prob_not_placed: float = round(float(class_probabilities[0]), 4)
        prob_placed:     float = round(float(class_probabilities[1]), 4)

        return PredictionResponse(
            model_used=self._MODEL_DISPLAY_NAME,
            placement_status=label,
            placement_label="Placed" if label == 1 else "Not Placed",
            probability_placed=prob_placed,
            probability_not_placed=prob_not_placed,
            risk_level=_derive_risk_level(prob_placed),
        )


# ── Concrete: Random Forest ──────────────────────────────────────────────────

class RandomForestPredictor(BasePredictor):
    """
    Inference back-end backed by:
      - preprocessor.joblib  (sklearn ColumnTransformer)
      - random_forest_best.joblib  (sklearn RandomForestClassifier)
    """

    _MODEL_DISPLAY_NAME = "Random Forest"

    def __init__(self, preprocessor, model) -> None:
        self._preprocessor = preprocessor
        self._model = model

    @property
    def model_display_name(self) -> str:
        return self._MODEL_DISPLAY_NAME

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

    @property
    def is_ready(self) -> bool:
        return self._preprocessor is not None and self._model is not None

    def predict(self, data: StudentInput) -> PredictionResponse:
        """
        Full inference pipeline:
          1. Convert StudentInput -> one-row DataFrame (preserves column names).
          2. Apply ColumnTransformer (StandardScaler + OneHotEncoder).
          3. Run RandomForestClassifier.predict + predict_proba.
          4. Build and return PredictionResponse.
        """
        df = _input_to_dataframe(data)
        X_transformed = self._preprocessor.transform(df)
        label: int = int(self._model.predict(X_transformed)[0])
        class_probabilities = self._model.predict_proba(X_transformed)[0]
        prob_not_placed: float = round(float(class_probabilities[0]), 4)
        prob_placed:     float = round(float(class_probabilities[1]), 4)

        return PredictionResponse(
            model_used=self._MODEL_DISPLAY_NAME,
            placement_status=label,
            placement_label="Placed" if label == 1 else "Not Placed",
            probability_placed=prob_placed,
            probability_not_placed=prob_not_placed,
            risk_level=_derive_risk_level(prob_placed),
        )


# ── Concrete: XGBoost ───────────────────────────────────────────────────────

class XGBoostPredictor(BasePredictor):
    """
    Inference back-end backed by:
      - preprocessor.joblib  (sklearn ColumnTransformer)
      - xgboost_best.joblib  (xgboost.XGBClassifier)
    """

    _MODEL_DISPLAY_NAME = "XGBoost"

    def __init__(self, preprocessor, model) -> None:
        self._preprocessor = preprocessor
        self._model = model

    @property
    def model_display_name(self) -> str:
        return self._MODEL_DISPLAY_NAME

    @classmethod
    def load(
        cls,
        preprocessor_path: Path,
        model_path: Path,
    ) -> "XGBoostPredictor":
        if not preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor artifact not found: {preprocessor_path}\n"
                "Run preprocessing.py to regenerate it."
            )
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {model_path}\n"
                "Run part3/xgboost_model.py to regenerate it."
            )

        preprocessor = joblib.load(preprocessor_path)
        model = joblib.load(model_path)
        return cls(preprocessor, model)

    @property
    def is_ready(self) -> bool:
        return self._preprocessor is not None and self._model is not None

    def predict(self, data: StudentInput) -> PredictionResponse:
        df = _input_to_dataframe(data)
        X_transformed = self._preprocessor.transform(df)
        label: int = int(self._model.predict(X_transformed)[0])
        class_probabilities = self._model.predict_proba(X_transformed)[0]
        prob_not_placed: float = round(float(class_probabilities[0]), 4)
        prob_placed:     float = round(float(class_probabilities[1]), 4)

        return PredictionResponse(
            model_used=self._MODEL_DISPLAY_NAME,
            placement_status=label,
            placement_label="Placed" if label == 1 else "Not Placed",
            probability_placed=prob_placed,
            probability_not_placed=prob_not_placed,
            risk_level=_derive_risk_level(prob_placed),
        )
