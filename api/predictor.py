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
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from api.schemas import PredictionResponse, StudentInput
from feature_engineering import (
    FEATURE_RANGES,
    RAW_CATEGORICAL_FEATURES,
    RAW_NUMERICAL_FEATURES,
    engineer_features,
)

# Sample valid student record used for startup smoke-testing
# Startup validation record, built from the midpoint of each feature's
# observed training range so it cannot drift out of sync with the schema.
# Raw only — engineer_features() is applied before the preprocessor sees it,
# exactly as it is for a real request.
_SAMPLE_VALID_RECORD = {
    name: [(lo + hi) / 2] for name, (lo, hi) in FEATURE_RANGES.items()
}
for _cat in RAW_CATEGORICAL_FEATURES:
    _SAMPLE_VALID_RECORD[_cat] = ["Yes"]


# ── Helpers ─────────────────────────────────────────────────────────────────


def _compute_sha256(filepath: Path) -> str:
    """Compute the SHA-256 checksum of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _input_to_dataframe(data: StudentInput) -> pd.DataFrame:
    row = {col: [getattr(data, col)] for col in RAW_NUMERICAL_FEATURES + RAW_CATEGORICAL_FEATURES}
    return engineer_features(pd.DataFrame(row))


def _derive_risk_level(prob_placed: float) -> str:
    """Map a placement probability to a qualitative risk label."""
    if prob_placed >= 0.8:
        return "High Probability of Placement (Low Risk)"
    if prob_placed >= 0.5:
        return "Moderate Probability of Placement (Medium Risk)"
    return "High Risk of Non-Placement"


def verify_and_load_bundle(
    model_key: str,
    preprocessor_path: Path,
    model_path: Path,
    manifest_path: Path | None = None,
) -> tuple[Any, Any, dict]:
    """
    Strict validation of a production model bundle:
      1. Verify artifact files exist.
      2. Verify manifest.json exists, matches schema, and model_name matches model_key.
      3. Verify SHA-256 checksums of model and preprocessor match manifest.
      4. Verify preprocessor can transform a valid sample input.
      5. Verify model supports predict() and predict_proba().
    """
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor file missing: {preprocessor_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file missing: {model_path}")

    manifest: dict = {}
    if manifest_path is not None:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file missing: {manifest_path}")

        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        manifest_model_name = manifest.get("model_name")
        if manifest_model_name != model_key:
            raise ValueError(
                f"Manifest model_name mismatch: expected '{model_key}', got '{manifest_model_name}'"
            )

        # Validate checksums
        expected_prep_sha = manifest.get("artifacts", {}).get("preprocessor", {}).get("sha256")
        expected_model_sha = manifest.get("artifacts", {}).get("model", {}).get("sha256")

        if not expected_prep_sha or not expected_model_sha:
            raise ValueError(f"Manifest in {manifest_path} is missing artifact SHA-256 entries.")

        actual_prep_sha = _compute_sha256(preprocessor_path)
        if actual_prep_sha != expected_prep_sha:
            raise ValueError(
                f"Preprocessor SHA-256 mismatch for {model_key}:\n"
                f"  Expected: {expected_prep_sha}\n"
                f"  Actual:   {actual_prep_sha}"
            )

        actual_model_sha = _compute_sha256(model_path)
        if actual_model_sha != expected_model_sha:
            raise ValueError(
                f"Model SHA-256 mismatch for {model_key}:\n"
                f"  Expected: {expected_model_sha}\n"
                f"  Actual:   {actual_model_sha}"
            )

    # Load artifacts
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)

    # Smoke-test transformation
    try:
        sample_df = engineer_features(pd.DataFrame(_SAMPLE_VALID_RECORD))
        X_sample = preprocessor.transform(sample_df)
    except Exception as exc:
        raise RuntimeError(f"Preprocessor validation failed for {model_key}: {exc}") from exc

    # Smoke-test prediction & probability
    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise TypeError(f"Model for {model_key} must implement predict() and predict_proba().")

    try:
        _ = model.predict(X_sample)
        _ = model.predict_proba(X_sample)
    except Exception as exc:
        raise RuntimeError(f"Model prediction validation failed for {model_key}: {exc}") from exc

    return preprocessor, model, manifest


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

    @property
    def model_version(self) -> str:
        """Model version string from manifest."""
        return getattr(self, "_model_version", "1.0.0")

    @classmethod
    @abc.abstractmethod
    def load(
        cls,
        preprocessor_path: Path,
        model_path: Path,
        manifest_path: Path | None = None,
    ) -> BasePredictor:
        """Load preprocessor and model artifacts from disk with manifest validation."""
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
    """Inference back-end for Logistic Regression."""

    _MODEL_DISPLAY_NAME = "Logistic Regression"

    def __init__(self, preprocessor: Any, model: Any, manifest: dict | None = None) -> None:
        self._preprocessor = preprocessor
        self._model = model
        self._manifest = manifest or {}
        self._model_version = self._manifest.get("model_version", "1.0.0")

    @property
    def model_display_name(self) -> str:
        return self._MODEL_DISPLAY_NAME

    @classmethod
    def load(
        cls,
        preprocessor_path: Path,
        model_path: Path,
        manifest_path: Path | None = None,
    ) -> LogisticRegressionPredictor:
        preprocessor, model, manifest = verify_and_load_bundle(
            "logistic_regression",
            preprocessor_path,
            model_path,
            manifest_path,
        )
        return cls(preprocessor, model, manifest)

    @property
    def is_ready(self) -> bool:
        return self._preprocessor is not None and self._model is not None

    def predict(self, data: StudentInput) -> PredictionResponse:
        df = _input_to_dataframe(data)
        X_transformed = self._preprocessor.transform(df)
        label: int = int(self._model.predict(X_transformed)[0])
        class_probabilities = self._model.predict_proba(X_transformed)[0]
        prob_not_placed: float = round(float(class_probabilities[0]), 4)
        prob_placed: float = round(float(class_probabilities[1]), 4)

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
    """Inference back-end for Random Forest."""

    _MODEL_DISPLAY_NAME = "Random Forest"

    def __init__(self, preprocessor: Any, model: Any, manifest: dict | None = None) -> None:
        self._preprocessor = preprocessor
        self._model = model
        self._manifest = manifest or {}
        self._model_version = self._manifest.get("model_version", "1.0.0")

    @property
    def model_display_name(self) -> str:
        return self._MODEL_DISPLAY_NAME

    @classmethod
    def load(
        cls,
        preprocessor_path: Path,
        model_path: Path,
        manifest_path: Path | None = None,
    ) -> RandomForestPredictor:
        preprocessor, model, manifest = verify_and_load_bundle(
            "random_forest",
            preprocessor_path,
            model_path,
            manifest_path,
        )
        return cls(preprocessor, model, manifest)

    @property
    def is_ready(self) -> bool:
        return self._preprocessor is not None and self._model is not None

    def predict(self, data: StudentInput) -> PredictionResponse:
        df = _input_to_dataframe(data)
        X_transformed = self._preprocessor.transform(df)
        label: int = int(self._model.predict(X_transformed)[0])
        class_probabilities = self._model.predict_proba(X_transformed)[0]
        prob_not_placed: float = round(float(class_probabilities[0]), 4)
        prob_placed: float = round(float(class_probabilities[1]), 4)

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
    """Inference back-end for XGBoost."""

    _MODEL_DISPLAY_NAME = "XGBoost"

    def __init__(self, preprocessor: Any, model: Any, manifest: dict | None = None) -> None:
        self._preprocessor = preprocessor
        self._model = model
        self._manifest = manifest or {}
        self._model_version = self._manifest.get("model_version", "1.0.0")

    @property
    def model_display_name(self) -> str:
        return self._MODEL_DISPLAY_NAME

    @classmethod
    def load(
        cls,
        preprocessor_path: Path,
        model_path: Path,
        manifest_path: Path | None = None,
    ) -> XGBoostPredictor:
        preprocessor, model, manifest = verify_and_load_bundle(
            "xgboost",
            preprocessor_path,
            model_path,
            manifest_path,
        )
        return cls(preprocessor, model, manifest)

    @property
    def is_ready(self) -> bool:
        return self._preprocessor is not None and self._model is not None

    def predict(self, data: StudentInput) -> PredictionResponse:
        df = _input_to_dataframe(data)
        X_transformed = self._preprocessor.transform(df)
        label: int = int(self._model.predict(X_transformed)[0])
        class_probabilities = self._model.predict_proba(X_transformed)[0]
        prob_not_placed: float = round(float(class_probabilities[0]), 4)
        prob_placed: float = round(float(class_probabilities[1]), 4)

        return PredictionResponse(
            model_used=self._MODEL_DISPLAY_NAME,
            placement_status=label,
            placement_label="Placed" if label == 1 else "Not Placed",
            probability_placed=prob_placed,
            probability_not_placed=prob_not_placed,
            risk_level=_derive_risk_level(prob_placed),
        )
