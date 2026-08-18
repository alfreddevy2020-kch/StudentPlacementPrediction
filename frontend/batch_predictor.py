"""
Batch Predictor Module for Student Placement Dashboard.
Loads existing preprocessor and model artifacts directly from disk for
batch inference across entire cohorts. Does NOT retrain or modify models.

This module enables the Streamlit dashboard to operate standalone
without the FastAPI backend server.
"""

from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd


# ── Artifact Paths ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
PREPROCESSOR_PATH = BASE_DIR / "part2" / "models" / "preprocessor.joblib"

MODEL_PATHS: Dict[str, Path] = {
    "Random Forest": BASE_DIR / "part2" / "models" / "random_forest_best.joblib",
    "Logistic Regression": BASE_DIR / "part2" / "models" / "logistic_regression_best.joblib",
    "XGBoost": BASE_DIR / "part3" / "models" / "xgboost_best.joblib",
}

# ── Feature columns expected by the fitted preprocessor ─────────────────────
# These must exactly match what the ColumnTransformer was fitted on.
NUMERICAL_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "gender",
    "extracurricular_activities",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# ── Risk thresholds ─────────────────────────────────────────────────────────
HIGH_RISK_THRESHOLD = 0.50
MODERATE_RISK_THRESHOLD = 0.75


class BatchPredictor:
    """
    Loads preprocessor and model from disk, provides batch and single
    inference without requiring the FastAPI server.
    """

    def __init__(self) -> None:
        self._preprocessor = None
        self._models: Dict[str, object] = {}
        self._active_model_name: str = ""
        self._loaded = False

    def load(self) -> None:
        """Load preprocessor and all available models from disk."""
        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(
                f"Preprocessor artifact not found: {PREPROCESSOR_PATH}\n"
                "Run preprocessing.py to generate it."
            )

        self._preprocessor = joblib.load(PREPROCESSOR_PATH)

        for name, path in MODEL_PATHS.items():
            if path.exists():
                m = joblib.load(path)
                # Auto-heal scikit-learn compatibility attributes across versions
                if hasattr(m, "__class__") and "LogisticRegression" in m.__class__.__name__:
                    if not hasattr(m, "multi_class"):
                        m.multi_class = "auto"
                self._models[name] = m

        if not self._models:
            raise FileNotFoundError(
                "No model artifacts found. Run the training pipeline first."
            )

        # Default to Random Forest if available, else first loaded model
        if "Random Forest" in self._models:
            self._active_model_name = "Random Forest"
        else:
            self._active_model_name = next(iter(self._models))

        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def available_models(self) -> list:
        return list(self._models.keys())

    @property
    def active_model_name(self) -> str:
        return self._active_model_name

    @active_model_name.setter
    def active_model_name(self, name: str) -> None:
        if name in self._models:
            self._active_model_name = name
        else:
            raise ValueError(
                f"Model '{name}' not loaded. Available: {self.available_models}"
            )

    @property
    def active_model(self):
        return self._models.get(self._active_model_name)

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a DataFrame with exactly the columns the preprocessor expects,
        in the correct order. Missing columns are filled with safe defaults.
        """
        result = pd.DataFrame(index=df.index)

        # Numerical features
        for col in NUMERICAL_FEATURES:
            if col in df.columns:
                result[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            else:
                result[col] = 0.0

        # Categorical features
        if "gender" in df.columns:
            result["gender"] = df["gender"].fillna("Male").astype(str)
        else:
            result["gender"] = "Male"

        if "extracurricular_activities" in df.columns:
            result["extracurricular_activities"] = (
                df["extracurricular_activities"].fillna("No").astype(str)
            )
        else:
            result["extracurricular_activities"] = "No"

        return result

    def predict_probabilities(
        self, df: pd.DataFrame, model_name: str | None = None
    ) -> np.ndarray:
        """
        Returns placement probabilities for each row in the DataFrame.
        Result shape: (n_samples,) with values in [0.0, 1.0].

        Parameters
        ----------
        model_name : str, optional
            Explicitly select which loaded model to use for inference.
            Falls back to ``self.active_model_name`` when *None*.
        """
        if not self._loaded:
            raise RuntimeError("BatchPredictor not loaded. Call load() first.")

        model = self._models.get(
            model_name or self._active_model_name
        )
        if model is None:
            raise ValueError(
                f"Model '{model_name}' not loaded. "
                f"Available: {self.available_models}"
            )

        features = self._prepare_features(df)
        X_transformed = self._preprocessor.transform(features)
        probs = model.predict_proba(X_transformed)[:, 1]
        return probs

    def predict_single(
        self, student_dict: dict, model_name: str | None = None
    ) -> float:
        """
        Predict placement probability for a single student given as a dict.
        Returns a float in [0.0, 1.0].

        Parameters
        ----------
        model_name : str, optional
            Explicitly select which loaded model to use for inference.
            Falls back to ``self.active_model_name`` when *None*.
        """
        df = pd.DataFrame([student_dict])
        return float(self.predict_probabilities(df, model_name=model_name)[0])

    @staticmethod
    def classify_risk(prob: float) -> str:
        """Map a probability to a risk tier label."""
        if prob < HIGH_RISK_THRESHOLD:
            return "High Risk"
        elif prob < MODERATE_RISK_THRESHOLD:
            return "Moderate Risk"
        return "Interview Ready"

    @staticmethod
    def load_dataset() -> pd.DataFrame:
        """Load the raw placement dataset from disk."""
        dataset_path = BASE_DIR / "data" / "raw" / "student_placement.csv"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        return pd.read_csv(dataset_path)
