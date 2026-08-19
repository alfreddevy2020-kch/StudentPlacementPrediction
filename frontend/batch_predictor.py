"""
Batch Predictor Module for Student Placement Dashboard.
Loads existing preprocessor and model artifacts directly from disk for
batch inference across entire cohorts. Does NOT retrain or modify models.

This module enables the Streamlit dashboard to operate standalone
without the FastAPI backend server.
"""

import sys
from pathlib import Path

# Ensure repository root is in sys.path for feature_engineering import
BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
import joblib
import numpy as np
import pandas as pd

from feature_engineering import (
    ALL_NUMERICAL_FEATURES,
    RAW_CATEGORICAL_FEATURES,
    RAW_NUMERICAL_FEATURES,
    engineer_features,
    load_raw_dataset,
)

# ── Artifact Paths ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # repo root

# Prefer the committed production bundles (available on Streamlit Cloud).
# Fall back to local dev paths (part2/models, part3/models) for local runs
# where the full training pipeline has been executed.
PROD_DIR = BASE_DIR / "artifacts" / "production"

def _resolve(prod_rel: str, dev_rel: str) -> Path:
    """Return the production path if it exists, otherwise the dev path."""
    prod = PROD_DIR / prod_rel
    return prod if prod.exists() else BASE_DIR / dev_rel

PREPROCESSOR_PATH: Path = _resolve(
    "random_forest/preprocessor.joblib",
    "part2/models/preprocessor.joblib",
)

STATS_PATH: Path = _resolve(
    "random_forest/normalization_stats.json",
    "data/processed/normalization_stats.json",
)

MODEL_PATHS: dict[str, Path] = {
    "Random Forest": _resolve(
        "random_forest/model.joblib",
        "part2/models/random_forest_best.joblib",
    ),
    "Logistic Regression": _resolve(
        "logistic_regression/model.joblib",
        "part2/models/logistic_regression_best.joblib",
    ),
    "XGBoost": _resolve(
        "xgboost/model.joblib",
        "part3/models/xgboost_best.joblib",
    ),
}


# ── Feature columns expected by the fitted preprocessor ─────────────────────
# These must exactly match what the ColumnTransformer was fitted on.
NUMERICAL_FEATURES = ALL_NUMERICAL_FEATURES
CATEGORICAL_FEATURES = RAW_CATEGORICAL_FEATURES

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
        self._stats: dict | None = None
        self._models: dict[str, object] = {}
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

        if STATS_PATH.exists():
            with open(STATS_PATH) as f:
                self._stats = json.load(f)
        else:
            self._stats = None

        for name, path in MODEL_PATHS.items():
            if path.exists():
                m = joblib.load(path)
                # Auto-heal scikit-learn compatibility attributes across versions
                if (
                    "LogisticRegression" in m.__class__.__name__
                    and not hasattr(m, "multi_class")
                ):
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
        df = engineer_features(df, stats=self._stats)  # adds the 18 derived columns first
        result = pd.DataFrame(index=df.index)
        for col in RAW_NUMERICAL_FEATURES + [c for c in NUMERICAL_FEATURES if c not in RAW_NUMERICAL_FEATURES]:
            result[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0) if col in df.columns else 0.0
        for col in CATEGORICAL_FEATURES:
            result[col] = df[col].astype(str) if col in df.columns else ""
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
        """Load the raw placement dataset with canonical snake_case columns."""
        return load_raw_dataset(BASE_DIR / "data" / "raw" / "student_placement.csv")
