"""
api/config.py
-------------
Central place for artifact paths and FastAPI application metadata.
Change paths here when swapping model artifacts.
"""

from pathlib import Path

# ── Artifact Paths ─────────────────────────────────────────────────────────
# Resolved relative to this file so the server works regardless of the
# working directory from which uvicorn is launched.
BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
ARTIFACT_ROOT = BASE_DIR / "artifacts" / "production"

MODEL_BUNDLES: dict[str, dict[str, Path]] = {
    "logistic_regression": {
        "preprocessor": (ARTIFACT_ROOT / "logistic_regression" / "preprocessor.joblib"),
        "model": (ARTIFACT_ROOT / "logistic_regression" / "model.joblib"),
        "manifest": (ARTIFACT_ROOT / "logistic_regression" / "manifest.json"),
        "baseline_metrics": (ARTIFACT_ROOT / "logistic_regression" / "baseline_metrics.json"),
    },
    "random_forest": {
        "preprocessor": (ARTIFACT_ROOT / "random_forest" / "preprocessor.joblib"),
        "model": (ARTIFACT_ROOT / "random_forest" / "model.joblib"),
        "manifest": (ARTIFACT_ROOT / "random_forest" / "manifest.json"),
        "baseline_metrics": (ARTIFACT_ROOT / "random_forest" / "baseline_metrics.json"),
    },
    "xgboost": {
        "preprocessor": (ARTIFACT_ROOT / "xgboost" / "preprocessor.joblib"),
        "model": (ARTIFACT_ROOT / "xgboost" / "model.joblib"),
        "manifest": (ARTIFACT_ROOT / "xgboost" / "manifest.json"),
        "baseline_metrics": (ARTIFACT_ROOT / "xgboost" / "baseline_metrics.json"),
    },
}

# ── OpenAPI / App Metadata ─────────────────────────────────────────────────
APP_TITLE = "Student Placement Prediction API"

APP_DESCRIPTION = """
## Overview
REST API that predicts whether a student will be placed based on academic
performance, skills, and professional experience.

## Inference Pipeline
```
Client → FastAPI (Pydantic validation)
       → ColumnTransformer  (StandardScaler + OneHotEncoder)
       → RandomForestClassifier
       → Structured JSON response
```

## Key Design Decisions
* **Artifacts loaded once at startup** — not on every request.
* **Preprocessing is server-side** — the client always sends raw feature values.
* **Modular predictor** — the model back-end can be replaced (e.g., XGBoost)
  without changing the API contract.

## Excluded Fields
`student_id` and `placement_status` are not accepted by the API.
`student_id` is an identifier with no predictive content, and
`placement_status` is the target itself.

## Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check — confirms artifacts are loaded |
| POST | `/api/v1/predict` | Placement prediction |
"""

APP_VERSION = "1.0.0"
