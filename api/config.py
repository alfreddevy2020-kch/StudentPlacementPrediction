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
BASE_DIR = Path(__file__).resolve().parent.parent   # repo root
PREPROCESSOR_PATH         = BASE_DIR / "part2" / "models" / "preprocessor.joblib"
LOGISTIC_REGRESSION_PATH  = BASE_DIR / "part2" / "models" / "logistic_regression_best.joblib"
RANDOM_FOREST_PATH        = BASE_DIR / "part2" / "models" / "random_forest_best.joblib"
XGBOOST_PATH              = BASE_DIR / "part3" / "models" / "xgboost_best.joblib"

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
`student_id` and `salary_package_lpa` are not accepted by the API.
`salary_package_lpa` is known only after placement and would cause data leakage.

## Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check — confirms artifacts are loaded |
| POST | `/api/v1/predict` | Placement prediction |
"""

APP_VERSION = "1.0.0"
