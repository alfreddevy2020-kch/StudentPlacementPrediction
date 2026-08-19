"""
api/main.py
-----------
FastAPI application entry point.

Start the server:
    uvicorn api.main:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs    (Swagger UI)
    http://localhost:8000/redoc   (ReDoc)
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from api.config import (
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    LOGISTIC_REGRESSION_PATH,
    PREPROCESSOR_PATH,
    RANDOM_FOREST_PATH,
    XGBOOST_PATH,
)
from api.predictor import (
    BasePredictor,
    LogisticRegressionPredictor,
    RandomForestPredictor,
    XGBoostPredictor,
)
from api.schemas import HealthResponse, ModelName, PredictionResponse, StudentInput

# ── Predictor registry ──────────────────────────────────────────────────────
# Loaded once at startup; shared across all requests.
_predictors: dict[str, BasePredictor] = {}
_preprocessor_loaded: bool = False


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Startup: deserialise preprocessor and all model artifacts.
    Shutdown: nothing to release (joblib objects are in-memory).
    """
    global _preprocessor_loaded

    model_configs: list[tuple[str, type[BasePredictor], type]] = [
        ("logistic_regression", LogisticRegressionPredictor, LOGISTIC_REGRESSION_PATH),
        ("random_forest", RandomForestPredictor, RANDOM_FOREST_PATH),
        ("xgboost", XGBoostPredictor, XGBOOST_PATH),
    ]

    for model_key, predictor_cls, model_path in model_configs:
        try:
            predictor = predictor_cls.load(PREPROCESSOR_PATH, model_path)
            _predictors[model_key] = predictor
            print(f"[API] [OK] {predictor.model_display_name} loaded: {model_path.name}")
        except FileNotFoundError as exc:
            print(f"[API] [FAIL] Artifact not found for {model_key}: {exc}")
        except Exception as exc:
            print(f"[API] [FAIL] Unexpected error loading {model_key}: {exc}")

    _preprocessor_loaded = PREPROCESSOR_PATH.exists()

    if _predictors:
        print(f"[API] [OK] {len(_predictors)}/3 models loaded successfully.")
    else:
        print("[API] [FAIL] No models loaded. Server is in degraded state.")

    yield   # server is now running

    # (shutdown cleanup — none required)


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ── Global exception handler ─────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for any unhandled exceptions so the server never returns a raw traceback."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {type(exc).__name__}: {exc}"},
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description=(
        "Liveness probe. Returns `status: healthy` when the preprocessor "
        "and at least one model are loaded and ready to serve predictions."
    ),
    tags=["System"],
)
def health_check() -> HealthResponse:
    models_loaded = {
        key: predictor is not None and predictor.is_ready
        for key, predictor in _predictors.items()
    }
    all_loaded = all(models_loaded.values()) and len(models_loaded) == 3
    return HealthResponse(
        status="healthy" if all_loaded else "degraded",
        preprocessor_loaded=_preprocessor_loaded,
        models_loaded=models_loaded,
    )


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    summary="Predict Student Placement",
    description=(
        "Accepts 10 raw student features and a model selection, applies the "
        "fitted preprocessor (StandardScaler + OneHotEncoder), and returns a "
        "placement prediction from the selected model.\n\n"
        "**Available models:** `logistic_regression`, `random_forest`, `xgboost`.\n\n"
        "**Fields NOT accepted:** `student_id`, `placement_status`."
    ),
    tags=["Inference"],
    responses={
        200: {"description": "Successful prediction"},
        422: {"description": "Validation error — check field types and allowed values"},
        503: {"description": "Model artifacts unavailable — check /health"},
        500: {"description": "Unexpected inference error"},
    },
)
def predict_placement(payload: StudentInput) -> PredictionResponse:
    """
    **Inference pipeline (server-side):**

    1. Pydantic validates the 15 raw features + model selection.
    2. The selected predictor's `predict()` builds a one-row DataFrame
       and applies the shared ColumnTransformer.
    3. The transformed array is passed to the chosen model's predict/predict_proba.
    4. The result is returned as a structured JSON response.
    """
    model_key = payload.model.value

    if model_key not in _predictors or not _predictors[model_key].is_ready:
        available = list(_predictors.keys())
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Model '{model_key}' is not available. "
                f"Loaded models: {available}. "
                "Check /health for details."
            ),
        )

    try:
        return _predictors[model_key].predict(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {type(exc).__name__}: {exc}",
        )
