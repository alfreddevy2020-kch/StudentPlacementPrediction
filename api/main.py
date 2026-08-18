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
    MODEL_BUNDLES,
)
from api.predictor import (
    BasePredictor,
    LogisticRegressionPredictor,
    RandomForestPredictor,
    XGBoostPredictor,
)
from api.schemas import HealthResponse, ModelsResponse, PredictionResponse, StudentInput

# ── Predictor mapping & registry ─────────────────────────────────────────────
PREDICTOR_TYPES: dict[str, type[BasePredictor]] = {
    "logistic_regression": LogisticRegressionPredictor,
    "random_forest": RandomForestPredictor,
    "xgboost": XGBoostPredictor,
}

_predictors: dict[str, BasePredictor] = {}
_model_errors: dict[str, str] = {}


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Startup: deserialise and validate each production model bundle.
    Shutdown: nothing to release (joblib objects are in-memory).
    """
    global _predictors, _model_errors
    _predictors.clear()
    _model_errors.clear()

    print("[API] Initialising production model bundles ...")

    for model_key, bundle in MODEL_BUNDLES.items():
        predictor_cls = PREDICTOR_TYPES.get(model_key)
        if not predictor_cls:
            continue

        try:
            predictor = predictor_cls.load(
                preprocessor_path=bundle["preprocessor"],
                model_path=bundle["model"],
                manifest_path=bundle.get("manifest"),
            )
            _predictors[model_key] = predictor
            print(f"[API] [OK] {predictor.model_display_name} loaded: {bundle['model'].name}")
        except Exception as exc:
            _model_errors[model_key] = str(exc)
            print(f"[API] [FAIL] Error loading model '{model_key}': {exc}")

    loaded_count = len(_predictors)
    total_count = len(MODEL_BUNDLES)

    if loaded_count == total_count:
        print(f"[API] [OK] All {loaded_count}/{total_count} models loaded successfully.")
    elif loaded_count > 0:
        print(f"[API] [WARN] Degraded state: {loaded_count}/{total_count} models loaded.")
    else:
        print("[API] [FAIL] Unavailable state: 0 models loaded.")

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
        "Liveness probe. Returns `status: healthy` when all three models "
        "are loaded, `degraded` when 1-2 models are loaded, and `unavailable` when none are loaded."
    ),
    tags=["System"],
)
def health_check() -> HealthResponse:
    models_loaded = {
        key: (key in _predictors and _predictors[key].is_ready)
        for key in MODEL_BUNDLES.keys()
    }
    loaded_count = sum(models_loaded.values())

    if loaded_count == len(MODEL_BUNDLES):
        status_str = "healthy"
    elif loaded_count > 0:
        status_str = "degraded"
    else:
        status_str = "unavailable"

    return HealthResponse(
        status=status_str,
        models_loaded=models_loaded,
    )


@app.get(
    "/api/v1/models",
    response_model=ModelsResponse,
    summary="List Available Models",
    description="Returns a list of all model identifiers currently loaded and available for prediction.",
    tags=["System"],
)
def list_available_models() -> ModelsResponse:
    available = [k for k, p in _predictors.items() if p.is_ready]
    return ModelsResponse(available_models=available)


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    summary="Predict Student Placement",
    description=(
        "Accepts 15 raw student features and a model selection, applies the "
        "fitted preprocessor (StandardScaler + OneHotEncoder), and returns a "
        "placement prediction from the selected model.\n\n"
        "**Available models:** `logistic_regression`, `random_forest`, `xgboost`.\n\n"
        "**Fields NOT accepted:** `student_id`, `salary_package_lpa`."
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
       and applies its verified ColumnTransformer.
    3. The transformed array is passed to the chosen model's predict/predict_proba.
    4. The result is returned as a structured JSON response.
    """
    model_key = payload.model.value
    predictor = _predictors.get(model_key)

    if predictor is None or not predictor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model '{model_key}' is currently unavailable.",
        )

    try:
        return predictor.predict(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {type(exc).__name__}: {exc}",
        )

