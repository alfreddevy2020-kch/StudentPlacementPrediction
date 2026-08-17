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

from api.config import APP_DESCRIPTION, APP_TITLE, APP_VERSION, MODEL_PATH, PREPROCESSOR_PATH
from api.predictor import RandomForestPredictor
from api.schemas import HealthResponse, PredictionResponse, StudentInput

# ── Singleton predictor ──────────────────────────────────────────────────────
# Loaded once at startup; shared across all requests.
_predictor: RandomForestPredictor | None = None


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Startup: deserialise preprocessor and model artifacts.
    Shutdown: nothing to release (joblib objects are in-memory).
    """
    global _predictor
    try:
        _predictor = RandomForestPredictor.load(PREPROCESSOR_PATH, MODEL_PATH)
        print(f"[API] [OK] Preprocessor loaded: {PREPROCESSOR_PATH.name}")
        print(f"[API] [OK] Model loaded:         {MODEL_PATH.name}")
    except FileNotFoundError as exc:
        # Artifact missing — server starts in a degraded state.
        # /health will report degraded; /predict will return 503.
        print(f"[API] [FAIL] Artifact not found: {exc}")
        _predictor = None
    except Exception as exc:
        print(f"[API] [FAIL] Unexpected error loading artifacts: {exc}")
        _predictor = None

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
        "Liveness probe. Returns `status: healthy` when both the preprocessor "
        "and the Random Forest model are loaded and ready to serve predictions."
    ),
    tags=["System"],
)
def health_check() -> HealthResponse:
    preprocessor_ok = _predictor is not None and _predictor._preprocessor is not None
    model_ok        = _predictor is not None and _predictor._model is not None
    return HealthResponse(
        status="healthy" if (preprocessor_ok and model_ok) else "degraded",
        preprocessor_loaded=preprocessor_ok,
        model_loaded=model_ok,
    )


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    summary="Predict Student Placement",
    description=(
        "Accepts 15 raw student features, applies the fitted preprocessor "
        "(StandardScaler + OneHotEncoder), and returns a placement prediction "
        "from the trained Random Forest model.\n\n"
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

    1. Pydantic validates the 15 raw features (types + value ranges + categorical literals).
    2. `predictor.predict()` builds a one-row DataFrame and applies the ColumnTransformer.
    3. The transformed array is passed to `RandomForestClassifier.predict / predict_proba`.
    4. The result is returned as a structured JSON response.
    """
    if _predictor is None or not _predictor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model artifacts are not loaded. "
                "Check /health for details and ensure the server started correctly."
            ),
        )

    try:
        return _predictor.predict(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {type(exc).__name__}: {exc}",
        )
