"""
tests/test_api.py
-----------------
Integration tests for the FastAPI application using TestClient.
Requires that artifacts/production/ contains valid .joblib files
(same requirement as the running server).

If artifacts are not available (e.g. in CI without .joblib),
tests that call the predict endpoint will be skipped automatically.
"""

import pytest
from fastapi.testclient import TestClient

from api.config import MODEL_BUNDLES

# Detect whether production artifacts exist
_ARTIFACTS_AVAILABLE = all(bundle["model"].exists() for bundle in MODEL_BUNDLES.values())

# Only import/create app if needed (avoids startup errors when not testing API)
try:
    from api.main import app

    _APP_IMPORTABLE = True
except Exception:
    _APP_IMPORTABLE = False

requires_artifacts = pytest.mark.skipif(
    not (_ARTIFACTS_AVAILABLE and _APP_IMPORTABLE),
    reason="Production model artifacts not present — skipping artifact-dependent tests",
)

VALID_PAYLOAD = {
    "model": "random_forest",
    "ssc_percentage": 75.5,
    "hsc_percentage": 78.0,
    "degree_percentage": 72.0,
    "cgpa": 8.2,
    "attendance_percentage": 90.0,
    "backlogs": 0,
    "entrance_exam_score": 85.0,
    "technical_skill_score": 80.0,
    "soft_skill_score": 75.0,
    "certifications": 3,
    "live_projects": 1,
    "internship_count": 2,
    "work_experience_months": 6,
    "gender": "Male",
    "extracurricular_activities": "Yes",
}


@pytest.fixture(scope="module")
def client():
    """Return a TestClient with a fully initialised app (lifespan runs)."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health + Models (no artifacts needed for schema check, but app must import)
# ---------------------------------------------------------------------------


@requires_artifacts
class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_schema(self, client: TestClient):
        data = client.get("/health").json()
        assert "status" in data
        assert "models_loaded" in data
        assert data["status"] in ("healthy", "degraded", "unavailable")

    def test_models_loaded_contains_all_keys(self, client: TestClient):
        data = client.get("/health").json()
        for key in MODEL_BUNDLES:
            assert key in data["models_loaded"]


@requires_artifacts
class TestModelsEndpoint:
    def test_returns_list(self, client: TestClient):
        r = client.get("/api/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert "available_models" in data
        assert isinstance(data["available_models"], list)


# ---------------------------------------------------------------------------
# Predict endpoint
# ---------------------------------------------------------------------------


@requires_artifacts
class TestPredictEndpoint:
    def test_valid_prediction_200(self, client: TestClient):
        r = client.post("/api/v1/predict", json=VALID_PAYLOAD)
        assert r.status_code == 200

    def test_response_schema(self, client: TestClient):
        data = client.post("/api/v1/predict", json=VALID_PAYLOAD).json()
        assert "placement_status" in data
        assert "probability_placed" in data
        assert "risk_level" in data
        assert data["placement_status"] in (0, 1)
        assert 0.0 <= data["probability_placed"] <= 1.0

    def test_xgboost_model(self, client: TestClient):
        payload = {**VALID_PAYLOAD, "model": "xgboost"}
        r = client.post("/api/v1/predict", json=payload)
        assert r.status_code == 200

    def test_logistic_regression_model(self, client: TestClient):
        payload = {**VALID_PAYLOAD, "model": "logistic_regression"}
        r = client.post("/api/v1/predict", json=payload)
        assert r.status_code == 200

    def test_missing_field_422(self, client: TestClient):
        bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "cgpa"}
        r = client.post("/api/v1/predict", json=bad)
        assert r.status_code == 422

    def test_out_of_range_cgpa_422(self, client: TestClient):
        bad = {**VALID_PAYLOAD, "cgpa": 99.9}
        r = client.post("/api/v1/predict", json=bad)
        assert r.status_code == 422

    def test_invalid_model_422(self, client: TestClient):
        bad = {**VALID_PAYLOAD, "model": "gpt4"}
        r = client.post("/api/v1/predict", json=bad)
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Drift + Log summary endpoints
# ---------------------------------------------------------------------------


@requires_artifacts
class TestDriftEndpoint:
    def test_drift_returns_200(self, client: TestClient):
        r = client.get("/api/v1/drift?model=xgboost")
        assert r.status_code == 200

    def test_drift_schema(self, client: TestClient):
        data = client.get("/api/v1/drift?model=xgboost").json()
        assert "status" in data
        assert data["status"] in ("ok", "warn", "alert", "insufficient_data", "error")

    def test_drift_invalid_model_422(self, client: TestClient):
        r = client.get("/api/v1/drift?model=neural_network")
        assert r.status_code == 422

    def test_log_summary_200(self, client: TestClient):
        r = client.get("/logs/summary")
        assert r.status_code == 200
        data = r.json()
        assert "total" in data
        assert "by_model" in data
