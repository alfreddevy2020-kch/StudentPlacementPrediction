"""
tests/test_logger.py
--------------------
Unit tests for PredictionLogger (SQLite).
Uses a temporary in-memory / temp-file database — never touches logs/predictions.db.
"""

from pathlib import Path

import pytest

from api.logger import PredictionLogger
from api.schemas import ModelName, PredictionResponse, StudentInput

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_INPUT = StudentInput(
    model=ModelName.random_forest,
    ssc_percentage=75.5,
    hsc_percentage=78.0,
    degree_percentage=72.0,
    cgpa=8.2,
    attendance_percentage=90.0,
    backlogs=0,
    entrance_exam_score=85.0,
    technical_skill_score=80.0,
    soft_skill_score=75.0,
    certifications=3,
    live_projects=1,
    internship_count=2,
    work_experience_months=6,
    gender="Male",
    extracurricular_activities="Yes",
)

VALID_RESPONSE = PredictionResponse(
    model_used="Random Forest",
    placement_status=1,
    placement_label="Placed",
    probability_placed=0.85,
    probability_not_placed=0.15,
    risk_level="High Probability of Placement (Low Risk)",
)


@pytest.fixture
def tmp_logger(tmp_path: Path) -> PredictionLogger:
    """PredictionLogger backed by a temp file (isolated per test)."""
    return PredictionLogger(db_path=tmp_path / "test_predictions.db")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPredictionLogger:
    def test_schema_created_on_init(self, tmp_logger: PredictionLogger):
        # Table must exist; total_count should return 0
        assert tmp_logger.total_count() == 0

    def test_log_single_prediction(self, tmp_logger: PredictionLogger):
        tmp_logger.log(VALID_INPUT, VALID_RESPONSE)
        assert tmp_logger.total_count() == 1

    def test_log_multiple_predictions(self, tmp_logger: PredictionLogger):
        for _ in range(5):
            tmp_logger.log(VALID_INPUT, VALID_RESPONSE)
        assert tmp_logger.total_count() == 5

    def test_recent_returns_correct_count(self, tmp_logger: PredictionLogger):
        for _ in range(10):
            tmp_logger.log(VALID_INPUT, VALID_RESPONSE)
        rows = tmp_logger.recent(n=5)
        assert len(rows) == 5

    def test_recent_filtered_by_model(self, tmp_logger: PredictionLogger):
        # Log 3 random_forest, 2 xgboost
        rf_input = VALID_INPUT.model_copy(update={"model": ModelName.random_forest})
        xgb_input = VALID_INPUT.model_copy(update={"model": ModelName.xgboost})

        for _ in range(3):
            tmp_logger.log(rf_input, VALID_RESPONSE)
        for _ in range(2):
            tmp_logger.log(xgb_input, VALID_RESPONSE)

        assert tmp_logger.total_count(model="random_forest") == 3
        assert tmp_logger.total_count(model="xgboost") == 2

    def test_recent_row_has_expected_keys(self, tmp_logger: PredictionLogger):
        tmp_logger.log(VALID_INPUT, VALID_RESPONSE)
        rows = tmp_logger.recent(n=1)
        row = rows[0]
        assert "probability_placed" in row
        assert "placement_status" in row
        assert "model_used" in row
        assert "timestamp" in row

    def test_log_probability_value_persisted(self, tmp_logger: PredictionLogger):
        tmp_logger.log(VALID_INPUT, VALID_RESPONSE)
        rows = tmp_logger.recent(n=1)
        assert rows[0]["probability_placed"] == pytest.approx(0.85)

    def test_log_never_raises_on_invalid(self, tmp_logger: PredictionLogger):
        """log() must silently swallow errors, never raise to caller."""
        tmp_logger._db_path = Path("/nonexistent/path/db.sqlite")
        if hasattr(tmp_logger._local, "conn"):
            tmp_logger._local.conn = None
        # Should not raise
        tmp_logger.log(VALID_INPUT, VALID_RESPONSE)
