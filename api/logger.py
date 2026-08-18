"""
api/logger.py
-------------
Thread-safe SQLite logger that persists every prediction to disk.

Schema
------
    predictions(
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp               TEXT,       -- ISO-8601 UTC
        model_used              TEXT,
        -- input features --
        ssc_percentage          REAL,
        hsc_percentage          REAL,
        degree_percentage       REAL,
        cgpa                    REAL,
        attendance_percentage   REAL,
        backlogs                INTEGER,
        entrance_exam_score     REAL,
        technical_skill_score   REAL,
        soft_skill_score        REAL,
        certifications          INTEGER,
        live_projects           INTEGER,
        internship_count        INTEGER,
        work_experience_months  INTEGER,
        gender                  TEXT,
        extracurricular_activities TEXT,
        -- output --
        probability_placed      REAL,
        placement_status        INTEGER
    )

Usage
-----
    logger = PredictionLogger()          # call once at startup
    logger.log(student_input, response)  # call after every prediction
    rows   = logger.recent(model="xgboost", n=200)
"""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from api.schemas import PredictionResponse, StudentInput

# Default DB path; override via PREDICTION_LOG_DB env var
_DEFAULT_DB = Path(__file__).resolve().parent.parent / "logs" / "predictions.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp                   TEXT    NOT NULL,
    model_used                  TEXT    NOT NULL,
    ssc_percentage              REAL,
    hsc_percentage              REAL,
    degree_percentage           REAL,
    cgpa                        REAL,
    attendance_percentage       REAL,
    backlogs                    INTEGER,
    entrance_exam_score         REAL,
    technical_skill_score       REAL,
    soft_skill_score            REAL,
    certifications              INTEGER,
    live_projects               INTEGER,
    internship_count            INTEGER,
    work_experience_months      INTEGER,
    gender                      TEXT,
    extracurricular_activities  TEXT,
    probability_placed          REAL,
    placement_status            INTEGER
);
"""

_INSERT_SQL = """
INSERT INTO predictions (
    timestamp, model_used,
    ssc_percentage, hsc_percentage, degree_percentage, cgpa,
    attendance_percentage, backlogs, entrance_exam_score,
    technical_skill_score, soft_skill_score, certifications,
    live_projects, internship_count, work_experience_months,
    gender, extracurricular_activities,
    probability_placed, placement_status
) VALUES (
    :timestamp, :model_used,
    :ssc_percentage, :hsc_percentage, :degree_percentage, :cgpa,
    :attendance_percentage, :backlogs, :entrance_exam_score,
    :technical_skill_score, :soft_skill_score, :certifications,
    :live_projects, :internship_count, :work_experience_months,
    :gender, :extracurricular_activities,
    :probability_placed, :placement_status
);
"""


class PredictionLogger:
    """Thread-safe, persistent prediction log backed by SQLite."""

    def __init__(self, db_path: Path | None = None) -> None:
        resolved = Path(os.getenv("PREDICTION_LOG_DB", str(db_path or _DEFAULT_DB)))
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = resolved
        self._local = threading.local()   # per-thread connection
        self._init_schema()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Return (or create) the per-thread SQLite connection."""
        if not getattr(self._local, "conn", None):
            self._local.conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,  # we manage threading ourselves
                isolation_level=None,     # autocommit
            )
            self._local.conn.execute("PRAGMA journal_mode=WAL;")
        return self._local.conn

    def _init_schema(self) -> None:
        conn = self._connect()
        conn.execute(_CREATE_TABLE_SQL)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, payload: StudentInput, response: PredictionResponse) -> None:
        """Persist one prediction row (best-effort; never raises to caller)."""
        try:
            conn = self._connect()
            conn.execute(
                _INSERT_SQL,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model_used": payload.model.value,
                    "ssc_percentage": payload.ssc_percentage,
                    "hsc_percentage": payload.hsc_percentage,
                    "degree_percentage": payload.degree_percentage,
                    "cgpa": payload.cgpa,
                    "attendance_percentage": payload.attendance_percentage,
                    "backlogs": payload.backlogs,
                    "entrance_exam_score": payload.entrance_exam_score,
                    "technical_skill_score": payload.technical_skill_score,
                    "soft_skill_score": payload.soft_skill_score,
                    "certifications": payload.certifications,
                    "live_projects": payload.live_projects,
                    "internship_count": payload.internship_count,
                    "work_experience_months": payload.work_experience_months,
                    "gender": payload.gender,
                    "extracurricular_activities": payload.extracurricular_activities,
                    "probability_placed": response.probability_placed,
                    "placement_status": response.placement_status,
                },
            )
        except Exception as exc:  # noqa: BLE001
            # Never let logging failures bubble up to the caller
            print(f"[Logger] Warning — could not write prediction log: {exc}")

    def recent(
        self,
        model: str | None = None,
        n: int = 200,
    ) -> list[dict]:
        """Return the most recent *n* rows, optionally filtered by model."""
        conn = self._connect()
        if model:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE model_used = ? ORDER BY id DESC LIMIT ?",
                (model, n),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
                (n,),
            ).fetchall()
        cols = [d[0] for d in conn.execute("SELECT * FROM predictions LIMIT 0").description or []]
        # Get column names properly
        cursor = conn.execute("SELECT * FROM predictions LIMIT 0")
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, row)) for row in rows]

    def total_count(self, model: str | None = None) -> int:
        """Return total number of logged predictions (optionally per model)."""
        conn = self._connect()
        if model:
            row = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE model_used = ?", (model,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()
        return row[0] if row else 0
