"""
api/logger.py
-------------
Thread-safe SQLite logger that persists every prediction to disk.

Schema
------
    predictions(
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp           TEXT,       -- ISO-8601 UTC
        model_used          TEXT,
        <one column per raw input feature>,
        probability_placed  REAL,
        placement_status    INTEGER
    )

The feature columns are generated from RAW_NUMERICAL_FEATURES /
RAW_CATEGORICAL_FEATURES in feature_engineering.py rather than hardcoded,
so a dataset change cannot silently desync the log schema from the API
schema. If an existing database predates the current feature set, it is
detected and rebuilt on startup (see _init_schema).

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
from feature_engineering import RAW_CATEGORICAL_FEATURES, RAW_NUMERICAL_FEATURES

# Default DB path; override via PREDICTION_LOG_DB env var
_DEFAULT_DB = Path(__file__).resolve().parent.parent / "logs" / "predictions.db"

_FEATURE_COLUMNS = RAW_NUMERICAL_FEATURES + RAW_CATEGORICAL_FEATURES
_FIXED_COLUMNS = ["timestamp", "model_used", "probability_placed", "placement_status"]

_feature_ddl = ",\n    ".join(
    f"{name:<28} {'TEXT' if name in RAW_CATEGORICAL_FEATURES else 'REAL'}"
    for name in _FEATURE_COLUMNS
)

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS predictions (
    id                           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp                    TEXT    NOT NULL,
    model_used                   TEXT    NOT NULL,
    {_feature_ddl},
    probability_placed           REAL,
    placement_status             INTEGER
);
"""

_insert_columns = ["timestamp", "model_used", *_FEATURE_COLUMNS,
                   "probability_placed", "placement_status"]

_INSERT_SQL = (
    f"INSERT INTO predictions ({', '.join(_insert_columns)}) "
    f"VALUES ({', '.join(':' + c for c in _insert_columns)});"
)


class PredictionLogger:
    """Thread-safe, persistent prediction log backed by SQLite."""

    def __init__(self, db_path: Path | None = None) -> None:
        resolved = Path(os.getenv("PREDICTION_LOG_DB", str(db_path or _DEFAULT_DB)))
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = resolved
        self._local = threading.local()  # per-thread connection
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
                isolation_level=None,  # autocommit
            )
            self._local.conn.execute("PRAGMA journal_mode=WAL;")
        return self._local.conn

    def _init_schema(self) -> None:
        conn = self._connect()
        # A database written against an older feature set would silently
        # reject every insert (log() swallows errors), so detect the
        # mismatch and rebuild rather than logging nothing forever.
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(predictions);").fetchall()
        }
        if existing and not set(_insert_columns).issubset(existing):
            print(
                "[Logger] Existing predictions table predates the current "
                "feature schema - archiving it as predictions_legacy."
            )
            conn.execute("DROP TABLE IF EXISTS predictions_legacy;")
            conn.execute("ALTER TABLE predictions RENAME TO predictions_legacy;")
        conn.execute(_CREATE_TABLE_SQL)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, payload: StudentInput, response: PredictionResponse) -> None:
        """Persist one prediction row (best-effort; never raises to caller)."""
        try:
            conn = self._connect()
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_used": payload.model.value,
                "probability_placed": response.probability_placed,
                "placement_status": response.placement_status,
            }
            for name in _FEATURE_COLUMNS:
                row[name] = getattr(payload, name)
            conn.execute(_INSERT_SQL, row)
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
