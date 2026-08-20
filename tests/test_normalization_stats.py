"""
tests/test_normalization_stats.py
---------------------------------
Regression tests for the frozen normalization maxima.

Background: these stats live in data/processed/, which is gitignored. When a
bundle did not carry its own copy, engineer_features() inferred the maxima
from whatever batch it was handed. For a single-row API request that divides
each value by itself, pinning every *_normalized feature to 1.0 and reporting
portfolio_strength as 100 instead of its true value — wrong answers, HTTP 200,
no warning.

These tests lock in the two properties that prevent a recurrence:
  1. Missing stats raise instead of being inferred.
  2. Every production bundle ships its own checksummed copy.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from api.config import MODEL_BUNDLES
from feature_engineering import (
    NORM_STATS_KEYS,
    engineer_features,
    load_normalization_stats,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

RAW_ROW = {
    "cgpa": [7.7],
    "ssc_percentage": [70.0],
    "hsc_percentage": [74.0],
    "degree_percentage": [72.0],
    "aptitude_test_score": [80.0],
    "technical_skill_score": [75.0],
    "soft_skills_rating": [4.4],
    "attendance_percentage": [85.0],
    "backlogs": [0],
    "internships": [1],
    "projects": [2],
    "certifications": [1],
    "work_experience_months": [6],
    "gender": ["Male"],
    "department": ["CS"],
    "extracurricular_activities": ["Yes"],
    "placement_training": ["Yes"],
}

# Matches data/processed/normalization_stats.json for the current dataset.
TRAINING_STATS = {
    "internships_max": 2.0,
    "projects_max": 4.0,
    "certifications_max": 2.0,
    "work_experience_months_max": 12.0,
}


class TestExplicitStats:
    def test_scales_by_the_frozen_maximum_not_the_batch(self):
        """A single row must be scaled by the training maxima, not its own."""
        out = engineer_features(pd.DataFrame(RAW_ROW), stats=TRAINING_STATS)
        assert out["internships_normalized"][0] == pytest.approx(1 / 2)
        assert out["projects_normalized"][0] == pytest.approx(2 / 4)
        assert out["certifications_normalized"][0] == pytest.approx(1 / 2)
        assert out["work_experience_normalized"][0] == pytest.approx(6 / 12)
        # (1/2 + 2/4 + 1/2 + 6/12) / 4 * 100 = 2.0 / 4 * 100 = 50.0
        assert out["portfolio_strength"][0] == pytest.approx(50.0)

    def test_single_row_matches_the_same_row_inside_a_cohort(self):
        """Cohort membership must not change a student's engineered features."""
        cohort = pd.DataFrame(
            {k: v * 3 for k, v in RAW_ROW.items()}
        )
        cohort.loc[1, "internships"] = 0
        cohort.loc[2, "internships"] = 2

        single = engineer_features(pd.DataFrame(RAW_ROW), stats=TRAINING_STATS)
        batch = engineer_features(cohort, stats=TRAINING_STATS)

        assert single["portfolio_strength"][0] == pytest.approx(
            batch["portfolio_strength"][0]
        )


class TestMissingStatsAreLoud:
    def test_missing_stats_raise_rather_than_infer(self, monkeypatch):
        """The silent batch-max fallback must stay removed."""
        monkeypatch.setattr(
            "feature_engineering._load_normalization_stats", lambda: None
        )
        with pytest.raises(RuntimeError, match="Normalization stats not found"):
            engineer_features(pd.DataFrame(RAW_ROW))

    def test_load_rejects_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_normalization_stats(tmp_path / "nope.json")

    def test_load_rejects_incomplete_file(self, tmp_path: Path):
        bad = tmp_path / "normalization_stats.json"
        bad.write_text(json.dumps({"internships_max": 2.0}), encoding="utf-8")
        with pytest.raises(ValueError, match="missing keys"):
            load_normalization_stats(bad)


class TestBundlesAreSelfContained:
    """Each production bundle must work on a clone with no data/ directory."""

    @pytest.mark.parametrize("model_key", sorted(MODEL_BUNDLES))
    def test_bundle_ships_normalization_stats(self, model_key: str):
        path = MODEL_BUNDLES[model_key]["normalization_stats"]
        assert path.exists(), (
            f"{model_key} bundle has no {path.name}. Re-package with "
            "scripts/package_model.py — data/processed/ is gitignored, so a "
            "bundle without its own copy breaks on a fresh clone."
        )
        stats = load_normalization_stats(path)
        assert all(k in stats for k in NORM_STATS_KEYS)

    @pytest.mark.parametrize("model_key", sorted(MODEL_BUNDLES))
    def test_manifest_records_the_stats_checksum(self, model_key: str):
        manifest_path = MODEL_BUNDLES[model_key]["manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest.get("artifacts", {}).get("normalization_stats")
        assert entry, f"{model_key} manifest does not checksum its stats file"
        assert entry.get("sha256")

    @pytest.mark.parametrize("model_key", sorted(MODEL_BUNDLES))
    def test_checksum_matches_the_file_on_disk(self, model_key: str):
        """Catch a stale manifest before CI does."""
        bundle = MODEL_BUNDLES[model_key]
        manifest = json.loads(bundle["manifest"].read_text(encoding="utf-8"))
        expected = manifest["artifacts"]["normalization_stats"]["sha256"]
        actual = hashlib.sha256(bundle["normalization_stats"].read_bytes()).hexdigest()
        assert actual == expected, (
            f"{model_key}: manifest checksum does not match the file. "
            "Re-run scripts/package_model.py."
        )

    @pytest.mark.parametrize("model_key", sorted(MODEL_BUNDLES))
    def test_stats_file_uses_lf_line_endings(self, model_key: str):
        """CRLF would change the SHA-256 between a Windows commit and a Linux
        CI checkout, so bundle verification would fail there but pass locally.
        .gitattributes marks artifacts/production/** as -text to stop git
        rewriting these bytes; this asserts the writer cooperates."""
        raw = MODEL_BUNDLES[model_key]["normalization_stats"].read_bytes()
        assert b"\r\n" not in raw, (
            f"{model_key} normalization_stats.json contains CRLF. It is "
            "checksummed, so its bytes must be identical on every platform."
        )
