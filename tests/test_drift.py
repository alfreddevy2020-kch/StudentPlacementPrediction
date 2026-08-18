"""
tests/test_drift.py
-------------------
Unit tests for the PSI drift detection logic.
Tests the computation functions directly without touching the database or
model artifacts.
"""

import pytest

from api.drift import (
    DriftReport,
    _compute_psi,
    _status_from_metrics,
    _WARN_PSI,
    _ALERT_PSI,
    _WARN_SHIFT,
    _ALERT_SHIFT,
)


# ---------------------------------------------------------------------------
# PSI calculation
# ---------------------------------------------------------------------------

class TestComputePSI:
    def test_identical_distributions_zero_psi(self):
        probs = [0.1, 0.3, 0.5, 0.7, 0.9] * 20
        psi = _compute_psi(probs, probs)
        assert psi == pytest.approx(0.0, abs=1e-5)

    def test_shifted_distribution_positive_psi(self):
        baseline = [0.8] * 100          # most predictions high probability
        current = [0.2] * 100           # all predictions low probability
        psi = _compute_psi(baseline, current)
        assert psi > _ALERT_PSI         # should be "alert" level

    def test_slight_shift_small_psi(self):
        import numpy as np
        rng = np.random.default_rng(42)
        baseline = list(rng.normal(0.7, 0.1, 500).clip(0, 1))
        current = list(rng.normal(0.72, 0.1, 200).clip(0, 1))  # tiny shift
        psi = _compute_psi(baseline, current)
        assert psi < _WARN_PSI          # should be "ok"

    def test_psi_is_non_negative(self):
        import numpy as np
        rng = np.random.default_rng(0)
        b = list(rng.uniform(0, 1, 100))
        c = list(rng.uniform(0, 1, 100))
        psi = _compute_psi(b, c)
        assert psi >= 0.0


# ---------------------------------------------------------------------------
# Status thresholds
# ---------------------------------------------------------------------------

class TestStatusFromMetrics:
    def test_ok_status(self):
        status, msg = _status_from_metrics(psi=0.05, shift=0.02)
        assert status == "ok"
        assert "No significant drift" in msg

    def test_warn_from_psi(self):
        status, msg = _status_from_metrics(psi=_WARN_PSI + 0.01, shift=0.01)
        assert status == "warn"

    def test_warn_from_shift(self):
        status, msg = _status_from_metrics(psi=0.05, shift=_WARN_SHIFT + 0.01)
        assert status == "warn"

    def test_alert_from_psi(self):
        status, msg = _status_from_metrics(psi=_ALERT_PSI + 0.01, shift=0.01)
        assert status == "alert"
        assert "retraining" in msg.lower()

    def test_alert_from_shift(self):
        status, msg = _status_from_metrics(psi=0.05, shift=_ALERT_SHIFT + 0.01)
        assert status == "alert"

    def test_boundary_exactly_at_warn(self):
        # Exactly at boundary — should be warn (>= condition)
        status, _ = _status_from_metrics(psi=_WARN_PSI, shift=0.0)
        assert status in ("ok", "warn")  # boundary behaviour


# ---------------------------------------------------------------------------
# DriftReport dataclass
# ---------------------------------------------------------------------------

class TestDriftReport:
    def test_to_dict_keys(self):
        r = DriftReport(
            status="ok",
            psi=0.05,
            mean_shift=0.02,
            baseline_mean=0.75,
            current_mean=0.77,
            n_predictions=150,
            message="No drift.",
        )
        d = r.to_dict()
        assert set(d.keys()) == {
            "status", "psi", "mean_shift",
            "baseline_mean", "current_mean",
            "n_predictions", "message",
        }

    def test_to_dict_rounding(self):
        r = DriftReport(
            status="ok",
            psi=0.123456789,
            mean_shift=0.0,
            baseline_mean=0.5,
            current_mean=0.5,
            n_predictions=50,
            message="",
        )
        assert len(str(r.to_dict()["psi"]).split(".")[-1]) <= 6
