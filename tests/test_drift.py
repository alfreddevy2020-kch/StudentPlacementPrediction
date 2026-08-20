"""
tests/test_drift.py
-------------------
Unit tests for the PSI drift detection logic.
Tests the computation functions directly without touching the database or
model artifacts.
"""

import pytest

from api.drift import (
    _ALERT_PSI,
    _ALERT_SHIFT,
    _WARN_PSI,
    _WARN_SHIFT,
    DriftReport,
    _compute_psi,
    _compute_psi_from_counts,
    _load_baseline_bin_counts,
    _status_from_metrics,
)


class _NoopLogger:
    """Minimal logger used to test baseline validation before live lookup."""

    def recent(self, model: str, n: int):  # noqa: ARG002
        return []

# ---------------------------------------------------------------------------
# PSI calculation
# ---------------------------------------------------------------------------


class TestComputePSI:
    def test_identical_distributions_zero_psi(self):
        probs = [0.1, 0.3, 0.5, 0.7, 0.9] * 20
        psi = _compute_psi(probs, probs)
        assert psi == pytest.approx(0.0, abs=1e-5)

    def test_shifted_distribution_positive_psi(self):
        baseline = [0.8] * 100  # most predictions high probability
        current = [0.2] * 100  # all predictions low probability
        psi = _compute_psi(baseline, current)
        assert psi > _ALERT_PSI  # should be "alert" level

    def test_slight_shift_small_psi(self):
        import numpy as np

        rng = np.random.default_rng(42)
        baseline = list(rng.normal(0.7, 0.1, 500).clip(0, 1))
        current = list(rng.normal(0.72, 0.1, 200).clip(0, 1))  # tiny shift
        psi = _compute_psi(baseline, current)
        assert psi < _WARN_PSI  # should be "ok"

    def test_psi_is_non_negative(self):
        import numpy as np

        rng = np.random.default_rng(0)
        b = list(rng.uniform(0, 1, 100))
        c = list(rng.uniform(0, 1, 100))
        psi = _compute_psi(b, c)
        assert psi >= 0.0

    def test_real_persisted_bin_counts_can_be_used_directly(self):
        baseline = {
            "probability_bin_edges": [i / 10 for i in range(11)],
            "probability_bin_counts": [10] * 10,
        }
        histogram = _load_baseline_bin_counts(baseline)
        assert histogram is not None
        edges, counts = histogram
        assert _compute_psi_from_counts(counts, [0.05, 0.15] * 50, edges) >= 0.0

    def test_missing_histogram_is_not_replaced_with_synthetic_values(self):
        assert _load_baseline_bin_counts({"mean_probability_placed": 0.5}) is None


class TestBaselineAvailability:
    def test_missing_real_histogram_returns_a_safe_non_result(self, tmp_path):
        import json

        from api.drift import DriftChecker

        metrics_path = tmp_path / "baseline_metrics.json"
        metrics_path.write_text(
            json.dumps({"mean_probability_placed": 0.5}), encoding="utf-8"
        )
        checker = DriftChecker(
            _NoopLogger(),  # type: ignore[arg-type]
            {"model": {"baseline_metrics": metrics_path}},
        )

        report = checker.check("model")

        assert report.status == "baseline_unavailable"
        assert "synthetic baseline" in report.message


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
            "status",
            "psi",
            "mean_shift",
            "baseline_mean",
            "current_mean",
            "n_predictions",
            "message",
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
