"""
api/drift.py
------------
Prediction-distribution shift monitor using Population Stability Index (PSI).

Overview
--------
Prediction-distribution shift is measured by comparing `probability_placed`
from the *most recent N predictions* against the **baseline distribution**
stored in each model's `baseline_metrics.json`.

PSI Formula
-----------
    PSI = Σ (actual% − expected%) × ln(actual% / expected%)

Thresholds (industry standard):
    PSI < 0.10  → no significant change  → status: "ok"
    PSI 0.10–0.20 → moderate change      → status: "warn"
    PSI > 0.20  → significant shift      → status: "alert"

Additionally we track the raw mean shift from the baseline. This is not direct
performance-drift monitoring: ROC-AUC, calibration, and false-negative rates
require later verified outcome feedback.

Public API
----------
    checker = DriftChecker(logger, model_bundles)
    result  = checker.check(model_key, window=200)
    # → DriftReport(status, psi, mean_shift, baseline_mean,
    #               current_mean, n_predictions, message)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from api.logger import PredictionLogger

# ── PSI configuration ──────────────────────────────────────────────────────
_N_BINS = 10
_BIN_EDGES = [i / _N_BINS for i in range(_N_BINS + 1)]  # 0.0 → 1.0, 10 equal bins

# Avoid log(0) by clipping proportions to a small epsilon
_EPS = 1e-6

# Status thresholds
_WARN_PSI = 0.10
_ALERT_PSI = 0.20
_WARN_SHIFT = 0.05
_ALERT_SHIFT = 0.10

# Minimum predictions required to compute a meaningful PSI
_MIN_PREDICTIONS = 20


# ── Data class for the drift report ───────────────────────────────────────
@dataclass
class DriftReport:
    status: str  # "ok" | "warn" | "alert" | "insufficient_data" | "baseline_unavailable"
    psi: float
    mean_shift: float  # |current_mean − baseline_mean|
    baseline_mean: float
    current_mean: float
    n_predictions: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "psi": round(self.psi, 6),
            "mean_shift": round(self.mean_shift, 6),
            "baseline_mean": round(self.baseline_mean, 6),
            "current_mean": round(self.current_mean, 6),
            "n_predictions": self.n_predictions,
            "message": self.message,
        }


# ── Helper: PSI calculation ────────────────────────────────────────────────
def _compute_psi(
    baseline_probs: list[float],
    current_probs: list[float],
    n_bins: int = _N_BINS,
) -> float:
    """
    Compute PSI between two distributions of probabilities in [0, 1].
    Uses equal-width bins spanning [0, 1].
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    baseline_counts, _ = np.histogram(baseline_probs, bins=edges)
    current_counts, _ = np.histogram(current_probs, bins=edges)

    baseline_pct = baseline_counts / max(len(baseline_probs), 1)
    current_pct = current_counts / max(len(current_probs), 1)

    # Clip to avoid log(0)
    baseline_pct = np.clip(baseline_pct, _EPS, None)
    current_pct = np.clip(current_pct, _EPS, None)

    psi = float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))
    return psi


def _load_baseline_bin_counts(baseline_data: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Validate persisted baseline probability bins without fabricating data."""
    edges = baseline_data.get("probability_bin_edges")
    counts = baseline_data.get("probability_bin_counts")
    if not isinstance(edges, list) or not isinstance(counts, list):
        return None
    if len(edges) != _N_BINS + 1 or len(counts) != _N_BINS + 1 - 1:
        return None
    try:
        edge_array = np.asarray(edges, dtype=float)
        count_array = np.asarray(counts, dtype=float)
    except (TypeError, ValueError):
        return None
    if (
        not np.allclose(edge_array, np.asarray(_BIN_EDGES, dtype=float))
        or (count_array < 0).any()
        or count_array.sum() <= 0
    ):
        return None
    return edge_array, count_array


def _compute_psi_from_counts(
    baseline_counts: np.ndarray, current_probs: list[float], bin_edges: np.ndarray
) -> float:
    """Compute PSI against real persisted histogram counts."""
    current_counts, _ = np.histogram(current_probs, bins=bin_edges)
    baseline_pct = baseline_counts / baseline_counts.sum()
    current_pct = current_counts / max(current_counts.sum(), 1)
    baseline_pct = np.clip(baseline_pct, _EPS, None)
    current_pct = np.clip(current_pct, _EPS, None)
    return float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))


def _status_from_metrics(psi: float, shift: float) -> tuple[str, str]:
    """Derive status string and human-readable message from PSI + shift."""
    if psi > _ALERT_PSI or shift > _ALERT_SHIFT:
        return "alert", (
            f"Significant distribution shift detected "
            f"(PSI={psi:.3f}, shift={shift:.3f}). "
            "Consider retraining the model."
        )
    if psi > _WARN_PSI or shift > _WARN_SHIFT:
        return "warn", (
            f"Moderate distribution shift observed "
            f"(PSI={psi:.3f}, shift={shift:.3f}). "
            "Monitor closely."
        )
    return "ok", (f"No significant drift detected (PSI={psi:.3f}, shift={shift:.3f}).")


# ── Main class ────────────────────────────────────────────────────────────
class DriftChecker:
    """
    Computes PSI drift metrics for a given model using logged predictions.

    Parameters
    ----------
    logger : PredictionLogger
        Must already be initialised (i.e. after `PredictionLogger()` call).
    model_bundles : dict
        The `MODEL_BUNDLES` dict from `api/config.py`.
    """

    def __init__(
        self,
        logger: PredictionLogger,
        model_bundles: dict[str, dict[str, Path]],
    ) -> None:
        self._logger = logger
        self._bundles = model_bundles
        # Cache baseline metrics at init time (they never change at runtime)
        self._baselines: dict[str, dict] = {}
        for model_key, paths in model_bundles.items():
            metrics_path = paths.get("baseline_metrics")
            if metrics_path and Path(metrics_path).exists():
                try:
                    with open(metrics_path) as f:
                        self._baselines[model_key] = json.load(f)
                except Exception as exc:
                    print(f"[Drift] Could not load baseline metrics for {model_key}: {exc}")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def check(self, model_key: str, window: int = 200) -> DriftReport:
        """
        Return a DriftReport for *model_key* using the last *window* predictions.
        """
        # ── Baseline ──────────────────────────────────────────────────
        baseline_data = self._baselines.get(model_key)
        if not baseline_data:
            return DriftReport(
                status="error",
                psi=0.0,
                mean_shift=0.0,
                baseline_mean=0.0,
                current_mean=0.0,
                n_predictions=0,
                message=f"No baseline metrics found for model '{model_key}'.",
            )

        baseline_mean: float = baseline_data.get("mean_probability_placed", 0.0)

        baseline_histogram = _load_baseline_bin_counts(baseline_data)
        if baseline_histogram is None:
            return DriftReport(
                status="baseline_unavailable",
                psi=0.0,
                mean_shift=0.0,
                baseline_mean=baseline_mean,
                current_mean=0.0,
                n_predictions=0,
                message=(
                    f"Model '{model_key}' has no persisted real baseline probability bins. "
                    "Run scripts/generate_baseline_metrics.py after evaluation; "
                    "a synthetic baseline is intentionally not used."
                ),
            )

        # ── Recent predictions ─────────────────────────────────────────
        rows = self._logger.recent(model=model_key, n=window)
        n = len(rows)

        if n < _MIN_PREDICTIONS:
            return DriftReport(
                status="insufficient_data",
                psi=0.0,
                mean_shift=0.0,
                baseline_mean=baseline_mean,
                current_mean=0.0,
                n_predictions=n,
                message=(
                    f"Only {n} predictions logged for '{model_key}' "
                    f"(minimum {_MIN_PREDICTIONS} required). "
                    "Accumulate more predictions before checking drift."
                ),
            )

        current_probs = [row["probability_placed"] for row in rows]
        current_mean = float(np.mean(current_probs))
        mean_shift = abs(current_mean - baseline_mean)

        bin_edges, baseline_counts = baseline_histogram
        psi = _compute_psi_from_counts(baseline_counts, current_probs, bin_edges)
        status, message = _status_from_metrics(psi, mean_shift)

        return DriftReport(
            status=status,
            psi=psi,
            mean_shift=mean_shift,
            baseline_mean=baseline_mean,
            current_mean=current_mean,
            n_predictions=n,
            message=message,
        )
