"""
api/drift.py
------------
Population Stability Index (PSI) drift detector.

Overview
--------
Drift is measured by comparing the distribution of `probability_placed`
from the *most recent N predictions* against the **baseline distribution**
stored in each model's `baseline_metrics.json`.

PSI Formula
-----------
    PSI = Σ (actual% − expected%) × ln(actual% / expected%)

Thresholds (industry standard):
    PSI < 0.10  → no significant change  → status: "ok"
    PSI 0.10–0.20 → moderate change      → status: "warn"
    PSI > 0.20  → significant shift      → status: "alert"

Additionally we track the raw mean shift from the baseline.

Public API
----------
    checker = DriftChecker(logger, model_bundles)
    result  = checker.check(model_key, window=200)
    # → DriftReport(status, psi, mean_shift, baseline_mean,
    #               current_mean, n_predictions, message)
"""

from __future__ import annotations

import json
import math
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
    status: str          # "ok" | "warn" | "alert" | "insufficient_data"
    psi: float
    mean_shift: float    # |current_mean − baseline_mean|
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
    return "ok", (
        f"No significant drift detected "
        f"(PSI={psi:.3f}, shift={shift:.3f})."
    )


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

        # Reconstruct a synthetic baseline distribution around the baseline mean
        # using the saved probability distribution if present, else synthesise
        baseline_probs: list[float] = baseline_data.get("probability_distribution", [])
        if not baseline_probs:
            # Fallback: generate a synthetic normal distribution around the mean
            rng = np.random.default_rng(42)
            std = baseline_data.get("std_probability_placed", 0.15)
            baseline_probs = list(
                np.clip(rng.normal(loc=baseline_mean, scale=std, size=1000), 0.0, 1.0)
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

        psi = _compute_psi(baseline_probs, current_probs)
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
