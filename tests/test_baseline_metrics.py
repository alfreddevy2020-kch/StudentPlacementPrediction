"""Tests for persisted real probability-bin monitoring baselines."""

from __future__ import annotations

import numpy as np

from scripts.generate_baseline_metrics import (
    PROBABILITY_BIN_EDGES,
    probability_bin_counts,
)


def test_probability_bin_counts_partition_all_held_out_predictions():
    probabilities = np.array([0.0, 0.05, 0.25, 0.75, 0.95, 1.0])
    counts = probability_bin_counts(probabilities)

    assert len(counts) == len(PROBABILITY_BIN_EDGES) - 1
    assert sum(counts) == len(probabilities)
    assert all(count >= 0 for count in counts)
