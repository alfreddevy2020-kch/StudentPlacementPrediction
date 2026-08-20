"""Regression tests for selected-model scenario scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from frontend.simulator import CohortWhatIfSimulator


class RecordingPredictor:
    """Predictor stub that records every requested model."""

    def __init__(self) -> None:
        self.models_requested: list[str | None] = []

    def predict_probabilities(self, df: pd.DataFrame, model_name: str | None = None) -> np.ndarray:
        self.models_requested.append(model_name)
        # The edited profile scores differently so both simulation paths run.
        return np.full(len(df), 0.40 if df["cgpa"].iloc[0] < 8.0 else 0.60)


@pytest.mark.parametrize("model_name", ["Logistic Regression", "Random Forest", "XGBoost"])
def test_simulator_forwards_selected_model_to_baseline_and_scenario(model_name: str):
    predictor = RecordingPredictor()
    simulator = CohortWhatIfSimulator(predictor)  # type: ignore[arg-type]
    cohort = pd.DataFrame(
        {
            "student_id": [1, 2],
            "cgpa": [7.0, 7.5],
            "placement_training": ["No", "Yes"],
        }
    )

    result = simulator.simulate_policy_intervention(
        cohort, interventions={"cgpa": 1.0}, model_name=model_name
    )

    assert predictor.models_requested == [model_name, model_name]
    assert result["cohort_size"] == 2
    assert result["newly_placed_count"] == 2
