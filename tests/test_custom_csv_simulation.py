import sys
from pathlib import Path

import numpy as np

# frontend/ modules import feature_engineering from the repo root, so both
# directories must be on sys.path before the local imports below.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "frontend"))

from batch_predictor import BatchPredictor
from simulator import CohortWhatIfSimulator

from feature_engineering import TARGET_COLUMN, normalize_columns


def test_custom_csv_simulation():
    # 1. Load trained BatchPredictor and Simulator
    predictor = BatchPredictor()
    predictor.load()
    default_df = predictor.load_dataset()
    simulator = CohortWhatIfSimulator(predictor)

    # 2. Simulate user uploading a custom CSV (e.g., 25 custom student rows)
    sample_df = default_df.head(25).copy()

    # Test case A: User CSV with standard raw column headers but no PlacementStatus (unlabelled inference)
    unlabelled_df = sample_df.drop(columns=[TARGET_COLUMN], errors="ignore")
    # Drop student_id to verify auto-synthesis
    if "student_id" in unlabelled_df.columns:
        unlabelled_df = unlabelled_df.drop(columns=["student_id"])

    # Clean / normalize as done in app.py
    unlabelled_df.columns = [str(c).strip() for c in unlabelled_df.columns]
    unlabelled_df = normalize_columns(unlabelled_df)
    if "student_id" not in unlabelled_df.columns:
        unlabelled_df.insert(0, "student_id", range(1, len(unlabelled_df) + 1))

    # Verify predictions
    probs = predictor.predict_probabilities(unlabelled_df, model_name="Random Forest")
    assert len(probs) == 25
    assert all(0.0 <= p <= 1.0 for p in probs)

    predicted_status = np.where(probs >= 0.50, "Placed", "Not Placed")
    risk_tiers = [predictor.classify_risk(p) for p in probs]
    assert len(predicted_status) == 25
    assert len(risk_tiers) == 25

    # 3. Test Cohort What-If Simulator on custom uploaded data
    interventions = {
        "aptitude_test_score": 10.0,
        "cgpa": 0.5,
        "soft_skills_rating": 0.3,
        "projects": 1.0,
        "workshops_certifications": 1.0,
        "internships": 1.0,
    }
    sim_results = simulator.simulate_policy_intervention(unlabelled_df, interventions)
    assert sim_results["cohort_size"] == 25
    assert "baseline_placement_rate" in sim_results
    assert "simulated_placement_rate" in sim_results
    assert "placement_uplift_pct" in sim_results
    assert not sim_results["student_transitions"].empty

    # Test all models: Logistic Regression and XGBoost
    for model_name in predictor.available_models:
        m_probs = predictor.predict_probabilities(unlabelled_df, model_name=model_name)
        assert len(m_probs) == 25
        assert all(0.0 <= p <= 1.0 for p in m_probs)

    print("All custom CSV prediction and simulation tests PASSED successfully!")

if __name__ == "__main__":
    test_custom_csv_simulation()
