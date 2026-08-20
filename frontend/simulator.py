"""
Cohort What-If Simulation Engine.
Enables placement officers to model macroscopic policy shifts and compute
baseline vs. simulated placement probabilities, risk migrations,
and candidate transitions.

Ported from prediction.txt and adapted to the current project's
dataset schema. No LLM or adaptive logic.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
from batch_predictor import HIGH_RISK_THRESHOLD, MODERATE_RISK_THRESHOLD, BatchPredictor

from feature_engineering import FEATURE_RANGES

# ── Default intervention levers ─────────────────────────────────────────────
# Hardcoded to the existing dataset's numerical columns.

INTERVENTION_KNOBS = [
    {
        "column": "technical_skill_score",
        "label": "Technical Upskilling BootCamp (+score)",
        "min": 0.0,
        "max": 25.0,
        "step": 1.0,
        "default": 0.0,
        "group": "technical",
    },
    {
        "column": "aptitude_test_score",
        "label": "Aptitude Coaching Programme (+points)",
        "min": 0.0,
        "max": 25.0,
        "step": 1.0,
        "default": 0.0,
        "group": "academic",
    },
    {
        "column": "attendance_percentage",
        "label": "Attendance Mentorship Drive (+%)",
        "min": 0.0,
        "max": 20.0,
        "step": 1.0,
        "default": 0.0,
        "group": "academic",
    },
    {
        "column": "cgpa",
        "label": "Academic Remediation Drive (+CGPA)",
        "min": 0.0,
        "max": 1.5,
        "step": 0.1,
        "default": 0.0,
        "group": "academic",
    },
    {
        "column": "soft_skills_rating",
        "label": "Communication Workshop (+rating)",
        "min": 0.0,
        "max": 1.5,
        "step": 0.1,
        "default": 0.0,
        "group": "academic",
    },
    {
        "column": "projects",
        "label": "Capstone Project Workshop (+projects)",
        "min": 0.0,
        "max": 3.0,
        "step": 1.0,
        "default": 0.0,
        "group": "experiential",
    },
    {
        "column": "certifications",
        "label": "Sponsored Certification Drive (+certs)",
        "min": 0.0,
        "max": 3.0,
        "step": 1.0,
        "default": 0.0,
        "group": "experiential",
    },
    {
        "column": "internships",
        "label": "Industry Internship Placement (+internships)",
        "min": 0.0,
        "max": 2.0,
        "step": 1.0,
        "default": 0.0,
        "group": "experiential",
    },
]


class CohortWhatIfSimulator:
    """
    Cohort filtering and what-if policy intervention simulator.
    Computes baseline vs. simulated placement metrics and risk tier migrations.
    """

    def __init__(self, predictor: BatchPredictor) -> None:
        self.predictor = predictor

    def predict_probabilities(self, df: pd.DataFrame) -> np.ndarray:
        """Pass-through to the batch predictor."""
        return self.predictor.predict_probabilities(df)

    def simulate_policy_intervention(
        self,
        cohort_df: pd.DataFrame,
        interventions: Optional[dict[str, float]] = None,
    ) -> dict[str, Any]:
        """
        Applies parameterized policy shifts on a cohort and computes
        baseline vs simulated placement metrics and risk tier migrations.

        Parameters
        ----------
        cohort_df : pd.DataFrame
            The student cohort to simulate on.
        interventions : dict, optional
            {column_name: delta} mapping. Positive values boost the feature.
            Deltas are clamped to the feature's observed training range.

        Returns
        -------
        dict with keys:
            cohort_size, baseline_placement_rate, simulated_placement_rate,
            placement_uplift_pct, newly_placed_count,
            net_transitioned_out_of_high_risk, risk_migration,
            student_transitions (DataFrame)
        """
        if cohort_df.empty:
            return {
                "cohort_size": 0,
                "baseline_placement_rate": 0.0,
                "simulated_placement_rate": 0.0,
                "placement_uplift_pct": 0.0,
                "newly_placed_count": 0,
                "net_transitioned_out_of_high_risk": 0,
                "risk_migration": {},
                "student_transitions": pd.DataFrame(),
            }

        # 1. Baseline Predictions
        base_probs = self.predict_probabilities(cohort_df)
        base_placed = (base_probs >= 0.50).astype(int)

        # 2. Apply Interventions
        sim_df = cohort_df.copy()
        if interventions:
            for col, delta in interventions.items():
                if (
                    col in sim_df.columns
                    and delta != 0
                    and pd.api.types.is_numeric_dtype(sim_df[col])
                ):
                    sim_df[col] = sim_df[col] + delta
                    # Clamp to the observed training range. Pushing a
                    # feature past what the model was trained on produces
                    # confident nonsense, so an intervention can only
                    # move students to the top of the real range.
                    lo, hi = FEATURE_RANGES.get(col, (None, None))
                    if lo is not None:
                        sim_df[col] = np.clip(sim_df[col], lo, hi)

        # 3. Post-Intervention Predictions
        sim_probs = self.predict_probabilities(sim_df)
        sim_placed = (sim_probs >= 0.50).astype(int)

        # 4. Compute Metrics
        cohort_size = len(cohort_df)
        base_rate = round(float(np.mean(base_placed) * 100.0), 1)
        sim_rate = round(float(np.mean(sim_placed) * 100.0), 1)
        uplift_pct = round(sim_rate - base_rate, 1)
        newly_placed = int(np.sum((base_placed == 0) & (sim_placed == 1)))

        # Risk tier migration
        base_high_risk = base_probs < HIGH_RISK_THRESHOLD
        sim_high_risk = sim_probs < HIGH_RISK_THRESHOLD
        transitioned_out = int(np.sum(base_high_risk & ~sim_high_risk))

        def get_risk_counts(probs: np.ndarray) -> dict[str, int]:
            return {
                "High Risk (<50%)": int(np.sum(probs < HIGH_RISK_THRESHOLD)),
                "Moderate Risk (50-75%)": int(
                    np.sum(
                        (probs >= HIGH_RISK_THRESHOLD)
                        & (probs < MODERATE_RISK_THRESHOLD)
                    )
                ),
                "Interview Ready (>=75%)": int(
                    np.sum(probs >= MODERATE_RISK_THRESHOLD)
                ),
            }

        risk_migration = {
            "baseline": get_risk_counts(base_probs),
            "simulated": get_risk_counts(sim_probs),
        }

        # Candidate-level transition log
        id_col = (
            "student_id" if "student_id" in cohort_df.columns
            else cohort_df.columns[0]
        )

        trans_dict = {
            "student_id": cohort_df[id_col].values,
            "baseline_prob": np.round(base_probs * 100, 1),
            "simulated_prob": np.round(sim_probs * 100, 1),
            "prob_gain": np.round((sim_probs - base_probs) * 100, 1),
            "baseline_status": np.where(
                base_placed == 1, "Placed", "At-Risk"
            ),
            "simulated_status": np.where(
                sim_placed == 1, "Placed", "At-Risk"
            ),
            "newly_shortlistable": (base_placed == 0) & (sim_placed == 1),
        }

        # Include cohort segmentation columns if available
        for group_col in ["placement_training", "extracurricular_activities"]:
            if group_col in cohort_df.columns:
                trans_dict[group_col] = cohort_df[group_col].values

        transitions_df = pd.DataFrame(trans_dict)

        return {
            "cohort_size": cohort_size,
            "baseline_placement_rate": base_rate,
            "simulated_placement_rate": sim_rate,
            "placement_uplift_pct": uplift_pct,
            "newly_placed_count": newly_placed,
            "net_transitioned_out_of_high_risk": transitioned_out,
            "average_prob_gain": round(
                float(np.mean(sim_probs - base_probs)) * 100.0, 1
            ),
            "risk_migration": risk_migration,
            "student_transitions": transitions_df,
        }
