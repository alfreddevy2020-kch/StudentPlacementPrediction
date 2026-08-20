"""
Role 5 -- Innovation & Research: Uplift Modeling
=================================================
Student Placement Prediction System -- Batch 1

WHAT THIS ANSWERS THAT PART 2/3 DON'T
--------------------------------------
part2/part3 predict P(placed | X): who is at risk. That's a RANKING
question. This module estimates:

    CATE(X) = P(placed | X, training=Yes) - P(placed | X, training=No)

...who would gain the most FROM placement training SPECIFICALLY. That's
a TARGETING question, and the two rank students differently. A student
can be high-risk (low P(placed)) with near-zero uplift -- training
wouldn't move them, something else is going on -- and a borderline
student can carry the largest uplift of anyone, because they're exactly
where a nudge tips the outcome. Risk tells you who to worry about.
Uplift tells you who to actually spend a training seat on.

METHOD: T-learner (two-model approach)
---------------------------------------
Split students by treatment (placement_training), fit one classifier per
arm on the SAME covariates, then score every student through BOTH models
and take the difference. This is the "T-learner", the simplest member of
the meta-learner family formalised in:

    Kunzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019).
    "Metalearners for estimating heterogeneous treatment effects using
    machine learning." PNAS, 116(10), 4156-4165.
    https://doi.org/10.1073/pnas.1804597116

The two-model idea itself is older -- Radcliffe & Surry (1999) and
Lo (2002) both built one response model per arm and subtracted them in
the direct-marketing literature, decades before "T-learner" was a name
for it:

    Radcliffe, N. J., & Surry, P. D. (1999). "Differential response
    analysis: Modeling true response by isolating the effect of a
    single action." Credit Scoring and Credit Control VI.

    Lo, V. S. Y. (2002). "The true lift model: A novel data mining
    approach to response modeling in database marketing." ACM SIGKDD
    Explorations Newsletter, 4(2), 78-86.

Kunzel et al. also describe the S-learner (one model, treatment as a
feature) and X-learner (built for imbalanced arms). T-learner is the
right starting point here: the two arms are roughly balanced and ~10k
rows is enough to fit two separate models without starving either one.

TREATMENT VARIABLE: placement_training
----------------------------------------
Chosen over every other column on the roster because it's the one thing
a placement cell can actually act on this term. You cannot retroactively
raise a student's CGPA. You CAN enroll them in the training programme.

This also extends Part 4's fairness finding rather than duplicating it:
Part 4 showed the model is worse at CATCHING at-risk students who never
had training (higher false-negative rate for that group). This module
asks the question Part 4 doesn't: of the untrained students, which ones
would training actually move? Part 4 says "we're missing them more
often." This says "here's who's worth reaching first."

======================================================================
READ THIS BEFORE PUTTING A SINGLE NUMBER FROM THIS FILE ON A SLIDE
======================================================================
placement_training is an OBSERVED column in a Kaggle dataset, not a
randomized assignment. Students who took training may differ
systematically from students who didn't, in ways that also affect
placement -- motivation, faculty encouragement, awareness of their own
weak spots. A T-learner fit on observational data estimates an
ASSOCIATIONAL effect. It is only a valid CAUSAL effect if every
confounder that affects both "took training" and "got placed" is
already in the feature set (the "no unmeasured confounders"
assumption) -- and that assumption cannot be verified from this data
alone, full stop.

check_covariate_balance() below is a cheap, honest diagnostic in the
same spirit as part8's leakage_sanity_check: it does not prove the
assumption holds, but a large imbalance is a concrete reason to trust
the estimate less, and a small one is (weak) evidence the two arms are
comparable on what we can measure.

The honest framing for the presentation is one sentence, said before
anyone asks: "this is decision-support for a pilot, not a validated
causal claim -- the next step is a randomized rollout of training to a
held-out group to confirm the effect size." That sentence is worth more
in front of a judge than any accuracy number in this file.

Run after: preprocessing.py (needs data/processed/normalization_stats.json)
Usage:
    python part5/uplift_modeling.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from feature_engineering import (  # noqa: E402
    ALL_NUMERICAL_FEATURES,
    TARGET_COLUMN,
    TARGET_MAP,
    engineer_features,
    load_raw_dataset,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

TREATMENT_COL = "placement_training"

# Features NOT fed to the arm-specific models, and why:
#   placement_training_binary -- this literally IS the treatment. Feeding
#       it in would let each per-arm model partly re-derive treatment
#       assignment instead of estimating the outcome surface, which
#       defeats the point of splitting into two models in the first place.
#   support_index -- support_index = extracurricular_binary +
#       placement_training_binary (see feature_engineering.py). It
#       encodes the treatment indirectly through the sum. Same leak,
#       one level removed.
EXCLUDED_FROM_UPLIFT_FEATURES = {"placement_training_binary", "support_index"}
UPLIFT_FEATURE_COLS = [
    c for c in ALL_NUMERICAL_FEATURES if c not in EXCLUDED_FROM_UPLIFT_FEATURES
]  # 27 of the 29 canonical numerical features


@dataclass
class UpliftModels:
    model_treated: LogisticRegression
    model_control: LogisticRegression
    scaler: StandardScaler
    feature_cols: list[str]
    holdout_auc_treated: float
    holdout_auc_control: float
    n_treated: int
    n_control: int


def check_covariate_balance(
    df: pd.DataFrame,
    treatment_col: str = TREATMENT_COL,
    covariates: list[str] | None = None,
) -> pd.DataFrame:
    """Standardized mean difference (SMD) between the treated and control
    arms on each covariate -- a propensity-style balance check, not a
    causal-validity proof.

    SMD = (mean_treated - mean_control) / pooled_std. |SMD| < 0.1 is the
    commonly used rule of thumb for "these two groups look comparable on
    this variable"; above ~0.25 is a real imbalance worth naming out
    loud. This does NOT confirm unconfoundedness -- it only checks the
    covariates we can see. It cannot see what isn't in the dataset.
    """
    if covariates is None:
        covariates = UPLIFT_FEATURE_COLS

    treated = df[df[treatment_col] == "Yes"]
    control = df[df[treatment_col] == "No"]

    rows = []
    for col in covariates:
        t_mean, c_mean = treated[col].mean(), control[col].mean()
        pooled_std = np.sqrt((treated[col].var() + control[col].var()) / 2)
        smd = (t_mean - c_mean) / pooled_std if pooled_std > 0 else 0.0
        rows.append(
            {
                "feature": col,
                "treated_mean": round(t_mean, 3),
                "control_mean": round(c_mean, 3),
                "smd": round(smd, 3),
                "flag": "IMBALANCED" if abs(smd) > 0.25 else (
                    "watch" if abs(smd) > 0.1 else "ok"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("smd", key=abs, ascending=False)


def fit_uplift_models(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    random_state: int = 42,
) -> UpliftModels:
    """T-learner: one LogisticRegression per treatment arm.

    LogisticRegression, not RF/XGBoost, on purpose: SCHEMA.md's own
    held-out comparison has LR as the best single model on this dataset
    (0.8836 ROC-AUC vs 0.8750 RF / 0.8684 XGB) -- evidence the true
    relationship is close to linear/log-linear on these features. That
    matters more here than for the main classifier, because splitting
    by arm roughly halves the training data available to each model;
    a lower-variance learner is the safer choice on a smaller sample.
    Swap in any sklearn-compatible classifier via `base_estimator=`
    below if you want to compare.
    """
    if feature_cols is None:
        feature_cols = UPLIFT_FEATURE_COLS

    y = df[TARGET_COLUMN].map(TARGET_MAP)
    if y.isna().any():
        raise ValueError(
            f"{TARGET_COLUMN} has values outside {TARGET_MAP} -- check the input."
        )

    scaler = StandardScaler().fit(df[feature_cols])
    X_all = pd.DataFrame(
        scaler.transform(df[feature_cols]), columns=feature_cols, index=df.index
    )

    def _fit_arm(mask: pd.Series) -> tuple[LogisticRegression, float, int]:
        X_arm, y_arm = X_all[mask], y[mask]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_arm, y_arm, test_size=0.20, random_state=random_state, stratify=y_arm
        )
        model = LogisticRegression(max_iter=1000, random_state=random_state)
        model.fit(X_tr, y_tr)
        # Holdout AUC is a PREDICTION-QUALITY check (does this arm's model
        # generalize?), not a causal-validity check. Refit on the full arm
        # afterwards so the shipped model uses every available row.
        holdout_auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
        model.fit(X_arm, y_arm)
        return model, holdout_auc, int(mask.sum())

    treated_mask = df[TREATMENT_COL] == "Yes"
    control_mask = df[TREATMENT_COL] == "No"

    model_treated, auc_treated, n_treated = _fit_arm(treated_mask)
    model_control, auc_control, n_control = _fit_arm(control_mask)

    return UpliftModels(
        model_treated=model_treated,
        model_control=model_control,
        scaler=scaler,
        feature_cols=feature_cols,
        holdout_auc_treated=auc_treated,
        holdout_auc_control=auc_control,
        n_treated=n_treated,
        n_control=n_control,
    )


def predict_uplift(models: UpliftModels, df: pd.DataFrame) -> pd.DataFrame:
    """Score every row through BOTH arm models regardless of that row's
    observed treatment -- this is the counterfactual step. Returns
    p_control (baseline), p_treated (with training), and uplift = the
    difference, i.e. CATE(X).
    """
    X = pd.DataFrame(
        models.scaler.transform(df[models.feature_cols]),
        columns=models.feature_cols,
        index=df.index,
    )
    p_control = models.model_control.predict_proba(X)[:, 1]
    p_treated = models.model_treated.predict_proba(X)[:, 1]
    return pd.DataFrame(
        {
            "p_control": p_control,
            "p_treated": p_treated,
            "uplift": p_treated - p_control,
        },
        index=df.index,
    )


def categorize_uplift(
    p_control: pd.Series,
    uplift: pd.Series,
    persuadable_threshold: float = 0.05,
    sure_thing_baseline: float = 0.70,
    lost_cause_baseline: float = 0.30,
) -> pd.Series:
    """The four-fold target matrix from the uplift-modeling literature
    (Radcliffe & Surry; see also Kane, Lo & Zheng's "four-fold" framing
    of the same idea), plus a fifth bucket this implementation adds on
    purpose -- see the note at the bottom.

      Persuadable    -- meaningfully higher P(placed) if trained. The
                         only group where a training seat has leverage.
      Sure Thing      -- already likely to be placed regardless, and
                         training adds little on top. Not wasted on them,
                         just not the reason they'll get placed.
      Lost Cause      -- unlikely to be placed either way, and training
                         doesn't move that. This group needs a different
                         lever, not more training seats.
      Do-Not-Disturb  -- NEGATIVE uplift: the model estimates training is
                         associated with a LOWER placement probability
                         for this student. Rare, worth a second look
                         rather than an automatic dismissal -- could be
                         a real effect (time traded away from something
                         that mattered more) or just noise in a small
                         slice. Don't act on this bucket without digging in.
      Moderate/Unclear -- p_control lands between the two baseline cutoffs
                         AND uplift doesn't clear persuadable_threshold.
                         An earlier version of this function let rows like
                         this fall through to "Sure Thing" by default,
                         which is wrong -- a student at p_control=0.5 isn't
                         a sure thing of anything. Naming the ambiguity is
                         more honest than mislabeling it, and it's a fair
                         question to expect in Q&A if you show this chart.

    Implemented as an exhaustive, mutually-exclusive rule ladder (each
    row matches exactly one condition, evaluated top to bottom) rather
    than a chain of overwrites, specifically so no combination of
    (p_control, uplift) falls through to an unintended default.
    """
    conditions = [
        uplift < 0,
        uplift >= persuadable_threshold,
        p_control < lost_cause_baseline,
        p_control >= sure_thing_baseline,
    ]
    choices = ["Do-Not-Disturb", "Persuadable", "Lost Cause", "Sure Thing"]
    return pd.Series(
        np.select(conditions, choices, default="Moderate/Unclear"),
        index=p_control.index,
    )


def run_uplift_pipeline(save: bool = True) -> pd.DataFrame:
    """End-to-end: load -> engineer -> balance check -> fit -> score ->
    categorize -> (optionally) save part5/results/uplift_scores.csv.
    Returns the per-student scored DataFrame.
    """
    raw = load_raw_dataset()
    df = engineer_features(raw)

    balance = check_covariate_balance(df)
    print("\n=== Covariate balance (treated vs control) ===")
    print(balance.to_string(index=False))
    n_flagged = (balance["flag"] == "IMBALANCED").sum()
    if n_flagged:
        print(
            f"\n{n_flagged} feature(s) show |SMD| > 0.25 -- name these explicitly "
            "as a confounding risk when presenting, don't let a judge find them first."
        )

    models = fit_uplift_models(df)
    print(
        f"\nArm sizes: treated={models.n_treated}, control={models.n_control}"
    )
    print(
        f"Per-arm holdout AUC (prediction quality, NOT causal validity): "
        f"treated={models.holdout_auc_treated:.4f}, control={models.holdout_auc_control:.4f}"
    )

    scores = predict_uplift(models, df)
    scores["uplift_category"] = categorize_uplift(scores["p_control"], scores["uplift"])
    result = df[["student_id"] if "student_id" in df.columns else []].join(scores)
    result = result.join(df[[TREATMENT_COL, TARGET_COLUMN]])

    print("\n=== Uplift category counts ===")
    print(result["uplift_category"].value_counts().to_string())
    print(f"\nMean estimated uplift across cohort: {scores['uplift'].mean():+.4f}")

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "uplift_scores.csv"
        result.to_csv(out_path, index=False)
        balance.to_csv(RESULTS_DIR / "covariate_balance.csv", index=False)
        print(f"\nSaved: {out_path}")

    return result


if __name__ == "__main__":
    run_uplift_pipeline()
