"""
Role 8 — Evaluation, Testing & Business Impact
================================================
Student Placement Prediction System — Group 1, Task 1

Built on top of what Roles 2 and 3 already shipped (part2/, part3/) rather
than duplicating it. See the top-of-file notes below for exactly what this
adds that isn't already in part2/model_comparison.py.

WHAT THIS ADDS THAT DOESN'T EXIST YET:
  1. At-risk-class recall/precision (pos_label=0) — model_comparison.py only
     reports sklearn's default (pos_label=1 = "Placed"), which answers the
     wrong question for this problem statement.
  2. Cost-sensitive threshold — existing thresholds are F1-optimal, not
     weighted by the real cost of missing an at-risk student.
  3. A true 3-way comparison including XGBoost (part2 only compares
     LogReg vs RF).
  4. A sanity check on XGBoost's suspiciously high test scores
     (ROC-AUC 0.9993 / F1 1.0 in xgb_metadata.csv) — train vs test gap.
  5. Edge-case stress tests (missing data, outliers, single-row inference).
  6. Business-impact translation for the pitch deck.

Run from the repo root, after part2 and part3 pipelines have both run:
    python part8/evaluation_suite.py
"""

import os
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG — matches the actual repo layout
# ---------------------------------------------------------------------------
TARGET_COL = "placement_status"  # 1 = Placed, 0 = Not Placed
TEST_PATH = "data/processed/test_processed.csv"
TRAIN_PATH = "data/processed/train_processed.csv"  # used only for the leakage sanity check

MODEL_PATHS = {
    "Logistic Regression": "part2/models/logistic_regression_best.joblib",
    "Random Forest": "part2/models/random_forest_best.joblib",
    "XGBoost": "part3/models/xgboost_best.joblib",
}

os.makedirs("part8/results", exist_ok=True)

# Business cost weights — state these as an explicit assumption in the deck.
# AT_RISK (class 0) is the class we care about catching early.
COST_MISSED_AT_RISK = 5  # false negative on at-risk detection: expensive
COST_FALSE_ALARM = 1  # flagging a fine student for extra attention: cheap


def load_model(path):
    return joblib.load(path)


def load_split(path):
    df = pd.read_csv(path)
    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL]).values
    return X, y


# ---------------------------------------------------------------------------
# 1. At-risk-focused metrics (the number missing from the existing deck)
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        # Default sklearn behaviour — "how well do we predict Placed"
        "Recall (Placed, class=1)": recall_score(y_test, y_pred, pos_label=1),
        "Precision (Placed, class=1)": precision_score(y_test, y_pred, pos_label=1),
        # The number that actually matters for this problem statement
        "Recall (AT-RISK, class=0)": recall_score(y_test, y_pred, pos_label=0),
        "Precision (AT-RISK, class=0)": precision_score(y_test, y_pred, pos_label=0),
        "F1 (weighted)": f1_score(y_test, y_pred, average="weighted"),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["At Risk (0)", "Placed (1)"]))
    return metrics, y_proba


def compare_all_three():
    X_test, y_test = load_split(TEST_PATH)
    results, roc_data = [], {}

    for name, path in MODEL_PATHS.items():
        model = load_model(path)
        metrics, y_proba = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, metrics["ROC-AUC"])

    results_df = pd.DataFrame(results).set_index("Model")
    print("\n=== 3-way comparison (this table doesn't exist anywhere else in the repo) ===")
    print(results_df.round(4).to_string())
    results_df.to_csv("part8/results/three_way_comparison.csv")

    loaded_models = {name: load_model(path) for name, path in MODEL_PATHS.items()}

    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC — LogReg vs RF vs XGBoost (3-way)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("part8/results/three_way_roc.png", dpi=150)
    print("Saved part8/results/three_way_roc.png")

    return results_df, loaded_models


# ---------------------------------------------------------------------------
# 2. Cost-sensitive threshold (existing thresholds are F1-optimal, not this)
# ---------------------------------------------------------------------------
def cost_sensitive_threshold(model, X_test, y_test, name):
    """
    Sweep thresholds and pick the one minimizing:
        cost = (missed at-risk students) * COST_MISSED_AT_RISK
             + (false alarms)            * COST_FALSE_ALARM
    Note the direction: with placement_status where 1=Placed, missing an
    at-risk student means the model predicts 1 (Placed) when truth is 0
    (Not Placed) — i.e. a FALSE POSITIVE in sklearn's default labeling.
    Don't get this backwards when you write the slide.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 200)

    best_t, best_cost = 0.5, np.inf
    costs = []
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
        # fp here = predicted Placed(1) but actually Not Placed(0) = missed at-risk student
        cost = fp * COST_MISSED_AT_RISK + fn * COST_FALSE_ALARM
        costs.append(cost)
        if cost < best_cost:
            best_cost, best_t = cost, t

    print(
        f"\n[{name}] Cost-sensitive optimal threshold: {best_t:.3f}  "
        f"(compare against the F1-optimal threshold already in {name}'s metadata.csv)"
    )

    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, costs)
    plt.axvline(best_t, color="red", linestyle="--", label=f"Cost-optimal = {best_t:.2f}")
    plt.axvline(0.5, color="gray", linestyle=":", label="Default 0.5")
    plt.xlabel("Decision threshold")
    plt.ylabel("Weighted business cost")
    plt.title(f"{name}: Cost-sensitive threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"part8/results/{name.replace(' ', '_').lower()}_cost_threshold.png", dpi=150)

    return best_t


# ---------------------------------------------------------------------------
# 3. Sanity-check XGBoost's suspiciously high scores
# ---------------------------------------------------------------------------
def leakage_sanity_check(model, name="XGBoost"):
    """
    xgb_metadata.csv shows test ROC-AUC=0.9993, F1=1.0. Before presenting
    this as a win, check the train score too. If train is also ~0.999, the
    synthetic dataset is just near-separable (common for generated Kaggle
    data, e.g. placement_status may follow a near-deterministic rule on
    backlogs + cgpa) — say so explicitly in the deck. If train is
    meaningfully different from test, something's actually wrong.
    """
    if not os.path.exists(TRAIN_PATH):
        print(f"\n[{name}] Skipping leakage check — {TRAIN_PATH} not found locally.")
        return
    X_train, y_train = load_split(TRAIN_PATH)
    X_test, y_test = load_split(TEST_PATH)

    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"\n[{name}] Train ROC-AUC: {train_auc:.4f} | Test ROC-AUC: {test_auc:.4f}")
    if abs(train_auc - test_auc) < 0.01:
        print(
            "  -> Gap is tiny. Likely a genuinely near-separable synthetic dataset, "
            "not overfitting. State this plainly in the deck rather than let the "
            "panel assume it's a bug."
        )
    else:
        print(
            "  -> Meaningful gap between train and test. Worth investigating further "
            "before presenting the test number as final."
        )


# ---------------------------------------------------------------------------
# 4. Edge-case stress tests
# ---------------------------------------------------------------------------
def edge_case_tests(model, X_test_df, name):
    print(f"\n=== Edge case tests: {name} ===")

    # Columns are one-hot/scaled with prefixes like "numerical__backlogs" —
    # match by substring so this doesn't break if exact names shift.
    def find_col(substr):
        matches = [c for c in X_test_df.columns if substr in c.lower()]
        return matches[0] if matches else None

    backlog_col = find_col("backlog")
    attendance_col = find_col("attendance")

    # 1. Missing values (post-scaling, simulate with the scaled mean = 0)
    X_missing = X_test_df.copy()
    for col in [c for c in [backlog_col, attendance_col] if c]:
        X_missing.loc[X_missing.sample(frac=0.1, random_state=1).index, col] = 0.0
    try:
        model.predict(X_missing.values)
        print("[PASS] Handles zero-imputed missing values")
    except Exception as e:
        print(f"[FAIL] {e}")

    # 2. Outlier profile — extreme backlog value
    if backlog_col:
        X_outlier = X_test_df.iloc[[0]].copy()
        X_outlier[backlog_col] = X_test_df[backlog_col].max() * 3
        try:
            pred = model.predict_proba(X_outlier.values)[:, 1][0]
            print(
                f"[PASS] Outlier backlog value handled, P(placed)={pred:.3f} — sanity-check manually"
            )
        except Exception as e:
            print(f"[FAIL] {e}")

    # 3. Single-row inference — the real dashboard/API use case
    try:
        model.predict_proba(X_test_df.iloc[[0]].values)
        print("[PASS] Single-student inference works (this is what the API actually does)")
    except Exception as e:
        print(f"[FAIL] {e}")


# ---------------------------------------------------------------------------
# 5. Business impact translation
# ---------------------------------------------------------------------------
def business_impact(model, X_test, y_test, threshold, name, current_manual_catch_rate=0.30):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    at_risk_total = tn + fp  # actual class-0 count
    caught = tn  # correctly predicted at-risk (class 0)
    manual_estimate = int(at_risk_total * current_manual_catch_rate)
    additional = max(caught - manual_estimate, 0)

    print(f"\n=== Business impact — {name} @ threshold {threshold:.3f} ===")
    print(f"At-risk students in test set: {at_risk_total}")
    print(f"Caught by model: {caught} ({caught / at_risk_total:.0%})")
    print(
        f"Estimated caught by current manual process: {manual_estimate} "
        f"({current_manual_catch_rate:.0%} — assumption, state this on the slide)"
    )
    print(f"Additional at-risk students caught early: ~{additional}")


if __name__ == "__main__":
    results_df, loaded_models = compare_all_three()

    # THE KEY DIAGNOSTIC: run this on all three models, not just the "winner".
    # A model that's perfect on test but NOT perfect on train (data it already
    # saw) is a red flag. A model that's perfect on BOTH is more likely just
    # an easy/rule-based synthetic dataset.
    print("\n" + "=" * 60)
    print("TRAIN vs TEST CHECK -- is 'perfect' too good to be true?")
    print("=" * 60)
    for name, model in loaded_models.items():
        leakage_sanity_check(model, name)

    # Use Logistic Regression for the cost-sensitive threshold + business
    # impact sections below -- it's the model actually showing real
    # trade-offs right now. Once the team confirms whether RF/XGBoost's
    # perfect scores are legitimate, swap this back to whichever model the
    # team decides to present.
    analysis_name = "Logistic Regression"
    analysis_model = loaded_models[analysis_name]

    X_test_arr, y_test = load_split(TEST_PATH)
    test_df_full = pd.read_csv(TEST_PATH)
    X_test_df = test_df_full.drop(columns=[TARGET_COL])

    best_threshold = cost_sensitive_threshold(analysis_model, X_test_arr, y_test, analysis_name)
    edge_case_tests(analysis_model, X_test_df, analysis_name)
    business_impact(analysis_model, X_test_arr, y_test, best_threshold, analysis_name)
