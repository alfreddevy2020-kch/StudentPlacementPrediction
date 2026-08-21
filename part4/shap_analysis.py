"""
Part 4 — SHAP Analysis for All Models
======================================

Generates SHAP explanations for:
    - Logistic Regression
    - Random Forest
    - XGBoost

on the held-out test set and writes per-model artifacts:

    part4/explainability_results/
        shap_values_<model>.csv
        shap_global_importance_<model>.csv
        shap_summary_<model>.png
        shap_bar_<model>.png
        shap_global_importance_all_models.csv

Run from the repo root after preprocessing + training:

    python download_dataset.py
    python preprocessing.py
    python part2/logistic_regression_model.py
    python part2/random_forest_model.py
    python part3/xgboost_model.py
    python part4/shap_analysis.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# ---------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------

import joblib
import matplotlib

# Use a non-interactive backend because this script saves PNG files.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from shap_explainer import (
    build_explainer,
    extract_base_value,
    extract_shap_values,
)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

warnings.filterwarnings("ignore")

RESULTS_DIR = REPO_ROOT / "part4" / "explainability_results"

TRAIN_PATH = (
    REPO_ROOT
    / "data"
    / "processed"
    / "train_processed.csv"
)

TEST_PATH = (
    REPO_ROOT
    / "data"
    / "processed"
    / "test_processed.csv"
)

# Production models are preferred.
PROD_DIR = REPO_ROOT / "artifacts" / "production"


# ---------------------------------------------------------------------
# Model sources
# ---------------------------------------------------------------------

MODEL_SOURCES = [
    (
        "Logistic_Regression",
        "logistic_regression",
        "part2/models/logistic_regression_best.joblib",
    ),
    (
        "Random_Forest",
        "random_forest",
        "part2/models/random_forest_best.joblib",
    ),
    (
        "XGBoost",
        "xgboost",
        "part3/models/xgboost_best.joblib",
    ),
]


# ---------------------------------------------------------------------
# SHAP configuration
# ---------------------------------------------------------------------

BACKGROUND_SIZE = 100


# ---------------------------------------------------------------------
# Plot configuration
# ---------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


# ---------------------------------------------------------------------
# Resolve model path
# ---------------------------------------------------------------------

def resolve_model_path(
    prod_rel: str,
    dev_rel: str,
) -> Path:
    """
    Return the production bundle path if it exists.
    Otherwise return the development model path.
    """

    prod = PROD_DIR / prod_rel / "model.joblib"

    if prod.exists():
        return prod

    return REPO_ROOT / dev_rel


# ---------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------

def load_models() -> dict[str, object]:
    """
    Load all three trained models.

    Production model bundles are preferred.
    Development model files are used as fallback.
    """

    models: dict[str, object] = {}

    for name, prod_rel, dev_rel in MODEL_SOURCES:

        path = resolve_model_path(
            prod_rel,
            dev_rel,
        )

        if not path.exists():
            print(
                f"  Warning: {path} not found "
                f"— skipping {name}."
            )
            continue

        try:
            models[name] = joblib.load(path)

            print(
                f"  Loaded {name} from "
                f"{path.relative_to(REPO_ROOT)}"
            )

        except Exception as exc:
            print(
                f"  Error loading {name}: {exc}"
            )

    return models


# ---------------------------------------------------------------------
# Load train/test matrices
# ---------------------------------------------------------------------

def load_matrices() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
]:
    """
    Load processed train/test datasets.

    Returns:
        X_train
        X_test
        feature_names
    """

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found:\n{TRAIN_PATH}\n\n"
            "Run preprocessing.py first."
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test data not found:\n{TEST_PATH}\n\n"
            "Run preprocessing.py first."
        )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    target = "placement_status"

    if target not in train_df.columns:
        raise ValueError(
            f"Target column '{target}' "
            "not found in training data."
        )

    if target not in test_df.columns:
        raise ValueError(
            f"Target column '{target}' "
            "not found in test data."
        )

    feature_names = [
        column
        for column in train_df.columns
        if column != target
    ]

    X_train = train_df.drop(
        columns=[target]
    )

    X_test = test_df.drop(
        columns=[target]
    )

    # Make sure train and test contain the same features.
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Training and test feature columns do not match."
        )

    return (
        X_train,
        X_test,
        feature_names,
    )


# ---------------------------------------------------------------------
# Plot SHAP summary
# ---------------------------------------------------------------------

def save_summary_plot(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    feature_names: list[str],
    model_name: str,
) -> None:
    """
    Save SHAP beeswarm summary plot.
    """

    output_path = (
        RESULTS_DIR
        / f"shap_summary_{model_name}.png"
    )

    plt.figure(figsize=(10, 8))

    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        show=False,
        max_display=15,
        cmap="RdBu",
    )

    plt.title(
        f"SHAP Summary — {model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"    Saved: {output_path.name}"
    )


# ---------------------------------------------------------------------
# Plot SHAP bar chart
# ---------------------------------------------------------------------

def save_bar_plot(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    feature_names: list[str],
    model_name: str,
) -> None:
    """
    Save SHAP mean absolute impact bar plot.

    IMPORTANT:
    The `color` argument is a normal color string.
    Do not pass plt.get_cmap(...) here because SHAP's
    bar plot expects a color rather than a colormap object.
    """

    output_path = (
        RESULTS_DIR
        / f"shap_bar_{model_name}.png"
    )

    plt.figure(figsize=(10, 7))

    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=15,
        color="#4F8EF7",
    )

    plt.title(
        f"SHAP Mean |Impact| — {model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"    Saved: {output_path.name}"
    )


# ---------------------------------------------------------------------
# Run SHAP for one model
# ---------------------------------------------------------------------

def run_shap_for_model(
    model_name: str,
    model: object,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Compute SHAP values for one model and save:

        - per-row SHAP values
        - global SHAP importance
        - SHAP beeswarm plot
        - SHAP bar plot
    """

    print(
        f"\n  Computing SHAP for {model_name} ..."
    )

    # -------------------------------------------------------------
    # Background dataset
    # -------------------------------------------------------------

    background_size = min(
        BACKGROUND_SIZE,
        len(X_train),
    )

    background = (
        X_train
        .sample(
            n=background_size,
            random_state=42,
        )
        .to_numpy()
    )

    # -------------------------------------------------------------
    # Build SHAP explainer
    # -------------------------------------------------------------

    explainer = build_explainer(
        model,
        background=background,
    )

    # -------------------------------------------------------------
    # Calculate SHAP values
    # -------------------------------------------------------------

    shap_output = explainer.shap_values(
        X_test.to_numpy()
    )

    shap_values = extract_shap_values(
        shap_output
    )

    base_value = extract_base_value(
        explainer.expected_value
    )

    # -------------------------------------------------------------
    # Validate SHAP matrix
    # -------------------------------------------------------------

    shap_values = np.asarray(
        shap_values,
        dtype=float,
    )

    if shap_values.ndim != 2:
        raise ValueError(
            f"Expected 2D SHAP matrix for "
            f"{model_name}, got shape "
            f"{shap_values.shape}"
        )

    if shap_values.shape[0] != len(X_test):
        raise ValueError(
            f"SHAP row count mismatch for "
            f"{model_name}: "
            f"{shap_values.shape[0]} vs "
            f"{len(X_test)}"
        )

    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            f"SHAP feature count mismatch for "
            f"{model_name}: "
            f"{shap_values.shape[1]} vs "
            f"{len(feature_names)}"
        )

    print(
        f"    SHAP matrix: {shap_values.shape} "
        f"| base value: {base_value:.4f}"
    )

    # -------------------------------------------------------------
    # Per-row SHAP contributions
    # -------------------------------------------------------------

    shap_df = pd.DataFrame(
        shap_values,
        columns=feature_names,
    )

    shap_df["base_value"] = base_value

    shap_df.insert(
        0,
        "row_id",
        np.arange(len(shap_df)),
    )

    shap_output_path = (
        RESULTS_DIR
        / f"shap_values_{model_name}.csv"
    )

    shap_df.to_csv(
        shap_output_path,
        index=False,
    )

    print(
        f"    Saved: {shap_output_path.name}"
    )

    # -------------------------------------------------------------
    # Global SHAP importance
    # -------------------------------------------------------------

    mean_abs_shap = (
        np.abs(shap_values)
        .mean(axis=0)
    )

    mean_shap = (
        shap_values
        .mean(axis=0)
    )

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
            "mean_shap": mean_shap,
        }
    )

    importance = (
        importance
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance_output_path = (
        RESULTS_DIR
        / f"shap_global_importance_{model_name}.csv"
    )

    importance.to_csv(
        importance_output_path,
        index=False,
    )

    print(
        f"    Saved: "
        f"{importance_output_path.name}"
    )

    # -------------------------------------------------------------
    # IMPORTANT:
    #
    # Keep X_test numeric.
    #
    # Do not convert one-hot encoded columns to strings
    # such as "Yes"/"No". SHAP plotting works more reliably
    # with the original numeric feature matrix.
    # -------------------------------------------------------------

    X_test_display = X_test.copy()

    # -------------------------------------------------------------
    # Beeswarm plot
    # -------------------------------------------------------------

    save_summary_plot(
        shap_values=shap_values,
        X_test=X_test_display,
        feature_names=feature_names,
        model_name=model_name,
    )

    # -------------------------------------------------------------
    # Bar plot
    # -------------------------------------------------------------

    save_bar_plot(
        shap_values=shap_values,
        X_test=X_test_display,
        feature_names=feature_names,
        model_name=model_name,
    )

    # -------------------------------------------------------------
    # Return importance table
    # -------------------------------------------------------------

    return importance.assign(
        model=model_name
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    # -------------------------------------------------------------
    # Create output directory
    # -------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[1] Loading models and processed matrices ..."
    )

    # -------------------------------------------------------------
    # Load models
    # -------------------------------------------------------------

    models = load_models()

    if not models:
        raise FileNotFoundError(
            "No model artifacts found. "
            "Run the training pipeline first."
        )

    # -------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------

    (
        X_train,
        X_test,
        feature_names,
    ) = load_matrices()

    print(
        f"    Train matrix: {X_train.shape} "
        f"| Test matrix: {X_test.shape}"
    )

    print(
        "[2] SHAP analysis for all models ..."
    )

    # -------------------------------------------------------------
    # Run SHAP for every available model
    # -------------------------------------------------------------

    importance_frames = []

    for name, model in models.items():

        try:

            importance = run_shap_for_model(
                model_name=name,
                model=model,
                X_train=X_train,
                X_test=X_test,
                feature_names=feature_names,
            )

            importance_frames.append(
                importance
            )

        except Exception as exc:

            print(
                f"\n  ERROR while processing "
                f"{name}: {exc}"
            )

            # Continue with the remaining models.
            continue

    # -------------------------------------------------------------
    # Check results
    # -------------------------------------------------------------

    if not importance_frames:
        raise RuntimeError(
            "SHAP analysis failed for all models."
        )

    # -------------------------------------------------------------
    # Combined global importance
    # -------------------------------------------------------------

    combined = pd.concat(
        importance_frames,
        ignore_index=True,
    )

    combined_output_path = (
        RESULTS_DIR
        / "shap_global_importance_all_models.csv"
    )

    combined.to_csv(
        combined_output_path,
        index=False,
    )

    print(
        f"\n  Saved: "
        f"{combined_output_path.name} "
        f"({len(combined)} rows)"
    )

    # -------------------------------------------------------------
    # Finished
    # -------------------------------------------------------------

    print(
        "\nPipeline complete. "
        "Check part4/explainability_results/!"
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
