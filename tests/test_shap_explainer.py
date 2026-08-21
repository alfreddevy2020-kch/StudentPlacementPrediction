"""Unit tests for the shared SHAP explainability module (shap_explainer.py)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from shap_explainer import (
    beeswarm_figure,
    build_explainer,
    extract_base_value,
    extract_shap_values,
    make_display_values,
    mean_abs_shap,
    mean_shap_bar_figure,
    prettify_feature_name,
    waterfall_figure,
)

pytest.importorskip("shap")


@pytest.fixture()
def binary_dataset():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "cgpa": rng.uniform(6.0, 9.5, 400),
            "aptitude_test_score": rng.uniform(50.0, 95.0, 400),
            "internships": rng.integers(0, 3, 400),
        }
    )
    y = (X["cgpa"] * 0.8 + X["aptitude_test_score"] * 0.1 + rng.normal(0, 1, 400) > 12).astype(int)
    return X, y


@pytest.fixture()
def fitted_models(binary_dataset):
    X, y = binary_dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    lr = LogisticRegression(max_iter=500).fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=30, random_state=42).fit(X_train, y_train)
    X_test_values = X_test.to_numpy()
    return {"lr": lr, "rf": rf, "X_test": X_test_values, "y_test": y_test}


class TestBuildExplainer:
    def test_linear_explainer_requires_background(self, fitted_models):
        with pytest.raises(ValueError):
            build_explainer(fitted_models["lr"], background=None)

    def test_linear_explainer_accepts_background(self, fitted_models):
        explainer = build_explainer(
            fitted_models["lr"], background=fitted_models["X_test"][:20]
        )
        assert "Explainer" in explainer.__class__.__name__

    def test_tree_explainer_no_background_needed(self, fitted_models):
        explainer = build_explainer(fitted_models["rf"], background=None)
        assert "TreeExplainer" in explainer.__class__.__name__


class TestExtractShapValues:
    def test_plain_array_passthrough(self):
        arr = np.ones((4, 3))
        result = extract_shap_values(arr)
        assert result.shape == (4, 3)
        assert result is arr

    def test_list_selects_positive_class(self):
        values = extract_shap_values([np.zeros((2, 3)), np.ones((2, 3))])
        assert np.allclose(values, 1.0)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            extract_shap_values([])

    def test_numpy_array_expected_value(self):
        assert extract_base_value(np.array([0.3, 0.7])) == 0.7
        assert extract_base_value(0.5) == 0.5


class TestMeanAbsShap:
    def test_sorts_descending(self, fitted_models):
        shap_values = np.array([[0.1, 0.9], [0.2, 0.4]])
        frame = mean_abs_shap(shap_values, ["a", "b"])
        assert frame["feature"].tolist() == ["b", "a"]
        assert frame["mean_abs_shap"].tolist() == pytest.approx([0.65, 0.15])


class TestDisplayValues:
    def test_maps_engineered_and_one_hot_columns(self):
        engineered = pd.DataFrame({"cgpa": [7.8], "placement_training": ["Yes"]})
        names = ["cgpa", "placement_training_Yes", "placement_training_No"]
        display = make_display_values(engineered, names)
        assert display["cgpa"].tolist() == [7.8]
        assert display["placement_training_Yes"].tolist() == ["Yes"]
        assert display["placement_training_No"].tolist() == ["No"]

    def test_unknown_column_is_nan(self):
        engineered = pd.DataFrame({"cgpa": [7.8]})
        display = make_display_values(engineered, ["unknown_feature"])
        assert np.isnan(display["unknown_feature"].iloc[0])


class TestFigures:
    @pytest.fixture(autouse=True)
    def _skip_plotly_if_missing(self):
        pytest.importorskip("plotly")

    def test_waterfall_figure_has_bars(self, fitted_models):
        explainer = build_explainer(fitted_models["rf"], background=None)
        shap_values = extract_shap_values(explainer.shap_values(fitted_models["X_test"]))
        fig = waterfall_figure(
            shap_values[0],
            extract_base_value(explainer.expected_value),
            ["cgpa", "aptitude_test_score", "internships"],
            display_values=[7.9, 81.0, 1],
        )
        traces = [t for t in fig.data if t.type == "bar"]
        assert len(traces) == 1
        assert len(traces[0].x) == 3

    def test_beeswarm_figure_one_trace_per_feature(self, fitted_models):
        explainer = build_explainer(fitted_models["rf"], background=None)
        shap_values = extract_shap_values(explainer.shap_values(fitted_models["X_test"]))
        fig = beeswarm_figure(
            shap_values,
            ["cgpa", "aptitude_test_score", "internships"],
            max_display=2,
        )
        scatter_traces = [t for t in fig.data if t.type == "scatter"]
        assert len(scatter_traces) == 2

    def test_mean_shap_bar_figure_sorted(self):
        shap_values = np.array([[0.1, 0.9], [0.2, 0.4]])
        fig = mean_shap_bar_figure(shap_values, ["a", "b"])
        bar = [t for t in fig.data if t.type == "bar"][0]
        assert list(bar.y) == ["A", "B"]

    def test_prettify(self):
        assert prettify_feature_name("internships_normalized") == "Internships Normalized"
