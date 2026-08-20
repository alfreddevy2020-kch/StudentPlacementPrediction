"""
Shared SHAP explainability helpers for all three models.

Single source of truth for the SHAP computations used by:
  - part4/shap_analysis.py     (offline pipeline artifacts: CSV + PNG)
  - frontend/app.py            (interactive waterfall / beeswarm / bar)

Every function in this module works on the **model input matrix** — the
output of the fitted ColumnTransformer — so contributions match exactly what
the served models see. Plain (prefix-stripped) feature names are expected;
``numerical__cgpa`` becomes ``cgpa`` and ``categorical__placement_training_Yes``
becomes ``placement_training_Yes``, mirroring ``preprocessing.py``.

The waterfall/beeswarm figures accept ``display_values``: the *unscaled*
engineered feature values (or "Yes"/"No" for one-hot categoricals) so charts
read in human units while the contributions themselves are computed on the
scaled matrix the model consumes.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap

# ── Explainer construction ─────────────────────────────────────────────────

LINEAR_MODEL_MARKERS = ("logistic", "linear", "ridge", "lasso")


def _shap_below_0_50() -> bool:
    """True when the installed shap predates the xgboost 2.0 fix."""
    from packaging.version import Version

    return Version(shap.__version__) < Version("0.50.0")


def is_linear_model(model: object) -> bool:
    """True for sklearn linear estimators (exact SHAP via coefficients)."""
    class_name = model.__class__.__name__.lower()
    return any(marker in class_name for marker in LINEAR_MODEL_MARKERS)


def build_explainer(model: object, background: Optional[np.ndarray] = None):
    """
    Build the fastest exact SHAP explainer for ``model``.

    Parameters
    ----------
    model : fitted estimator
        Must be one of the three served model types (LogisticRegression,
        RandomForestClassifier, XGBClassifier). Unknown estimators fall back
        to ``shap.Explainer`` (Permutation) which is exact but slower.
    background : np.ndarray, optional
        Model-input matrix used for the expected value of linear models and
        the Permutation fallback. Required for linear estimators.
    """
    if is_linear_model(model):
        if background is None:
            raise ValueError(
                "Linear models require a background sample to compute "
                "expected values. Pass the transformed matrix."
            )
        return shap.LinearExplainer(model, background)

    class_name = model.__class__.__name__
    if "RandomForest" in class_name or "XGB" in class_name or "GradientBoost" in class_name:
        if "XGB" in class_name and _shap_below_0_50():
            raise ValueError(
                "SHAP < 0.50 cannot explain XGBoost models trained with "
                "xgboost >= 2.0 (base_score is serialized as a vector string). "
                "Upgrade shap: pip install -U 'shap>=0.50.0'"
            )
        return shap.TreeExplainer(model)

    if background is None:
        raise ValueError(
            f"Cannot auto-select an explainer for {class_name}; pass a "
            "background sample to use the permutation explainer."
        )
    return shap.Explainer(model, background)


def extract_shap_values(shap_output: object) -> np.ndarray:
    """
    Normalize the output of ``explainer.shap_values(X)``.

    Binary sklearn classifiers return a list of two arrays (class 0, class 1)
    in some shap versions and a single ``(n_samples, n_features, 2)`` array
    in others. This always returns the **positive-class** contributions with
    shape (n_samples, n_features).
    """
    if isinstance(shap_output, (list, tuple)):
        if len(shap_output) == 0:
            raise ValueError("shap_values returned an empty list.")
        arr = np.asarray(shap_output[-1])
    else:
        arr = np.asarray(shap_output)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        arr = arr[..., 1]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected shap_values shape: {arr.shape}")
    return arr


def extract_base_value(expected_value: object) -> float:
    """Normalize ``explainer.expected_value`` to a scalar for binary models."""
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        if len(expected_value) == 0:
            raise ValueError("expected_value is empty.")
        return float(expected_value[-1])
    return float(expected_value)


# ── Aggregations ───────────────────────────────────────────────────────────


def mean_abs_shap(
    shap_values: np.ndarray, feature_names: Sequence[str]
) -> pd.DataFrame:
    """Global importance: mean |SHAP| per feature, sorted descending."""
    importance = np.abs(shap_values).mean(axis=0)
    frame = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": importance}
    ).sort_values("mean_abs_shap", ascending=False)
    return frame.reset_index(drop=True)


# ── Display-value mapping (unscaled human-readable chart labels) ───────────


def make_display_values(
    engineered_frame: pd.DataFrame, feature_names: Sequence[str]
) -> pd.DataFrame:
    """
    Map transformed feature columns back to unscaled display values.

    ``engineered_frame`` is the output of
    ``BatchPredictor._prepare_features``: the 29 numerical (raw + engineered)
    columns plus the 2 raw categoricals, in ``ALL_FEATURES`` order. Numerical
    one-hot categorical columns (``placement_training_Yes``) display as the
    category label itself.
    """
    display = pd.DataFrame(index=engineered_frame.index)
    for name in feature_names:
        if name in engineered_frame.columns:
            display[name] = engineered_frame[name]
        elif "_Yes" in name or "_No" in name:
            display[name] = name.rsplit("_", 1)[-1]
        else:
            display[name] = np.nan
    return display


def prettify_feature_name(name: str) -> str:
    """``internships_normalized`` -> ``Internships Normalized``."""
    return name.replace("_", " ").title()


# ── Plotly figures (dark theme, aligned with the dashboard palette) ────────

POSITIVE_COLOR = "#F87171"
NEGATIVE_COLOR = "#60A5FA"
BASE_COLOR = "#64748B"


def _numeric_feature_values(values: object) -> np.ndarray:
    """
    Coerce display values to floats for beeswarm coloring. One-hot
    categorical displays ("Yes"/"No") map to 1.0/0.0; anything
    non-numeric falls back to 0.0.
    """
    values = np.asarray(values)
    if values.dtype.kind in "fiub":
        return np.nan_to_num(values.astype(float), nan=0.0)
    coerced = np.zeros(len(values), dtype=float)
    for i, value in enumerate(values):
        if isinstance(value, str):
            coerced[i] = 1.0 if value.lower() in ("yes", "true", "1") else 0.0
        else:
            try:
                coerced[i] = float(value)
            except (TypeError, ValueError):
                coerced[i] = 0.0
    return coerced


def _apply_dark_layout(fig: go.Figure, height: int, title: str) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": "#CBD5E1", "size": 12},
        title={
            "text": title,
            "font": {"family": "Inter, sans-serif", "size": 15, "color": "#F8FAFC"},
        },
        height=height,
        margin={"l": 24, "r": 24, "t": 52, "b": 24},
        xaxis={
            "gridcolor": "#334155",
            "zerolinecolor": "#334155",
            "tickfont": {"color": "#94A3B8"},
        },
        yaxis={
            "gridcolor": "#334155",
            "zerolinecolor": "#334155",
            "tickfont": {"color": "#CBD5E1"},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"color": "#CBD5E1"},
            "bgcolor": "rgba(0,0,0,0)",
        },
    )
    return fig


def waterfall_figure(
    shap_values_row: np.ndarray,
    base_value: float,
    feature_names: Sequence[str],
    display_values: Optional[Sequence[object]] = None,
    max_display: int = 12,
    title: str = "SHAP waterfall — how each feature moved the prediction",
) -> go.Figure:
    """
    Local explanation: base rate plus the top contributions that produced
    this student's predicted probability (log-odds scale).
    """
    shap_values_row = np.asarray(shap_values_row, dtype=float)
    if display_values is None:
        display_values = feature_names

    order = np.argsort(-np.abs(shap_values_row))[:max_display]
    contributions = shap_values_row[order]
    names = [prettify_feature_name(feature_names[i]) for i in order]
    values = [display_values[i] for i in order]

    cumulative = np.concatenate(
        ([base_value], np.cumsum(contributions) + base_value)
    )
    starts = cumulative[:-1]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=contributions,
            y=names,
            base=starts,
            orientation="h",
            marker_color=[
                POSITIVE_COLOR if c >= 0 else NEGATIVE_COLOR for c in contributions
            ],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Value: %{customdata}<br>"
                "Contribution: %{x:.3f}<extra></extra>"
            ),
            customdata=[[str(v)] for v in values],
            name="Feature contribution",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[starts[0], cumulative[-1]],
            y=[len(names) - 0.5, -0.5],
            mode="lines",
            line={"color": "#F8FAFC", "width": 1.5, "dash": "dot"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[base_value],
            y=[len(names) - 0.5],
            mode="markers",
            marker={"color": BASE_COLOR, "size": 9, "symbol": "diamond"},
            name=f"Base rate ({base_value:.3f})",
            hovertemplate=f"Base rate: {base_value:.3f}<extra></extra>",
        )
    )
    fig.add_vline(
        x=cumulative[-1],
        line_dash="dash",
        line_color="#34D399",
        annotation_text=f"f(x) = {cumulative[-1]:.3f}",
        annotation_font_color="#34D399",
    )
    fig.update_xaxes(title_text="log-odds contribution to placement probability")
    return _apply_dark_layout(fig, height=420, title=title)


def beeswarm_figure(
    shap_values: np.ndarray,
    feature_names: Sequence[str],
    display_values: Optional[np.ndarray] = None,
    max_display: int = 15,
    title: str = "SHAP beeswarm — global feature impact direction",
) -> go.Figure:
    """
    Global explanation: one dot per sample, colored by feature value.
    Red = high feature value, blue = low. Right of zero pushes toward
    placement; left pushes away.
    """
    shap_values = np.asarray(shap_values, dtype=float)
    mean_abs = mean_abs_shap(shap_values, feature_names)
    top = mean_abs.head(max_display)
    top_idx = [
        list(feature_names).index(f) for f in top["feature"]
    ]

    fig = go.Figure()
    for pos, i in enumerate(top_idx):
        values = shap_values[:, i]
        feat_vals = (
            display_values[:, i]
            if display_values is not None
            else shap_values[:, i]
        )
        feat_vals = _numeric_feature_values(feat_vals)

        order = np.argsort(feat_vals, kind="stable")
        jitter = np.linspace(-0.35, 0.35, len(order))
        span = np.ptp(feat_vals) if np.ptp(feat_vals) > 0 else 1.0
        color = (feat_vals[order] - feat_vals[order].min()) / span

        fig.add_trace(
            go.Scatter(
                x=values[order],
                y=jitter + pos,
                mode="markers",
                marker={
                    "size": 5,
                    "color": color,
                    "colorscale": [[0.0, NEGATIVE_COLOR], [1.0, POSITIVE_COLOR]],
                    "opacity": 0.7,
                    "line": {"width": 0},
                },
                customdata=feat_vals[order],
                hovertemplate=(
                    f"<b>{prettify_feature_name(feature_names[i])}</b><br>"
                    "SHAP: %{x:.3f}<br>"
                    "Value: %{customdata:.3f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.add_vline(
        x=0,
        line_dash="solid",
        line_color="#94A3B8",
        line_width=1,
        annotation_text="pushes toward NOT placed ← 0 → pushes toward placed",
        annotation_font_color="#94A3B8",
        annotation_position="top",
    )
    fig.update_yaxes(
        tickvals=list(range(len(top_idx))),
        ticktext=[prettify_feature_name(f) for f in top["feature"]],
        range=[-0.6, len(top_idx) - 0.4],
        title_text="",
    )
    fig.update_xaxes(title_text="SHAP value (log-odds)")
    return _apply_dark_layout(fig, height=460, title=title)


def mean_shap_bar_figure(
    shap_values: np.ndarray,
    feature_names: Sequence[str],
    max_display: int = 15,
    title: str = "SHAP mean |impact| — top features",
) -> go.Figure:
    """Global importance bar: mean absolute SHAP per feature."""
    importance = mean_abs_shap(shap_values, feature_names).head(max_display)
    importance = importance.iloc[::-1]

    fig = go.Figure(
        go.Bar(
            x=importance["mean_abs_shap"],
            y=[prettify_feature_name(f) for f in importance["feature"]],
            orientation="h",
            marker={
                "color": importance["mean_abs_shap"],
                "colorscale": [[0.0, NEGATIVE_COLOR], [1.0, POSITIVE_COLOR]],
            },
            hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Mean |SHAP value| (log-odds)")
    return _apply_dark_layout(fig, height=440, title=title)