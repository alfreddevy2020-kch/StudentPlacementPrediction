"""
CampusReady — Multi-Tab Executive Dashboard
===================================================
Premium Streamlit dashboard with 3 tabs:
  Tab 1: Departmental Pulse & Readiness Analytics
  Tab 2: Per-Student Diagnostic & Skill-Gap Analysis
  Tab 3: Cohort What-If Policy Simulator

Plus: Multi-Model Benchmark Comparison expander

Ported from prediction.txt design system.

Usage:
    cd frontend
    streamlit run app.py

Prerequisites:
    - Model artifacts in artifacts/production/ (or part2/models/ and
      part3/models/ as a local-dev fallback)
    - Dataset at data/raw/student_placement.csv
"""

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add repo root and frontend dir to sys.path so local imports work
REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

from batch_predictor import (
    BatchPredictor,
)
from simulator import INTERVENTION_KNOBS, CohortWhatIfSimulator

from feature_engineering import (
    FEATURE_RANGES,
    TARGET_COLUMN,
    TARGET_MAP,
    normalize_columns,
)

# =============================================================================
# 1. PAGE SETUP & THEME STYLING
# =============================================================================
st.set_page_config(
    page_title="CampusReady | Student Placement Readiness & Policy Simulator",
    page_icon=":material/school:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# NOTE: All visual styling is handled natively via .streamlit/config.toml
# and st.container(border=True). No custom CSS injection needed.


def get_plotly_layout(height: int = 320, title: Optional[str] = None) -> dict[str, Any]:
    """Returns a unified executive dark Plotly layout aligned with the theme."""
    layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, sans-serif", "color": "#CBD5E1", "size": 12},
        "margin": {"l": 24, "r": 24, "t": 36 if title else 16, "b": 24},
        "height": height,
        "xaxis": {
            "gridcolor": "#334155",
            "zerolinecolor": "#334155",
            "tickfont": {"color": "#94A3B8"},
            "title": {"font": {"color": "#CBD5E1"}},
        },
        "yaxis": {
            "gridcolor": "#334155",
            "zerolinecolor": "#334155",
            "tickfont": {"color": "#94A3B8"},
            "title": {"font": {"color": "#CBD5E1"}},
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"color": "#CBD5E1"},
            "bgcolor": "rgba(0,0,0,0)",
        },
    }
    if title:
        layout["title"] = {
            "text": title,
            "font": {"family": "Inter, sans-serif", "size": 15, "color": "#F8FAFC"},
        }
    return layout


# Risk tier color palette (shared across all charts)
RISK_COLORS = {
    "High Risk": "#F87171",
    "Moderate Risk": "#FBBF24",
    "Interview Ready": "#34D399",
}


# =============================================================================
# 2. SYSTEM BOOTSTRAP & CACHING
# =============================================================================

@st.cache_resource(show_spinner="Loading ML Models & Dataset...")
def load_system():
    """Load dataset, preprocessor, and all model artifacts once."""
    predictor = BatchPredictor()
    predictor.load()
    default_df = predictor.load_dataset()
    simulator = CohortWhatIfSimulator(predictor)
    return predictor, default_df, simulator


try:
    predictor, default_raw_df, simulator = load_system()
    system_loaded = True
except Exception as load_err:
    system_loaded = False
    st.error(
        f":material/error: **System Load Error:** {load_err}\n\n"
        "Ensure model artifacts exist in `part2/models/` and `part3/models/`, "
        "and the dataset is at `data/raw/student_placement.csv`."
    )
    st.stop()


# =============================================================================
# 3. SIDEBAR CONTROLS
# =============================================================================
with st.sidebar:
    st.markdown("### :material/school: CampusReady")
    st.caption("Student placement readiness & policy simulator")
    st.space("small")

    # Dataset Source & Upload
    st.subheader(":material/dataset: Student dataset")
    uploaded_file = st.file_uploader(
        "Upload student cohort CSV",
        type=["csv"],
        help="Upload a CSV with the same structure as the training dataset.",
    )

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            user_df.columns = [str(c).strip() for c in user_df.columns]
            user_df = normalize_columns(user_df)
            if "student_id" not in user_df.columns:
                user_df.insert(0, "student_id", range(1, len(user_df) + 1))
            raw_df = user_df
            st.success(
                f"**Uploaded cohort:** {len(raw_df):,} students • {len(raw_df.columns)} features",
                icon=":material/check_circle:",
            )
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            raw_df = default_raw_df
    else:
        raw_df = default_raw_df
        st.caption(f"**Default dataset:** {len(raw_df):,} students • {len(raw_df.columns)} features")

    # Sample template download
    sample_csv = default_raw_df.head(20).to_csv(index=False).encode("utf-8")
    st.download_button(
        ":material/download: Download sample CSV",
        data=sample_csv,
        file_name="sample_student_placement.csv",
        mime="text/csv",
        help="Download a 20-row sample CSV matching the expected schema.",
    )

    # Model selector — stored per-session, never mutates the cached singleton
    st.space("small")
    st.subheader(":material/psychology: Inference engine")
    model_names = predictor.available_models
    st.session_state.setdefault("active_model", predictor.active_model_name)
    selected_model_name = st.selectbox(
        "Active model",
        options=model_names,
        index=(
            model_names.index(st.session_state["active_model"])
            if st.session_state["active_model"] in model_names
            else 0
        ),
        help="Select classification model to drive all dashboard predictions.",
    )
    st.session_state["active_model"] = selected_model_name

    # Cohort filters — pills for binary/ternary options
    st.space("small")
    st.subheader(":material/filter_alt: Cohort filters")

    # This dataset carries no demographic or department columns, so cohorts
    # are segmented on the two institutional-support flags it does provide.
    training_options = (
        ["ALL"] + sorted(raw_df["placement_training"].dropna().astype(str).unique().tolist())
        if "placement_training" in raw_df.columns
        else ["ALL"]
    )
    selected_training = st.pills(
        "Placement training",
        training_options,
        default="ALL",
        key="filter_training",
    )

    extra_options = (
        ["ALL"] + sorted(raw_df["extracurricular_activities"].dropna().astype(str).unique().tolist())
        if "extracurricular_activities" in raw_df.columns
        else ["ALL"]
    )
    selected_extra = st.pills(
        "Extracurricular activities",
        extra_options,
        default="ALL",
        key="filter_extra",
    )

    # Academic band filter — derived, gives a department-style breakdown
    cgpa_band_options = ["ALL", "< 7.0", "7.0 – 8.0", "> 8.0"]
    selected_band = st.pills(
        "CGPA band", cgpa_band_options, default="ALL", key="filter_band"
    )

    # Apply filters
    filtered_df = raw_df.copy()
    if selected_training and selected_training != "ALL" and "placement_training" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["placement_training"] == selected_training]
    if selected_extra and selected_extra != "ALL" and "extracurricular_activities" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["extracurricular_activities"] == selected_extra
        ]
    if selected_band and selected_band != "ALL" and "cgpa" in filtered_df.columns:
        if selected_band == "< 7.0":
            filtered_df = filtered_df[filtered_df["cgpa"] < 7.0]
        elif selected_band == "7.0 – 8.0":
            filtered_df = filtered_df[
                (filtered_df["cgpa"] >= 7.0) & (filtered_df["cgpa"] <= 8.0)
            ]
        else:
            filtered_df = filtered_df[filtered_df["cgpa"] > 8.0]

    cohort_ratio = f"{len(filtered_df):,} / {len(raw_df):,}"
    st.caption(f":material/groups: Active cohort: **{cohort_ratio}** students")


# =============================================================================
# 4. BATCH PREDICTIONS ON FILTERED COHORT
# =============================================================================
if not filtered_df.empty:
    try:
        cohort_probs = predictor.predict_probabilities(
            filtered_df, model_name=st.session_state.get("active_model")
        )
    except Exception as pred_err:
        st.error(
            f":material/error: **Prediction pipeline failed:** `{pred_err}`\n\n"
            "The loaded preprocessor/model artifacts in `part2/models/` and "
            "`part3/models/` don't match the columns `feature_engineering.py` "
            "produces from `data/raw/student_placement.csv`. Showing a "
            "placeholder probability for every student would silently hide "
            "this mismatch, so the dashboard stops here instead.\n\n"
            "Regenerate the artifacts for the current schema: "
            "`python download_dataset.py` → `python preprocessing.py` → "
            "retrain the models (e.g. `python scripts/train_models_fast.py`), "
            "then reload this page."
        )
        st.stop()

    filtered_df = filtered_df.copy()
    filtered_df["placement_prob"] = np.round(cohort_probs * 100, 1)
    filtered_df["predicted_status"] = np.where(
        cohort_probs >= 0.50, "Placed", "Not Placed"
    )
    filtered_df["risk_tier"] = [
        predictor.classify_risk(p) for p in cohort_probs
    ]


# =============================================================================
# 5. DASHBOARD HEADER
# =============================================================================
st.badge(
    "Student Placement Prediction System",
    icon=":material/auto_awesome:",
    color="blue",
)
st.title("Student placement prediction & readiness system")
st.caption(
    f"Production ML classification engine, diagnostic radar, & macro cohort simulator • "
    f"Driven by **{selected_model_name}**"
)

# 3 Primary Tabs — Material icons, sentence casing (per design.md)
tab1, tab2, tab3 = st.tabs([
    ":material/bar_chart: Departmental pulse & readiness",
    ":material/person_search: Per-student diagnostic & skill-gaps",
    ":material/tune: Cohort what-if policy simulator",
])


# =============================================================================
# TAB 1: DEPARTMENTAL PULSE & READINESS ANALYTICS
# =============================================================================
with tab1:
    if filtered_df.empty:
        st.warning("No students match the active filter criteria. Adjust filters in the sidebar.")
    else:
        # Executive KPIs Row
        with st.container(border=True):
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            total_count = len(filtered_df)
            placed_count = int((filtered_df["predicted_status"] == "Placed").sum())
            placement_rate = (
                round((placed_count / total_count) * 100, 1) if total_count > 0 else 0.0
            )
            critical_risk_count = int((filtered_df["placement_prob"] < 40.0).sum())
            avg_score = round(
                float(filtered_df["placement_prob"].mean()), 1
            )

            with kpi1:
                st.metric(
                    "Total cohort size",
                    f"{total_count:,}",
                    help="Total students in current filter",
                )
            with kpi2:
                st.metric(
                    "Projected placement rate",
                    f"{placement_rate}%",
                    f"{placed_count} placed",
                )
            with kpi3:
                st.metric(
                    "Critical risk (<40%)",
                    f"{critical_risk_count}",
                    f"{round((critical_risk_count / max(1, total_count)) * 100, 1)}% of cohort",
                    delta_color="inverse",
                )
            with kpi4:
                # Sparkline showing probability distribution shape
                prob_sparkline = sorted(filtered_df["placement_prob"].tolist())
                st.metric(
                    "Mean placement likelihood",
                    f"{avg_score}%",
                    help="Average predicted probability across cohort",
                    chart_data=prob_sparkline[-30:],
                    chart_type="bar",
                )

        # Department Readiness & Risk Donut
        col_dept, col_donut = st.columns([3, 2])

        with col_dept, st.container(border=True):
            st.markdown("#### :material/bar_chart: Readiness by CGPA band")
            st.caption(
                "This dataset has no department column, so cohorts are "
                "banded by CGPA — the closest available grouping."
            )
            band_df = filtered_df.copy()
            band_df["cohort"] = pd.cut(
                band_df["cgpa"],
                bins=[0, 7.0, 7.5, 8.0, 8.5, 10.0],
                labels=["< 7.0", "7.0–7.5", "7.5–8.0", "8.0–8.5", "> 8.5"],
                include_lowest=True,
            )
            dept_stats = (
                band_df.groupby("cohort", observed=True)
                .agg(
                    student_count=("student_id", "count"),
                    placement_rate=(
                        "placement_prob",
                        lambda x: np.round(np.mean(x >= 50.0) * 100, 1),
                    ),
                    avg_likelihood=(
                        "placement_prob",
                        lambda x: np.round(np.mean(x), 1),
                    ),
                )
                .reset_index()
            )
            dept_stats["cohort"] = dept_stats["cohort"].astype(str)

            fig_bar = go.Figure()
            fig_bar.add_trace(
                go.Bar(
                    x=dept_stats["cohort"],
                    y=dept_stats["placement_rate"],
                    name="Placement Rate (%)",
                    marker_color="#3B82F6",
                    text=dept_stats["placement_rate"].astype(str) + "%",
                    textposition="auto",
                )
            )
            fig_bar.add_trace(
                go.Bar(
                    x=dept_stats["cohort"],
                    y=dept_stats["avg_likelihood"],
                    name="Avg Placement Likelihood (%)",
                    marker_color="#60A5FA",
                    text=dept_stats["avg_likelihood"].astype(str) + "%",
                    textposition="auto",
                )
            )
            layout_bar = get_plotly_layout(height=280)
            layout_bar["barmode"] = "group"
            layout_bar["yaxis"]["range"] = [0, 100]
            fig_bar.update_layout(layout_bar)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_donut, st.container(border=True):
            st.markdown("#### :material/pie_chart: Risk tier distribution")
            risk_dist = (
                filtered_df["risk_tier"].value_counts().reset_index()
            )
            risk_dist.columns = ["Risk Tier", "Count"]

            fig_pie = px.pie(
                risk_dist,
                names="Risk Tier",
                values="Count",
                color="Risk Tier",
                color_discrete_map=RISK_COLORS,
                hole=0.55,
            )
            fig_pie.update_traces(
                textinfo="percent+label",
                textfont={"color": "#F8FAFC", "size": 11},
            )
            layout_pie = get_plotly_layout(height=280)
            layout_pie["showlegend"] = False
            fig_pie.update_layout(layout_pie)
            st.plotly_chart(fig_pie, use_container_width=True)

        # Placement Probability Distribution
        with st.container(border=True):
            st.markdown(
                "#### :material/stacked_line_chart: Placement probability distribution & thresholds"
            )
            fig_dist = px.histogram(
                filtered_df,
                x="placement_prob",
                color="risk_tier",
                nbins=30,
                color_discrete_map=RISK_COLORS,
                labels={
                    "placement_prob": "Predicted Placement Likelihood (%)",
                    "count": "Student Count",
                },
                opacity=0.85,
            )
            fig_dist.add_vline(
                x=40,
                line_dash="dot",
                line_color="#F87171",
                annotation_text="Critical Risk (<40%)",
                annotation_font_color="#F87171",
            )
            fig_dist.add_vline(
                x=50,
                line_dash="dash",
                line_color="#FBBF24",
                annotation_text="Decision Cutoff (50%)",
                annotation_font_color="#FBBF24",
            )
            fig_dist.add_vline(
                x=75,
                line_dash="dash",
                line_color="#34D399",
                annotation_text="Interview Ready (≥75%)",
                annotation_font_color="#34D399",
            )
            fig_dist.update_layout(get_plotly_layout(height=260))
            st.plotly_chart(fig_dist, use_container_width=True)

        # Student roster table
        with st.container(border=True):
            st.markdown("#### :material/table_chart: Cohort candidate readiness roster")

            display_cols = [
                "student_id",
                "cgpa",
                "ssc_marks",
                "hsc_marks",
                "aptitude_test_score",
                "soft_skills_rating",
                "internships",
                "projects",
                "workshops_certifications",
                "placement_training",
                "placement_prob",
                "risk_tier",
                "predicted_status",
            ]
            display_cols = [c for c in display_cols if c in filtered_df.columns]

            column_configs = {
                "student_id": st.column_config.TextColumn(
                    "Student ID", pinned=True,
                ),
                "cgpa": st.column_config.ProgressColumn(
                    "CGPA", format="%.1f", min_value=0, max_value=10,
                ),
                "ssc_marks": st.column_config.NumberColumn("SSC %", format="%.0f"),
                "hsc_marks": st.column_config.NumberColumn("HSC %", format="%.0f"),
                "aptitude_test_score": st.column_config.ProgressColumn(
                    "Aptitude", format="%.0f", min_value=0, max_value=100,
                ),
                "soft_skills_rating": st.column_config.ProgressColumn(
                    "Soft skills", format="%.1f", min_value=0, max_value=5,
                ),
                "internships": st.column_config.NumberColumn("Internships"),
                "projects": st.column_config.NumberColumn("Projects"),
                "workshops_certifications": st.column_config.NumberColumn("Certs"),
                "placement_training": st.column_config.TextColumn("Training"),
                "placement_prob": st.column_config.ProgressColumn(
                    "Placement likelihood",
                    help="Model-projected placement probability",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "risk_tier": st.column_config.TextColumn("Risk classification"),
                "predicted_status": st.column_config.TextColumn("Projected status"),
            }

            roster_df = (
                filtered_df[display_cols]
                .sort_values(by="placement_prob", ascending=False)
            )

            st.dataframe(
                roster_df,
                column_config=column_configs,
                use_container_width=True,
                hide_index=True,
            )

            # Download button for export
            csv_data = roster_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                ":material/download: Download roster CSV",
                data=csv_data,
                file_name="cohort_readiness_roster.csv",
                mime="text/csv",
                type="tertiary",
            )


# =============================================================================
# TAB 2: PER-STUDENT DIAGNOSTIC & SKILL-GAP ANALYSIS
# =============================================================================
with tab2:
    st.markdown("### :material/person_search: Candidate diagnostic & skill-gap diagnosis")
    st.caption(
        "Multi-dimensional readiness radar and prescriptive remediation engine"
    )

    diag_mode = st.segmented_control(
        "Candidate input mode",
        options=["Existing candidate", "Custom profiler"],
        default="Existing candidate",
    )

    if diag_mode == "Existing candidate":
        stu_options = filtered_df["student_id"].tolist()
        if not stu_options:
            st.warning("No students available. Adjust sidebar filters.")
            st.stop()
        selected_id = st.selectbox(
            "Search & select student ID", options=stu_options, index=0
        )
        student_data = (
            filtered_df[filtered_df["student_id"] == selected_id]
            .iloc[0]
            .to_dict()
        )
    else:
        st.caption(
            ":material/tune: Tune candidate metrics to test hypothetical profiles"
        )

        with st.container(border=True):
            student_data = {"student_id": 99999}

            st.caption(
                "Slider ranges match the model's training data. Values at the "
                "extremes are still within what the model has seen."
            )

            # Academic Performance
            st.markdown("**Academic record**")
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                student_data["cgpa"] = st.slider(
                    "Current CGPA",
                    6.5, 9.1,
                    value=7.7,
                    step=0.1,
                    key="diag_cgpa_slider",
                )
            with ac2:
                student_data["ssc_marks"] = st.slider(
                    "SSC / Class 10 %",
                    55.0, 90.0,
                    value=70.0,
                    step=1.0,
                    key="diag_ssc_slider",
                )
            with ac3:
                student_data["hsc_marks"] = st.slider(
                    "HSC / Class 12 %",
                    57.0, 88.0,
                    value=74.0,
                    step=1.0,
                    key="diag_hsc_slider",
                )

            # Skills & Test Scores
            st.markdown("**Skills & assessment**")
            sk1, sk2 = st.columns(2)
            with sk1:
                student_data["aptitude_test_score"] = st.slider(
                    "Aptitude test score", 60.0, 90.0, 80.0, 1.0, key="diag_aptitude"
                )
            with sk2:
                student_data["soft_skills_rating"] = st.slider(
                    "Soft skills rating (0–5)", 3.0, 4.8, 4.3, 0.1, key="diag_soft"
                )

            # Experience & portfolio
            st.markdown("**Experience & portfolio**")
            ex1, ex2, ex3 = st.columns(3)
            with ex1:
                student_data["internships"] = st.number_input(
                    "Internships", 0, 2,
                    value=1,
                    key="diag_intern_input"
                )
            with ex2:
                student_data["projects"] = st.number_input(
                    "Projects", 0, 3,
                    value=2,
                    key="diag_proj_input"
                )
            with ex3:
                student_data["workshops_certifications"] = st.number_input(
                    "Workshops / certifications", 0, 3,
                    value=1,
                    key="diag_certs_input"
                )

            # Institutional support
            st.markdown("**Institutional support**")
            su1, su2 = st.columns(2)
            with su1:
                student_data["placement_training"] = st.selectbox(
                    "Placement training", options=["Yes", "No"], key="diag_training"
                )
            with su2:
                student_data["extracurricular_activities"] = st.selectbox(
                    "Extracurricular activities", options=["Yes", "No"], key="diag_extra"
                )

    # Evaluate single candidate
    candidate_prob = predictor.predict_single(
        student_data, model_name=st.session_state.get("active_model")
    )
    candidate_prob_pct = round(candidate_prob * 100, 1)

    diag_left, diag_right = st.columns([1, 2])

    with diag_left, st.container(border=True):
        st.markdown("#### :material/speed: Placement probability")
        bar_color = (
            "#F87171"
            if candidate_prob_pct < 50
            else ("#FBBF24" if candidate_prob_pct < 75 else "#34D399")
        )
        display_label = str(student_data.get("student_id", "Student"))

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=candidate_prob_pct,
                number={
                    "suffix": "%",
                    "font": {
                        "size": 38,
                        "color": "#F8FAFC",
                        "family": "JetBrains Mono, monospace",
                    },
                },
                title={
                    "text": f"<b>Student {display_label}</b>",
                    "font": {"size": 14, "color": "#94A3B8"},
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": "#334155",
                    },
                    "bar": {"color": bar_color, "thickness": 0.35},
                    "bgcolor": "#1E293B",
                    "steps": [
                        {
                            "range": [0, 50],
                            "color": "rgba(248, 113, 113, 0.15)",
                        },
                        {
                            "range": [50, 75],
                            "color": "rgba(251, 191, 36, 0.15)",
                        },
                        {
                            "range": [75, 100],
                            "color": "rgba(52, 211, 153, 0.15)",
                        },
                    ],
                    "threshold": {
                        "line": {"color": "#60A5FA", "width": 3},
                        "thickness": 0.75,
                        "value": candidate_prob_pct,
                    },
                },
            )
        )
        fig_gauge.update_layout(get_plotly_layout(height=240))
        st.plotly_chart(fig_gauge, use_container_width=True)

        if candidate_prob_pct >= 75:
            st.success(
                "**Interview ready** — High likelihood of shortlisting.",
                icon=":material/verified:",
            )
        elif candidate_prob_pct >= 50:
            st.warning(
                "**Moderate risk** — Viable profile requiring focused prep.",
                icon=":material/warning:",
            )
        else:
            st.error(
                "**Critical risk** — Significant preparedness gaps identified.",
                icon=":material/error:",
            )

    with diag_right, st.container(border=True):
        st.markdown("#### :material/radar: Multi-dimensional competency radar")

        # Define radar axes using available features, each rescaled to 0-100
        radar_specs = [
            {"column": "cgpa", "label": "CGPA (×10)", "scale_factor": 10.0},
            {"column": "ssc_marks", "label": "SSC %", "scale_factor": 1.0},
            {"column": "hsc_marks", "label": "HSC %", "scale_factor": 1.0},
            {"column": "aptitude_test_score", "label": "Aptitude", "scale_factor": 1.0},
            {"column": "soft_skills_rating", "label": "Soft skills (×20)", "scale_factor": 20.0},
            {"column": "projects", "label": "Projects (×33)", "scale_factor": 33.3},
            {"column": "workshops_certifications", "label": "Certs (×33)", "scale_factor": 33.3},
        ]

        # Compute placed peers benchmark. If raw_df has ground-truth PlacementStatus,
        # use placed students from it; otherwise fall back to default training dataset placed peers.
        target_col = TARGET_COLUMN
        placed_mask = (
            raw_df[target_col].astype(str) == "Placed"
            if target_col in raw_df.columns
            else None
        )
        if placed_mask is not None and placed_mask.any():
            placed_peers = raw_df[placed_mask]
        elif target_col in default_raw_df.columns:
            placed_peers = default_raw_df[default_raw_df[target_col].astype(str) == "Placed"]
        else:
            placed_peers = raw_df

        radar_categories = []
        cand_radar_vals = []
        placed_radar_vals = []

        for r_spec in radar_specs:
            col_name = r_spec["column"]
            scale_f = float(r_spec["scale_factor"])
            label = r_spec["label"]
            radar_categories.append(label)

            # Candidate value
            cand_val = min(
                100.0,
                max(0.0, float(student_data.get(col_name, 0.0)) * scale_f),
            )
            cand_radar_vals.append(cand_val)

            # Placed peers average
            if col_name in placed_peers.columns:
                peer_val = min(
                    100.0,
                    max(0.0, float(placed_peers[col_name].mean()) * scale_f),
                )
            else:
                peer_val = 0.0
            placed_radar_vals.append(peer_val)

        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=placed_radar_vals,
                theta=radar_categories,
                fill="toself",
                name="Placed Peers Benchmark",
                line_color="#64748B",
                fillcolor="rgba(100, 116, 139, 0.2)",
                opacity=0.5,
            )
        )
        fig_radar.add_trace(
            go.Scatterpolar(
                r=cand_radar_vals,
                theta=radar_categories,
                fill="toself",
                name="Selected Candidate",
                line_color="#60A5FA",
                fillcolor="rgba(96, 165, 250, 0.35)",
                opacity=0.9,
            )
        )
        radar_layout = get_plotly_layout(height=300)
        radar_layout["polar"] = {
            "radialaxis": {
                "visible": True,
                "range": [0, 100],
                "gridcolor": "#334155",
                "tickfont": {"color": "#94A3B8"},
            },
            "angularaxis": {
                "gridcolor": "#334155",
                "tickfont": {"color": "#CBD5E1", "size": 11},
            },
            "bgcolor": "rgba(0,0,0,0)",
        }
        radar_layout["showlegend"] = True
        fig_radar.update_layout(radar_layout)
        st.plotly_chart(fig_radar, use_container_width=True)

    # Prescriptive remediation engine
    st.space("small")
    st.markdown("### :material/lightbulb: Prescriptive remediation & targeted interventions")
    st.caption("Quantified action recommendations with simulated probability uplifts")

    # sim_value bounds stay inside the training ranges in FEATURE_RANGES —
    # recommending a value the model has never seen would produce a
    # confident but meaningless uplift number.
    remediation_rules = [
        {
            "condition": lambda s: s.get("placement_training", "Yes") == "No",
            "priority": "CRITICAL",
            "title": "Enrol in Placement Training",
            "action": "Join the institutional placement-training programme — the single strongest controllable factor in this cohort.",
            "sim_column": "placement_training",
            "sim_op": "set",
            "sim_value": "Yes",
        },
        {
            "condition": lambda s: s.get("aptitude_test_score", 100) < 78,
            "priority": "HIGH",
            "title": "Raise Aptitude Test Score",
            "action": "Complete structured aptitude coaching and timed mock tests to lift quantitative and reasoning scores.",
            "sim_column": "aptitude_test_score",
            "sim_op": "add",
            "sim_value": 10.0,
        },
        {
            "condition": lambda s: s.get("soft_skills_rating", 5.0) < 4.3,
            "priority": "HIGH",
            "title": "Strengthen Communication & Soft Skills",
            "action": "Attend group-discussion and mock-interview workshops to improve the soft-skills rating.",
            "sim_column": "soft_skills_rating",
            "sim_op": "add",
            "sim_value": 0.5,
        },
        {
            "condition": lambda s: s.get("projects", 10) < 2,
            "priority": "MEDIUM",
            "title": "Build End-to-End Projects",
            "action": "Complete at least two deployed projects to demonstrate practical capability.",
            "sim_column": "projects",
            "sim_op": "add",
            "sim_value": 1.0,
        },
        {
            "condition": lambda s: s.get("workshops_certifications", 10) < 2,
            "priority": "MEDIUM",
            "title": "Pursue Industry Certifications",
            "action": "Earn additional industry certifications (AWS, Azure, Google Cloud, etc.).",
            "sim_column": "workshops_certifications",
            "sim_op": "add",
            "sim_value": 1.0,
        },
        {
            "condition": lambda s: s.get("cgpa", 10) < 7.5,
            "priority": "MEDIUM",
            "title": "Improve CGPA Above 7.5",
            "action": "Focus on upcoming semester exams — many companies set a CGPA cutoff at shortlisting.",
            "sim_column": "cgpa",
            "sim_op": "add",
            "sim_value": 0.5,
        },
        {
            "condition": lambda s: s.get("internships", 10) < 1,
            "priority": "MEDIUM",
            "title": "Secure an Internship",
            "action": "Apply for at least one internship to gain industry exposure before placements.",
            "sim_column": "internships",
            "sim_op": "add",
            "sim_value": 1.0,
        },
        {
            "condition": lambda s: s.get("extracurricular_activities", "Yes") == "No",
            "priority": "LOW",
            "title": "Join Extracurricular Activities",
            "action": "Participate in clubs or events to build the collaborative profile recruiters screen for.",
            "sim_column": "extracurricular_activities",
            "sim_op": "set",
            "sim_value": "Yes",
        },
    ]

    remediations = []
    for rule in remediation_rules:
        try:
            if rule["condition"](student_data):
                # Simulate the intervention
                sim_s = dict(student_data)
                col = rule["sim_column"]
                if col in sim_s:
                    if rule["sim_op"] == "subtract":
                        sim_s[col] = sim_s[col] - rule["sim_value"]
                    elif rule["sim_op"] == "add":
                        sim_s[col] = sim_s[col] + rule["sim_value"]
                    elif rule["sim_op"] == "set":
                        sim_s[col] = rule["sim_value"]
                    # Keep the simulated profile inside the training range so
                    # the quoted uplift reflects something the model has seen.
                    lo, hi = FEATURE_RANGES.get(col, (None, None))
                    if lo is not None:
                        sim_s[col] = min(max(sim_s[col], lo), hi)

                new_p = predictor.predict_single(
                    sim_s, model_name=st.session_state.get("active_model")
                )
                gain = round((new_p - candidate_prob) * 100, 1)

                color = (
                    "#F87171"
                    if rule["priority"] == "CRITICAL"
                    else ("#FB923C" if rule["priority"] == "HIGH" else "#60A5FA")
                )
                remediations.append({
                    "priority": rule["priority"],
                    "title": rule["title"],
                    "action": rule["action"],
                    # Sign-aware: a negative gain must not render as "+-4.5%".
                    "uplift": f"{gain:+.1f}% Placement Uplift",
                    "color": color,
                })
        except Exception:
            pass

    if not remediations:
        with st.container(border=True):
            st.success(
                "**Optimal profile** — This candidate already satisfies "
                "all institutional competency benchmarks!",
                icon=":material/celebration:",
            )
    else:
        for rem in remediations:
            with st.container(border=True):
                r_left, r_mid, r_right = st.columns([1, 4, 2])
                with r_left:
                    if rem["priority"] == "CRITICAL":
                        st.badge(
                            "Critical priority",
                            icon=":material/error:",
                            color="red",
                        )
                    elif rem["priority"] == "HIGH":
                        st.badge(
                            "High priority",
                            icon=":material/warning:",
                            color="orange",
                        )
                    else:
                        st.badge(
                            "Medium priority",
                            icon=":material/info:",
                            color="blue",
                        )
                with r_mid:
                    st.markdown(f"**{rem['title']}**")
                    st.write(rem["action"])
                with r_right:
                    st.metric("Uplift", rem["uplift"])


# =============================================================================
# TAB 3: COHORT WHAT-IF POLICY SIMULATOR
# =============================================================================
with tab3:
    st.markdown("### :material/tune: Cohort policy intervention & what-if simulator")
    st.caption(
        "Simulate institutional training interventions, risk migrations, "
        "and candidate conversions before allocating budget."
    )

    sim_left, sim_right = st.columns([1, 2])

    with sim_left:
        with st.container(border=True):
            st.markdown("#### 1. Define target segment")

            # No demographic columns in this dataset, so the target segment
            # is defined by who has already received institutional support.
            sim_segment = st.pills(
                "Target segment",
                ["ALL COHORTS", "Untrained only", "Trained only", "No extracurriculars"],
                default="ALL COHORTS",
                key="sim_segment",
            )

            target_slice = raw_df.copy()
            if sim_segment == "Untrained only":
                target_slice = target_slice[target_slice["placement_training"] == "No"]
            elif sim_segment == "Trained only":
                target_slice = target_slice[target_slice["placement_training"] == "Yes"]
            elif sim_segment == "No extracurriculars":
                target_slice = target_slice[
                    target_slice["extracurricular_activities"] == "No"
                ]

            st.caption(
                f":material/groups: Target cohort: **{len(target_slice):,}** candidates"
            )

        def render_knobs(knobs, store):
            """Render one group of intervention sliders into `store`."""
            for knob in knobs:
                k_col = knob["column"]
                if k_col not in target_slice.columns:
                    continue
                val = st.slider(
                    knob["label"],
                    float(knob["min"]), float(knob["max"]),
                    float(knob["default"]), float(knob["step"]),
                    key=f"sim_knob_{k_col}",
                )
                store[k_col] = -val if knob.get("invert", False) else val

        interventions_dict = {}

        # Grouped intervention sliders — grouping comes from the knob
        # definitions in simulator.py, not a hardcoded column list.
        with st.container(border=True):
            st.markdown("#### 2. Academic interventions")
            render_knobs(
                [k for k in INTERVENTION_KNOBS if k.get("group") == "academic"],
                interventions_dict,
            )

        with st.container(border=True):
            st.markdown("#### 3. Experiential interventions")
            render_knobs(
                [k for k in INTERVENTION_KNOBS if k.get("group") == "experiential"],
                interventions_dict,
            )

    with sim_right:
        sim_outcomes = simulator.simulate_policy_intervention(
            cohort_df=target_slice,
            interventions=interventions_dict,
        )

        with st.container(border=True):
            st.markdown("#### 4. Baseline vs. simulated impact")
            res1, res2, res3, res4 = st.columns(4)
            with res1:
                st.metric(
                    "Baseline placement rate",
                    f"{sim_outcomes['baseline_placement_rate']}%",
                )
            with res2:
                st.metric(
                    "Simulated placement rate",
                    f"{sim_outcomes['simulated_placement_rate']}%",
                    f"+{sim_outcomes['placement_uplift_pct']}% uplift",
                )
            with res3:
                st.metric(
                    "Newly placed students",
                    f"+{sim_outcomes['newly_placed_count']}",
                    "New conversions",
                )
            with res4:
                st.metric(
                    "Transitioned out of high-risk",
                    f"+{sim_outcomes['net_transitioned_out_of_high_risk']}",
                    "Risk reductions",
                )

        # Risk Migration Chart
        with st.container(border=True):
            st.markdown("#### :material/compare_arrows: Risk tier migration breakdown")
            risk_data = sim_outcomes["risk_migration"]
            if risk_data:
                categories = list(risk_data["baseline"].keys())
                base_values = [risk_data["baseline"][c] for c in categories]
                sim_values = [risk_data["simulated"][c] for c in categories]

                fig_mig = go.Figure()
                fig_mig.add_trace(
                    go.Bar(
                        x=categories,
                        y=base_values,
                        name="Baseline (Pre-Intervention)",
                        marker_color="#64748B",
                        text=base_values,
                        textposition="auto",
                    )
                )
                fig_mig.add_trace(
                    go.Bar(
                        x=categories,
                        y=sim_values,
                        name="Simulated (Post-Intervention)",
                        marker_color="#34D399",
                        text=sim_values,
                        textposition="auto",
                    )
                )
                layout_mig = get_plotly_layout(height=280)
                layout_mig["barmode"] = "group"
                fig_mig.update_layout(layout_mig)
                st.plotly_chart(fig_mig, use_container_width=True)

    # Student transitions table
    with st.container(border=True):
        st.markdown("#### :material/table_rows: Candidate transition & uplift log")
        transitions_table = sim_outcomes["student_transitions"]
        if not transitions_table.empty:
            newly_placed_df = transitions_table[
                transitions_table["newly_shortlistable"]
            ]
            st.success(
                f"**{len(newly_placed_df)} at-risk students** "
                "will successfully cross the shortlisting cutoff under this simulated policy!",
                icon=":material/verified:",
            )

            trans_configs = {
                "baseline_prob": st.column_config.ProgressColumn(
                    "Pre-intervention prob",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "simulated_prob": st.column_config.ProgressColumn(
                    "Post-intervention prob",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "prob_gain": st.column_config.NumberColumn(
                    "Probability gain", format="+%.1f%%"
                ),
                "newly_shortlistable": st.column_config.CheckboxColumn(
                    "Converted?"
                ),
            }

            st.dataframe(
                transitions_table.sort_values(
                    by="prob_gain", ascending=False
                ).head(50),
                column_config=trans_configs,
                use_container_width=True,
                hide_index=True,
            )



# =============================================================================
# MULTI-MODEL BENCHMARK COMPARISON EXPANDER
# =============================================================================

@st.cache_data(show_spinner="Computing Multi-Model Benchmark Metrics & Validation...")
def compute_benchmark_suite(dataset_len: int) -> dict:
    """
    Evaluates Logistic Regression, Random Forest, and XGBoost on a held-out
    test split with 5-fold cross validation. Cached to avoid UI latency.
    """
    import time

    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

    bench_df = raw_df.copy()
    has_labels = False
    if "placement_target" in bench_df.columns:
        y_bench = bench_df["placement_target"]
        has_labels = len(y_bench.dropna().unique()) > 1
    elif TARGET_COLUMN in bench_df.columns:
        col = bench_df[TARGET_COLUMN]
        # Check for a non-numeric dtype rather than `== object`: pandas may
        # back string columns with pyarrow, which is not object dtype.
        y_bench = col if pd.api.types.is_numeric_dtype(col) else col.map(TARGET_MAP)
        has_labels = not y_bench.isnull().any() and len(y_bench.dropna().unique()) > 1
        if has_labels:
            y_bench = y_bench.astype(int)

    if not has_labels:
        # Fall back to default reference dataset for ground-truth benchmark
        bench_df = default_raw_df.copy()
        col = bench_df[TARGET_COLUMN]
        y_bench = col.map(TARGET_MAP).astype(int)

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        bench_df, y_bench, test_size=0.20, random_state=42, stratify=y_bench
    )

    X_train_proc = predictor._preprocessor.transform(predictor._prepare_features(X_train_b))
    X_test_proc = predictor._preprocessor.transform(predictor._prepare_features(X_test_b))

    # Feature names come from the shared preprocessor, so they are identical
    # for every model - resolve them once, ahead of the per-model loop.
    feat_names_out = []
    for name, trans, cols in predictor._preprocessor.transformers_:
        if name == "numerical":
            feat_names_out.extend(cols)
        elif name == "categorical":
            if hasattr(trans, "get_feature_names_out"):
                feat_names_out.extend(trans.get_feature_names_out(cols).tolist())
            elif hasattr(trans, "categories_"):
                for i, col in enumerate(cols):
                    for cat in trans.categories_[i]:
                        feat_names_out.append(f"{col}_{cat}")

    def extract_top_features(model_obj) -> list:
        """Top-10 global importances (%) for one model, or [] if unsupported."""
        if hasattr(model_obj, "feature_importances_"):
            raw_imp = model_obj.feature_importances_
        elif hasattr(model_obj, "coef_"):
            raw_imp = np.abs(model_obj.coef_[0])
        else:
            return []
        total = np.sum(raw_imp)
        normalized_imp = (raw_imp / total) * 100.0 if total > 0 else raw_imp
        f_names = (
            feat_names_out
            if len(feat_names_out) == len(normalized_imp)
            else [f"Feature_{i}" for i in range(len(normalized_imp))]
        )
        return sorted(
            zip(f_names, np.round(normalized_imp, 2)),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

    comparison_matrix = []
    detailed_metrics = {}

    for m_name, m_model in predictor._models.items():
        # Ensure scikit-learn compatibility
        if (
            "LogisticRegression" in m_model.__class__.__name__
            and not hasattr(m_model, "multi_class")
        ):
            m_model.multi_class = "auto"

        y_pred = m_model.predict(X_test_proc)
        y_prob = m_model.predict_proba(X_test_proc)[:, 1]

        test_auc = float(roc_auc_score(y_test_b, y_prob))
        precision = float(precision_score(y_test_b, y_pred, zero_division=0))
        recall_val = float(recall_score(y_test_b, y_pred, zero_division=0))
        f1_val = float(f1_score(y_test_b, y_pred, zero_division=0))
        float(accuracy_score(y_test_b, y_pred))

        # Latency benchmark
        t0 = time.perf_counter()
        for _ in range(50):
            _ = m_model.predict_proba(X_test_proc)
        latency_ms = round(
            ((time.perf_counter() - t0) / (50 * max(1, len(X_test_proc)))) * 1000, 3
        )

        # 5-fold CV
        try:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                m_model, X_train_proc, y_train_b, cv=cv, scoring="roc_auc"
            )
            mean_cv = float(np.mean(cv_scores))
        except Exception:
            mean_cv = test_auc

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test_b, y_prob)
        step_r = max(1, len(fpr) // 30)
        roc_data = {
            "fpr": [round(float(v), 4) for v in fpr[::step_r].tolist()] + [1.0],
            "tpr": [round(float(v), 4) for v in tpr[::step_r].tolist()] + [1.0],
        }

        cm = confusion_matrix(y_test_b, y_pred).tolist()

        comparison_matrix.append({
            "Model": m_name,
            "Mean CV ROC-AUC": round(mean_cv, 4),
            "Test ROC-AUC": round(test_auc, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall_val, 4),
            "F1-Score": round(f1_val, 4),
            "Inference Latency (ms)": latency_ms,
        })
        detailed_metrics[m_name] = {
            "test_roc_auc": round(test_auc, 4),
            "roc_curve": roc_data,
            "confusion_matrix": cm,
            "top_features": extract_top_features(m_model),
        }

    best_model_name = max(comparison_matrix, key=lambda x: x["Test ROC-AUC"])["Model"]

    return {
        "comparison_matrix": comparison_matrix,
        "detailed_metrics": detailed_metrics,
        "best_model_name": best_model_name,
    }


bench_expander = st.expander(
    ":material/query_stats: Multi-model benchmark comparison matrix & performance validation",
    expanded=False,
)
with bench_expander:
    st.markdown("### Formal multi-model benchmark comparison matrix")
    st.caption(
        "Compare Logistic Regression, Random Forest, and XGBoost "
        "across accuracy, ROC-AUC, precision, recall, F1, and latency."
    )

    # Gated behind an explicit toggle rather than the expander's own open
    # state: toggling an expander is client-side only and never reruns the
    # script, so an `expander.open` check stays False forever and the panel
    # renders empty. A checkbox does trigger a rerun.
    run_benchmark = st.checkbox(
        "Run benchmark evaluation",
        key="run_benchmark",
        help="Evaluates all three models with a held-out split and 5-fold CV.",
    )

    if not run_benchmark:
        st.info(
            "Tick **Run benchmark evaluation** to compute the comparison "
            "matrix. Results are cached after the first run.",
            icon=":material/info:",
        )

    if run_benchmark:
      try:
        if TARGET_COLUMN not in raw_df.columns or len(raw_df[TARGET_COLUMN].dropna().unique()) <= 1:
            st.info(
                "Uploaded cohort has no ground-truth `PlacementStatus` labels. "
                "Benchmark metrics are evaluated on the reference dataset.",
                icon=":material/info:",
            )
        bench_data = compute_benchmark_suite(len(raw_df))
        comparison_matrix = bench_data["comparison_matrix"]
        detailed_metrics = bench_data["detailed_metrics"]
        best_model_name = bench_data["best_model_name"]

        # The single-model panels below follow the sidebar selection, not the
        # best scorer: picking Random Forest previously still rendered the
        # Logistic Regression breakdown whenever LR won on test ROC-AUC.
        display_model_name = st.session_state.get("active_model", best_model_name)
        if display_model_name not in detailed_metrics:
            display_model_name = best_model_name
        display_metrics = detailed_metrics[display_model_name]
        top_features = display_metrics.get("top_features", [])

        model_colors = {
            "Logistic Regression": "#94A3B8",
            "Random Forest": "#34D399",
            "XGBoost": "#60A5FA",
        }

        # Display comparison matrix
        matrix_configs = {
            "Mean CV ROC-AUC": st.column_config.ProgressColumn(
                "Mean CV ROC-AUC", format="%.4f", min_value=0, max_value=1
            ),
            "Test ROC-AUC": st.column_config.ProgressColumn(
                "Test ROC-AUC", format="%.4f", min_value=0, max_value=1
            ),
            "Precision": st.column_config.NumberColumn("Precision", format="%.3f"),
            "Recall": st.column_config.NumberColumn("Recall", format="%.3f"),
            "F1-Score": st.column_config.ProgressColumn(
                "F1-Score", format="%.3f", min_value=0, max_value=1
            ),
            "Inference Latency (ms)": st.column_config.NumberColumn(
                "Latency (ms)", format="%.3f ms"
            ),
        }
        st.dataframe(
            pd.DataFrame(comparison_matrix),
            column_config=matrix_configs,
            use_container_width=True,
            hide_index=True,
        )

        # ROC Curves & Confusion Matrix
        col_roc, col_cm = st.columns(2)

        with col_roc, st.container(border=True):
            st.markdown("#### :material/show_chart: Multi-model ROC curves")
            fig_roc = go.Figure()
            fig_roc.add_shape(
                type="line",
                line={"dash": "dash", "color": "#64748B"},
                x0=0, x1=1, y0=0, y1=1,
            )

            for m_name, m_met in detailed_metrics.items():
                r_curve = m_met["roc_curve"]
                fig_roc.add_trace(
                    go.Scatter(
                        x=r_curve["fpr"],
                        y=r_curve["tpr"],
                        name=f"{m_name} (AUC={m_met['test_roc_auc']:.3f})",
                        mode="lines",
                        line={
                            "color": model_colors.get(m_name, "#60A5FA"),
                            "width": 2.5,
                        },
                    )
                )
            layout_roc = get_plotly_layout(height=300)
            layout_roc["xaxis"]["title"] = "False Positive Rate"
            layout_roc["yaxis"]["title"] = "True Positive Rate"
            fig_roc.update_layout(layout_roc)
            st.plotly_chart(fig_roc, use_container_width=True)

        with col_cm, st.container(border=True):
            st.markdown(
                f"#### :material/grid_on: Confusion matrix ({display_model_name})"
            )
            active_cm = display_metrics["confusion_matrix"]
            fig_cm = px.imshow(
                active_cm,
                text_auto=True,
                labels={"x": "Predicted class", "y": "Actual class", "color": "Count"},
                x=["Not placed (0)", "Placed (1)"],
                y=["Not placed (0)", "Placed (1)"],
                color_continuous_scale="Blues",
            )
            fig_cm.update_layout(get_plotly_layout(height=300))
            st.plotly_chart(fig_cm, use_container_width=True)

        # Feature importance
        if top_features:
            with st.container(border=True):
                st.markdown(f"#### :material/leaderboard: Global feature importance attribution ({display_model_name})")
                f_df = pd.DataFrame(top_features, columns=["Feature", "Importance (%)"]).sort_values(by="Importance (%)", ascending=True)

                fig_feat = px.bar(
                    f_df,
                    x="Importance (%)",
                    y="Feature",
                    orientation="h",
                    color="Importance (%)",
                    color_continuous_scale="Blues",
                )
                layout_feat = get_plotly_layout(height=300)
                layout_feat["showlegend"] = False
                fig_feat.update_layout(layout_feat)
                st.plotly_chart(fig_feat, use_container_width=True)

      except Exception as bench_err:
        st.warning(
            f":material/warning: Could not generate benchmark metrics: {bench_err}"
        )
