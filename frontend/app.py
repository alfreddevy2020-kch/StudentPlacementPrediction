"""
PlacementPulse AI — Multi-Tab Executive Dashboard
===================================================
Premium Streamlit dashboard with 3 tabs:
  Tab 1: Departmental Pulse & Readiness Analytics
  Tab 2: Per-Student Diagnostic & Skill-Gap Analysis
  Tab 3: Cohort What-If Policy Simulator

Plus: Multi-Model Benchmark Comparison expander

Ported from prediction.txt design system. Preserves resume upload
functionality from original dashboard.

Usage:
    cd frontend
    streamlit run app.py

Prerequisites:
    - Model artifacts in part2/models/ and part3/models/
    - Dataset at data/raw/student_placement.csv
"""

import io
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add frontend dir to path so local imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from batch_predictor import (
    BatchPredictor,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    HIGH_RISK_THRESHOLD,
    MODERATE_RISK_THRESHOLD,
)
from simulator import CohortWhatIfSimulator, INTERVENTION_KNOBS

# Try importing pdfplumber for resume parsing (optional)
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


# =============================================================================
# 1. PAGE SETUP & THEME STYLING
# =============================================================================
st.set_page_config(
    page_title="PlacementPulse AI | Student Placement Readiness & Policy Simulator",
    page_icon=":material/school:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Theme CSS Polish for subtle accents & badge styling
st.markdown("""
<style>
    /* Metric Card Polish */
    div[data-testid="stMetric"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px 18px;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #60A5FA;
        box-shadow: 0 4px 12px rgba(96, 165, 250, 0.08);
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] div {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
    }

    /* Tab Polish */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #0F172A;
        padding: 4px;
        border-radius: 8px;
        border: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
        color: #94A3B8;
        background-color: transparent;
        border: none;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #F8FAFC;
        background-color: #1E293B;
    }
    .stTabs [aria-selected="true"] {
        color: #60A5FA !important;
        background-color: #1E293B !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }

    /* Header Title Badge */
    .domain-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #1E293B;
        border: 1px solid #3B82F6;
        color: #93C5FD;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def get_plotly_layout(height: int = 320, title: Optional[str] = None) -> Dict[str, Any]:
    """Returns a unified executive dark Plotly layout aligned with the theme."""
    return dict(
        title=dict(
            text=title,
            font=dict(family="Inter, sans-serif", size=15, color="#F8FAFC"),
        )
        if title
        else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#CBD5E1", size=12),
        margin=dict(l=24, r=24, t=36 if title else 16, b=24),
        height=height,
        xaxis=dict(
            gridcolor="#334155",
            zerolinecolor="#334155",
            tickfont=dict(color="#94A3B8"),
            title=dict(font=dict(color="#CBD5E1")),
        ),
        yaxis=dict(
            gridcolor="#334155",
            zerolinecolor="#334155",
            tickfont=dict(color="#94A3B8"),
            title=dict(font=dict(color="#CBD5E1")),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#CBD5E1"),
            bgcolor="rgba(0,0,0,0)",
        ),
    )


# Risk tier color palette (shared across all charts)
RISK_COLORS = {
    "High Risk": "#F87171",
    "Moderate Risk": "#FBBF24",
    "Interview Ready": "#34D399",
}


# =============================================================================
# 2. RESUME PARSING (preserved from original dashboard)
# =============================================================================

def extract_resume_data(pdf_bytes: bytes) -> dict:
    """
    Extract structured student data from a resume PDF using regex
    heuristics. Returns a dict with keys matching sidebar widget keys;
    values are None when the pattern is not found.
    """
    if not HAS_PDFPLUMBER:
        return {}

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    result = {
        "name": None,
        "cgpa": None,
        "ssc_percentage": None,
        "hsc_percentage": None,
        "degree_percentage": None,
        "certifications": None,
        "internship_count": None,
        "live_projects": None,
        "work_experience_months": None,
    }

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # -- Name: first plausible line in the header --
    skip_keywords = [
        "@", "phone", "email", "address", "linkedin", "section",
        "objective", "summary", "education", "experience", "skills",
        "certification", "project", "contact", "profile",
    ]
    for line in lines[:6]:
        if any(kw in line.lower() for kw in skip_keywords):
            continue
        cleaned = re.sub(
            r"^(name|candidate|mr\.|ms\.|miss|shri)\s*[:\-]?\s*",
            "", line, flags=re.IGNORECASE,
        )
        if cleaned and len(cleaned.split()) >= 2:
            result["name"] = cleaned.strip().title()
            break

    # -- CGPA --
    m = re.search(r"(?:cgpa|gpa)[:\s]*(\d+\.?\d*)", text, re.IGNORECASE)
    if m:
        result["cgpa"] = min(float(m.group(1)), 10.0)

    # -- SSC (10th) percentage --
    m = re.search(
        r"(?:10th|tenth|ssc|class\s*x)[:\s]*(\d+\.?\d*)\s*%?",
        text, re.IGNORECASE,
    )
    if m:
        result["ssc_percentage"] = min(float(m.group(1)), 100.0)

    # -- HSC (12th) percentage --
    m = re.search(
        r"(?:12th|twelfth|hsc|class\s*xii|xii)[:\s]*(\d+\.?\d*)\s*%?",
        text, re.IGNORECASE,
    )
    if m:
        result["hsc_percentage"] = min(float(m.group(1)), 100.0)

    # -- Degree percentage --
    m = re.search(
        r"(?:degree|graduation|b\.?tech|b\.?e\.?)\s*(?:percentage|marks)?"
        r"[:\s]*(\d+\.?\d*)\s*%?",
        text, re.IGNORECASE,
    )
    if m:
        result["degree_percentage"] = min(float(m.group(1)), 100.0)

    # -- Certifications count --
    cert_match = re.search(
        r"(?:certification|certificate|courses?)[\\s:]*\n"
        r"((?:[-•*]\s*.+\n?)+)",
        text, re.IGNORECASE,
    )
    if cert_match:
        result["certifications"] = len(
            re.findall(r"[-•*]\s*.+", cert_match.group(1))
        )

    # -- Internship count --
    exp_match = re.search(
        r"(?:experience|internship|work\s*history)[\s:]*\n"
        r"((?:[-•*]\s*.+\n?)+)",
        text, re.IGNORECASE,
    )
    if exp_match:
        result["internship_count"] = len(
            re.findall(r"[-•*]\s*.+", exp_match.group(1))
        )

    # -- Live projects --
    proj_match = re.search(
        r"(?:projects?|portfolio)[\s:]*\n((?:[-•*]\s*.+\n?)+)",
        text, re.IGNORECASE,
    )
    if proj_match:
        result["live_projects"] = len(
            re.findall(r"[-•*]\s*.+", proj_match.group(1))
        )

    return result


# =============================================================================
# 3. SYSTEM BOOTSTRAP & CACHING
# =============================================================================

@st.cache_resource(show_spinner="Loading ML Models & Dataset...")
def load_system():
    """Load dataset, preprocessor, and all model artifacts once."""
    predictor = BatchPredictor()
    predictor.load()
    raw_df = predictor.load_dataset()
    simulator = CohortWhatIfSimulator(predictor)
    return predictor, raw_df, simulator


try:
    predictor, raw_df, simulator = load_system()
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
# 4. SIDEBAR CONTROLS
# =============================================================================
with st.sidebar:
    st.markdown("### :material/school: PlacementPulse AI")
    st.caption("Student Placement Readiness & Policy Simulator")
    st.markdown("---")

    # Dataset Info
    st.subheader(":material/dataset: Dataset")
    st.caption(f"**{len(raw_df):,}** students • **{len(raw_df.columns)}** features")

    # Model Selector
    st.markdown("---")
    st.subheader(":material/psychology: Inference Engine")
    model_names = predictor.available_models
    default_idx = (
        model_names.index(predictor.active_model_name)
        if predictor.active_model_name in model_names
        else 0
    )
    selected_model_name = st.selectbox(
        "Active Model",
        options=model_names,
        index=default_idx,
        help="Select classification model to drive all dashboard predictions.",
    )
    predictor.active_model_name = selected_model_name

    # Cohort Filters
    st.markdown("---")
    st.subheader(":material/filter_alt: Cohort Filters")

    # Gender filter
    gender_options = ["ALL"] + sorted(raw_df["gender"].dropna().unique().tolist())
    selected_gender = st.selectbox("Filter by Gender", gender_options, index=0)

    # Extracurricular filter
    extra_options = ["ALL"] + sorted(
        raw_df["extracurricular_activities"].dropna().unique().tolist()
    )
    selected_extra = st.selectbox(
        "Filter by Extracurriculars", extra_options, index=0
    )

    # Apply filters
    filtered_df = raw_df.copy()
    if selected_gender != "ALL":
        filtered_df = filtered_df[filtered_df["gender"] == selected_gender]
    if selected_extra != "ALL":
        filtered_df = filtered_df[
            filtered_df["extracurricular_activities"] == selected_extra
        ]

    st.caption(
        f":material/groups: Active Cohort: **{len(filtered_df):,}** / {len(raw_df):,} students"
    )

    # Resume Upload
    st.markdown("---")
    st.subheader(":material/upload_file: Resume Upload")
    if HAS_PDFPLUMBER:
        resume_file = st.file_uploader(
            "Upload Resume (PDF)",
            type=["pdf"],
            help="Upload a student resume to auto-fill Tab 2 fields.",
        )
        if resume_file is not None:
            if st.session_state.get("_last_resume_file") != resume_file.name:
                parsed = extract_resume_data(resume_file.getvalue())
                st.session_state["_parsed"] = parsed
                st.session_state["_last_resume_file"] = resume_file.name
                for key, val in parsed.items():
                    if val is not None and key != "name":
                        st.session_state[f"diag_{key}"] = val
                if parsed.get("name"):
                    st.success(f"Parsed resume for **{parsed['name']}**")
    else:
        st.info("Install `pdfplumber` for resume auto-fill support.")
        resume_file = None


# =============================================================================
# 5. BATCH PREDICTIONS ON FILTERED COHORT
# =============================================================================
if not filtered_df.empty:
    try:
        cohort_probs = predictor.predict_probabilities(filtered_df)
    except Exception as pred_err:
        st.warning(f":material/warning: Prediction error: {pred_err}")
        cohort_probs = np.full(len(filtered_df), 0.5)

    filtered_df = filtered_df.copy()
    filtered_df["placement_prob"] = np.round(cohort_probs * 100, 1)
    filtered_df["predicted_status"] = np.where(
        cohort_probs >= 0.50, "Placed", "Not Placed"
    )
    filtered_df["risk_tier"] = [
        predictor.classify_risk(p) for p in cohort_probs
    ]


# =============================================================================
# 6. DASHBOARD HEADER
# =============================================================================
st.markdown(
    '<div class="domain-badge">:material/auto_awesome: Student Placement Prediction System</div>',
    unsafe_allow_html=True,
)
st.title("Student Placement Prediction & Readiness System")
st.caption(
    f"Production ML Classification Engine, Diagnostic Radar, & Macro Cohort Simulator • "
    f"Driven by **{selected_model_name}**"
)

# 3 Primary Tabs
tab1, tab2, tab3 = st.tabs([
    "📊 Departmental Pulse & Readiness",
    "🎯 Per-Student Diagnostic & Skill-Gaps",
    "🧪 Cohort What-If Policy Simulator",
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
                    "Total Cohort Size",
                    f"{total_count:,}",
                    help="Total students in current filter",
                )
            with kpi2:
                st.metric(
                    "Projected Placement Rate",
                    f"{placement_rate}%",
                    f"{placed_count} Placed",
                )
            with kpi3:
                st.metric(
                    "Critical Risk (<40%)",
                    f"{critical_risk_count}",
                    f"{round((critical_risk_count / max(1, total_count)) * 100, 1)}% of cohort",
                    delta_color="inverse",
                )
            with kpi4:
                st.metric(
                    "Mean Placement Likelihood",
                    f"{avg_score}%",
                    help="Average predicted probability across cohort",
                )

        # Department Readiness & Risk Donut
        col_dept, col_donut = st.columns([3, 2])

        with col_dept:
            with st.container(border=True):
                st.markdown("#### :material/bar_chart: Readiness by Gender")
                dept_stats = (
                    filtered_df.groupby("gender")
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

                fig_bar = go.Figure()
                fig_bar.add_trace(
                    go.Bar(
                        x=dept_stats["gender"],
                        y=dept_stats["placement_rate"],
                        name="Placement Rate (%)",
                        marker_color="#3B82F6",
                        text=dept_stats["placement_rate"].astype(str) + "%",
                        textposition="auto",
                    )
                )
                fig_bar.add_trace(
                    go.Bar(
                        x=dept_stats["gender"],
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

        with col_donut:
            with st.container(border=True):
                st.markdown("#### :material/pie_chart: Risk Tier Distribution")
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
                    textfont=dict(color="#F8FAFC", size=11),
                )
                layout_pie = get_plotly_layout(height=280)
                layout_pie["showlegend"] = False
                fig_pie.update_layout(layout_pie)
                st.plotly_chart(fig_pie, use_container_width=True)

        # Placement Probability Distribution
        with st.container(border=True):
            st.markdown(
                "#### :material/stacked_line_chart: Placement Probability Distribution & Thresholds"
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

        # Student Roster Table
        with st.container(border=True):
            st.markdown("#### :material/table_chart: Cohort Candidate Readiness Roster")

            display_cols = [
                "student_id",
                "gender",
                "cgpa",
                "attendance_percentage",
                "backlogs",
                "technical_skill_score",
                "certifications",
                "placement_prob",
                "risk_tier",
                "predicted_status",
            ]
            display_cols = [c for c in display_cols if c in filtered_df.columns]

            column_configs = {
                "placement_prob": st.column_config.ProgressColumn(
                    "Placement Likelihood",
                    help="Model-projected placement probability",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "risk_tier": st.column_config.TextColumn("Risk Classification"),
                "predicted_status": st.column_config.TextColumn("Projected Status"),
            }

            st.dataframe(
                filtered_df[display_cols]
                .sort_values(by="placement_prob", ascending=False)
                .head(100),
                column_config=column_configs,
                use_container_width=True,
                hide_index=True,
            )


# =============================================================================
# TAB 2: PER-STUDENT DIAGNOSTIC & SKILL-GAP ANALYSIS
# =============================================================================
with tab2:
    st.markdown("### :material/person_search: Candidate Diagnostic & Skill-Gap Diagnosis")
    st.caption(
        "Multi-dimensional readiness radar and prescriptive remediation engine"
    )

    diag_mode = st.radio(
        "Candidate Input Mode:",
        [
            ":material/person: Select Existing Candidate from Cohort",
            ":material/tune: Interactive Custom Candidate Profiler",
        ],
        horizontal=True,
    )

    if "Select Existing Candidate" in diag_mode:
        stu_options = filtered_df["student_id"].tolist()
        if not stu_options:
            st.warning("No students available. Adjust sidebar filters.")
            st.stop()
        selected_id = st.selectbox(
            "Search & Select Student ID:", options=stu_options, index=0
        )
        student_data = (
            filtered_df[filtered_df["student_id"] == selected_id]
            .iloc[0]
            .to_dict()
        )
    else:
        st.info(
            ":material/tune: Tune candidate metrics to test hypothetical profiles:"
        )

        with st.container(border=True):
            student_data = {"student_id": 99999}

            # Categorical attributes
            st.markdown("**Categorical Attributes:**")
            cat_c1, cat_c2 = st.columns(2)
            with cat_c1:
                student_data["gender"] = st.selectbox(
                    "Gender",
                    options=["Male", "Female"],
                    key="diag_gender",
                )
            with cat_c2:
                student_data["extracurricular_activities"] = st.selectbox(
                    "Extracurricular Activities",
                    options=["Yes", "No"],
                    key="diag_extra",
                )

            # Numerical attributes
            st.markdown("**Academic Performance:**")
            ac1, ac2, ac3, ac4 = st.columns(4)
            with ac1:
                student_data["cgpa"] = st.slider(
                    "Current CGPA",
                    0.0, 10.0,
                    value=st.session_state.get("diag_cgpa", 7.0),
                    step=0.1,
                    key="diag_cgpa_slider",
                )
            with ac2:
                student_data["ssc_percentage"] = st.slider(
                    "SSC (10th) %",
                    0.0, 100.0,
                    value=float(st.session_state.get("diag_ssc_percentage", 75.0)),
                    step=1.0,
                    key="diag_ssc_slider",
                )
            with ac3:
                student_data["hsc_percentage"] = st.slider(
                    "HSC (12th) %",
                    0.0, 100.0,
                    value=float(st.session_state.get("diag_hsc_percentage", 75.0)),
                    step=1.0,
                    key="diag_hsc_slider",
                )
            with ac4:
                student_data["degree_percentage"] = st.slider(
                    "Degree %",
                    0.0, 100.0,
                    value=float(st.session_state.get("diag_degree_percentage", 72.0)),
                    step=1.0,
                    key="diag_degree_slider",
                )

            st.markdown("**Skills & Tests:**")
            sk1, sk2, sk3, sk4 = st.columns(4)
            with sk1:
                student_data["entrance_exam_score"] = st.slider(
                    "Entrance Exam Score",
                    0.0, 100.0, 80.0, 1.0,
                    key="diag_entrance",
                )
            with sk2:
                student_data["technical_skill_score"] = st.slider(
                    "Technical Skill Score",
                    0.0, 100.0, 75.0, 1.0,
                    key="diag_tech",
                )
            with sk3:
                student_data["soft_skill_score"] = st.slider(
                    "Soft Skill Score",
                    0.0, 100.0, 75.0, 1.0,
                    key="diag_soft",
                )
            with sk4:
                student_data["attendance_percentage"] = st.slider(
                    "Attendance %",
                    0.0, 100.0, 85.0, 1.0,
                    key="diag_attendance",
                )

            st.markdown("**Experience & Credentials:**")
            ex1, ex2, ex3, ex4, ex5 = st.columns(5)
            with ex1:
                student_data["backlogs"] = st.number_input(
                    "Active Backlogs", 0, 20, 0, key="diag_backlogs"
                )
            with ex2:
                student_data["certifications"] = st.number_input(
                    "Certifications",
                    0, 20,
                    value=int(st.session_state.get("diag_certifications", 1)),
                    key="diag_certs_input",
                )
            with ex3:
                student_data["live_projects"] = st.number_input(
                    "Live Projects",
                    0, 20,
                    value=int(st.session_state.get("diag_live_projects", 1)),
                    key="diag_projects_input",
                )
            with ex4:
                student_data["internship_count"] = st.number_input(
                    "Internships",
                    0, 10,
                    value=int(st.session_state.get("diag_internship_count", 0)),
                    key="diag_intern_input",
                )
            with ex5:
                student_data["work_experience_months"] = st.number_input(
                    "Work Exp (months)",
                    0, 60,
                    value=int(st.session_state.get("diag_work_experience_months", 0)),
                    key="diag_workexp_input",
                )

    # Evaluate single candidate
    candidate_prob = predictor.predict_single(student_data)
    candidate_prob_pct = round(candidate_prob * 100, 1)

    diag_left, diag_right = st.columns([1, 2])

    with diag_left:
        with st.container(border=True):
            st.markdown("#### :material/speed: Placement Probability")
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
                    ":material/verified: **Interview Ready**: High likelihood of shortlisting."
                )
            elif candidate_prob_pct >= 50:
                st.warning(
                    ":material/warning: **Moderate Risk**: Viable profile requiring focused prep."
                )
            else:
                st.error(
                    ":material/error: **Critical Risk**: Significant preparedness gaps identified."
                )

    with diag_right:
        with st.container(border=True):
            st.markdown("#### :material/radar: Multi-Dimensional Competency Radar")

            # Define radar axes using available features
            radar_specs = [
                {"column": "cgpa", "label": "CGPA (×10)", "scale_factor": 10.0},
                {"column": "attendance_percentage", "label": "Attendance %", "scale_factor": 1.0},
                {"column": "technical_skill_score", "label": "Technical Skill", "scale_factor": 1.0},
                {"column": "soft_skill_score", "label": "Soft Skill", "scale_factor": 1.0},
                {"column": "entrance_exam_score", "label": "Entrance Exam", "scale_factor": 1.0},
                {"column": "certifications", "label": "Certs (×20)", "scale_factor": 20.0},
            ]

            # Compute placed peers benchmark
            target_col = "placement_status"
            placed_peers = (
                raw_df[raw_df[target_col] == 1]
                if target_col in raw_df.columns and (raw_df[target_col] == 1).any()
                else raw_df
            )

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
            radar_layout["polar"] = dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor="#334155",
                    tickfont=dict(color="#94A3B8"),
                ),
                angularaxis=dict(
                    gridcolor="#334155",
                    tickfont=dict(color="#CBD5E1", size=11),
                ),
                bgcolor="rgba(0,0,0,0)",
            )
            radar_layout["showlegend"] = True
            fig_radar.update_layout(radar_layout)
            st.plotly_chart(fig_radar, use_container_width=True)

    # Prescriptive Remediation Engine
    st.markdown("---")
    st.markdown("### :material/lightbulb: Prescriptive Remediation & Targeted Interventions")
    st.caption("Quantified action recommendations with simulated probability uplifts")

    remediation_rules = [
        {
            "condition": lambda s: s.get("backlogs", 0) > 0,
            "priority": "CRITICAL",
            "title": "Clear Active Backlogs",
            "action": "Enroll in fast-track backlog clearance before corporate shortlisting begins.",
            "sim_column": "backlogs",
            "sim_op": "subtract",
            "sim_value": 1.0,
        },
        {
            "condition": lambda s: s.get("attendance_percentage", 100) < 75.0,
            "priority": "HIGH",
            "title": "Raise Attendance to 75%+ Target",
            "action": "Complete remedial attendance coursework to pass corporate screening filters.",
            "sim_column": "attendance_percentage",
            "sim_op": "set",
            "sim_value": 78.0,
        },
        {
            "condition": lambda s: s.get("technical_skill_score", 100) < 70,
            "priority": "HIGH",
            "title": "Boost Technical Skills",
            "action": "Complete coding bootcamp and hackathon practice to improve technical readiness.",
            "sim_column": "technical_skill_score",
            "sim_op": "add",
            "sim_value": 15.0,
        },
        {
            "condition": lambda s: s.get("certifications", 10) < 2,
            "priority": "MEDIUM",
            "title": "Pursue Industry Certifications",
            "action": "Earn at least 2 industry certifications (AWS, Azure, Google Cloud, etc.).",
            "sim_column": "certifications",
            "sim_op": "add",
            "sim_value": 2.0,
        },
        {
            "condition": lambda s: s.get("live_projects", 10) < 1,
            "priority": "MEDIUM",
            "title": "Build Live Projects",
            "action": "Complete at least one end-to-end deployed project to demonstrate practical skills.",
            "sim_column": "live_projects",
            "sim_op": "add",
            "sim_value": 1.0,
        },
        {
            "condition": lambda s: s.get("cgpa", 10) < 7.0,
            "priority": "MEDIUM",
            "title": "Improve CGPA Above 7.0",
            "action": "Focus on upcoming semester exams — many companies set 7.0 CGPA as the minimum cutoff.",
            "sim_column": "cgpa",
            "sim_op": "set",
            "sim_value": 7.2,
        },
        {
            "condition": lambda s: s.get("internship_count", 10) < 1,
            "priority": "MEDIUM",
            "title": "Secure an Internship",
            "action": "Apply for at least one internship to gain industry exposure before placements.",
            "sim_column": "internship_count",
            "sim_op": "add",
            "sim_value": 1.0,
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
                        sim_s[col] = max(0, sim_s[col] - rule["sim_value"])
                    elif rule["sim_op"] == "add":
                        sim_s[col] = sim_s[col] + rule["sim_value"]
                    elif rule["sim_op"] == "set":
                        sim_s[col] = rule["sim_value"]

                new_p = predictor.predict_single(sim_s)
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
                    "uplift": f"+{gain}% Placement Uplift",
                    "color": color,
                })
        except Exception:
            pass

    if not remediations:
        with st.container(border=True):
            st.success(
                ":material/celebration: **Optimal Profile**: This candidate already satisfies "
                "all institutional competency benchmarks!"
            )
    else:
        for rem in remediations:
            with st.container(border=True):
                r_left, r_mid, r_right = st.columns([1, 4, 2])
                with r_left:
                    st.markdown(
                        f"<span style='color:{rem['color']};font-weight:700;"
                        f"font-size:0.85rem;'>{rem['priority']} PRIORITY</span>",
                        unsafe_allow_html=True,
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
    st.markdown("### :material/tune: Cohort Policy Intervention & What-If Simulator")
    st.caption(
        "Simulate institutional training interventions, risk migrations, "
        "and candidate conversions before allocating budget."
    )

    sim_left, sim_right = st.columns([1, 2])

    with sim_left:
        with st.container(border=True):
            st.markdown("#### 1. Define Target Segment")

            # Gender segment
            sim_gender_opts = ["ALL COHORTS"] + sorted(
                raw_df["gender"].dropna().unique().tolist()
            )
            sim_target_gender = st.selectbox(
                "Target Gender Segment",
                sim_gender_opts,
                index=0,
                key="sim_gender",
            )

            target_slice = raw_df.copy()
            if sim_target_gender != "ALL COHORTS":
                target_slice = target_slice[
                    target_slice["gender"] == sim_target_gender
                ]

            st.caption(
                f":material/groups: Target Cohort: **{len(target_slice):,}** candidates"
            )

            st.markdown("#### 2. Policy Interventions")
            interventions_dict = {}

            for knob in INTERVENTION_KNOBS:
                k_col = knob["column"]
                if k_col in target_slice.columns:
                    label = knob["label"]
                    min_v = float(knob["min"])
                    max_v = float(knob["max"])
                    step_v = float(knob["step"])
                    def_v = float(knob["default"])
                    invert = knob.get("invert", False)

                    val = st.slider(
                        label, min_v, max_v, def_v, step_v,
                        key=f"sim_knob_{k_col}",
                    )
                    interventions_dict[k_col] = -val if invert else val

    with sim_right:
        sim_outcomes = simulator.simulate_policy_intervention(
            cohort_df=target_slice,
            interventions=interventions_dict,
        )

        with st.container(border=True):
            st.markdown("#### 3. Baseline vs. Simulated Impact")
            res1, res2, res3, res4 = st.columns(4)
            with res1:
                st.metric(
                    "Baseline Placement Rate",
                    f"{sim_outcomes['baseline_placement_rate']}%",
                )
            with res2:
                st.metric(
                    "Simulated Placement Rate",
                    f"{sim_outcomes['simulated_placement_rate']}%",
                    f"+{sim_outcomes['placement_uplift_pct']}% Uplift",
                )
            with res3:
                st.metric(
                    "Newly Placed Students",
                    f"+{sim_outcomes['newly_placed_count']}",
                    "New Conversions",
                )
            with res4:
                st.metric(
                    "Transitioned Out of High-Risk",
                    f"+{sim_outcomes['net_transitioned_out_of_high_risk']}",
                    "Risk Reductions",
                )

        # Risk Migration Chart
        with st.container(border=True):
            st.markdown("#### :material/compare_arrows: Risk Tier Migration Breakdown")
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

    # Student Transitions Table
    with st.container(border=True):
        st.markdown("#### :material/table_rows: Candidate Transition & Uplift Log")
        transitions_table = sim_outcomes["student_transitions"]
        if not transitions_table.empty:
            newly_placed_df = transitions_table[
                transitions_table["newly_shortlistable"] == True
            ]
            st.success(
                f":material/verified: **{len(newly_placed_df)} at-risk students** "
                "will successfully cross the shortlisting cutoff under this simulated policy!"
            )

            trans_configs = {
                "baseline_prob": st.column_config.ProgressColumn(
                    "Pre-Intervention Prob",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "simulated_prob": st.column_config.ProgressColumn(
                    "Post-Intervention Prob",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "prob_gain": st.column_config.NumberColumn(
                    "Probability Gain", format="+%.1f%%"
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
with st.expander(
    ":material/query_stats: Multi-Model Benchmark Comparison Matrix & Performance Validation",
    expanded=False,
):
    st.markdown("### Formal Multi-Model Benchmark Comparison Matrix")
    st.caption(
        "Compare Logistic Regression, Random Forest, and XGBoost "
        "across accuracy, ROC-AUC, precision, recall, F1, and latency."
    )

    # Generate benchmark metrics on-the-fly from loaded models and dataset
    try:
        from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
        from sklearn.metrics import (
            roc_auc_score, accuracy_score, precision_score,
            recall_score, f1_score, confusion_matrix, roc_curve,
        )
        import time

        # Prepare data
        bench_df = raw_df.copy()
        bench_df["placement_target"] = bench_df["placement_status"]
        features_for_bench = [c for c in NUMERICAL_FEATURES + CATEGORICAL_FEATURES if c in bench_df.columns]

        X_bench = bench_df[features_for_bench]
        y_bench = bench_df["placement_target"]

        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
            X_bench, y_bench, test_size=0.20, random_state=42, stratify=y_bench
        )

        # Transform using loaded preprocessor
        X_train_proc = predictor._preprocessor.transform(X_train_b)
        X_test_proc = predictor._preprocessor.transform(X_test_b)

        comparison_matrix = []
        detailed_metrics = {}
        model_colors = {
            "Logistic Regression": "#94A3B8",
            "Random Forest": "#34D399",
            "XGBoost": "#60A5FA",
        }

        for m_name, m_model in predictor._models.items():
            y_pred = m_model.predict(X_test_proc)
            y_prob = m_model.predict_proba(X_test_proc)[:, 1]

            test_auc = float(roc_auc_score(y_test_b, y_prob))
            precision = float(precision_score(y_test_b, y_pred, zero_division=0))
            recall_val = float(recall_score(y_test_b, y_pred, zero_division=0))
            f1_val = float(f1_score(y_test_b, y_pred, zero_division=0))
            accuracy = float(accuracy_score(y_test_b, y_pred))

            # Inference latency
            t0 = time.perf_counter()
            for _ in range(50):
                _ = m_model.predict_proba(X_test_proc)
            latency_ms = round(
                ((time.perf_counter() - t0) / (50 * max(1, len(X_test_proc)))) * 1000, 3
            )

            # CV score
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

        with col_roc:
            with st.container(border=True):
                st.markdown("#### :material/show_chart: Multi-Model ROC Curves")
                fig_roc = go.Figure()
                fig_roc.add_shape(
                    type="line",
                    line=dict(dash="dash", color="#64748B"),
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
                            line=dict(
                                color=model_colors.get(m_name, "#60A5FA"),
                                width=2.5,
                            ),
                        )
                    )
                layout_roc = get_plotly_layout(height=300)
                layout_roc["xaxis"]["title"] = "False Positive Rate"
                layout_roc["yaxis"]["title"] = "True Positive Rate"
                fig_roc.update_layout(layout_roc)
                st.plotly_chart(fig_roc, use_container_width=True)

        with col_cm:
            with st.container(border=True):
                best_model_name = max(
                    comparison_matrix, key=lambda x: x["Test ROC-AUC"]
                )["Model"]
                st.markdown(
                    f"#### :material/grid_on: Confusion Matrix ({best_model_name})"
                )
                best_cm = detailed_metrics[best_model_name]["confusion_matrix"]
                fig_cm = px.imshow(
                    best_cm,
                    text_auto=True,
                    labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                    x=["Not Placed (0)", "Placed (1)"],
                    y=["Not Placed (0)", "Placed (1)"],
                    color_continuous_scale="Blues",
                )
                fig_cm.update_layout(get_plotly_layout(height=300))
                st.plotly_chart(fig_cm, use_container_width=True)

        # Feature Importance
        best_model_obj = predictor._models[best_model_name]
        if hasattr(best_model_obj, "feature_importances_"):
            with st.container(border=True):
                st.markdown("#### :material/leaderboard: Global Feature Importance Attribution")

                # Get feature names after preprocessing
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

                raw_imp = best_model_obj.feature_importances_
                total = np.sum(raw_imp)
                normalized_imp = (raw_imp / total) * 100.0 if total > 0 else raw_imp

                if len(feat_names_out) == len(normalized_imp):
                    f_names = feat_names_out
                else:
                    f_names = [f"Feature_{i}" for i in range(len(normalized_imp))]

                f_df = (
                    pd.DataFrame({
                        "Feature": f_names,
                        "Importance (%)": np.round(normalized_imp, 2),
                    })
                    .sort_values(by="Importance (%)", ascending=True)
                    .tail(10)
                )

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
