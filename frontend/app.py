"""
CampusReady — Multi-Tab Executive Dashboard
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

# Add repo root and frontend dir to sys.path so local imports work
REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

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
    page_title="CampusReady | Student Placement Readiness & Policy Simulator",
    page_icon=":material/school:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# NOTE: All visual styling is handled natively via .streamlit/config.toml
# and st.container(border=True). No custom CSS injection needed.


def get_plotly_layout(height: int = 320, title: Optional[str] = None) -> Dict[str, Any]:
    """Returns a unified executive dark Plotly layout aligned with the theme."""
    layout = dict(
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
    if title:
        layout["title"] = dict(
            text=title,
            font=dict(family="Inter, sans-serif", size=15, color="#F8FAFC"),
        )
    return layout


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
    st.markdown("### :material/school: CampusReady")
    st.caption("Student placement readiness & policy simulator")
    st.space("small")

    # Dataset info
    st.subheader(":material/dataset: Dataset")
    st.caption(f"**{len(raw_df):,}** students • **{len(raw_df.columns)}** features")

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

    # Gender filter
    gender_options = ["ALL"] + sorted(raw_df["gender"].dropna().unique().tolist())
    selected_gender = st.pills(
        "Gender", gender_options, default="ALL", key="filter_gender"
    )

    # Branch filter
    branch_options = ["ALL"] + sorted(raw_df["branch"].dropna().unique().tolist()) if "branch" in raw_df.columns else ["ALL"]
    selected_branch = st.pills(
        "Branch", branch_options, default="ALL", key="filter_branch"
    )

    # College Tier filter
    tier_options = ["ALL"] + sorted(raw_df["college_tier"].dropna().unique().tolist()) if "college_tier" in raw_df.columns else ["ALL"]
    selected_tier = st.pills(
        "College Tier", tier_options, default="ALL", key="filter_tier"
    )

    # Volunteer Experience filter
    vol_col = "volunteer_experience" if "volunteer_experience" in raw_df.columns else "gender"
    vol_options = ["ALL"] + sorted(raw_df[vol_col].dropna().unique().tolist())
    selected_vol = st.pills(
        "Volunteer Experience", vol_options, default="ALL", key="filter_vol"
    )

    # Apply filters
    filtered_df = raw_df.copy()
    if selected_gender and selected_gender != "ALL":
        filtered_df = filtered_df[filtered_df["gender"] == selected_gender]
    if selected_branch and selected_branch != "ALL":
        filtered_df = filtered_df[filtered_df["branch"] == selected_branch]
    if selected_tier and selected_tier != "ALL":
        filtered_df = filtered_df[filtered_df["college_tier"] == selected_tier]
    if selected_vol and selected_vol != "ALL" and "volunteer_experience" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["volunteer_experience"] == selected_vol
        ]

    cohort_ratio = f"{len(filtered_df):,} / {len(raw_df):,}"
    st.caption(f":material/groups: Active cohort: **{cohort_ratio}** students")

    # Resume upload
    st.space("small")
    st.subheader(":material/upload_file: Resume upload")
    if HAS_PDFPLUMBER:
        resume_file = st.file_uploader(
            "Upload resume (PDF)",
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
                    st.success(
                        f"Parsed resume for **{parsed['name']}**",
                        icon=":material/check_circle:",
                    )
    else:
        st.caption(":material/info: Install `pdfplumber` for resume auto-fill support.")
        resume_file = None


# =============================================================================
# 5. BATCH PREDICTIONS ON FILTERED COHORT
# =============================================================================
if not filtered_df.empty:
    try:
        cohort_probs = predictor.predict_probabilities(
            filtered_df, model_name=st.session_state.get("active_model")
        )
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

        with col_dept:
            with st.container(border=True):
                st.markdown("#### :material/bar_chart: Readiness by gender")
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
                    textfont=dict(color="#F8FAFC", size=11),
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
                "gender",
                "branch",
                "college_tier",
                "cgpa",
                "attendance_percentage",
                "backlogs",
                "coding_skill_score",
                "certifications_count",
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
                "coding_skill_score": st.column_config.ProgressColumn(
                    "Coding Score", format="%.0f", min_value=0, max_value=100,
                ),
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

            # Categorical attributes
            st.markdown("**Demographics & Institution**")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                student_data["gender"] = st.selectbox(
                    "Gender", options=["Male", "Female"], key="diag_gender"
                )
            with c2:
                branch_list = sorted(raw_df["branch"].dropna().unique().tolist()) if "branch" in raw_df.columns else ["CSE", "IT", "ECE", "EEE", "Mechanical", "Civil"]
                student_data["branch"] = st.selectbox(
                    "Branch", options=branch_list, key="diag_branch"
                )
            with c3:
                tier_list = sorted(raw_df["college_tier"].dropna().unique().tolist()) if "college_tier" in raw_df.columns else ["Tier 1", "Tier 2", "Tier 3"]
                student_data["college_tier"] = st.selectbox(
                    "College Tier", options=tier_list, key="diag_tier"
                )
            with c4:
                student_data["volunteer_experience"] = st.selectbox(
                    "Volunteer Experience", options=["Yes", "No"], key="diag_vol"
                )

            # Academic Performance
            st.markdown("**Academic & Core Profile**")
            ac1, ac2, ac3, ac4 = st.columns(4)
            with ac1:
                student_data["cgpa"] = st.slider(
                    "Current CGPA",
                    0.0, 10.0,
                    value=float(st.session_state.get("diag_cgpa", 7.5)),
                    step=0.1,
                    key="diag_cgpa_slider",
                )
            with ac2:
                student_data["attendance_percentage"] = st.slider(
                    "Attendance %",
                    0.0, 100.0,
                    value=float(st.session_state.get("diag_attendance", 85.0)),
                    step=1.0,
                    key="diag_att_slider",
                )
            with ac3:
                student_data["backlogs"] = st.number_input(
                    "Active Backlogs", 0, 20,
                    value=int(st.session_state.get("diag_backlogs", 0)),
                    key="diag_backlogs_input"
                )
            with ac4:
                student_data["age"] = st.number_input(
                    "Age", 18, 40,
                    value=int(st.session_state.get("diag_age", 21)),
                    key="diag_age_input"
                )

            # Skills & Test Scores
            st.markdown("**Skills & Assessment Scores**")
            sk1, sk2, sk3, sk4, sk5 = st.columns(5)
            with sk1:
                student_data["coding_skill_score"] = st.slider(
                    "Coding Score", 0.0, 100.0, 75.0, 1.0, key="diag_coding"
                )
            with sk2:
                student_data["aptitude_score"] = st.slider(
                    "Aptitude Score", 0.0, 100.0, 75.0, 1.0, key="diag_aptitude"
                )
            with sk3:
                student_data["communication_skill_score"] = st.slider(
                    "Communication Score", 0.0, 100.0, 75.0, 1.0, key="diag_comm"
                )
            with sk4:
                student_data["logical_reasoning_score"] = st.slider(
                    "Logical Reasoning", 0.0, 100.0, 75.0, 1.0, key="diag_logical"
                )
            with sk5:
                student_data["mock_interview_score"] = st.slider(
                    "Mock Interview", 0.0, 100.0, 70.0, 1.0, key="diag_mock"
                )

            # Experience & Professional Presence
            st.markdown("**Experience & Professional Presence**")
            ex1, ex2, ex3, ex4, ex5, ex6 = st.columns(6)
            with ex1:
                student_data["internships_count"] = st.number_input(
                    "Internships", 0, 10,
                    value=int(st.session_state.get("diag_internships_count", 1)),
                    key="diag_intern_input"
                )
            with ex2:
                student_data["projects_count"] = st.number_input(
                    "Projects", 0, 20,
                    value=int(st.session_state.get("diag_projects_count", 2)),
                    key="diag_proj_input"
                )
            with ex3:
                student_data["certifications_count"] = st.number_input(
                    "Certifications", 0, 20,
                    value=int(st.session_state.get("diag_certifications_count", 2)),
                    key="diag_certs_input"
                )
            with ex4:
                student_data["hackathons_participated"] = st.number_input(
                    "Hackathons", 0, 15, 1, key="diag_hack_input"
                )
            with ex5:
                student_data["github_repos"] = st.number_input(
                    "GitHub Repos", 0, 100, 8, key="diag_git_input"
                )
            with ex6:
                student_data["linkedin_connections"] = st.number_input(
                    "LinkedIn Conn.", 0, 1000, 200, key="diag_li_input"
                )

            # Leadership, Activities & Habits
            st.markdown("**Leadership, Extracurricular & Lifestyle**")
            ls1, ls2, ls3, ls4 = st.columns(4)
            with ls1:
                student_data["extracurricular_score"] = st.slider(
                    "Extracurricular Score", 0.0, 100.0, 65.0, 1.0, key="diag_extra_score"
                )
            with ls2:
                student_data["leadership_score"] = st.slider(
                    "Leadership Score", 0.0, 100.0, 60.0, 1.0, key="diag_lead_score"
                )
            with ls3:
                student_data["study_hours_per_day"] = st.slider(
                    "Study Hours / Day", 0.0, 16.0, 4.0, 0.5, key="diag_study_hours"
                )
            with ls4:
                student_data["sleep_hours"] = st.slider(
                    "Sleep Hours / Day", 0.0, 16.0, 7.0, 0.5, key="diag_sleep_hours"
                )

    # Evaluate single candidate
    candidate_prob = predictor.predict_single(
        student_data, model_name=st.session_state.get("active_model")
    )
    candidate_prob_pct = round(candidate_prob * 100, 1)

    diag_left, diag_right = st.columns([1, 2])

    with diag_left:
        with st.container(border=True):
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

    with diag_right:
        with st.container(border=True):
            st.markdown("#### :material/radar: Multi-dimensional competency radar")

            # Define radar axes using available features
            radar_specs = [
                {"column": "cgpa", "label": "CGPA (×10)", "scale_factor": 10.0},
                {"column": "attendance_percentage", "label": "Attendance %", "scale_factor": 1.0},
                {"column": "coding_skill_score", "label": "Coding Skill", "scale_factor": 1.0},
                {"column": "communication_skill_score", "label": "Communication", "scale_factor": 1.0},
                {"column": "aptitude_score", "label": "Aptitude", "scale_factor": 1.0},
                {"column": "certifications_count", "label": "Certs (×20)", "scale_factor": 20.0},
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

    # Prescriptive remediation engine
    st.space("small")
    st.markdown("### :material/lightbulb: Prescriptive remediation & targeted interventions")
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
            "condition": lambda s: s.get("coding_skill_score", 100) < 70,
            "priority": "HIGH",
            "title": "Boost Coding & Technical Skills",
            "action": "Complete coding bootcamp and competitive programming to improve technical readiness.",
            "sim_column": "coding_skill_score",
            "sim_op": "add",
            "sim_value": 15.0,
        },
        {
            "condition": lambda s: s.get("certifications_count", 10) < 2,
            "priority": "MEDIUM",
            "title": "Pursue Industry Certifications",
            "action": "Earn at least 2 industry certifications (AWS, Azure, Google Cloud, etc.).",
            "sim_column": "certifications_count",
            "sim_op": "add",
            "sim_value": 2.0,
        },
        {
            "condition": lambda s: s.get("projects_count", 10) < 2,
            "priority": "MEDIUM",
            "title": "Build Live Projects",
            "action": "Complete at least two end-to-end deployed projects to demonstrate practical skills.",
            "sim_column": "projects_count",
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
            "condition": lambda s: s.get("internships_count", 10) < 1,
            "priority": "MEDIUM",
            "title": "Secure an Internship",
            "action": "Apply for at least one internship to gain industry exposure before placements.",
            "sim_column": "internships_count",
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
                    "uplift": f"+{gain}% Placement Uplift",
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

            # Gender segment — pills for small option set
            sim_gender_opts = ["ALL COHORTS"] + sorted(
                raw_df["gender"].dropna().unique().tolist()
            )
            sim_target_gender = st.pills(
                "Target segment",
                sim_gender_opts,
                default="ALL COHORTS",
                key="sim_gender",
            )

            target_slice = raw_df.copy()
            if sim_target_gender and sim_target_gender != "ALL COHORTS":
                target_slice = target_slice[
                    target_slice["gender"] == sim_target_gender
                ]

            st.caption(
                f":material/groups: Target cohort: **{len(target_slice):,}** candidates"
            )

        # Grouped intervention sliders
        with st.container(border=True):
            st.markdown("#### 2. Academic interventions")
            academic_knobs = [k for k in INTERVENTION_KNOBS if k["column"] in ("attendance_percentage", "backlogs", "aptitude_score")]
            interventions_dict = {}
            for knob in academic_knobs:
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

        with st.container(border=True):
            st.markdown("#### 3. Experiential interventions")
            exp_knobs = [k for k in INTERVENTION_KNOBS if k["column"] in ("coding_skill_score", "certifications_count", "projects_count")]
            for knob in exp_knobs:
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
                transitions_table["newly_shortlistable"] == True
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
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, precision_score,
        recall_score, f1_score, confusion_matrix, roc_curve,
    )
    import time

    bench_df = raw_df.copy()
    if "placement_target" in bench_df.columns:
        y_bench = bench_df["placement_target"]
    elif "placement_status" in bench_df.columns:
        if bench_df["placement_status"].dtype == object:
            y_bench = bench_df["placement_status"].map({"Not Placed": 0, "Placed": 1})
        else:
            y_bench = bench_df["placement_status"]
    else:
        y_bench = pd.Series(0, index=bench_df.index)

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        bench_df, y_bench, test_size=0.20, random_state=42, stratify=y_bench
    )

    X_train_proc = predictor._preprocessor.transform(predictor._prepare_features(X_train_b))
    X_test_proc = predictor._preprocessor.transform(predictor._prepare_features(X_test_b))

    comparison_matrix = []
    detailed_metrics = {}

    for m_name, m_model in predictor._models.items():
        # Ensure scikit-learn compatibility
        if hasattr(m_model, "__class__") and "LogisticRegression" in m_model.__class__.__name__:
            if not hasattr(m_model, "multi_class"):
                m_model.multi_class = "auto"

        y_pred = m_model.predict(X_test_proc)
        y_prob = m_model.predict_proba(X_test_proc)[:, 1]

        test_auc = float(roc_auc_score(y_test_b, y_prob))
        precision = float(precision_score(y_test_b, y_pred, zero_division=0))
        recall_val = float(recall_score(y_test_b, y_pred, zero_division=0))
        f1_val = float(f1_score(y_test_b, y_pred, zero_division=0))
        accuracy = float(accuracy_score(y_test_b, y_pred))

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
        }

    # Best model feature importance extraction
    best_model_name = max(comparison_matrix, key=lambda x: x["Test ROC-AUC"])["Model"]
    best_model_obj = predictor._models[best_model_name]
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

    top_features = []
    if hasattr(best_model_obj, "feature_importances_"):
        raw_imp = best_model_obj.feature_importances_
        total = np.sum(raw_imp)
        normalized_imp = (raw_imp / total) * 100.0 if total > 0 else raw_imp
        f_names = feat_names_out if len(feat_names_out) == len(normalized_imp) else [f"Feature_{i}" for i in range(len(normalized_imp))]
        top_features = sorted(zip(f_names, np.round(normalized_imp, 2)), key=lambda x: x[1], reverse=True)[:10]
    elif hasattr(best_model_obj, "coef_"):
        raw_imp = np.abs(best_model_obj.coef_[0])
        total = np.sum(raw_imp)
        normalized_imp = (raw_imp / total) * 100.0 if total > 0 else raw_imp
        f_names = feat_names_out if len(feat_names_out) == len(normalized_imp) else [f"Feature_{i}" for i in range(len(normalized_imp))]
        top_features = sorted(zip(f_names, np.round(normalized_imp, 2)), key=lambda x: x[1], reverse=True)[:10]

    return {
        "comparison_matrix": comparison_matrix,
        "detailed_metrics": detailed_metrics,
        "best_model_name": best_model_name,
        "top_features": top_features,
    }


bench_expander = st.expander(
    ":material/query_stats: Multi-model benchmark comparison matrix & performance validation",
    expanded=False,
)
if bench_expander.open:
  with bench_expander:
    st.markdown("### Formal multi-model benchmark comparison matrix")
    st.caption(
        "Compare Logistic Regression, Random Forest, and XGBoost "
        "across accuracy, ROC-AUC, precision, recall, F1, and latency."
    )

    try:
        bench_data = compute_benchmark_suite(len(raw_df))
        comparison_matrix = bench_data["comparison_matrix"]
        detailed_metrics = bench_data["detailed_metrics"]
        best_model_name = bench_data["best_model_name"]
        top_features = bench_data.get("top_features", [])

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

        with col_roc:
            with st.container(border=True):
                st.markdown("#### :material/show_chart: Multi-model ROC curves")
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
                st.markdown(
                    f"#### :material/grid_on: Confusion matrix ({best_model_name})"
                )
                best_cm = detailed_metrics[best_model_name]["confusion_matrix"]
                fig_cm = px.imshow(
                    best_cm,
                    text_auto=True,
                    labels=dict(x="Predicted class", y="Actual class", color="Count"),
                    x=["Not placed (0)", "Placed (1)"],
                    y=["Not placed (0)", "Placed (1)"],
                    color_continuous_scale="Blues",
                )
                fig_cm.update_layout(get_plotly_layout(height=300))
                st.plotly_chart(fig_cm, use_container_width=True)

        # Feature importance
        if top_features:
            with st.container(border=True):
                st.markdown(f"#### :material/leaderboard: Global feature importance attribution ({best_model_name})")
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
