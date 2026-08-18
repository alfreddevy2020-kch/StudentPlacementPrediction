"""
Student Placement Prediction Dashboard
=======================================
A Streamlit-based interactive dashboard that communicates with a FastAPI
backend to predict student placement probability using an XGBoost model.

Usage:
    streamlit run app.py

Prerequisites:
    - FastAPI backend running on http://localhost:8000
    - Python packages: streamlit, requests, plotly, pandas, pdfplumber
"""

import io
import re

import pdfplumber
import plotly.graph_objects as go
import requests
import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND_URL = "http://localhost:8000/api/v1/predict"

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Resume Parsing (client-side heuristic extraction)
# ---------------------------------------------------------------------------

def extract_resume_data(pdf_bytes: bytes) -> dict:
    """
    Extract structured student data from a resume PDF using regex
    heuristics. Returns a dict with keys matching sidebar widget keys;
    values are None when the pattern is not found.
    """
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

    # -- Name: first plausible line in the header ---------------------------
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

    # -- CGPA ---------------------------------------------------------------
    m = re.search(r"(?:cgpa|gpa)[:\s]*(\d+\.?\d*)", text, re.IGNORECASE)
    if m:
        result["cgpa"] = min(float(m.group(1)), 10.0)

    # -- SSC (10th) percentage ----------------------------------------------
    m = re.search(
        r"(?:10th|tenth|ssc|class\s*x)[:\s]*(\d+\.?\d*)\s*%?",
        text, re.IGNORECASE,
    )
    if m:
        result["ssc_percentage"] = min(float(m.group(1)), 100.0)

    # -- HSC (12th) percentage ----------------------------------------------
    m = re.search(
        r"(?:12th|twelfth|hsc|class\s*xii|xii)[:\s]*(\d+\.?\d*)\s*%?",
        text, re.IGNORECASE,
    )
    if m:
        result["hsc_percentage"] = min(float(m.group(1)), 100.0)

    # -- Degree percentage --------------------------------------------------
    m = re.search(
        r"(?:degree|graduation|b\.?tech|b\.?e\.?)\s*(?:percentage|marks)?"
        r"[:\s]*(\d+\.?\d*)\s*%?",
        text, re.IGNORECASE,
    )
    if m:
        result["degree_percentage"] = min(float(m.group(1)), 100.0)

    # -- Certifications count (bullet items under certification heading) ----
    cert_match = re.search(
        r"(?:certification|certificate|courses?)[\s:]*\n"
        r"((?:[-•*]\s*.+\n?)+)",
        text, re.IGNORECASE,
    )
    if cert_match:
        result["certifications"] = len(
            re.findall(r"[-•*]\s*.+", cert_match.group(1))
        )

    # -- Internship count (entries under experience heading) ----------------
    exp_match = re.search(
        r"(?:experience|internship|work\s*history)[\s:]*\n"
        r"((?:[-•*]\s*.+\n?)+)",
        text, re.IGNORECASE,
    )
    if exp_match:
        result["internship_count"] = len(
            re.findall(r"[-•*]\s*.+", exp_match.group(1))
        )

    # -- Live projects (entries under projects heading) ---------------------
    proj_match = re.search(
        r"(?:projects?|portfolio)[\s:]*\n((?:[-•*]\s*.+\n?)+)",
        text, re.IGNORECASE,
    )
    if proj_match:
        result["live_projects"] = len(
            re.findall(r"[-•*]\s*.+", proj_match.group(1))
        )

    return result


# ---------------------------------------------------------------------------
# Sidebar – Input Controls
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[dict, object]:
    """
    Render all sidebar input widgets and return a dictionary whose keys
    match the JSON contract expected by the FastAPI /predict endpoint.

    Uses st.session_state for resume auto-fill: when a new PDF is uploaded,
    parsed values are written into session_state BEFORE the widgets are
    created, so the widgets pick them up as their current values.  The user
    can still drag sliders / change numbers to override.
    """
    # -- Resume upload (placed first so parsing runs before widgets) --------
    st.sidebar.header("Resume")
    resume_file = st.sidebar.file_uploader(
        "Upload Resume (PDF)",
        type=["pdf"],
        help="Upload the student's resume to auto-fill fields below",
    )

    # Parse new resume and push values into session_state -------------------
    if resume_file is not None:
        if st.session_state.get("_last_resume_file") != resume_file.name:
            parsed = extract_resume_data(resume_file.getvalue())
            st.session_state["_parsed"] = parsed
            st.session_state["_last_resume_file"] = resume_file.name
            _AUTO_FILL_MAP = {
                "cgpa": "cgpa",
                "ssc_percentage": "ssc_percentage",
                "hsc_percentage": "hsc_percentage",
                "degree_percentage": "degree_percentage",
                "certifications": "certifications",
                "internship_count": "internship_count",
                "live_projects": "live_projects",
                "work_experience_months": "work_experience_months",
            }
            for parsed_key, widget_key in _AUTO_FILL_MAP.items():
                if parsed[parsed_key] is not None:
                    st.session_state[widget_key] = parsed[parsed_key]

    # -- Prediction Model ---------------------------------------------------
    st.sidebar.header("Prediction Model")
    prediction_model = st.sidebar.selectbox(
        "Prediction Model",
        key="prediction_model",
        options=["random_forest", "logistic_regression", "xgboost"],
        index=0,
        format_func=lambda x: {
            "random_forest": "Random Forest",
            "logistic_regression": "Logistic Regression",
            "xgboost": "XGBoost",
        }[x],
        help="Select which trained ML model to use for prediction",
    )

    # -- Academic Performance -----------------------------------------------
    st.sidebar.header("Academic Performance")
    cgpa = st.sidebar.slider(
        "Current CGPA", key="cgpa",
        min_value=0.0, max_value=10.0, value=7.0, step=0.1,
        help="Semester GPA on a 0-10 scale",
    )
    ssc_percentage = st.sidebar.slider(
        "SSC (10th) Marks %", key="ssc_percentage",
        min_value=0.0, max_value=100.0, value=80.0, step=0.5,
        help="Percentage scored in 10th standard",
    )
    hsc_percentage = st.sidebar.slider(
        "HSC (12th) Marks %", key="hsc_percentage",
        min_value=0.0, max_value=100.0, value=75.0, step=0.5,
        help="Percentage scored in 12th standard",
    )
    degree_percentage = st.sidebar.slider(
        "Degree Marks %", key="degree_percentage",
        min_value=0.0, max_value=100.0, value=75.0, step=0.5,
        help="Overall degree / graduation percentage",
    )
    attendance_percentage = st.sidebar.slider(
        "Attendance %", key="attendance_percentage",
        min_value=0.0, max_value=100.0, value=85.0, step=1.0,
        help="Overall class attendance percentage",
    )

    # -- Academic Standing ---------------------------------------------------
    st.sidebar.header("Academic Standing")
    backlogs = st.sidebar.number_input(
        "Active Backlogs", key="backlogs",
        min_value=0, max_value=20, value=0, step=1,
        help="Number of currently pending backlogs",
    )

    # -- Entrance & Skills --------------------------------------------------
    st.sidebar.header("Entrance & Skills")
    entrance_exam_score = st.sidebar.slider(
        "Entrance Exam Score", key="entrance_exam_score",
        min_value=0.0, max_value=100.0, value=80.0, step=1.0,
        help="Score in entrance examination",
    )
    technical_skill_score = st.sidebar.slider(
        "Technical Skill Score", key="technical_skill_score",
        min_value=0.0, max_value=100.0, value=75.0, step=1.0,
        help="Self-assessed or tested technical skill score",
    )
    soft_skill_score = st.sidebar.slider(
        "Soft Skill Score", key="soft_skill_score",
        min_value=0.0, max_value=100.0, value=75.0, step=1.0,
        help="Communication, teamwork, etc.",
    )

    # -- Experience & Credentials --------------------------------------------
    st.sidebar.header("Experience & Credentials")
    certifications = st.sidebar.number_input(
        "Number of Certifications", key="certifications",
        min_value=0, max_value=20, value=0, step=1,
        help="Industry / online certifications earned",
    )
    live_projects = st.sidebar.number_input(
        "Live Projects", key="live_projects",
        min_value=0, max_value=20, value=0, step=1,
        help="Number of live / deployed projects",
    )
    internship_count = st.sidebar.number_input(
        "Internship Count", key="internship_count",
        min_value=0, max_value=10, value=0, step=1,
        help="Total internships completed",
    )
    work_experience_months = st.sidebar.number_input(
        "Work Experience (Months)", key="work_experience_months",
        min_value=0, max_value=60, value=0, step=1,
        help="Prior full-time work experience in months",
    )

    # -- Personal -----------------------------------------------------------
    st.sidebar.header("Personal")
    gender = st.sidebar.selectbox(
        "Gender", key="gender",
        options=["Male", "Female", "Other"], index=0,
    )
    extracurricular_activities = st.sidebar.selectbox(
        "Extracurricular Activities", key="extracurricular_activities",
        options=["No", "Yes"], index=0,
        help="Participation in clubs, sports, volunteering, etc.",
    )

    # -- Structured payload --------------------------------------------------
    # Keys match the FastAPI /predict contract exactly.
    payload = {
        "model": prediction_model,
        "ssc_percentage": ssc_percentage,
        "hsc_percentage": hsc_percentage,
        "degree_percentage": degree_percentage,
        "cgpa": cgpa,
        "attendance_percentage": attendance_percentage,
        "backlogs": backlogs,
        "entrance_exam_score": entrance_exam_score,
        "technical_skill_score": technical_skill_score,
        "soft_skill_score": soft_skill_score,
        "certifications": certifications,
        "live_projects": live_projects,
        "internship_count": internship_count,
        "work_experience_months": work_experience_months,
        "gender": gender,
        "extracurricular_activities": extracurricular_activities,
    }

    return payload, resume_file


# ---------------------------------------------------------------------------
# API Communication
# ---------------------------------------------------------------------------

def get_prediction(
    payload: dict, resume_file=None, resume_text: str = ""
) -> dict | None:
    """
    Send the student payload (and optional resume PDF + extracted text)
    to the FastAPI backend and return the JSON response.  Returns None
    if the request fails so the caller can handle the error gracefully.
    """
    try:
        '''if resume_file is not None:
            files = {
                "resume": (resume_file.name, resume_file, "application/pdf")
            }
            data = {
                "payload_json": pd.Series(payload).to_json(),
                "resume_text": resume_text,
            }
            response = requests.post(
                BACKEND_URL, data=data, files=files, timeout=10
            )
        else:
            response = requests.post(BACKEND_URL, json=payload, timeout=10) '''
        response = requests.post(BACKEND_URL, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot reach the prediction backend. "
            "Make sure the FastAPI server is running on "
            "`http://localhost:8000`."
        )
        return None
    except requests.exceptions.Timeout:
        st.error("The backend request timed out. Please try again.")
        return None
    except requests.exceptions.HTTPError as exc:
        st.error(f"Backend returned an error: {exc}")
        return None
    except requests.exceptions.RequestException as exc:
        st.error(f"Unexpected request error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Visualisation Helpers
# ---------------------------------------------------------------------------

def build_gauge_chart(probability: float) -> go.Figure:
    """
    Build a Plotly gauge chart (0-100 %) with three colour zones:
        Red    – 0–40 %
        Yellow – 40–70 %
        Green  – 70–100 %
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,  # convert 0-1 fraction to percentage
            number={"suffix": "%", "font": {"size": 48}},
            title={"text": "Placement Probability", "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#1f77b4", "thickness": 0.3},
                "steps": [
                    {"range": [0, 40], "color": "#ff4b4b"},    # Red
                    {"range": [40, 70], "color": "#ffa726"},   # Yellow
                    {"range": [70, 100], "color": "#66bb6a"},  # Green
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": probability * 100,
                },
            },
        )
    )
    fig.update_layout(height=350, margin=dict(t=40, b=10, l=30, r=30))
    return fig


def get_recommendation(risk_level: str, payload: dict) -> str:
    """
    Return a human-readable recommendation string based on the backend's
    risk_level classification and the student's current metrics.
    """
    base = f"**{risk_level}**"

    tips: list[str] = []

    if payload["backlogs"] > 0:
        tips.append(
            f"- Clear **{payload['backlogs']} active backlog(s)** as the highest priority."
        )
    if payload["certifications"] < 2:
        tips.append(
            "- Pursue at least **2 industry certifications** to boost employability."
        )
    if payload["internship_count"] < 1:
        tips.append(
            "- Secure **at least one internship** before placement season."
        )
    if payload["cgpa"] < 7.0:
        tips.append(
            "- Aim to raise CGPA **above 7.0** — many companies set this as a cutoff."
        )
    if payload["attendance_percentage"] < 75:
        tips.append(
            "- Improve attendance to **>= 75 %** to avoid disqualification."
        )
    if payload["live_projects"] < 1:
        tips.append(
            "- Build **at least one live project** to demonstrate practical skills."
        )
    if payload["technical_skill_score"] < 70:
        tips.append(
            "- Work on **technical skills** — consider online courses or hackathons."
        )

    recommendation = base
    if tips:
        recommendation += "\n\n**Action Items:**\n" + "\n".join(tips)

    return recommendation


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

def main() -> None:
    # -- Header --------------------------------------------------------------
    st.title("Student Placement Prediction Dashboard")
    st.caption(
        "Adjust the student metrics in the sidebar to run what-if "
        "simulations and see how each factor affects placement probability."
    )

    # -- Collect inputs ------------------------------------------------------
    payload, resume_file = render_sidebar()

    # -- Extract resume text client-side -------------------------------------
    resume_text = ""
    if resume_file is not None:
        with pdfplumber.open(io.BytesIO(resume_file.getvalue())) as pdf:
            resume_text = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )

    # -- Student name from resume --------------------------------------------
    student_name = st.session_state.get("_parsed", {}).get("name")
    if student_name:
        st.info(f"**Student:** {student_name}")

    # -- Call backend --------------------------------------------------------
    result = get_prediction(payload, resume_file, resume_text)

    if result is None:
        # Backend is unreachable — error already displayed by get_prediction
        return

    # Backend response contract:
    # {
    #     "model_used": "<string>",
    #     "placement_status": 1 | 0,
    #     "placement_label": "Placed" | "Not Placed",
    #     "probability_placed": <float 0-1>,
    #     "probability_not_placed": <float 0-1>,
    #     "risk_level": "<string>"
    # }
    model_used: str = result.get("model_used", "Unknown")
    probability_placed: float = result.get("probability_placed", 0.0)
    placement_label: str = result.get("placement_label", "Unknown")
    risk_level: str = result.get("risk_level", "Unknown")

    # -- Primary metric display ----------------------------------------------
    col_title, col_prob = st.columns([2, 1])
    with col_title:
        st.subheader("Predicted Outcome")
    with col_prob:
        st.metric(
            label="Placement Probability",
            value=f"{probability_placed * 100:.1f}%",
        )

    # -- Prediction model & placement status ---------------------------------
    st.markdown(
        f"**Prediction Model:** {model_used}"
    )
    st.markdown(
        f"**Status:** {placement_label} &nbsp; | &nbsp; **Risk:** {risk_level}"
    )

    # -- Gauge chart ---------------------------------------------------------
    gauge_fig = build_gauge_chart(probability_placed)
    st.plotly_chart(gauge_fig, use_container_width=True)

    # -- Dynamic recommendation ----------------------------------------------
    recommendation = get_recommendation(risk_level, payload)
    st.markdown("---")
    st.subheader("Recommendation")
    st.markdown(recommendation)

    # -- Debug expanders -----------------------------------------------------
    if resume_text:
        with st.expander("Extracted Resume Text"):
            st.text_area("Content", resume_text, height=300, disabled=True)

    with st.expander("View Raw Payload Sent to Backend"):
        st.json(payload)

    with st.expander("Raw Backend Response"):
        st.json(result)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
