# Data Flow & Variables

## End-to-End Flow

```
User uploads resume PDF (optional)
        │
        ▼
extract_resume_data() ──► st.session_state ──► auto-fills sidebar widgets
        │
        ▼
User adjusts widgets in Streamlit sidebar
        │
        ▼
render_sidebar() builds `payload` dict + `resume_file`
        │
        ├── (with resume) ──► multipart POST: payload_json + resume_text + resume file
        └── (without) ──────► JSON POST
        │
        ▼
get_prediction() returns `result` dict   ◄── { placement_status, placement_label,
        │                                      probability_placed, probability_not_placed,
        │                                      risk_level }
        │
        ├─► st.metric               shows probability_placed
        ├─► build_gauge_chart()      renders Plotly gauge (probability_placed)
        └─► get_recommendation()     uses risk_level + payload for advice text
```

Every sidebar interaction triggers a full Streamlit rerun, so the pipeline
above fires on every slider drag or dropdown change.

## Resume Parsing

`extract_resume_data()` (`app.py:62`) uses regex heuristics on the raw PDF
text to fill these fields automatically:

| Field               | Heuristic                                                  |
|---------------------|------------------------------------------------------------|
| `name`              | First non-empty line in first 6 lines that looks like a person name (≥2 words, not a section header) |
| `cgpa`              | `CGPA: X.XX` or `GPA: X.XX` pattern                      |
| `ssc_percentage`    | `10th: XX` / `SSC: XX` / `Class X: XX` pattern            |
| `hsc_percentage`    | `12th: XX` / `HSC: XX` / `Class XII: XX` pattern          |
| `degree_percentage` | `Degree: XX` / `B.Tech: XX` pattern                       |
| `certifications`    | Count of `[-•*]` bullet items under a Certifications heading |
| `internship_count`  | Count of bullet items under an Experience/Internship heading |
| `live_projects`     | Count of bullet items under a Projects heading             |

All other fields (entrance_exam_score, technical_skill_score, soft_skill_score,
gender, extracurricular_activities, backlogs, attendance_percentage) are NOT
auto-filled — the user must set them manually.

Parsed values are stored in `st.session_state["_parsed"]` and pushed into
widget keys. User can override any auto-filled value by dragging/changing
the widget.

## Payload (frontend → backend)

Built by `render_sidebar()`. Keys match the FastAPI `/predict` contract.

| Key                          | Type    | Range         | Widget         | Auto-fillable |
|------------------------------|---------|---------------|----------------|---------------|
| `ssc_percentage`             | float   | 0.0 – 100.0  | slider         | Yes           |
| `hsc_percentage`             | float   | 0.0 – 100.0  | slider         | Yes           |
| `degree_percentage`          | float   | 0.0 – 100.0  | slider         | Yes           |
| `cgpa`                       | float   | 0.0 – 10.0   | slider         | Yes           |
| `attendance_percentage`      | float   | 0.0 – 100.0  | slider         | No            |
| `backlogs`                   | int     | 0 – 20        | number_input   | No            |
| `entrance_exam_score`        | float   | 0.0 – 100.0  | slider         | No            |
| `technical_skill_score`      | float   | 0.0 – 100.0  | slider         | No            |
| `soft_skill_score`           | float   | 0.0 – 100.0  | slider         | No            |
| `certifications`             | int     | 0 – 20        | number_input   | Yes           |
| `live_projects`              | int     | 0 – 20        | number_input   | Yes           |
| `internship_count`           | int     | 0 – 10        | number_input   | Yes           |
| `work_experience_months`     | int     | 0 – 60        | number_input   | Yes           |
| `gender`                     | string  | Male/Female/Other | selectbox   | No            |
| `extracurricular_activities` | string  | "Yes"/"No"    | selectbox      | No            |

###  Backend Call

`POST /predict` with `application/json` body.

## Response (backend → frontend)

Expected shape from `POST /predict`:

```json
{
    "placement_status": 1,
    "placement_label": "Placed",
    "probability_placed": 0.9646,
    "probability_not_placed": 0.0354,
    "risk_level": "High Probability of Placement (Low Risk)"
}
```

| Field                    | Type    | Values / Range        | Used in                            |
|--------------------------|---------|-----------------------|------------------------------------|
| `placement_status`       | int     | `1` (placed) / `0`    | (available but not directly used)  |
| `placement_label`        | string  | "Placed" / "Not Placed" | Status line display             |
| `probability_placed`     | float   | 0.0 – 1.0            | `st.metric`, gauge chart           |
| `probability_not_placed` | float   | 0.0 – 1.0            | (available, not displayed)         |
| `risk_level`             | string  | Free-text risk label  | Status line + recommendation base  |

Defaults if fields are missing: `probability_placed` → `0.0`, `placement_label` → `"Unknown"`, `risk_level` → `"Unknown"`.

## Recommendation Logic

`get_recommendation()` produces advice based on the backend's `risk_level`
string (displayed as the base message) plus conditional tips from the payload:

**Conditional tips** (checked independently):
| Condition                      | Tip                                            |
|--------------------------------|------------------------------------------------|
| `backlogs > 0`                 | Clear backlogs (highest priority)              |
| `certifications < 2`           | Pursue at least 2 certifications               |
| `internship_count < 1`         | Secure at least one internship                 |
| `cgpa < 7.0`                   | Raise CGPA above 7.0                           |
| `attendance_percentage < 75`   | Improve attendance to >= 75%                   |
| `live_projects < 1`            | Build at least one live project                |
| `technical_skill_score < 70`   | Work on technical skills                       |

## Gauge Chart Zones

`build_gauge_chart()` converts the 0–1 probability to 0–100%:

| Range      | Colour | Hex       |
|------------|--------|-----------|
| 0 – 40%   | Red    | `#ff4b4b` |
| 40 – 70%  | Yellow | `#ffa726` |
| 70 – 100% | Green  | `#66bb6a` |

## Config Constants

| Constant      | Value                              | Location  |
|---------------|------------------------------------|-----------|
| `BACKEND_URL` | `http://localhost:8000/predict`    | `app.py:30` |
