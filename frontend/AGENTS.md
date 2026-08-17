# AGENTS.md

## Project Structure

Single-file Streamlit app (`app.py`) that proxies to an external FastAPI backend. No build system, no tests, no CI in this repo.

```
dashboard/
  app.py          # Streamlit frontend — the only file that matters
  AGENTS.md
  flow.md         # Data flow & variable reference
```

## Running

```bash
# Install deps (no requirements.txt exists — install manually)
pip install streamlit requests plotly pandas pdfplumber

# Start the frontend
streamlit run app.py
```

The FastAPI backend must be running separately at `http://localhost:8000` (expected endpoint: `POST /predict`). The backend is **not** part of this repo.

## Key Gotchas

- **Streamlit reruns the entire script on every widget change.** There is no explicit state machine; reactivity is implicit. `st.session_state` is not used in the current code.
- **The app will show an error banner if the backend is down** — it does not crash. Do not add `sys.exit()` or `st.stop()` for backend failures; the existing `get_prediction()` handles this gracefully.
- **Backend response contract**: expects `{"placement_status", "placement_label", "probability_placed", "probability_not_placed", "risk_level"}`. The frontend reads `probability_placed` for the gauge/metric, `placement_label` and `risk_level` for display. Missing fields default gracefully via `.get()`.
- **JSON payload keys must stay in sync with the backend.** The keys are: `ssc_percentage`, `hsc_percentage`, `degree_percentage`, `cgpa`, `attendance_percentage`, `backlogs`, `entrance_exam_score`, `technical_skill_score`, `soft_skill_score`, `certifications`, `live_projects`, `internship_count`, `work_experience_months`, `gender`, `extracurricular_activities`. See `render_sidebar()` for the canonical mapping.
- **`pandas` is used for multipart serialization** — `pd.Series(payload).to_json()` encodes the payload when a resume is attached. Do not remove it.
- **`honours` is a string `"Yes"`/`"No"`, not a bool.** The backend must handle string comparison.
- **Resume upload changes the request format.** When a PDF is uploaded, `get_prediction()` switches from `json=` to `data=` + `files=` (multipart/form-data). The backend must accept both plain JSON and multipart requests.
- **Resume text is extracted client-side.** `pdfplumber` reads the PDF in the Streamlit process and sends the extracted text as `resume_text` to the backend. The backend receives both the raw PDF and the pre-extracted text.
- **`st.session_state` is used for resume auto-fill.** When a new PDF is uploaded, parsed values are written to `st.session_state[key]` BEFORE the widgets are created. This causes widgets to show the parsed values. User can override by dragging/changing widgets — their manual value persists across reruns.

## Conventions

- No linter or formatter is configured. Follow the existing style: section headers with `# ---` dividers, docstrings on all functions, type hints on return values.
- Functions are grouped by responsibility in this order: Config → Sidebar → API → Visualisation → Main.
