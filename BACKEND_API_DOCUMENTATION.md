# Backend API Documentation

> **Scope:** This file documents the FastAPI prediction backend and its
> integration with the Streamlit frontend — implemented as my individual
> contribution to the Student Placement Prediction project.
>
> **Existing team documentation is left unchanged:**
> - `README.md` — project overview (Member 1)
> - `frontend/AGENTS.md` — frontend agent guidelines (Member 3)
> - `frontend/flow.md` — frontend data-flow reference (Member 3)

---

## Integration / My Contribution

I designed and implemented the entire backend serving layer for the
Student Placement Prediction system. My work covers:

| Area | What I built |
|---|---|
| **API serving** | FastAPI application with lifespan management, global error handling, and OpenAPI docs |
| **Inference integration** | Modular predictor layer that loads joblib artifacts and runs end-to-end inference |
| **Preprocessing / model loading** | Deserialising `preprocessor.joblib` and `random_forest_best.joblib` once at startup |
| **Pydantic validation** | Strict request/response schemas that mirror training-data constraints |
| **Frontend-backend integration** | JSON contract consumed directly by the Streamlit frontend via `requests.post()` |

All four backend files (`api/main.py`, `api/config.py`, `api/schemas.py`,
`api/predictor.py`) and the `pyrightconfig.json` type-checker configuration
were created by me. The `preprocessing.py` update (saving
`preprocessor.joblib` to `part2/models/`) was also my addition.

---

## Repository Layout (backend files I created)

```
StudentPlacementPrediction/
├── api/
│   ├── __init__.py          ← makes api/ a Python package
│   ├── config.py            ← artifact paths + OpenAPI metadata
│   ├── main.py              ← FastAPI app, routes, lifespan
│   ├── predictor.py         ← inference logic (load → transform → predict)
│   └── schemas.py           ← Pydantic request / response models
├── part2/
│   └── models/
│       ├── preprocessor.joblib          ← saved by preprocessing.py
│       └── random_forest_best.joblib    ← saved by part2/random_forest_model.py
└── frontend/
    └── app.py               ← Streamlit UI (calls POST /api/v1/predict)
```

---

## Inference Architecture

```
Streamlit Frontend  (frontend/app.py)
        │
        │  requests.post("http://localhost:8000/api/v1/predict", json=payload)
        ▼
FastAPI  (api/main.py)
        │
        │  Pydantic validates 15 raw fields
        ▼
StudentInput  (api/schemas.py)
        │
        │  _input_to_dataframe()  →  pd.DataFrame (1 row × 15 named columns)
        ▼
preprocessor.joblib  (ColumnTransformer)
        │  StandardScaler   →  13 numerical features  (z-scored)
        │  OneHotEncoder    →   2 categorical features (4 dummy columns)
        │  Output: NumPy array (1 × 17)
        ▼
RandomForestClassifier  (random_forest_best.joblib)
        │  .predict()       →  label ∈ {0, 1}
        │  .predict_proba() →  [P(Not Placed), P(Placed)]
        ▼
PredictionResponse  (api/schemas.py)
        │  JSON serialised by FastAPI
        ▼
Streamlit Frontend  (displays gauge chart, label, risk level)
```

---

## File Responsibilities

### `api/config.py`

Central configuration. Resolves artifact paths relative to the file's own
location, so the server works from any working directory.

```python
BASE_DIR          = Path(__file__).resolve().parent.parent
PREPROCESSOR_PATH = BASE_DIR / "part2" / "models" / "preprocessor.joblib"
MODEL_PATH        = BASE_DIR / "part2" / "models" / "random_forest_best.joblib"
```

Also holds `APP_TITLE`, `APP_DESCRIPTION`, and `APP_VERSION` used by the
OpenAPI specification.

---

### `api/schemas.py`

Defines the full API contract using Pydantic v2 models.

#### `StudentInput` — POST request body

15 raw student features expected by `POST /api/v1/predict`.
`student_id` and `salary_package_lpa` are **not accepted** (the latter
would cause data leakage).

**Numerical fields (13) — validated with `ge` / `le` bounds:**

| Field | Type | Range | Description |
|---|---|---|---|
| `ssc_percentage` | `float` | 0–100 | Secondary school (10th) percentage |
| `hsc_percentage` | `float` | 0–100 | Higher secondary (12th) percentage |
| `degree_percentage` | `float` | 0–100 | Undergraduate degree percentage |
| `cgpa` | `float` | 0–10 | College CGPA on a 10-point scale |
| `attendance_percentage` | `float` | 0–100 | College attendance percentage |
| `entrance_exam_score` | `float` | 0–100 | Entrance examination score |
| `technical_skill_score` | `float` | 0–100 | Technical / coding skills score |
| `soft_skill_score` | `float` | 0–100 | Soft skills assessment score |
| `backlogs` | `int` | ≥ 0 | Active academic backlogs |
| `certifications` | `int` | ≥ 0 | Professional certifications earned |
| `live_projects` | `int` | ≥ 0 | Live / capstone projects completed |
| `internship_count` | `int` | ≥ 0 | Internships completed |
| `work_experience_months` | `int` | ≥ 0 | Prior work experience in months |

**Categorical fields (2) — validated with `Literal`:**

| Field | Type | Accepted values |
|---|---|---|
| `gender` | `str` | `"Male"`, `"Female"` |
| `extracurricular_activities` | `str` | `"Yes"`, `"No"` |

#### `PredictionResponse` — POST response body

| Field | Type | Description |
|---|---|---|
| `placement_status` | `int` | Binary class label: `1` = Placed, `0` = Not Placed |
| `placement_label` | `str` | Human-readable: `"Placed"` or `"Not Placed"` |
| `probability_placed` | `float` | Model confidence (0–1) the student will be placed |
| `probability_not_placed` | `float` | Model confidence (0–1) the student will NOT be placed |
| `risk_level` | `str` | Qualitative risk label (see table below) |

**`risk_level` mapping** (derived from `probability_placed`):

| `probability_placed` | `risk_level` |
|---|---|
| ≥ 0.8 | `"High Probability of Placement (Low Risk)"` |
| 0.5 – 0.79 | `"Moderate Probability of Placement (Medium Risk)"` |
| < 0.5 | `"High Risk of Non-Placement"` |

#### `HealthResponse` — GET /health response body

| Field | Type | Description |
|---|---|---|
| `status` | `str` | `"healthy"` when both artifacts are loaded; `"degraded"` otherwise |
| `preprocessor_loaded` | `bool` | `True` when `preprocessor.joblib` loaded at startup |
| `model_loaded` | `bool` | `True` when `random_forest_best.joblib` loaded at startup |

---

### `api/predictor.py`

Modular inference layer. Contains an abstract base class and the concrete
Random Forest implementation.

```
BasePredictor  (ABC)
    └── RandomForestPredictor   ← current production backend
```

**Key internals:**

```python
NUMERICAL_FEATURES = [
    "ssc_percentage", "hsc_percentage", "degree_percentage", "cgpa",
    "entrance_exam_score", "technical_skill_score", "soft_skill_score",
    "internship_count", "live_projects", "work_experience_months",
    "certifications", "attendance_percentage", "backlogs",
]

CATEGORICAL_FEATURES = ["gender", "extracurricular_activities"]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES  # 15 total
```

**`RandomForestPredictor.load()`** — called once at startup:
```python
preprocessor = joblib.load(preprocessor_path)   # ColumnTransformer
model        = joblib.load(model_path)           # RandomForestClassifier
```

**`RandomForestPredictor.predict()`** — called per request:
```python
df            = _input_to_dataframe(data)           # StudentInput → DataFrame
X_transformed = self._preprocessor.transform(df)   # 15 cols → 17 scaled cols
label         = int(self._model.predict(X_transformed)[0])
proba         = self._model.predict_proba(X_transformed)[0]
# proba = [P(class=0), P(class=1)]  i.e.  [P(Not Placed), P(Placed)]
```

---

### `api/main.py`

FastAPI application entry point.

**Singleton predictor** — loaded once, shared across all requests:
```python
_predictor: RandomForestPredictor | None = None
```

**Lifespan handler** — startup deserialises both artifacts:
```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    global _predictor
    _predictor = RandomForestPredictor.load(PREPROCESSOR_PATH, MODEL_PATH)
    yield  # server now serving
```

If either artifact is missing, the server starts in a **degraded** state
rather than crashing. `GET /health` reports `"status": "degraded"` and
`POST /api/v1/predict` returns `503`.

**Error handling layers:**

| Layer | Trigger | HTTP code |
|---|---|---|
| Pydantic | Wrong type / out-of-range / missing field | `422` |
| Route guard | `_predictor` not ready | `503` |
| Inference | Exception inside `predict()` | `500` |
| Global handler | Any other unhandled exception | `500` |

---

## API Endpoints

### `GET /health`

Liveness probe. Confirms both artifacts are loaded.

**Example response (`200 OK`):**
```json
{
  "status": "healthy",
  "preprocessor_loaded": true,
  "model_loaded": true
}
```

---

### `POST /api/v1/predict`

Main inference endpoint. Accepts 15 raw student features and returns a
placement prediction.

**Example request body:**
```json
{
  "ssc_percentage": 75.5,
  "hsc_percentage": 78.0,
  "degree_percentage": 72.0,
  "cgpa": 8.2,
  "attendance_percentage": 90.0,
  "backlogs": 0,
  "entrance_exam_score": 85.0,
  "technical_skill_score": 80.0,
  "soft_skill_score": 75.0,
  "certifications": 3,
  "live_projects": 1,
  "internship_count": 2,
  "work_experience_months": 6,
  "gender": "Male",
  "extracurricular_activities": "Yes"
}
```

**Example response (`200 OK`):**
```json
{
  "placement_status": 1,
  "placement_label": "Placed",
  "probability_placed": 0.938,
  "probability_not_placed": 0.062,
  "risk_level": "High Probability of Placement (Low Risk)"
}
```

**Error responses:**

| Code | Cause |
|---|---|
| `422` | Invalid field type, out-of-range value, or missing required field |
| `503` | Model artifacts not loaded (check `GET /health`) |
| `500` | Unexpected inference error |

---

## Frontend Integration

The Streamlit frontend (`frontend/app.py`) communicates with the backend
using the `requests` library:

```python
BACKEND_URL = "http://localhost:8000/api/v1/predict"

response = requests.post(BACKEND_URL, json=payload, timeout=10)
result   = response.json()
```

The `payload` dict built by `render_sidebar()` contains exactly the 15
fields required by `StudentInput`. No preprocessing is done client-side —
all scaling and encoding happens server-side inside the API.

**Response fields consumed by the frontend:**

| Field | Used for |
|---|---|
| `probability_placed` | `st.metric` display + Plotly gauge chart (0–100 %) |
| `placement_label` | Status line: `"Placed"` / `"Not Placed"` |
| `risk_level` | Risk label display + base of recommendation text |

---

## Model Replacement

The current production model is **Random Forest** (`random_forest_best.joblib`).

The predictor abstraction (`BasePredictor` ABC in `api/predictor.py`) allows
the model backend to be replaced without changing the API contract.

**To replace Random Forest with XGBoost (or any other model):**

1. Train and save the new model as a `.joblib` file.
2. Create a new class in `api/predictor.py` that inherits from `BasePredictor`:
   ```python
   class XGBoostPredictor(BasePredictor):
       @classmethod
       def load(cls, preprocessor_path, model_path): ...
       def predict(self, data: StudentInput) -> PredictionResponse: ...
       @property
       def is_ready(self) -> bool: ...
   ```
3. Update `api/config.py` to point `MODEL_PATH` at the new artifact.
4. In `api/main.py`, swap `RandomForestPredictor` for `XGBoostPredictor`.

The `StudentInput` → `PredictionResponse` API contract, the preprocessing
pipeline, and the Streamlit frontend remain **unchanged** provided the
replacement model:
- Accepts the same 17-column preprocessed NumPy array as input.
- Exposes `.predict()` and `.predict_proba()` (scikit-learn-compatible API).

---

## Testing

### Start the server

```powershell
venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000
```

---

### `GET /health` — liveness check

```powershell
curl http://localhost:8000/health
```

Expected:
```json
{"status":"healthy","preprocessor_loaded":true,"model_loaded":true}
```

---

### Swagger UI — interactive browser testing

Navigate to:
```
http://localhost:8000/docs
```

Click **POST /api/v1/predict → Try it out → Execute**.

ReDoc (read-only reference):
```
http://localhost:8000/redoc
```

---

### `POST /api/v1/predict` — curl (PowerShell)

**Strong candidate (expected: Placed):**
```powershell
curl -X POST http://localhost:8000/api/v1/predict `
  -H "Content-Type: application/json" `
  -d '{
    "ssc_percentage":85.0,"hsc_percentage":88.0,"degree_percentage":80.0,
    "cgpa":9.0,"attendance_percentage":95.0,"backlogs":0,
    "entrance_exam_score":90.0,"technical_skill_score":88.0,
    "soft_skill_score":82.0,"certifications":4,"live_projects":3,
    "internship_count":2,"work_experience_months":6,
    "gender":"Male","extracurricular_activities":"Yes"
  }'
```

**Weak candidate (expected: Not Placed):**
```powershell
curl -X POST http://localhost:8000/api/v1/predict `
  -H "Content-Type: application/json" `
  -d '{
    "ssc_percentage":45.0,"hsc_percentage":48.0,"degree_percentage":42.0,
    "cgpa":4.5,"attendance_percentage":60.0,"backlogs":5,
    "entrance_exam_score":35.0,"technical_skill_score":30.0,
    "soft_skill_score":28.0,"certifications":0,"live_projects":0,
    "internship_count":0,"work_experience_months":0,
    "gender":"Female","extracurricular_activities":"No"
  }'
```

**Validation error test (expected: 422):**
```powershell
curl -X POST http://localhost:8000/api/v1/predict `
  -H "Content-Type: application/json" `
  -d '{"cgpa": 15.0}'
```

---

### Python script

```python
import requests

payload = {
    "ssc_percentage": 75.5, "hsc_percentage": 78.0,
    "degree_percentage": 72.0, "cgpa": 8.2,
    "attendance_percentage": 90.0, "backlogs": 0,
    "entrance_exam_score": 85.0, "technical_skill_score": 80.0,
    "soft_skill_score": 75.0, "certifications": 3,
    "live_projects": 1, "internship_count": 2,
    "work_experience_months": 6,
    "gender": "Male", "extracurricular_activities": "Yes",
}

r = requests.post("http://localhost:8000/api/v1/predict", json=payload)
print(r.json())
```

---

## Preprocessor Details

The `preprocessor.joblib` artifact is a fitted scikit-learn `ColumnTransformer`
produced by `preprocessing.py` during the training pipeline.

| Branch | Transformer | Input columns | Output columns |
|---|---|---|---|
| `numerical` | `StandardScaler` | 13 numerical features | 13 z-scored floats |
| `categorical` | `OneHotEncoder` | `gender`, `extracurricular_activities` | 4 dummy columns |
| **Total** | | **15** raw features | **17** processed features |

**Output column names (in order):**
```
numerical__ssc_percentage
numerical__hsc_percentage
numerical__degree_percentage
numerical__cgpa
numerical__entrance_exam_score
numerical__technical_skill_score
numerical__soft_skill_score
numerical__internship_count
numerical__live_projects
numerical__work_experience_months
numerical__certifications
numerical__attendance_percentage
numerical__backlogs
categorical__gender_Female
categorical__gender_Male
categorical__extracurricular_activities_No
categorical__extracurricular_activities_Yes
```

The `ColumnTransformer` selects columns **by name**, so the column order
in the one-row inference DataFrame does not need to match the transformer's
internal order.
