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
        │  Pydantic validates 10 raw fields
        ▼
StudentInput  (api/schemas.py)
        │
        │  _input_to_dataframe()  →  pd.DataFrame (1 row × 10 raw columns)
        │  engineer_features()    →  + 21 derived columns  (31 total)
        ▼
preprocessor.joblib  (ColumnTransformer)
        │  StandardScaler   →  29 numerical features  (z-scored)
        │  OneHotEncoder    →   2 categorical features (4 dummy columns)
        │  Output: NumPy array (1 × 33)
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

10 raw student features expected by `POST /api/v1/predict`.
`student_id` and `placement_status` are **not accepted** (the latter is the
target itself).

**Numerical fields (8) — validated with `ge` / `le` bounds:**

Accepted bounds are the widest sensible range per field. The model's actual
training range is narrower (the "Trained on" column); values outside it
validate successfully but are extrapolation.

| Field | Type | Accepted | Trained on | Description |
|---|---|---|---|---|
| `cgpa` | `float` | 0–10 | 6.5–9.1 | College CGPA on a 10-point scale |
| `ssc_marks` | `float` | 0–100 | 55–90 | Secondary school (class 10) percentage |
| `hsc_marks` | `float` | 0–100 | 57–88 | Higher secondary (class 12) percentage |
| `aptitude_test_score` | `float` | 0–100 | 60–90 | Aptitude / mock-test score |
| `soft_skills_rating` | `float` | 0–5 | 3.0–4.8 | Soft-skills rating on a 5-point scale |
| `internships` | `int` | 0–10 | 0–2 | Internships completed |
| `projects` | `int` | 0–20 | 0–3 | Projects completed |
| `workshops_certifications` | `int` | 0–20 | 0–3 | Workshops / certifications earned |

**Categorical fields (2) — validated with `Literal`:**

| Field | Type | Accepted values |
|---|---|---|
| `extracurricular_activities` | `str` | `"Yes"`, `"No"` |
| `placement_training` | `str` | `"Yes"`, `"No"` |

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

Defined once in `feature_engineering.py` and imported everywhere else —
never redeclared locally.

```python
RAW_NUMERICAL_FEATURES = [
    "cgpa", "ssc_marks", "hsc_marks",
    "aptitude_test_score", "soft_skills_rating",
    "internships", "projects", "workshops_certifications",
]

RAW_CATEGORICAL_FEATURES = ["extracurricular_activities", "placement_training"]

ALL_RAW_FEATURES = RAW_NUMERICAL_FEATURES + RAW_CATEGORICAL_FEATURES  # 10 total
```

`engineer_features()` then adds 21 derived columns, giving the 29 numerical
+ 2 categorical (33 after one-hot) that the fitted preprocessor expects.

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

Main inference endpoint. Accepts 10 raw student features and returns a
placement prediction.

**Example request body:**
```json
{
  "model": "random_forest",
  "cgpa": 7.7,
  "ssc_marks": 70.0,
  "hsc_marks": 74.0,
  "aptitude_test_score": 80.0,
  "soft_skills_rating": 4.4,
  "internships": 1,
  "projects": 2,
  "workshops_certifications": 1,
  "extracurricular_activities": "Yes",
  "placement_training": "Yes"
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

**Strong candidate (expected: Placed, ~0.93):**
```powershell
curl -X POST http://localhost:8000/api/v1/predict `
  -H "Content-Type: application/json" `
  -d '{
    "cgpa":8.9,"ssc_marks":78.0,"hsc_marks":82.0,
    "aptitude_test_score":90.0,"soft_skills_rating":4.6,
    "internships":2,"projects":3,"workshops_certifications":3,
    "extracurricular_activities":"Yes","placement_training":"Yes"
  }'
```

**Weak candidate (expected: Not Placed, ~0.01):**
```powershell
curl -X POST http://localhost:8000/api/v1/predict `
  -H "Content-Type: application/json" `
  -d '{
    "cgpa":6.6,"ssc_marks":56.0,"hsc_marks":58.0,
    "aptitude_test_score":61.0,"soft_skills_rating":3.1,
    "internships":0,"projects":0,"workshops_certifications":0,
    "extracurricular_activities":"No","placement_training":"No"
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
    "model": "random_forest",
    "cgpa": 7.7, "ssc_marks": 70.0, "hsc_marks": 74.0,
    "aptitude_test_score": 80.0, "soft_skills_rating": 4.4,
    "internships": 1, "projects": 2, "workshops_certifications": 1,
    "extracurricular_activities": "Yes", "placement_training": "Yes",
}

r = requests.post("http://localhost:8000/api/v1/predict", json=payload)
print(r.json())
```

---

## Preprocessor Details

The `preprocessor.joblib` artifact is a fitted scikit-learn `ColumnTransformer`
produced by `preprocessing.py` during the training pipeline.

The transformer is fitted on the **engineered** frame, not the raw request
body: `engineer_features()` expands the 10 raw fields into 29 numerical
columns before the `ColumnTransformer` runs.

| Branch | Transformer | Input columns | Output columns |
|---|---|---|---|
| `numerical` | `StandardScaler` | 29 numerical features (8 raw + 21 engineered) | 29 z-scored floats |
| `categorical` | `OneHotEncoder` | `extracurricular_activities`, `placement_training` | 4 dummy columns |
| **Total** | | **10** raw fields → 31 engineered | **33** processed features |

**Output column names:** the 8 raw numerical features, then the 21 in
`ENGINEERED_NUMERICAL_FEATURES` (each prefixed `numerical__`), then the
four `categorical__*_No` / `categorical__*_Yes` dummies. The authoritative
list is `preprocessor.get_feature_names_out()`.

The `ColumnTransformer` selects columns **by name**, so the column order
in the one-row inference DataFrame does not need to match the transformer's
internal order.
