# Student Placement Prediction

A production-grade Machine Learning system that predicts whether a student is likely to be placed, based on academic performance, technical skills, soft skills, internships, projects, work experience, certifications, attendance, and backlogs.

[![CI](https://github.com/alfreddevy2020-kch/StudentPlacementPrediction/actions/workflows/ci.yml/badge.svg)](https://github.com/alfreddevy2020-kch/StudentPlacementPrediction/actions/workflows/ci.yml)

---

## 📌 Project Overview

The system follows a complete end-to-end Machine Learning lifecycle:

```
Raw Dataset → EDA → Visualisation → Preprocessing
    → Model Training (LR · RF · XGBoost)
    → Explainability & Fairness (SHAP)
    → Artifact Packaging → REST API → Streamlit Dashboard
    → CI/CD (GitHub Actions) → Cloud Deployment (Render · Streamlit Cloud)
    → Prediction Logging (SQLite) → Drift Monitoring (PSI)
```

---

## 🎯 Objective

Build a machine learning system that predicts:

- `0` → Not Placed
- `1` → Placed

The system helps placement cells identify students who need additional preparation, and provides explainable, auditable predictions.

---

## 📊 Dataset

The dataset contains **5,000 student records** with 18 columns.

### Features Used (15 input features)

| Feature | Type | Description |
|---|---|---|
| `gender` | Categorical | Student gender |
| `ssc_percentage` | Numerical | Secondary school (10th) percentage |
| `hsc_percentage` | Numerical | Higher secondary (12th) percentage |
| `degree_percentage` | Numerical | Undergraduate degree percentage |
| `cgpa` | Numerical | College CGPA (0–10 scale) |
| `attendance_percentage` | Numerical | College attendance percentage |
| `backlogs` | Numerical | Number of active academic backlogs |
| `entrance_exam_score` | Numerical | Entrance examination score |
| `technical_skill_score` | Numerical | Technical / coding assessment score |
| `soft_skill_score` | Numerical | Soft skills assessment score |
| `certifications` | Numerical | Number of professional certifications |
| `live_projects` | Numerical | Number of capstone/live projects |
| `internship_count` | Numerical | Number of internships completed |
| `work_experience_months` | Numerical | Prior work experience in months |
| `extracurricular_activities` | Categorical | Participation in extracurricular activities |

### Removed Columns

| Column | Reason |
|---|---|
| `student_id` | Identifier only — no predictive value |
| `salary_package_lpa` | Known only after placement → data leakage |

### Class Distribution

| Class | Count | Percentage |
|---|---|---|
| Not Placed | 4,134 | 82.68 % |
| Placed | 866 | 17.32 % |

The target is **imbalanced** — all models use `scale_pos_weight` / class weighting and are evaluated on F1/PR-AUC, not accuracy.

---

## 🗂️ Repository Structure

```
StudentPlacementPrediction/
├── .github/
│   └── workflows/
│       └── ci.yml                  ← GitHub Actions CI pipeline
│
├── api/                            ← FastAPI prediction service
│   ├── config.py                   ← artifact registry & app metadata
│   ├── drift.py                    ← PSI drift detector
│   ├── logger.py                   ← SQLite prediction logger
│   ├── main.py                     ← FastAPI app + endpoints
│   ├── predictor.py                ← model loader/predictor classes
│   └── schemas.py                  ← Pydantic request/response schemas
│
├── artifacts/
│   └── production/                 ← ✅ committed — reviewed model bundles
│       ├── logistic_regression/    (model.joblib · preprocessor.joblib · manifest.json · baseline_metrics.json)
│       ├── random_forest/          (model.joblib · preprocessor.joblib · manifest.json · baseline_metrics.json)
│       └── xgboost/                (model.joblib · preprocessor.joblib · manifest.json · baseline_metrics.json)
│
├── docs/
│   └── DEPLOYMENT.md               ← deployment guide & retraining policy
│
├── frontend/
│   └── app.py                      ← Streamlit dashboard
│
├── logs/
│   └── .gitkeep                    ← predictions.db written here at runtime (gitignored)
│
├── part2/                          ← Logistic Regression & Random Forest training
├── part3/                          ← XGBoost training
├── part4/                          ← SHAP explainability & fairness audit
│
├── scripts/
│   ├── package_model.py            ← packages a model into artifacts/production/
│   └── smoke_test_models.py        ← CI model health check
│
├── tests/
│   ├── test_schemas.py             ← Pydantic validation tests
│   ├── test_drift.py               ← PSI logic tests
│   ├── test_logger.py              ← SQLite logger tests
│   └── test_api.py                 ← FastAPI integration tests
│
├── .streamlit/
│   └── secrets.toml                ← BACKEND_URL (gitignored)
│
├── pyproject.toml                  ← Project metadata, Ruff linter & Pytest config
├── render.yaml                     ← Render free-tier deployment blueprint
├── requirements.txt                ← full dependencies
├── requirements-ci.txt             ← minimal CI dependencies (CPU-only)
└── SETUP.md                        ← detailed local setup guide
```

---

## 🤖 Part 2 — Model Training (Logistic Regression + Random Forest)

All Part 2 files live in `part2/`.

### Quick Start

```bash
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
part2\run_pipeline.bat      # downloads data → preprocesses → trains → evaluates
```

### Key Results

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| ROC-AUC | 0.9354 | 1.0000 |
| F1 Score (tuned threshold) | 0.6949 | 1.0000 |
| Avg Precision (PR-AUC) | 0.7718 | 1.0000 |
| Brier Score | 0.1110 | 0.0042 |
| Optimal threshold | 0.773 | 0.173 |

**Top predictive features (Permutation Importance, RF):** `backlogs` · `cgpa` · `technical_skill_score` · `soft_skill_score`

---

## ⚡ Part 3 — XGBoost Model

All Part 3 files live in `part3/`.

### Quick Start

```bash
python part3/xgboost_model.py
# or
part3\run_pipeline.bat
```

XGBoost is trained with CUDA GPU acceleration (falls back to CPU automatically). `scale_pos_weight` handles class imbalance. Hyperparameter tuning uses `RandomizedSearchCV` → `GridSearchCV`.

---

## 🔍 Part 4 — Explainability & Fairness

All Part 4 files live in `part4/`.

### Quick Start

```bash
python download_dataset.py
python preprocessing.py
python part3/xgboost_model.py
part4\run_pipeline.bat
```

| Technique | Purpose |
|---|---|
| SHAP TreeExplainer | Local + global explanations — why THIS student got their score |
| Platt scaling / Isotonic regression | Calibrate raw XGBoost scores into trustworthy probabilities |
| Group-wise FNR audit | Compare false negative rates across gender & extracurricular groups |
| Mitigation report | Proposed fixes if disparity detected |

---

## 🚀 Part 6 — REST API (FastAPI)

### Architecture

```
Streamlit Frontend
      │ POST /api/v1/predict (JSON)
      ▼
FastAPI  ─── Pydantic validation
      │
      ▼
ColumnTransformer (StandardScaler + OneHotEncoder)
      │
      ▼
Classifier (LR / RF / XGBoost — selectable per request)
      │
      ▼
JSON response  +  SQLite prediction log
```

### Running Locally

```bash
# Terminal 1 — API
venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000

# Terminal 2 — Dashboard
venv\Scripts\streamlit run frontend\app.py
```

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — confirms all 3 models are loaded |
| `GET` | `/api/v1/models` | List available model identifiers |
| `POST` | `/api/v1/predict` | Predict placement (body: 15 features + model choice) |
| `GET` | `/api/v1/drift` | PSI drift report for a model (`?model=xgboost&window=200`) |
| `GET` | `/logs/summary` | Total prediction counts per model |

Interactive docs: http://localhost:8000/docs

### Selecting a Model

Every prediction request includes a `model` field:

```json
{
  "model": "xgboost",       // "logistic_regression" | "random_forest" | "xgboost"
  "cgpa": 8.2,
  "ssc_percentage": 75.5,
  ...
}
```

All three models are loaded at startup and served from the same API.

---

## 📦 Artifact Packaging

Production artifacts live in `artifacts/production/` and are **committed to git** (allowed by `.gitignore` negation rules). Each model bundle contains:

```
artifacts/production/<model_name>/
├── model.joblib          ← trained classifier
├── preprocessor.joblib   ← fitted ColumnTransformer
├── manifest.json         ← training metadata (date, features, params)
└── baseline_metrics.json ← F1, ROC-AUC, mean probability (used by drift checker)
```

To package a newly trained model:

```bash
python scripts/package_model.py \
    --model part3/models/xgboost_best.joblib \
    --preprocessor part3/models/preprocessor.joblib \
    --output-dir artifacts/production/xgboost \
    --overwrite
```

---

## 🔄 CI/CD — GitHub Actions

Every push and pull request to `main` triggers the CI pipeline defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

| Step | Tool | Blocks merge? |
|---|---|:---:|
| Lint | `ruff check .` | ✅ Yes |
| Type check | `pyright` | ⚠️ Advisory |
| Unit tests | `pytest tests/` | ✅ Yes |
| Model smoke test | `python scripts/smoke_test_models.py` | ✅ Yes |

The smoke test loads all three production `.joblib` bundles and runs a sample prediction through each. **A broken model artifact cannot be merged.**

### Running Linting & Tests Locally

```bash
# 1. Install test and dev dependencies
pip install pytest httpx pytest-asyncio ruff pyright

# 2. Run linter & formatter
ruff check .
ruff format . --check

# 3. Run unit tests (no artifacts needed)
pytest tests/test_schemas.py tests/test_drift.py tests/test_logger.py -v

# 4. Run full integration tests (requires artifacts/production/)
pytest tests/ -v

# 5. Run model smoke test
python scripts/smoke_test_models.py
```

---

## 📊 Prediction Logging & Drift Monitoring

### Prediction Logging

Every successful prediction is recorded in `logs/predictions.db` (SQLite, WAL mode, thread-safe).

```bash
# Check log counts
curl http://localhost:8000/logs/summary
# → {"total": 42, "by_model": {"xgboost": 30, "random_forest": 10, "logistic_regression": 2}}
```

### Drift Detection (PSI)

```bash
curl "http://localhost:8000/api/v1/drift?model=xgboost&window=200"
# → {"status": "ok", "psi": 0.042, "mean_shift": 0.018, ...}
```

| Status | Condition | Action |
|---|---|---|
| `ok` | PSI < 0.10, shift < 0.05 | None |
| `warn` | PSI 0.10–0.20 or shift 0.05–0.10 | Monitor closely |
| `alert` | PSI > 0.20 or shift > 0.10 | Initiate retraining review |
| `insufficient_data` | < 20 predictions logged | Accumulate more traffic |

---

## ☁️ Cloud Deployment

### Backend API — Render (Free Tier, $0/month)

1. Push this repo to GitHub (artifacts must be committed).
2. Go to [render.com](https://render.com) → **New → Blueprint**.
3. Connect your GitHub repo — Render auto-detects `render.yaml`.
4. The API will be live at `https://student-placement-api.onrender.com`.

> Free tier sleeps after 15 min of inactivity (~30 s cold start). See `docs/DEPLOYMENT.md` for upgrade options.

### Frontend Dashboard — Streamlit Community Cloud (Free Tier, $0/month)

1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
2. Connect your GitHub repo, set main file to `frontend/app.py`.
3. Under **Secrets**, add:
   ```toml
   BACKEND_URL = "https://student-placement-api.onrender.com/api/v1/predict"
   ```
4. Click **Deploy**.

Full deployment guide and retraining policy: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)

---

## 🔁 Retraining Policy

| Trigger | Threshold |
|---|---|
| Drift alert | PSI > 0.20 OR mean shift > 0.10 for 3+ consecutive days |
| F1 regression | F1 drops below 0.80 on held-out validation set |
| Data volume | New labeled data > 20 % of original training set |
| Calendar | Every academic semester (~6 months) |

See `docs/DEPLOYMENT.md` for the full retraining procedure.

---

## 🛠️ Local Setup

See **[SETUP.md](SETUP.md)** for detailed step-by-step setup instructions.

**Quick start:**

```bash
# Clone & set up environment
git clone https://github.com/alfreddevy2020-kch/StudentPlacementPrediction.git
cd StudentPlacementPrediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Download dataset
python download_dataset.py

# Start the API
venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000

# Start the dashboard (new terminal)
venv\Scripts\streamlit run frontend\app.py
```

---

## 📋 Tech Stack

| Layer | Technology |
|---|---|
| ML Models | scikit-learn (LR, RF), XGBoost |
| Explainability | SHAP |
| API | FastAPI + Pydantic v2 + uvicorn |
| Frontend | Streamlit + Plotly |
| Prediction Logging | SQLite (WAL mode) |
| Drift Detection | PSI (Population Stability Index) |
| CI/CD | GitHub Actions |
| Linting & Formatting | Ruff |
| Type checking | pyright |
| Testing | pytest + httpx |
| Deployment (API) | Render free tier |
| Deployment (UI) | Streamlit Community Cloud |

---

## 🔒 Data & Privacy

The following are **never committed** to git:

- Raw student data (`data/raw/`)
- Processed datasets (`data/processed/`)
- Student IDs or PII
- API tokens or secrets (`.env`, `.streamlit/secrets.toml`)
- Prediction logs (`logs/predictions.db`, `*.sqlite`, `*.db`)
- Experimental / unapproved model artifacts

Only explicitly reviewed production artifacts in `artifacts/production/` are committed.