# SETUP.md — Fresh Clone to Running System

> **Purpose:** the single source of truth for taking a fresh clone from zero
> to a working system. Everything here is verified against the current
> codebase.
>
> **Related docs:**
>
> | Topic | Doc |
> |---|---|
> | Dataset choice, schema, feature engineering | `SCHEMA.md` |
> | Project overview & model results | `README.md` |
> | REST API contract | `BACKEND_API_DOCUMENTATION.md` |
> | Cloud deployment (Render / Streamlit Cloud) | `docs/DEPLOYMENT.md` |
> | XGBoost workstream | `part3/work.md` |
> | Explainability & fairness workstream | `part4/work.md` |
> | Frontend data flow | `frontend/flow.md` |

---

## 0. TL;DR

```bash
git clone https://github.com/alfreddevy2020-kch/StudentPlacementPrediction.git
cd StudentPlacementPrediction

python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt      # Windows
# venv/bin/python -m pip install -r requirements.txt            # Mac/Linux

venv\Scripts\python.exe download_dataset.py     # Kaggle -> data/raw/
venv\Scripts\python.exe preprocessing.py        # REQUIRED — see Section 5
venv\Scripts\python.exe scripts\train_models_fast.py
```

That gets you a working dashboard and a correct API. Details below.

---

## 1. What ships in the repo vs. what you generate

This is the section that explains every "it doesn't work on my machine".

### Committed — present on a fresh clone

| Path | Notes |
|---|---|
| `artifacts/production/{model}/model.joblib` | The three trained models the **API** serves |
| `artifacts/production/{model}/preprocessor.joblib` | Fitted `ColumnTransformer` |
| `artifacts/production/{model}/normalization_stats.json` | Frozen scaling maxima for that model |
| `artifacts/production/{model}/manifest.json` | Version + SHA-256 checksums, verified at startup |
| `artifacts/production/{model}/baseline_metrics.json` | Reference distribution for drift monitoring |
| `part3/models/xgboost_best.json` | XGBoost native format (not `.joblib`, so not ignored) |
| `part3/model_results/*` | 10 files — plots + CSVs |
| `part8/results/*` | 4 files — evaluation outputs |
| `visualizations/*` | 14 EDA PNGs |

`.gitignore` ignores `*.joblib` globally, then re-allows
`artifacts/production/**/*.joblib` — that negation is deliberate, it is what
lets the API run on a fresh clone.

### Generated — you must create these

| Path | Created by | Needed for |
|---|---|---|
| `venv/` | Section 3 | everything |
| `data/raw/student_placement.csv` | `download_dataset.py` | everything downstream |
| `data/processed/*.csv` | `preprocessing.py` | model training, part4, part8 |
| `data/processed/normalization_stats.json` | `preprocessing.py` | dashboard predictions, and packaging new bundles (Section 5) |
| `part2/models/*.joblib` | Section 6 | dashboard, part8 |
| `part2/model_results/*` | Section 10 | Part 2 reporting |
| `part3/models/xgboost_best.joblib` | Section 6 | dashboard, part4, part8 |
| `part4/explainability_results/*` | Section 10 | Part 4 reporting |
| `logs/predictions.db` | created on first API call | prediction logging (self-creating) |

### The two entry points have different requirements

They are **independent processes**. The dashboard does **not** call the API —
it loads model artifacts directly from disk via `frontend/batch_predictor.py`.
There is no HTTP between them.

| | Reads from | Works on a bare clone? |
|---|---|---|
| **FastAPI** (`api.main:app`) | `artifacts/production/` (committed) | **Yes** — bundles are self-contained |
| **Streamlit** (`frontend/app.py`) | `part2/models/`, `part3/models/`, `data/raw/` (all ignored) | No — needs Sections 4–6 |

> `.streamlit/secrets.toml` contains a `BACKEND_URL` key. It is a leftover
> from an earlier architecture and is not read by the current dashboard.

---

## 2. Prerequisites

### 2a. Python

`pyproject.toml` declares `requires-python = ">=3.9"`. CI validates on
**3.11**. Development has also been done on 3.13.

```bash
python --version        # Windows
python3 --version       # Mac/Linux
```

If your default interpreter is older than 3.9, point venv creation at a
newer binary — a venv does not change the interpreter version:

```bash
py -3.11 -m venv venv          # Windows
python3.11 -m venv venv        # Mac/Linux
```

### 2b. Kaggle account (free)

`download_dataset.py` uses `kagglehub`. **No manual API key setup is
needed** — the dataset is public, and if credentials are ever required
`kagglehub` prompts in the terminal with a browser link.

> **Fallback**, if the interactive prompt fails:
> 1. kaggle.com → **Settings** → **API** → **Create New Token**
> 2. Save `kaggle.json` to `C:\Users\<you>\.kaggle\kaggle.json` (Windows)
>    or `~/.kaggle/kaggle.json` (Mac/Linux)
> 3. Mac/Linux only: `chmod 600 ~/.kaggle/kaggle.json`

### 2c. GPU (optional, Part 3 only)

`part3/xgboost_model.py` auto-detects CUDA and falls back to CPU
automatically. **Not required.**

---

## 3. Clone, create the venv, install

```bash
git clone https://github.com/alfreddevy2020-kch/StudentPlacementPrediction.git
cd StudentPlacementPrediction
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Mac/Linux:**
```bash
python3 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.txt
```

Activating (`venv\Scripts\activate` / `source venv/bin/activate`) is
optional — every command below can be prefixed with the venv python path
instead. **All examples use the explicit path**, because the single most
common setup failure is running with the system Python.

### What `requirements.txt` covers

| Group | Packages |
|---|---|
| Core ML | `numpy`, `pandas`, `scikit-learn==1.6.1`, `scipy`, `joblib`, `imbalanced-learn` |
| Models | `xgboost` |
| Data | `kagglehub` |
| Plots | `matplotlib`, `seaborn`, `plotly` |
| API | `fastapi`, `uvicorn`, `pydantic`, `httpx` |
| Dashboard | `streamlit`, `plotly` |
| Explainability | `shap` |
| Dev/test | `pytest`, `pytest-asyncio`, `ruff` |

`requirements-ci.txt` is a smaller pinned subset used only by GitHub Actions.

---

## 4. Download the dataset

```bash
venv\Scripts\python.exe download_dataset.py      # Windows
venv/bin/python download_dataset.py              # Mac/Linux
```

Downloads `ruchikakumbhar/placement-prediction-dataset` and writes it to
`data/raw/student_placement.csv` (**10,000 rows × 12 columns**).

See `SCHEMA.md` for why this dataset was chosen over the alternatives.

**Verify:**
```bash
# Windows
if exist data\raw\student_placement.csv (echo RAW OK) else (echo MISSING)
# Mac/Linux
test -f data/raw/student_placement.csv && echo "RAW OK" || echo "MISSING"
```

---

## 5. Preprocess

```bash
venv\Scripts\python.exe preprocessing.py      # Windows
venv/bin/python preprocessing.py              # Mac/Linux
```

Required before training anything, and before the **dashboard** will work.
The **API** does not need it — each production bundle ships its own copy of
the normalization stats (see below).

### About `normalization_stats.json`

`preprocessing.py` writes `data/processed/normalization_stats.json`, holding
the training-set maxima used to scale the count features. `data/processed/`
is gitignored, so that file is absent on a fresh clone.

Those maxima are a fitted parameter of the pipeline, exactly like
`preprocessor.joblib`, so `scripts/package_model.py` copies them into each
production bundle and records their SHA-256 in the manifest. The API loads
the per-bundle copy, which keeps a served model pinned to the scale it was
trained on and makes bundles self-contained.

If no stats are available at all, `engineer_features()` **raises**. It will
not infer a maximum from the batch it is given: for a single row that
divides each value by itself, pinning every `*_normalized` feature to 1.0
and reporting `portfolio_strength` as 100 instead of 50 — wrong answers with
no error. `tests/test_normalization_stats.py` locks this behaviour in.

### What it does

1. Loads `data/raw/student_placement.csv` and snake_cases the headers
2. Drops `student_id`; this dataset has no post-outcome column, so the text
   target is the only other field removed
3. Engineers 21 derived features on top of the 8 raw numerical ones
4. Stratified 80/20 split (`random_state=42`)
5. Fits a `ColumnTransformer` **on the training split only**:
   `StandardScaler` over 29 numerical, `OneHotEncoder` over 2 categorical
6. Reports class balance (58.03% Not Placed / 41.97% Placed) and SMOTE stats

### Outputs

```
data/processed/train_processed.csv        8,000 × 34  (33 features + target)
data/processed/test_processed.csv         2,000 × 34
data/processed/class_weights.csv
data/processed/normalization_stats.json   <- also copied into each bundle
part2/models/preprocessor.joblib
```

---

## 6. Train the models

### Fast path (recommended) — Random Forest + XGBoost

```bash
venv\Scripts\python.exe scripts\train_models_fast.py      # Windows
venv/bin/python scripts/train_models_fast.py              # Mac/Linux
```

Trains in seconds and writes:
```
part2/models/random_forest_best.joblib
part3/models/xgboost_best.joblib
part3/models/xgboost_best.json
part3/models/preprocessor.joblib      (synced copy)
```

> This script does **not** produce Logistic Regression. Add it with:
> `venv\Scripts\python.exe part2\logistic_regression_model.py`
> (a few minutes — it runs a full `GridSearchCV` plus bootstrap CIs).

The dashboard tolerates missing models and shows whichever are present, so
the fast path alone is enough to get it running.

### Current held-out performance (n = 2,000)

| Model | ROC-AUC | Accuracy | F1 |
|---|---|---|---|
| Logistic Regression | 0.8836 | 0.7965 | 0.7697 |
| Random Forest | 0.8750 | 0.7890 | 0.7564 |
| XGBoost | 0.8684 | 0.7855 | 0.7542 |

### Refreshing the API's committed artifacts

The `artifacts/production/` bundles are only rebuilt on demand:

```bash
venv\Scripts\python.exe scripts\package_model.py --model-name random_forest ^
  --model-version 2026.08.19-rf.2 ^
  --preprocessor part2\models\preprocessor.joblib ^
  --model part2\models\random_forest_best.joblib --overwrite

venv\Scripts\python.exe scripts\generate_baseline_metrics.py
```

`package_model.py` recomputes the SHA-256 checksums in `manifest.json`;
`generate_baseline_metrics.py` refreshes the drift-monitoring baselines from
the current test split. Run both after retraining, or the API's startup
checksum verification will reject the bundle.

---

## 7. Verify you are ready

### Windows (PowerShell)

```powershell
@(
  "data\raw\student_placement.csv",
  "data\processed\train_processed.csv",
  "data\processed\test_processed.csv",
  "data\processed\class_weights.csv",
  "data\processed\normalization_stats.json",
  "part2\models\preprocessor.joblib",
  "part2\models\random_forest_best.joblib",
  "part3\models\xgboost_best.joblib"
) | ForEach-Object {
  if (Test-Path $_) { Write-Host "OK       $_" }
  else { Write-Host "MISSING  $_" -ForegroundColor Red }
}
```

### Mac/Linux (Bash)

```bash
for f in \
  data/raw/student_placement.csv \
  data/processed/train_processed.csv \
  data/processed/test_processed.csv \
  data/processed/class_weights.csv \
  data/processed/normalization_stats.json \
  part2/models/preprocessor.joblib \
  part2/models/random_forest_best.joblib \
  part3/models/xgboost_best.joblib; do
  [ -f "$f" ] && echo "OK       $f" || echo "MISSING  $f"
done
```

---

## 8. Run the API

```bash
venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000   # Windows
venv/bin/python -m uvicorn api.main:app --reload --port 8000           # Mac/Linux
```

Startup verifies each bundle: manifest present, `model_name` matches,
SHA-256 checksums match, the preprocessor transforms a sample row, and the
model exposes `predict` / `predict_proba`. Expect:

```
[API] [OK] Logistic Regression loaded: model.joblib
[API] [OK] Random Forest loaded: model.joblib
[API] [OK] XGBoost loaded: model.joblib
[API] [OK] All 3/3 models loaded successfully.
[API] Drift checker initialised.
```

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | `{"status", "models_loaded"}` |
| GET | `/api/v1/models` | list available model identifiers |
| POST | `/api/v1/predict` | placement prediction |
| GET | `/api/v1/drift?model=<name>` | PSI drift report vs. baseline |
| GET | `/logs/summary` | prediction-log counts |

```bash
curl http://localhost:8000/health
# {"status":"healthy","models_loaded":{"logistic_regression":true,"random_forest":true,"xgboost":true}}

curl -X POST http://localhost:8000/api/v1/predict -H "Content-Type: application/json" -d "{\"cgpa\":7.7,\"ssc_marks\":70.0,\"hsc_marks\":74.0,\"aptitude_test_score\":80.0,\"soft_skills_rating\":4.4,\"internships\":1,\"projects\":2,\"workshops_certifications\":1,\"extracurricular_activities\":\"Yes\",\"placement_training\":\"Yes\"}"
```

Swagger UI: <http://localhost:8000/docs>

Full field reference and bounds: `BACKEND_API_DOCUMENTATION.md`.

---

## 9. Run the dashboard

```bash
venv\Scripts\streamlit run frontend\app.py      # Windows
venv/bin/streamlit run frontend/app.py          # Mac/Linux
```

Opens at <http://localhost:8501>.

> **The API does not need to be running.** The dashboard loads models
> directly from `part2/models/` and `part3/models/` through
> `frontend/batch_predictor.py`. Its requirement is Sections 4–6, not a
> live backend.

Three tabs: departmental pulse, per-student diagnostic and skill gaps, and
the cohort what-if simulator. The multi-model benchmark panel at the bottom
is behind a **Run benchmark evaluation** checkbox — it is not computed on
page load.

---

## 10. Optional workstreams

### Part 2 — full Logistic Regression + Random Forest reporting

```bash
part2\run_pipeline.bat        # Windows: download -> preprocess -> LR -> RF -> comparison -> summary
```

Mac/Linux (same order, individually):
```bash
venv/bin/python part2/logistic_regression_model.py
venv/bin/python part2/random_forest_model.py
venv/bin/python part2/model_comparison.py
venv/bin/python part2/model_summary_report.py
```

**Order matters.** `model_comparison.py` reads both `.joblib` models *and*
`logreg_metadata.csv` / `rf_metadata.csv`; `model_summary_report.py` reads
the outputs of all three. Skipping a step raises `FileNotFoundError`.

Standalone (not part of the pipeline, not needed by anything else):
```bash
venv\Scripts\python.exe part2\hp_sensitivity_analysis.py
```

### Part 3 — XGBoost with full tuning

```bash
part3\run_pipeline.bat        # Windows: xgboost_model.py -> predict_sample.py
venv/bin/python part3/xgboost_model.py       # Mac/Linux
```

Runs `RandomizedSearchCV` then `GridSearchCV`. Considerably slower than
`scripts/train_models_fast.py`; use it when you want the tuned model and the
full plot set.

### Part 4 — explainability & fairness (SHAP)

```bash
part4\run_pipeline.bat                                    # Windows
venv/bin/python part4/explainability_fairness.py          # Mac/Linux
```

Needs `data/processed/*`, `data/raw/*` and `part3/models/xgboost_best.joblib`.
Produces SHAP importance, calibration curves, and a group-fairness audit in
`part4/explainability_results/`.

> This dataset carries no demographic attributes, so the audit uses
> `placement_training` and `extracurricular_activities` as its equity axes —
> measuring fairness of *access to institutional support* rather than
> demographic parity. See `SCHEMA.md`.

### Part 8 — three-way evaluation suite

```bash
venv\Scripts\python.exe part8\evaluation_suite.py
```

Needs `data/processed/*` plus all three trained `.joblib` models
(`part2/models/`, `part3/models/`). Writes to `part8/results/`.

### Exploratory analysis

```bash
venv\Scripts\python.exe data_analysis.py     # console statistics
venv\Scripts\python.exe visualization.py     # regenerates visualizations/*.png
```

Both are driven by the canonical feature lists in `feature_engineering.py`,
so they need no per-column edits when the schema changes.

---

## 11. Tests and CI

```bash
venv\Scripts\python.exe -m pytest tests\ -q          # full suite
venv\Scripts\python.exe -m ruff check .              # lint
venv\Scripts\python.exe scripts\smoke_test_models.py # production bundle check
```

`.github/workflows/ci.yml` runs on pushes to `main` and PRs into `main`:

| Step | Blocking |
|---|---|
| `ruff check .` | yes |
| `pyright` | no — advisory |
| `pytest tests/ --ignore=tests/test_api.py` | yes |
| `python scripts/smoke_test_models.py` | yes |

`test_api.py` is excluded in CI and self-skips when production artifacts are
absent. It runs locally once `artifacts/production/` is populated — which it
is, out of the box.

---

## 12. Repository layout

```
StudentPlacementPrediction/
├── .github/workflows/ci.yml            ← lint · type-check · test · smoke-test
├── .streamlit/
│   ├── config.toml                     ← dark theme for the dashboard
│   └── secrets.toml                    ← legacy BACKEND_URL (unused)
│
├── api/                                ← FastAPI serving layer
│   ├── config.py                       ← artifact paths + OpenAPI metadata
│   ├── main.py                         ← app, routes, lifespan
│   ├── predictor.py                    ← bundle verification + inference
│   ├── schemas.py                      ← Pydantic request/response models
│   ├── logger.py                       ← SQLite prediction log
│   └── drift.py                        ← PSI drift monitoring
│
├── frontend/                           ← Streamlit dashboard (loads models directly)
│   ├── app.py                          ← 3-tab dashboard
│   ├── batch_predictor.py              ← cohort inference from disk artifacts
│   ├── simulator.py                    ← what-if policy engine
│   └── flow.md, AGENTS.md
│
├── artifacts/production/               ← [COMMITTED] what the API serves
│   └── {logistic_regression,random_forest,xgboost}/
│       ├── model.joblib
│       ├── preprocessor.joblib
│       ├── manifest.json               ← version + SHA-256
│       ├── normalization_stats.json    ← frozen scaling maxima
│       └── baseline_metrics.json       ← drift reference
│
├── scripts/
│   ├── train_models_fast.py            ← quick RF + XGB training
│   ├── package_model.py                ← build a production bundle
│   ├── generate_baseline_metrics.py    ← refresh drift baselines
│   └── smoke_test_models.py            ← validate committed bundles
│
├── tests/                              ← pytest suite (api, schemas, logger,
│                                          drift, normalization stats)
├── docs/DEPLOYMENT.md                  ← Render + Streamlit Cloud
│
├── part2/                              ← Logistic Regression + Random Forest
├── part3/                              ← XGBoost  (models/xgboost_best.json committed)
├── part4/                              ← SHAP explainability + fairness audit
├── part8/                              ← three-way evaluation suite
│
├── data/                               ← [GITIGNORED — you generate all of this]
│   ├── raw/student_placement.csv
│   └── processed/{train,test}_processed.csv, class_weights.csv,
│                 normalization_stats.json
│
├── visualizations/                     ← [COMMITTED] 14 EDA plots
├── logs/                               ← [GITIGNORED] predictions.db, self-creating
│
├── feature_engineering.py              ← SINGLE SOURCE OF TRUTH for the schema
├── preprocessing.py                    ← pipeline + artifact generation
├── data_analysis.py                    ← EDA to console
├── visualization.py                    ← EDA plots
├── download_dataset.py                 ← kagglehub -> data/raw/
│
├── SCHEMA.md                           ← dataset rationale + schema reference
├── README.md                           ← project overview
├── BACKEND_API_DOCUMENTATION.md        ← API contract
├── SETUP.md                            ← THIS FILE
├── requirements.txt                    ← full dependency set
├── requirements-ci.txt                 ← pinned CI subset
├── pyproject.toml                      ← ruff + pytest config
├── pyrightconfig.json                  ← type-checker config
└── render.yaml                         ← Render deployment
```

---

## 13. Dependency graph

```
download_dataset.py
        │
        ▼
data/raw/student_placement.csv
        │
        ▼
preprocessing.py
        ├─► data/processed/{train,test}_processed.csv
        ├─► data/processed/class_weights.csv
        ├─► data/processed/normalization_stats.json ──┐  (dashboard reads
        └─► part2/models/preprocessor.joblib          │   this directly)
        │                                             │
        ├──► scripts/train_models_fast.py             │
        │      ├─► part2/models/random_forest_best.joblib
        │      └─► part3/models/xgboost_best.joblib   │
        │                                             │
        ├──► part2/*.py  (full LR/RF reporting)       │
        ├──► part3/xgboost_model.py  (tuned XGB)      │
        ├──► part4/explainability_fairness.py         │
        ├──► part8/evaluation_suite.py                │
        │                                             │
        │      ┌──────────────────────────────────────┘
        │      │
        ├──► frontend/app.py ◄── part2/models/ + part3/models/ + data/raw/
        │            └─► http://localhost:8501
        │
        └──► scripts/package_model.py
                     │   copies preprocessor + model + normalization_stats
                     │   into the bundle and checksums all three
                     ▼
               artifacts/production/   [COMMITTED, self-contained]
                     │
                     ▼
               api.main:app  (needs no data/ directory)
                     └─► http://localhost:8000

NOTE: the two servers are independent. The dashboard does not call the API.
```

---

## 14. Troubleshooting

### `ModuleNotFoundError` for any package

You are not using the venv interpreter. Prefix with
`venv\Scripts\python.exe` (Windows) or `venv/bin/python` (Mac/Linux).

### `FileNotFoundError: data/raw/student_placement.csv`

Run `download_dataset.py` (Section 4). Every downstream script reads that
exact path.

### `RuntimeError: Normalization stats not found at ...`

Run `preprocessing.py`. This is deliberate: the alternative was inferring the
scaling maxima from the input, which silently corrupts every `*_normalized`
feature. See Section 5.

### `Normalization stats SHA-256 mismatch` at API startup

A bundle's `normalization_stats.json` was edited without re-running
`scripts/package_model.py`, so the manifest checksum no longer matches.
Re-package (Section 6).

### Dashboard: "Prediction pipeline failed" / KeyError on a column name

The artifacts in `part2/models/` and `part3/models/` were trained on an older
schema. Regenerate: `preprocessing.py` then `scripts/train_models_fast.py`.
The dashboard stops with an explicit error rather than showing placeholder
probabilities, which is intentional.

### `InconsistentVersionWarning: ... from version 1.9.0 when using version 1.6.1`

Expected and safe. `requirements.txt` pins scikit-learn 1.6.1; the committed
bundles were serialised by a newer scikit-learn. `api/predictor.py` restores
the attribute the older version expects, and predictions are identical across
both versions. Only a concern if it escalates to an actual error.

### API returns 503 / startup reports a checksum mismatch

`artifacts/production/` was modified without re-running
`scripts/package_model.py`, so `manifest.json` no longer matches the files.
Re-package (Section 6).

### `xgboost_model.py` reports "GPU not available, falling back to CPU"

Normal without an NVIDIA GPU. Slower, identical results.

### `part8/evaluation_suite.py` raises `FileNotFoundError`

It needs all three `.joblib` models. `scripts/train_models_fast.py` only
produces two — also run `part2/logistic_regression_model.py`.

### `run_pipeline.bat` fails immediately

The `.bat` files are Windows-only and assume `venv\Scripts\python.exe`
exists and that you are running from the repo root.

---

## 15. Copy-paste: full setup

### Windows (PowerShell)

```powershell
git clone https://github.com/alfreddevy2020-kch/StudentPlacementPrediction.git
cd StudentPlacementPrediction

python -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt

venv\Scripts\python.exe download_dataset.py
venv\Scripts\python.exe preprocessing.py
venv\Scripts\python.exe scripts\train_models_fast.py

# Optional: add Logistic Regression (a few minutes)
# venv\Scripts\python.exe part2\logistic_regression_model.py

# Verify
venv\Scripts\python.exe -m pytest tests\ -q
venv\Scripts\python.exe scripts\smoke_test_models.py

# Run either (or both) — they are independent
# venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000
# venv\Scripts\streamlit run frontend\app.py
```

### Mac/Linux (Bash)

```bash
git clone https://github.com/alfreddevy2020-kch/StudentPlacementPrediction.git
cd StudentPlacementPrediction

python3 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.txt

venv/bin/python download_dataset.py
venv/bin/python preprocessing.py
venv/bin/python scripts/train_models_fast.py

# Optional: add Logistic Regression (a few minutes)
# venv/bin/python part2/logistic_regression_model.py

# Verify
venv/bin/python -m pytest tests/ -q
venv/bin/python scripts/smoke_test_models.py

# Run either (or both) — they are independent
# venv/bin/python -m uvicorn api.main:app --reload --port 8000
# venv/bin/streamlit run frontend/app.py
```
