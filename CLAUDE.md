# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Binary classifier predicting student placement (`Placed` / `NotPlaced`) from the Kaggle
`ruchikakumbhar/placement-prediction-dataset` (10,000 rows × 12 cols). Three models are trained
and served side by side: Logistic Regression, Random Forest, XGBoost.

There are **two independent consumers** of the trained models:

- `api/` — FastAPI service, loads bundles at startup, one prediction per request, logs to SQLite.
- `frontend/` — Streamlit dashboard, **standalone**: it loads the same artifacts from disk and runs
  inference in-process. It does **not** call the API. There is no HTTP client and no `BACKEND_URL`;
  the dashboard runs fine with the API stopped.

## Commands

```bash
# Setup (Windows venv is what the .bat runners assume)
pip install -r requirements.txt

# Full training pipeline — MUST be run from the repo root, in this order.
python download_dataset.py        # -> data/raw/student_placement.csv
python preprocessing.py           # -> data/processed/*.csv, normalization_stats.json,
                                  #    part2/models/preprocessor.joblib
python part2/logistic_regression_model.py
python part2/random_forest_model.py
python part3/xgboost_model.py
python part4/explainability_fairness.py    # SHAP + fairness
python part8/evaluation_suite.py           # three-way comparison, cost-sensitive thresholds

# Windows convenience runners (each cd's to the repo root itself, then uses venv\Scripts\python.exe)
part2\run_pipeline.bat    # download -> preprocess -> LR -> RF -> comparison -> report
part3\run_pipeline.bat
part4\run_pipeline.bat

# Fast retrain (skips hyperparameter search; reads whatever preprocessing.py last wrote)
python scripts/train_models_fast.py

# Promote a trained model into artifacts/production/ (computes SHA-256s, writes manifest.json,
# copies normalization_stats.json into the bundle)
python scripts/package_model.py --model-name logistic_regression --model-version 2026.08.18-lr.1 \
  --preprocessor part2/models/preprocessor.joblib \
  --model part2/models/logistic_regression_best.joblib --overwrite

# Regenerate drift baselines — required whenever models or the dataset change
python scripts/generate_baseline_metrics.py

# Run
uvicorn api.main:app --reload --port 8000    # /docs, /redoc
streamlit run frontend/app.py

# Test / lint / type-check
pytest tests/ -v                             # full suite (test_api.py auto-skips without artifacts)
pytest tests/test_drift.py -v                # single file
pytest tests/test_drift.py::TestComputePSI::test_identical_distributions_zero_psi -v  # single test
ruff check .                                 # run before finishing any change
pyright --project pyrightconfig.json         # advisory in CI, does not block

python scripts/smoke_test_models.py          # loads all 3 production bundles, one prediction each
```

CI (`.github/workflows/ci.yml`) runs ruff → pyright (advisory) → pytest (excluding `test_api.py`) →
`smoke_test_models.py`. The smoke test is a hard gate and only passes because
`artifacts/production/**/*.joblib` are committed via `.gitignore` negation rules.

## Architecture

### The artifact bundle contract

`artifacts/production/<model>/` is the unit of deployment. Each bundle contains four files that must
stay mutually consistent:

| File | Role |
|---|---|
| `model.joblib` | the fitted estimator |
| `preprocessor.joblib` | the `ColumnTransformer` (StandardScaler + OneHotEncoder) fitted in `preprocessing.py` |
| `normalization_stats.json` | frozen count-feature maxima — a **fitted parameter**, not config |
| `manifest.json` | `model_name`, `model_version`, and SHA-256 of the other three |

`verify_and_load_bundle()` in `api/predictor.py` validates all of this at startup: existence,
manifest/model-key match, every checksum, a transform smoke-test through `engineer_features()`, and
a `predict`/`predict_proba` smoke-test. A failing bundle does not crash the server — `api/main.py`
records the error and `/health` reports `degraded` (1–2 loaded) or `unavailable` (0 loaded).

Bundles are self-contained on purpose: `data/processed/` is gitignored, so a fresh clone can serve
without ever running `preprocessing.py`.

### feature_engineering.py is the single source of truth

Everything — API, dashboard, simulator, training scripts, `predict_sample.py` — imports
`engineer_features()` from the repo root. **Never reimplement these formulas locally**; that
duplication is what broke the last schema migration.

The flow is: raw CSV headers → `normalize_columns()` (mixed-case → snake_case, done exactly once) →
8 raw numerical + 2 raw categorical → `engineer_features()` adds 21 derived columns → 29 numerical +
2 categorical reach the fitted `ColumnTransformer`. Raw columns are **not** model inputs on their own.

`FEATURE_RANGES` supplies both dashboard slider bounds and API validation bounds; values outside it
are extrapolation.

### Normalization stats

The count features (`internships`, `projects`, `workshops_certifications`) have no natural upper
bound, so their `*_normalized` forms are scaled by maxima fitted **once on the full training set** by
`preprocessing.py`. `engineer_features()` reuses that frozen constant so one student, a filtered
cohort, and the full dataset all land on the same scale.

If stats are missing, `engineer_features()` **raises rather than inferring** them from the batch at
hand. Inferring divides each value by itself, pinning every `*_normalized` feature to 1.0 and
`portfolio_strength` to 100 — wrong answers with no error. Do not add a fallback.

`fit_normalization_stats()` writes the file with `newline="\n"` deliberately: the file is SHA-256'd
into every manifest, and CRLF on Windows would break the hash when checked out on Linux CI.

### API layer

`BasePredictor` (ABC) → `LogisticRegressionPredictor` / `RandomForestPredictor` / `XGBoostPredictor`.
All three take a `StudentInput`, build the same one-row engineered frame (using **their own bundle's**
stats), and return a common `PredictionResponse`. Adding a model means adding a `MODEL_BUNDLES` entry
in `api/config.py` plus a `PREDICTOR_TYPES` entry in `api/main.py`.

Endpoints: `GET /health`, `GET /api/v1/models`, `POST /api/v1/predict`, `GET /api/v1/drift`,
`GET /logs/summary`. `student_id` and `placement_status` are deliberately not accepted.

`api/logger.py` persists every prediction to SQLite (best-effort, never raises); path overridable via
the `PREDICTION_LOG_DB` env var. `api/drift.py` computes PSI between the recent prediction window and
each bundle's `baseline_metrics.json` — stale baselines make drift detection meaningless.

## Gotchas

- **`preprocessing.py` and the `part*/` scripts are top-level scripts with relative paths and no
  `__main__` guard.** They must be invoked from the repo root.
- **sklearn version skew is a live concern.** `requirements.txt` pins `1.9.0` but `requirements-ci.txt`
  pins `1.6.1` to match the serialized artifacts. sklearn dropped `LogisticRegression.multi_class` in
  1.7 while older runtimes still read it during `predict()`, so both `api/predictor.py` and
  `frontend/batch_predictor.py` restore the attribute after load. Keep the shim in both.
- **`BatchPredictor._resolve()` prefers `artifacts/production/<model>/` and falls back to the local
  training outputs** in `part2/models/` and `part3/models/`. A stale local file can silently shadow
  what you expect if production is absent.
- **Streamlit reruns the whole script on every widget change.** `app.py` is top-level script code,
  not a function tree.
  - `st.session_state["active_model"]` is the single source of truth for which model runs; every
    inference call must forward it. Panels that ignore it silently render a different model than the
    one the user picked.
  - `load_system()` is `@st.cache_resource` — shared across sessions, never mutate what it returns.
  - `compute_benchmark_suite()` is `@st.cache_data` keyed on dataset length alone; anything
    model-specific must be computed for all models inside it and selected at render time.
  - Expander open/close is client-side and triggers no rerun, so expensive work sits behind an
    explicit `st.checkbox`.
  - A failed prediction calls `st.stop()` on purpose, so an artifact/schema mismatch surfaces instead
    of hiding behind placeholder probabilities. Do not substitute defaults.
- **The derived features do not raise ROC-AUC** (0.8780 → 0.8778). They exist to power the skill-gap
  radar, readiness scores, the what-if simulator, and legible SHAP output. Don't delete them as dead
  weight, and don't expect tuning them to move the metric.
- **Uploaded CSVs accept either header style** — `normalize_columns()` maps raw mixed-case headers, so
  validation runs on normalized names and maps back through `COLUMN_RENAME_MAP` for user-facing errors.

## Conventions

- ruff at the repo root: line length 100, rules `E,W,F,I,UP,B,C4,SIM`, first-party `api` and
  `feature_engineering`.
- `E402` is per-file-ignored for the five modules that live outside the root package and must insert
  the repo root onto `sys.path` before importing `feature_engineering`. Keep that bootstrap above the
  local imports.
- `frontend/app.py` sections use a `# ===` divider with a numbered title; tabs use the same divider
  with a `TAB N:` title. Keep numbering contiguous when adding or removing one.
- Streamlit UI copy: material icons in labels (`:material/bar_chart:`) and sentence casing.
