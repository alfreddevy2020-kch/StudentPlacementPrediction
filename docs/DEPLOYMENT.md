# Deployment Guide & Retraining Policy

## Architecture Overview

```
GitHub ──push/PR──► GitHub Actions CI
                          │
                   lint · test · smoke
                          │
                   merge to main
                          │
              ┌───────────┴───────────┐
              │                       │
         Render (free)        Streamlit Community Cloud
         FastAPI API          Streamlit Dashboard
         :8000                :8501
              │                       │
              └───────────┬───────────┘
                          │
                    logs/predictions.db  (SQLite on Render disk)
                          │
                    /api/v1/drift        (prediction-distribution shift endpoint)
```

---

## 1. GitHub Actions CI/CD

### What runs on every push / pull request

| Step | Tool | Blocks merge? |
|------|------|:---:|
| Lint | `ruff check .` | ✅ Yes |
| Type check | `pyright` | ⚠️ Advisory only |
| Unit tests | `pytest tests/` (excluding API integration) | ✅ Yes |
| Model smoke test | `python scripts/smoke_test_models.py` | ✅ Yes |

### What the smoke test checks

`scripts/smoke_test_models.py` loads all three production `.joblib` bundles from
`artifacts/production/` and runs one sample prediction through each. If any model:

- fails to deserialise
- raises during preprocessing or inference
- returns a probability outside `[0, 1]`
- returns a class label outside `{0, 1}`

…the CI step exits with code `1` and the pull request **cannot be merged**.

This ensures a regression in model packaging is caught before it reaches production.

---

## 2. Deployment Stack & Cost

### Backend API — Render (Free Tier)

| Property | Value |
|----------|-------|
| Platform | [render.com](https://render.com) |
| Plan | Free |
| Cost | **$0/month** |
| Region | Singapore (closest free region to India) |
| Runtime | Python 3.11, `uvicorn api.main:app` |
| Cold start | ~30 s after 15 min of inactivity |
| Disk | Ephemeral (logs reset on deploy) |

> **For always-on logs**: Upgrade to Render Starter ($7/month) to add a persistent disk,
> or export `PREDICTION_LOG_DB` to a mounted volume / external Postgres.

#### Deploy steps

1. Push this repo to GitHub (ensure `artifacts/production/**/*.joblib` are committed).
2. Go to [render.com](https://render.com) → **New → Blueprint**.
3. Connect your GitHub repository — Render auto-detects `render.yaml`.
4. Click **Apply** — the API will be live at `https://student-placement-api.onrender.com`.

### Frontend Dashboard — Streamlit Community Cloud

| Property | Value |
|----------|-------|
| Platform | [share.streamlit.io](https://share.streamlit.io) |
| Plan | Free |
| Cost | **$0/month** |
| Cold start | ~10 s |

#### Deploy steps

1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
2. Connect your GitHub repo, set **Main file path** to `frontend/app.py`.
3. Under **Advanced settings → Secrets**, add:
   ```toml
   BACKEND_URL = "https://student-placement-api.onrender.com/api/v1/predict"
   ```
4. Click **Deploy**.

The dashboard reads `BACKEND_URL` from `st.secrets`, so no code changes are needed.

---

## 3. Prediction Logging

Every successful call to `POST /api/v1/predict` is recorded in a SQLite database at
`logs/predictions.db` (configurable via `PREDICTION_LOG_DB` env var).

**Schema** (key columns):

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | TEXT | ISO-8601 UTC |
| `model_used` | TEXT | `xgboost`, `random_forest`, `logistic_regression` |
| `cgpa`, `ssc_marks`, … | REAL/INT | All 10 raw input features |
| `probability_placed` | REAL | Model output probability |
| `placement_status` | INTEGER | 0 or 1 |

**Monitoring endpoints:**

```
GET /logs/summary
    → { "total": 1234, "by_model": { "xgboost": 800, ... } }

GET /api/v1/drift?model=xgboost&window=200
    → { "status": "ok", "psi": 0.04, "mean_shift": 0.02, ... }
```

---

## 4. Prediction-distribution shift monitoring

### Method: Population Stability Index (PSI)

PSI measures how much the distribution of `probability_placed` has shifted from
the held-out baseline. This is prediction-distribution monitoring, not direct
model-performance monitoring.

**Formula:**

```
PSI = Σ (actual_% − expected_%) × ln(actual_% / expected_%)
```

10 equal-width bins spanning [0, 1] are used.

**Thresholds:**

| Status | Condition | Meaning |
|--------|-----------|---------|
| `ok` | PSI < 0.10 **and** shift < 0.05 | No action needed |
| `warn` | PSI 0.10–0.20 **or** shift 0.05–0.10 | Monitor; inspect inputs |
| `alert` | PSI > 0.20 **or** shift > 0.10 | Trigger retraining review |
| `insufficient_data` | < 20 predictions logged | Wait for more traffic |
| `baseline_unavailable` | Held-out histogram is missing | Run `scripts/generate_baseline_metrics.py` |

The baseline uses persisted real held-out probability bins in
`baseline_metrics.json`. The service will not synthesize a distribution from a
mean and standard deviation. Capture verified outcomes separately to monitor
ROC-AUC, calibration, and false-negative rate.

---

## 5. Retraining Trigger Policy

Retraining is a deliberate decision — not fully automated — to prevent
unreviewed models from reaching production.

### Triggers

| Trigger | Threshold | Priority |
|---------|-----------|----------|
| **Drift alert** | PSI > 0.20 **or** mean shift > 0.10 for 3+ consecutive days | High |
| **F1 regression** | F1 on held-out validation set drops below 0.80 | High |
| **Data volume** | New labeled data > 20 % of original training set size | Medium |
| **Calendar** | Every academic semester (≈ 6 months) | Low |
| **Manual trigger** | Team decision after cohort intake or curriculum change | Any time |

### Retraining Process

```
1. Collect new labeled data (post-placement outcomes)
2. Merge with original training set (or use rolling window)
3. Re-run training pipeline:
       python part2/random_forest_model.py
       python part3/xgboost_model.py
4. Compare metrics against current baseline_metrics.json
       → new model must meet or exceed baseline F1
5. Package artifacts:
       python scripts/package_model.py \
           --model part3/models/xgboost_best.joblib \
           --preprocessor part3/models/preprocessor.joblib \
           --output-dir artifacts/production/xgboost \
           --overwrite
6. Run smoke test locally:
       python scripts/smoke_test_models.py
7. Open a pull request → CI smoke test must pass → merge to main → Render auto-deploys
```

> **Key principle**: No model reaches production without a passing CI smoke test
> and explicit team review of the baseline metrics comparison.

---

## 6. Cost Summary

| Service | Plan | Monthly Cost |
|---------|------|:---:|
| GitHub Actions | Free (2,000 min/month) | **$0** |
| Render (API) | Free tier | **$0** |
| Streamlit Community Cloud | Free | **$0** |
| SQLite (disk) | In-process, ephemeral on free tier | **$0** |
| **Total** | | **$0/month** |

This makes the system viable for a real college placement cell with no infrastructure budget.
A production upgrade path exists: Render Starter ($7/month) adds persistent disks and
always-on instances, eliminating cold starts.
