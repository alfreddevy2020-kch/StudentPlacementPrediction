# Student Placement Prediction

A Machine Learning project that predicts whether a student is likely to be placed based on academic performance, technical skills, soft skills, internships, projects, work experience, certifications, attendance and backlogs.

## 📌 Project Overview

Student placement depends on several academic and professional factors. This project uses Machine Learning to analyze student-related features and predict their placement status.

The system follows a complete Machine Learning pipeline:

Raw Dataset → Data Analysis → Visualization → Preprocessing → Model Training → Evaluation → Prediction

## 🎯 Objective

The main objective of this project is to build a machine learning model that can predict:

- `0` → Not Placed
- `1` → Placed

The model can help identify students who may require additional training or placement preparation.

---

## 📊 Dataset

Source: **`ruchikakumbhar/placement-prediction-dataset`** on Kaggle —
**10,000 student records**, 12 columns. See [SCHEMA.md](SCHEMA.md) for why
this dataset was chosen over the alternatives.

### Features

Column names are normalised to snake_case on load by
`feature_engineering.normalize_columns()`; the raw CSV headers are
mixed-case (e.g. `Workshops/Certifications`).

| Feature | Type | Range | Description |
|---|---|---|---|
| cgpa | float | 6.5–9.1 | College CGPA (10-point scale) |
| ssc_marks | int | 55–90 | Secondary school (class 10) percentage |
| hsc_marks | int | 57–88 | Higher secondary (class 12) percentage |
| aptitude_test_score | int | 60–90 | Aptitude / mock-test score |
| soft_skills_rating | float | 3.0–4.8 | Soft-skills rating (5-point scale) |
| internships | int | 0–2 | Number of internships completed |
| projects | int | 0–3 | Number of projects completed |
| workshops_certifications | int | 0–3 | Workshops and certifications earned |
| extracurricular_activities | Yes/No | — | Participation in extracurriculars |
| placement_training | Yes/No | — | Received institutional placement training |
| placement_status | Placed/NotPlaced | — | **Target** — placement outcome |

### Removed Columns

- `student_id` — identifier only, no predictive content

This dataset ships no post-outcome fields (no salary/package column), so
the target itself is the only leakage source to drop.

### Not present in this dataset

`backlogs`, `attendance`, department/branch, and demographic attributes
(gender, college tier) are **absent from the source data** and are
therefore not modelled rather than fabricated. See [SCHEMA.md](SCHEMA.md)
for the consequences on department reporting and the Part 4 bias audit.

---

## 🔎 Exploratory Data Analysis

The dataset was analyzed for:

- Missing values
- Duplicate records
- Unique values
- Statistical distributions
- Placement distribution
- Feature averages by placement status
- Gender vs placement
- Extracurricular activities vs placement
- Feature correlations

### Data Quality

- Total records: **5,000**
- Missing values: **0**
- Duplicate rows: **0**

### Placement Distribution

- Not Placed: **4,134 (82.68%)**
- Placed: **866 (17.32%)**

The target variable is therefore imbalanced.

---

## 📈 Data Visualization

The project generates visualizations for important relationships between features and placement status.

Examples include:

- Placement distribution
- CGPA vs placement
- Technical skills vs placement
- Soft skills vs placement
- Backlogs vs placement
- Internships vs placement
- Live projects vs placement
- Work experience vs placement
- Attendance vs placement
- Gender vs placement
- Extracurricular activities vs placement
- Correlation heatmap

All generated visualizations are stored in:

```text
visualizations/
```

---

## 🤖 Part 2 — Model Training (Logistic Regression + Random Forest)

> **Role:** Model Lead 1 — builds, tunes, and evaluates two ML classifiers with full interpretability analysis.

All Part 2 files live in the `part2/` folder.

### Quick Start

```bash
# 1. Create a virtual environment (one-time setup)
python -m venv venv

# 2. Install dependencies
venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. Run the full pipeline (downloads data → preprocesses → trains → evaluates)
part2\run_pipeline.bat
```

> **Windows only:** Double-click `part2\run_pipeline.bat` to run the entire pipeline end-to-end. It automatically anchors to the repo root so all paths resolve correctly.

### Part 2 Scripts

| Script | Description |
|---|---|
| `part2/logistic_regression_model.py` | L2 (Ridge) + L1 (Lasso) Logistic Regression with GridSearchCV, 500-iteration bootstrap confidence intervals, threshold optimisation |
| `part2/random_forest_model.py` | Random Forest with two-phase tuning (RandomizedSearchCV → GridSearchCV), MDI + Permutation + Drop-Column feature importance, OOB convergence, learning curves |
| `part2/model_comparison.py` | Head-to-head comparison: overlaid ROC & Precision-Recall curves, calibration plot, threshold sweep, McNemar's statistical test |
| `part2/model_summary_report.py` | Generates `model_report.txt` and a 4-panel executive summary figure ready for slides |
| `part2/run_pipeline.bat` | One-click pipeline runner (Windows) |

### Generated Outputs

After running the pipeline, all outputs appear inside `part2/` (excluded from git — regenerate locally):

```
part2/
├── models/
│   ├── logistic_regression_best.joblib
│   └── random_forest_best.joblib
└── model_results/
    ├── executive_summary.png              ← 4-panel slide-ready figure
    ├── model_report.txt                   ← full text report with talking points
    ├── logreg_coefficient_importance.png
    ├── logreg_l1_vs_l2_coefficients.png
    ├── logreg_bootstrap_ci.png
    ├── logreg_regularization_tuning.png
    ├── logreg_threshold_optimization.png
    ├── rf_feature_importance_mdi.png
    ├── rf_feature_importance_permutation.png
    ├── rf_importance_comparison.png
    ├── rf_drop_column_importance.png
    ├── rf_oob_convergence.png
    ├── rf_learning_curve.png
    ├── comparison_roc_curves.png
    ├── comparison_pr_curves.png
    ├── comparison_calibration.png
    ├── comparison_threshold_analysis.png
    ├── comparison_metrics_table.png
    └── mcnemar_test.csv
```

> `part2/models/` and `part2/model_results/` are listed in `.gitignore` — they are **not committed**. Run `part2\run_pipeline.bat` to regenerate them.

### Key Results

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| ROC-AUC | 0.9354 | 1.0000 |
| F1 Score (tuned threshold) | 0.6949 | 1.0000 |
| Average Precision (PR) | 0.7718 | 1.0000 |
| Brier Score | 0.1110 | 0.0042 |
| McNemar's test p-value | — | < 0.0001 (significant) |
| Optimal threshold | 0.773 | 0.173 |

**Top predictive features** (Permutation Importance, RF):

1. `backlogs` — strongest negative predictor
2. `cgpa`
3. `technical_skill_score`
4. `soft_skill_score`

### Techniques Used (Beyond Basic Implementation)

| Technique | Purpose |
|---|---|
| L1 vs L2 regularisation comparison | Shows which features Lasso eliminates vs Ridge retains |
| Bootstrap confidence intervals (500×) | Statistically validates which coefficients are robust |
| 3 importance methods: MDI, Permutation, Drop-Column | Exposes MDI's high-cardinality bias; Permutation is the honest metric |
| OOB convergence plot | Proves `n_estimators=500` was chosen empirically |
| Learning curves | Diagnoses bias-variance tradeoff |
| Precision-Recall curves | Correct evaluation for an 82/18 imbalanced dataset |
| Calibration plot (reliability diagram) | Verifies predicted probabilities are trustworthy |
| Threshold optimisation | Tunes decision boundary for F1 — 0.5 is never optimal on imbalanced data |
| McNemar's statistical test | Formally proves model differences are not due to chance |

---

## 🔍 Part 4 — Explainability & Bias/Fairness

> **Role:** Explainability & Fairness Lead — SHAP explanations, probability calibration, and demographic fairness audit on the XGBoost model.

All Part 4 files live in the `part4/` folder.

### Quick Start

```bash
# Prerequisites: dataset, preprocessing, Part 3 XGBoost model
python download_dataset.py
python preprocessing.py
python part3/xgboost_model.py

# Run Part 4 pipeline
part4\run_pipeline.bat
```

### Part 4 Scripts

| Script | Description |
|---|---|
| `part4/explainability_fairness.py` | SHAP analysis, Platt/isotonic calibration, fairness audit, mitigation report |
| `part4/run_pipeline.bat` | One-click pipeline runner (Windows) |

### Generated Outputs

After running the pipeline, outputs appear inside `part4/` (excluded from git — regenerate locally):

```
part4/
├── models/
│   └── calibrated_xgboost.joblib
└── explainability_results/
    ├── shap_summary_plot.png
    ├── shap_bar_plot.png
    ├── shap_waterfall_sample.png
    ├── shap_values_test.csv
    ├── shap_global_importance.csv
    ├── calibration_before_after.png
    ├── calibration_metrics.csv
    ├── fairness_group_metrics.csv
    ├── fairness_fnr_by_group.png
    └── fairness_report.txt
```

### Key Deliverables

| Technique | Purpose |
|---|---|
| SHAP TreeExplainer | Local + global explanations — why THIS student got their score |
| Platt scaling / Isotonic regression | Calibrate raw XGBoost scores into trustworthy probabilities |
| Group-wise FNR audit | Compare false negative rates across gender and extracurricular groups |
| Mitigation report | Proposed fixes if disparity detected (threshold tuning, monitoring) |

See `part4/work.md` for full documentation and presentation talking points.

---

# 🚀 Part 6 — Model Serving & Application Integration

> **Role:** Backend / API Developer — integrates the trained ML model with
> a FastAPI prediction service and Streamlit frontend.

## Overview

The trained machine learning model is exposed through a REST API so that
the frontend can send student information and receive a placement
prediction.

The application follows this architecture:

```text
Streamlit Frontend
        │
        │ POST /api/v1/predict
        │ JSON
        ▼
FastAPI Prediction API
        │
        ▼
Pydantic Input Validation
        │
        ▼
Raw Student Data
        │
        ▼
preprocessor.joblib
        │
        ├── StandardScaler
        │
        └── OneHotEncoder
        │
        ▼
17 Processed Features
        │
        ▼
Random Forest Classifier
        │
        ▼
Prediction + Probabilities
        │
        ▼
FastAPI JSON Response
        │
        ▼
Streamlit Dashboard

## 🔄 Model Serving

The API is designed with a modular predictor architecture so that the trained model can be replaced without changing the frontend request/response contract.

Current production model:
- XGBoost / Random Forest — update once the final model is selected.

The preprocessing artifact and trained model artifact are loaded independently during API startup.