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

The dataset contains **5,000 student records** and initially contains 18 columns.

### Features

| Feature | Description |
|---|---|
| gender | Student gender |
| ssc_percentage | Secondary school percentage |
| hsc_percentage | Higher secondary percentage |
| degree_percentage | Degree percentage |
| cgpa | College CGPA |
| entrance_exam_score | Entrance examination score |
| technical_skill_score | Technical skill assessment score |
| soft_skill_score | Soft skill assessment score |
| internship_count | Number of internships |
| live_projects | Number of live projects |
| work_experience_months | Previous work experience |
| certifications | Number of certifications |
| attendance_percentage | Attendance percentage |
| backlogs | Number of academic backlogs |
| extracurricular_activities | Participation in extracurricular activities |
| placement_status | Placement outcome |

### Removed Columns

The following columns were removed during preprocessing:

- `student_id` — identifier only
- `salary_package_lpa` — removed to prevent data leakage because salary is known after placement

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