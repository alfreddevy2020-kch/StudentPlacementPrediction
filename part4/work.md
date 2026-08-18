# Part 4 — Explainability & Bias/Fairness

> **Role:** Explainability & Fairness Lead — SHAP explanations, probability calibration, and demographic fairness audit.

## Overview

This module makes the XGBoost model **trustworthy and accountable** for placement officers:

1. **SHAP** — explains individual predictions (local) and global feature drivers
2. **Calibration** — ensures displayed probabilities are reliable (Platt / isotonic)
3. **Fairness audit** — compares false negative rates across gender and extracurricular groups

---

## Quick Start

```bash
# Prerequisites: dataset, preprocessing, and Part 3 XGBoost model
python download_dataset.py
python preprocessing.py
python part3/xgboost_model.py

# Run Part 4 pipeline
part4\run_pipeline.bat
```

Or manually:

```bash
venv\Scripts\python.exe -m pip install shap
venv\Scripts\python.exe part4/explainability_fairness.py
```

---

## Scripts

| Script | Description |
|---|---|
| `part4/explainability_fairness.py` | Full pipeline: SHAP + calibration + fairness audit + report |
| `part4/run_pipeline.bat` | One-click Windows runner |

---

## Generated Outputs

After running, artifacts appear in `part4/explainability_results/` (regenerate locally):

```
part4/
├── models/
│   └── calibrated_xgboost.joblib       ← best calibrated model for API/dashboard
└── explainability_results/
    ├── shap_summary_plot.png             ← global beeswarm (slide-ready)
    ├── shap_bar_plot.png                 ← mean |SHAP| bar chart
    ├── shap_waterfall_sample.png         ← local explanation for one student
    ├── shap_values_test.csv              ← per-student SHAP values
    ├── shap_global_importance.csv        ← ranked global drivers
    ├── calibration_before_after.png      ← reliability diagram
    ├── calibration_metrics.csv           ← Brier scores (raw vs Platt vs isotonic)
    ├── fairness_group_metrics.csv        ← accuracy, FNR, recall by group
    ├── fairness_fnr_by_group.png         ← FNR comparison chart
    └── fairness_report.txt               ← talking points + mitigations
```

> `part4/models/` and `part4/explainability_results/` are gitignored — run the pipeline locally.

---

## Key Talking Points (Presentation Q&A)

### SHAP vs Feature Importance

| | Global RF Importance | SHAP |
|---|---|---|
| Scope | All students averaged | One specific student |
| Direction | Magnitude only | Positive/negative contribution |
| Use case | Model debugging | Explaining a decision to a student/officer |

### Why Calibration Matters

Raw XGBoost scores are optimized for ranking, not probability accuracy. When the dashboard shows **"72% placement likelihood"**, calibration (Platt or isotonic) ensures that among students scored ~0.72, roughly 72% are actually placed.

### Fairness Audit

- **Primary metric:** False Negative Rate (FNR) — missing a placement-ready student is costlier than a false alarm
- **Groups audited:** `gender`, `extracurricular_activities`
- **Note:** This Kaggle dataset has no `department` or `category` column; we audit available demographic fields
- **Mitigation:** Group-specific thresholds, re-weighting, production monitoring (handoff to Role 7)

---

## Integration with Other Roles

| Role | Handoff |
|---|---|
| **Role 3 (Dashboard)** | Use `shap_waterfall_sample.png` and `shap_global_importance.csv` for per-student drivers |
| **Role 6 (API)** | Swap in `calibrated_xgboost.joblib` for trustworthy probabilities |
| **Role 8 (Evaluation)** | Include `fairness_group_metrics.csv` in final metrics report |

---

## Reproduction

```bash
python download_dataset.py
python preprocessing.py
python part3/xgboost_model.py
python part4/explainability_fairness.py
```
