# Part 3 — XGBoost with CUDA GPU Acceleration

## Overview

This document describes all steps involved in training an XGBoost classifier with GPU acceleration (CUDA) on an NVIDIA GeForce RTX 4050 for student placement prediction.

---

## Environment Setup

### 1. System & Hardware
- **OS:** Linux (WSL2 on Windows)
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)
- **CUDA Version:** 12.9
- **Driver Version:** 576.88
- **Python:** 3.12.3

### 2. Virtual Environment
```bash
python3 -m venv venv
venv/bin/python -m ensurepip
```

### 3. Installed Packages
```bash
venv/bin/pip install numpy pandas scikit-learn xgboost joblib kagglehub matplotlib seaborn
```

| Package    | Version | Purpose                        |
|------------|---------|--------------------------------|
| xgboost    | 3.4.1   | Gradient boosted trees + CUDA  |
| pandas     | 3.0.5   | Data manipulation              |
| scikit-learn| 1.9.0  | Preprocessing, metrics, CV     |
| joblib     | 1.5.3   | Model serialization            |
| kagglehub  | 1.0.2   | Dataset download               |
| numpy      | 2.5.2   | Numerical computing            |
| matplotlib | 3.11.1  | Plotting                       |
| seaborn    | 0.13.2  | Statistical plots              |

---

## Step 1 — Dataset Acquisition

**Source:** [Kaggle — Student Academic Placement Performance Dataset](https://www.kaggle.com/datasets/suvidyasonawane/student-academic-placement-performance-dataset)

```bash
venv/bin/python download_dataset.py
```

- Downloaded to `~/.cache/kagglehub/...`
- Copied to `data/raw/student_placement.csv`

**Dataset characteristics:**
- 5000 rows, 18 columns
- Target: `placement_status` (0 = Not Placed, 1 = Placed)
- Class distribution: 82.68% Not Placed (4134) vs 17.32% Placed (866) — **imbalanced**

---

## Step 2 — Preprocessing

```bash
venv/bin/python preprocessing.py
```

### Actions performed:
1. **Removed leakage columns:** `student_id`, `salary_package_lpa` (known after placement)
2. **Separated features/target:** 15 features, 1 target (`placement_status`)
3. **Train/test split:** 80/20, stratified by target (random_state=42)
   - Training: 4000 samples
   - Testing: 1000 samples
4. **Feature types:**
   - Numerical (13): `ssc_percentage`, `hsc_percentage`, `degree_percentage`, `cgpa`, `entrance_exam_score`, `technical_skill_score`, `soft_skill_score`, `internship_count`, `live_projects`, `work_experience_months`, `certifications`, `attendance_percentage`, `backlogs`
   - Categorical (2): `gender`, `extracurricular_activities`
5. **Scaling:** `StandardScaler` on numerical features
6. **Encoding:** `OneHotEncoder` on categorical features (handle_unknown="ignore")
7. **Output features:** 17 processed features
8. **Class weights:** Class 0 = 0.605, Class 1 = 2.886

### Saved files:
- `data/processed/train_processed.csv`
- `data/processed/test_processed.csv`
- `data/processed/class_weights.csv`

---

## Step 3 — Model Training (XGBoost + CUDA)

```bash
venv/bin/python part3/xgboost_model.py
```

### 3a. GPU Verification
- XGBoost `device="cuda"` parameter routes computation to the GPU
- Verified CUDA availability with a quick test train on 10 samples
- Training ran entirely on **NVIDIA RTX 4050**

### 3b. Class Imbalance Handling
- Used `scale_pos_weight = count_neg / count_pos = 3307 / 693 = 4.772`
- This upweights the minority class (Placed) during gradient computation

### 3c. Hyperparameter Tuning — Two-Phase Approach

#### Phase 1: RandomizedSearchCV (broad sweep)
- **100 random parameter combinations**, 5-fold stratified CV
- Scoring metric: **F1** (appropriate for imbalanced data)
- Time: ~38 seconds on GPU
- Best CV F1: **0.9993**

**Search space:**
| Parameter          | Values                                  |
|--------------------|-----------------------------------------|
| n_estimators       | 100, 200, 300, 500, 700, 1000          |
| max_depth          | 3, 4, 5, 6, 8, 10, 12                  |
| learning_rate      | 0.01, 0.02, 0.05, 0.1, 0.15, 0.2      |
| subsample          | 0.6, 0.7, 0.8, 0.9, 1.0               |
| colsample_bytree   | 0.6, 0.7, 0.8, 0.9, 1.0               |
| min_child_weight   | 1, 3, 5, 7, 10                         |
| gamma              | 0, 0.1, 0.2, 0.3, 0.5                  |
| reg_alpha          | 0, 0.01, 0.1, 1, 10                    |
| reg_lambda         | 0, 0.01, 0.1, 1, 10                    |

#### Phase 2: GridSearchCV (fine-tuning)
- Narrow grid around best random-search params (neighbor values)
- **54 combinations**, 5-fold CV
- Time: ~3.2 seconds

### 3d. Final Hyperparameters

| Parameter          | Value   |
|--------------------|---------|
| n_estimators       | 200     |
| max_depth          | 10      |
| learning_rate      | 0.2     |
| subsample          | 0.7     |
| colsample_bytree   | 0.7     |
| min_child_weight   | 7       |
| gamma              | 0.5     |
| reg_alpha          | 0       |
| reg_lambda         | 0.01    |
| scale_pos_weight   | 4.772   |
| tree_method        | hist    |
| device             | cuda    |

---

## Step 4 — Test Set Evaluation

| Metric              | Value    |
|---------------------|----------|
| Accuracy            | 1.0000   |
| Precision (Placed)  | 1.0000   |
| Recall (Placed)     | 1.0000   |
| F1 (Placed)         | 1.0000   |
| ROC-AUC             | 1.0000   |
| Average Precision   | 1.0000   |
| Brier Score         | 0.0002   |
| Optimal Threshold   | 0.054    |
| Optimal F1          | 1.0000   |

### Confusion Matrix
```
           Predicted
Actual  TN=827  FP=0
        FN=0    TP=173
```

---

## Step 5 — Feature Importance

### Top Features by Gain (total loss reduction):
1. **backlogs** — 144.82
2. **technical_skill_score** — 80.45
3. **cgpa** — 57.48
4. **soft_skill_score** — 39.44
5. **work_experience_months** — 2.54

### Top Features by Weight (split frequency):
1. **cgpa** — 67 splits
2. **soft_skill_score** — 64 splits
3. **backlogs** — 52 splits
4. **technical_skill_score** — 45 splits
5. **entrance_exam_score** — 28 splits

---

## Step 6 — Sample Prediction

### Input:
```json
{
  "ssc_percentage": 82.5,
  "hsc_percentage": 79.2,
  "degree_percentage": 76.8,
  "cgpa": 8.1,
  "attendance_percentage": 88,
  "backlogs": 0,
  "entrance_exam_score": 84,
  "technical_skill_score": 78,
  "soft_skill_score": 81,
  "certifications": 4,
  "live_projects": 3,
  "internship_count": 2,
  "work_experience_months": 0,
  "gender": "Male",
  "extracurricular_activities": "Yes"
}
```

### Output:
```json
{
  "placement_status": 1,
  "placement_label": "Placed",
  "probability_placed": 0.9977,
  "probability_not_placed": 0.0023,
  "risk_level": "High Probability of Placement (Low Risk)"
}
```

---

## Step 7 — Artifacts Saved

### Model files (`part3/models/`):
| File | Description |
|------|-------------|
| `xgboost_best.json` | XGBoost native format |
| `xgboost_best.joblib` | Scikit-learn wrapper |
| `preprocessor.joblib` | Fitted ColumnTransformer |

### Results (`part3/model_results/`):
| File | Description |
|------|-------------|
| `xgb_metadata.csv` | Hyperparameters and metrics |
| `xgb_importance_gain.csv` | Gain feature importance |
| `xgb_importance_weight.csv` | Weight feature importance |
| `xgb_feature_importance_gain.png` | Gain importance plot |
| `xgb_feature_importance_weight.png` | Weight importance plot |
| `xgb_importance_comparison.png` | Gain vs Weight side-by-side |
| `xgb_roc_curve.png` | ROC curve |
| `xgb_pr_curve.png` | Precision-Recall curve |
| `xgb_confusion_matrix.png` | Confusion matrix heatmap |
| `xgb_threshold_optimization.png` | Threshold vs F1 |

---

## Reproduction

```bash
# 1. Setup venv
python3 -m venv venv
venv/bin/python -m ensurepip
venv/bin/pip install numpy pandas scikit-learn xgboost joblib kagglehub matplotlib seaborn

# 2. Download dataset
venv/bin/python download_dataset.py

# 3. Preprocess
venv/bin/python preprocessing.py

# 4. Train XGBoost on GPU
venv/bin/python part3/xgboost_model.py

# 5. Run sample prediction
venv/bin/python part3/predict_sample.py
```
