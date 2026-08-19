# Dataset & Schema

## Current dataset

**`ruchikakumbhar/placement-prediction-dataset`** — 10,000 rows × 12 columns.

Fetched by `download_dataset.py` into `data/raw/student_placement.csv`.

## Why this dataset

The use case calls for a *Kaggle Campus Recruitment / Placement dataset
(academics, skills, certifications)* and sets an explicit **ROC-AUC ≥ 0.80**
target. Five candidates were screened on three criteria: real signal,
absence of a degenerate labelling rule, and coverage of the feature
families the brief names.

| Dataset | Rows | Feature families | Best AUC | Depth-4 tree acc. | Verdict |
|---|---|---|---|---|---|
| `sehaj1104/student-placement-prediction-dataset-2026` | 100,000 | 8/8 | **0.577** | — | No signal — fails the target |
| `suvidyasonawane/student-academic-placement-performance` | 5,000 | 8/8 | **1.0000** | 0.9998 | Deterministic rule — reject |
| `benroshan/factors-affecting-campus-placement` | 215 | 3/8 | 0.938 | 0.879 | Real, but tiny and thin |
| `tejashvi14/engineering-placements-prediction` | 2,966 | 3/8 | 0.939 | 0.872 | Real, but only 8 columns |
| **`ruchikakumbhar/placement-prediction-dataset`** | **10,000** | **6/8** | **0.884** | **0.785** | **Selected** |

Two datasets were rejected for opposite reasons, and both are worth
recording because each looks fine on a casual glance:

- The **100k dataset** has every feature the brief asks for, but every
  feature correlates with the target at |r| ≤ 0.08 and the placement rate
  is a flat ~54.5% across every subgroup. No model can beat ~0.58 on it.
- The **5k dataset** scores a perfect 1.0000 ROC-AUC — because placement is
  a hand-coded rule that a depth-4 tree reproduces at 99.98% accuracy:
  `backlogs ≤ 2 AND technical_skill > 59.5 AND cgpa > 7.02 AND soft_skill > 54.5`.
  A perfect score here is a red flag, not a result.

The selected dataset clears 0.80 with margin without being trivially
separable, and every one of its features carries graded signal
(correlations 0.26–0.52), which is what makes the skill-gap radar and SHAP
explanations meaningful.

## Held-out performance (20% test split, n = 2,000)

| Model | ROC-AUC | Accuracy | F1 |
|---|---|---|---|
| Logistic Regression | **0.8836** | 0.7965 | 0.7697 |
| Random Forest | 0.8750 | 0.7890 | 0.7564 |
| XGBoost | 0.8684 | 0.7855 | 0.7542 |

All three clear the 0.80 milestone.

## Schema

Raw CSV headers are mixed-case and one (`Workshops/Certifications`) is not
a valid Python identifier. `feature_engineering.normalize_columns()` maps
them to snake_case **once**, so the API, dashboard and training scripts all
speak the same names. Always load via `load_raw_dataset()`.

| Raw header | Canonical name | Role |
|---|---|---|
| `StudentID` | `student_id` | dropped (identifier) |
| `CGPA` | `cgpa` | numerical |
| `SSC_Marks` | `ssc_marks` | numerical |
| `HSC_Marks` | `hsc_marks` | numerical |
| `AptitudeTestScore` | `aptitude_test_score` | numerical |
| `SoftSkillsRating` | `soft_skills_rating` | numerical |
| `Internships` | `internships` | numerical |
| `Projects` | `projects` | numerical |
| `Workshops/Certifications` | `workshops_certifications` | numerical |
| `ExtracurricularActivities` | `extracurricular_activities` | categorical |
| `PlacementTraining` | `placement_training` | categorical |
| `PlacementStatus` | `placement_status` | **target** (`Placed` / `NotPlaced`) |

**Feature counts:** 8 raw numerical + 21 engineered = 29 numerical, plus 2
categorical → **33 columns** after one-hot encoding.

## Feature engineering

`engineer_features()` adds 21 derived columns (see
`ENGINEERED_NUMERICAL_FEATURES`).

**These do not raise ROC-AUC on this dataset** — measured 0.8780 → 0.8778.
The raw features are already close to sufficient, and composites of
existing columns add no new information to a linear model. State this
honestly rather than claiming an accuracy gain.

They earn their place elsewhere, and demonstrably so: SHAP ranks
`placement_readiness_score`, `support_index` and `interview_readiness_score`
as the top three global drivers, and the dashboard's skill-gap radar,
per-student readiness scores and what-if simulator are all built on them.

`FEATURE_RANGES` records each feature's observed training range. The
simulator and the remediation engine clamp to it, so a suggested
intervention never quotes an uplift from a profile the model has never seen.

## Normalization stats

The three count features (`internships`, `projects`,
`workshops_certifications`) have no natural upper bound, so their `0-1`
normalized forms are divided by maxima frozen from the full training set.
`preprocessing.py` fits these once and writes
`data/processed/normalization_stats.json`.

They are a fitted parameter of the pipeline, exactly like
`preprocessor.joblib`, and are handled the same way:

- `scripts/package_model.py` copies them into each production bundle and
  records their SHA-256 in `manifest.json` (`schema_version: 2`).
- `api/predictor.py` loads the per-bundle copy and passes it into
  `engineer_features(..., stats=...)` on every request, so a served model is
  always scaled the way it was trained — and bundles stay self-contained on a
  clone where `data/processed/` does not exist.
- With no stats available, `engineer_features()` **raises**. It will not
  infer maxima from the batch it is given: for a single row that divides each
  value by itself, pinning every `*_normalized` feature to 1.0 and reporting
  `portfolio_strength` as 100 instead of 50 — wrong answers, HTTP 200, no
  warning. `tests/test_normalization_stats.py` guards both properties.

**Retraining changes these maxima if the data changes.** Re-run
`scripts/package_model.py` after retraining, or the API's startup checksum
verification will reject the bundle.

## Known gaps

The source data has no `backlogs`, `attendance`, department/branch, or
demographic columns. Two consequences, both documented rather than
worked around:

1. **Department trends** — the dashboard bands cohorts by CGPA and segments
   on `placement_training` / `extracurricular_activities` instead.
2. **Bias audit** (`part4/`) — audits equity of *access to institutional
   support* rather than demographic parity. This surfaced a real finding:
   false-negative rate is **0.47 for untrained students vs 0.19 for
   trained** (and 0.69 vs 0.15 on extracurriculars). The model misses
   placement-ready students who lacked institutional support at 2.5–4.6×
   the rate — an access problem, not a capability one.

A production deployment on real student records should re-run the audit
against actual demographic fields.

## Changing the dataset again

The schema lives in `feature_engineering.py`. To migrate:

1. `COLUMN_RENAME_MAP`, `RAW_NUMERICAL_FEATURES`, `RAW_CATEGORICAL_FEATURES`,
   `FEATURE_RANGES`, `TARGET_COLUMN`, `TARGET_MAP`
2. `engineer_features()` and `ENGINEERED_NUMERICAL_FEATURES`
3. `download_dataset.py` — the Kaggle slug
4. `api/schemas.py` — `StudentInput` fields
5. `frontend/app.py` — Tab 2 inputs, `radar_specs`, `remediation_rules`,
   sidebar filters; `frontend/simulator.py` — `INTERVENTION_KNOBS`
6. Re-run: `download_dataset.py` → `preprocessing.py` → training scripts

`data_analysis.py` and `visualization.py` are driven by the canonical
feature lists and need no per-column edits.
