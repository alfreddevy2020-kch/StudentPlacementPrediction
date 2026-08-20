# Data Flow & Module Reference

## End-to-End Flow

Inference is in-process. Nothing in this flow crosses a network boundary.

```
load_system()  @st.cache_resource                     ── once per process ──
        │
        ├─► BatchPredictor.load()   reads preprocessor + 3 models + normalization stats
        ├─► BatchPredictor.load_dataset()   data/raw/student_placement.csv
        └─► CohortWhatIfSimulator(predictor)
        │
        ▼
Sidebar (section 3): active model + cohort filters
        │
        ├─► st.session_state["active_model"]  ──► every inference call
        └─► filtered_df  (placement training / extracurricular / CGPA band)
        │
        ▼
Section 4: predictor.predict_probabilities(filtered_df, model_name=...)
        │
        └─► filtered_df gains: placement_prob, predicted_status, risk_tier
        │
        ▼
   ┌────┴─────┬──────────────┬──────────────┬─────────────────┐
 Tab 1      Tab 2          Tab 3          Tab 4        Benchmark expander
 cohort     per-student    what-if        CSV upload   compute_benchmark_suite()
 analytics  diagnostic     simulator                   @st.cache_data
```

Every widget interaction reruns the whole script, so the pipeline above fires on
every slider drag or filter change. The two caches are what keep that cheap.

## Feature Pipeline

```
raw CSV  ──normalize_columns()──►  snake_case headers
         ──engineer_features()──►  8 raw numerical + 21 derived = 29 numerical
                                   + 2 categorical
         ──ColumnTransformer────►  model.predict_proba()[:, 1]
```

`engineer_features(df, stats=...)` takes the frozen normalization maxima from
the production bundle's `normalization_stats.json`, so served features are on
the same scale the model was trained on.

| Group       | Columns |
|-------------|---------|
| Raw numerical (8) | `cgpa`, `ssc_marks`, `hsc_marks`, `aptitude_test_score`, `soft_skills_rating`, `internships`, `projects`, `workshops_certifications` |
| Raw categorical (2) | `extracurricular_activities`, `placement_training` |
| Derived (21) | added by `engineer_features()` — e.g. `skill_composite`, `support_index`, `placement_readiness_score`, `overall_academic_score`, `interview_readiness_score` |

## `BatchPredictor` (`batch_predictor.py`)

| Member | Returns | Notes |
|---|---|---|
| `load()` | — | Loads preprocessor, stats, and all three models. Raises if none are found. |
| `available_models` | `list[str]` | `["Random Forest", "Logistic Regression", "XGBoost"]`, subject to what is on disk. |
| `active_model_name` | `str` | Defaults to Random Forest. The UI does **not** set this; it passes `model_name` per call instead. |
| `predict_probabilities(df, model_name=None)` | `np.ndarray` `(n,)` in `[0,1]` | Falls back to `active_model_name` when `model_name` is None. |
| `predict_single(student_dict, model_name=None)` | `float` | Wraps the dict in a one-row frame. |
| `classify_risk(prob)` | `str` | Static. Risk tier label. |
| `load_dataset()` | `pd.DataFrame` | Static. Raw dataset with canonical snake_case columns. |

Artifact resolution (`_resolve()`) prefers `artifacts/production/<model>/` and
falls back to `part2/models/` (Random Forest, Logistic Regression, preprocessor)
and `part3/models/` (XGBoost).

### Risk tiers

| Tier | Probability | Colour |
|---|---|---|
| High Risk | `< 0.50` | `#F87171` |
| Moderate Risk | `0.50 – 0.75` | `#FBBF24` |
| Interview Ready | `>= 0.75` | `#34D399` |

The same 0.50 cut also drives `predicted_status` (`Placed` / `Not Placed`).

## `CohortWhatIfSimulator` (`simulator.py`)

`simulate_policy_intervention(cohort_df, interventions)` takes a
`{column: delta}` mapping, applies the deltas, clamps each column to its
observed training range from `FEATURE_RANGES`, and re-scores the cohort.
Clamping is deliberate: pushing a feature past the trained range produces
confident nonsense, so an intervention can only move students to the top of the
real range.

Returns a dict with:

| Key | Type | Meaning |
|---|---|---|
| `cohort_size` | int | Rows simulated |
| `baseline_placement_rate` | float | % placed before intervention |
| `simulated_placement_rate` | float | % placed after |
| `placement_uplift_pct` | float | Difference in percentage points |
| `newly_placed_count` | int | Crossed 0.50 upward |
| `net_transitioned_out_of_high_risk` | int | Left the `< 0.50` tier |
| `risk_migration` | dict | `{baseline, simulated}` → tier counts |
| `student_transitions` | DataFrame | Per-candidate before/after log |

An empty cohort returns the same keys with zeroed values rather than raising.

### Intervention knobs (`INTERVENTION_KNOBS`)

| Column | Label | Max delta | Group |
|---|---|---|---|
| `aptitude_test_score` | Aptitude Coaching Programme | +25.0 | academic |
| `cgpa` | Academic Remediation Drive | +1.5 | academic |
| `soft_skills_rating` | Communication Workshop | +1.5 | academic |
| `projects` | Capstone Project Workshop | +3.0 | experiential |
| `workshops_certifications` | Sponsored Certification Drive | +3.0 | experiential |
| `internships` | Industry Internship Placement | +2.0 | experiential |

## Benchmark Suite

`compute_benchmark_suite(dataset_len)` is `@st.cache_data` and gated behind the
"Run benchmark evaluation" checkbox. It holds out 20% stratified, then for each
loaded model computes test ROC-AUC, precision, recall, F1, per-row inference
latency, 5-fold CV ROC-AUC, an ROC curve, a confusion matrix, and the top-10
global feature importances.

```
{
  "comparison_matrix": [ {Model, Mean CV ROC-AUC, Test ROC-AUC,
                          Precision, Recall, F1-Score,
                          Inference Latency (ms)}, ... ],
  "detailed_metrics": { <model>: {test_roc_auc, roc_curve,
                                  confusion_matrix, top_features} },
  "best_model_name": <highest Test ROC-AUC>,
}
```

The cache key is dataset length only, so **every** model's metrics are computed
up front and the panels pick one at render time from
`st.session_state["active_model"]`, with `best_model_name` as the fallback.
Importances come from `feature_importances_` (tree models) or `abs(coef_[0])`
(logistic regression), normalized to percentages.

## Tabs

| Tab | Content | Input |
|---|---|---|
| 1 — Departmental pulse | Cohort KPIs, distributions, risk breakdown | `filtered_df` |
| 2 — Per-student diagnostic | Radar chart + skill gaps. "Existing candidate" picks from the cohort; "Custom profiler" takes manual slider input | `filtered_df` or manual |
| 3 — Cohort what-if simulator | Policy knobs → uplift and risk migration | `filtered_df` + knob deltas |
| 4 — Upload & analyze cohort | CSV upload, schema validation, template download, scored analytics | uploaded CSV |

Tab 4 is the bulk-input path. `normalize_columns()` accepts either raw
mixed-case headers (`AptitudeTestScore`) or snake_case; missing required columns
are reported back in the user's original naming via a reversed
`COLUMN_RENAME_MAP`.

## Session State

| Key | Set by | Read by |
|---|---|---|
| `active_model` | Sidebar selectbox (section 3) | Every `predict_*` call, the benchmark panels, the header caption |

Everything else in `st.session_state` is Streamlit's own widget storage, keyed
by the `key=` argument on each widget.
