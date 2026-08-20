# Role 5 and Core-Correctness Implementation Plan

## Decision

Proceed with Role 5, but complete two immediate correctness and credibility
fixes first:

1. Make the cohort what-if simulator use the model selected in the sidebar.
2. Replace causal wording in the existing simulator with scenario wording.

The existing simulator is useful **scenario scoring**: it re-scores an edited
student profile with a predictive classifier. It is not formal uplift
modelling. Role 5 will add a separate, explicitly labelled **observational
treatment-effect research module** and a production-safe skill-gap clustering
feature.

## Current Part 6 Audit and Integration Decision

`part6/` already contains a Role 5 prototype. It is not currently imported by
the production Streamlit entry point (`frontend/app.py`), API, test suite, CI,
or deployment setup. The current dashboard therefore remains four tabs, with
the fourth tab reserved for **Upload & analyze cohort**.

### What can be reused

| Part 6 item | Status | Integration decision |
|---|---|---|
| `uplift_modeling.py` | T-learner prototype runs on the current 10,000-row schema | Retain as an associational baseline; add causal diagnostics and a cross-fitted learner before presenting estimates as CATE. |
| `skill_gap_clustering.py` | K-means, silhouette search, and reproducible archetype naming run | Retain the framework, but change the feature set before integration. |
| `literature_review.md` | Useful Role 5 source mapping | Correct path references and add causal-inference/diagnostic references. |
| `work.md` | Useful handoff rationale | Correct the treatment-arm description and the stale paths. |
| `app_py_role5.patch` and `app_with_tab4.py` | Legacy frontend prototype | Do **not** apply or copy over the current app. It replaces Tab 4 with Role 5 and would remove Upload & analyze cohort. |

### Defects that must be resolved before reuse

1. All frontend imports and documentation say `part5`, but no `part5/`
   directory exists; the implementation lives in `part6/`.
2. The legacy app copy has four tabs, but its fourth tab is Role 5. The current
   app's fourth tab is Upload & analyze cohort. Replacing `frontend/app.py`
   with `part6/app_with_tab4.py` would remove the existing upload feature.
3. `support_index` is a current clustering feature, but it includes
   `placement_training`, the proposed treatment. It must be excluded from
   skill-gap clustering so archetypes do not encode treatment assignment.
4. The prototype T-learner has no propensity/overlap adjustment or
   cross-fitting. It is valuable as a transparent association baseline, but
   not sufficient by itself for a formal causal claim.
5. The claim that treatment arms are roughly balanced is inaccurate: current
   data contains 7,318 trained and 2,682 untrained students. The imbalance
   and baseline covariate differences must be shown in the methodology.

### Upload & analyze cohort preservation contract

The following are non-negotiable acceptance criteria for every Role 5 change:

1. `frontend/app.py` remains the deployed app entry point.
2. Existing Tab 4 label, `cohort_csv_uploader` widget key, CSV template,
   schema normalization, predictions, analytics, outcome comparison, and CSV
   export stay intact.
3. Role 5 is added as a fifth tab or a separately gated view; it never replaces
   the fourth tab and does not import from, mutate, or write to Tab 4 state.
4. Role 5 state keys use a `role5_` prefix; Tab 4 retains its existing keys.
5. Role 5 computations consume a copy of the built-in reference cohort only;
   uploaded data remains independently processed unless a later, explicit
   product decision adds uploaded-cohort segmentation.
6. No legacy `app_with_tab4.py` or patch file is used as the integration source.

`docs/implementation_plan.md` itself is documentation only. It has no runtime,
CI, API, deployment, or frontend reference, so it cannot affect the current
Upload & analyze cohort flow.

## Scientific Scope and Guardrails

### Current data contract

- Treatment: `placement_training` (`Yes` / `No`)
- Outcome: `placement_status` (`Placed` / `NotPlaced`)
- Baseline covariates: CGPA, SSC/HSC marks, aptitude, soft skills,
  internships, projects, workshops/certifications, extracurricular status

The dataset is observational. Placement training is strongly associated with
baseline readiness, so no output may be described as proof that training
causes a student to be placed. The UI and documentation must consistently say:

> Estimates adjust for observed baseline factors only. Unmeasured selection
> factors may remain; use this evidence to prioritise programme evaluation,
> not to automate or deny student support.

### Required distinction

| Component | Question answered | Permitted language |
|---|---|---|
| Existing what-if simulator | How does the classifier's score change when a profile is edited? | Modelled scenario estimate |
| Role 5 uplift module | What is the estimated incremental effect of placement training under stated observational assumptions? | Observational estimated treatment effect |
| Skill-gap clusters | Which readiness patterns recur in the cohort? | Archetype / profile |

## Architecture

```text
Raw student data
  ├─> Feature preparation
  │     ├─> K-means + validation ──> skill-gap archetypes
  │     └─> cross-fitted causal pipeline ──> ATE/CATE + diagnostics
  ├─> Existing classification models ──> scenario planner
  └─> Streamlit dashboard
        ├─> cohort/archetype view
        ├─> per-student archetype explanation
        ├─> scenario planner (non-causal)
        └─> programme-insights research view
```

Proposed module layout:

```text
role5/
  __init__.py
  features.py
  skill_gap_clustering.py
  uplift_modeling.py
  reporting.py
  train_role5.py
tests/
  test_role5_features.py
  test_skill_gap_clustering.py
  test_uplift_modeling.py
```

## Phase 0 — Baseline Verification

1. Inspect the worktree without overwriting unrelated changes.
2. Set up the project environment from `requirements.txt`.
3. Run linting, the test suite, API tests, and the artifact smoke test.
4. Record current model versions and held-out metrics.
5. Add a dedicated upload-cohort regression test before Role 5 integration:
   valid raw-header and snake-case CSVs must normalize, score every row once,
   and keep every probability in `[0, 1]`; missing required columns must retain
   the current validation error.
6. Manually verify all four existing tabs, especially CSV upload, output table,
   actual-vs-predicted panel, and export download.

**Acceptance:** known baseline state before changes begin.

## Phase 1 — Immediate Correctness and Credibility Fixes

1. Update `frontend/simulator.py` so `predict_probabilities()` accepts an
   optional `model_name` and forwards it to `BatchPredictor`.
2. Update `simulate_policy_intervention()` to receive and use that
   `model_name` for both baseline and simulated predictions.
3. Update `frontend/app.py` Tab 3 to pass the active sidebar model.
4. Add tests proving each model selection reaches the simulator path.
5. Change deterministic claims such as “will successfully cross” to
   “is estimated to cross in this modelled scenario.”
6. Add visible wording that the scenario planner is not causal-effect
   estimation.

**Acceptance:** selecting Logistic Regression, Random Forest, or XGBoost
changes all relevant dashboard calculations; the app no longer claims an
intervention will cause an outcome.

## Phase 2 — Default Model and Single Source of Truth

1. Make Logistic Regression the default in `api/schemas.py` and
   `frontend/batch_predictor.py` because it currently has the strongest
   held-out ROC-AUC and F1.
2. Keep Random Forest and XGBoost selectable as comparison alternatives.
3. Update README and API examples to match the default and 10-feature schema.
4. Regenerate or remove the stale 15-feature/perfect-score Part 8 and Part 3
   artifacts from the active demo/documentation path.

**Acceptance:** UI, API, README, evaluation material, and source schema agree
on input fields and actual metrics.

## Phase 3 — Skill-Gap Clustering

### Data and method

1. Build a six-dimension readiness frame scaled to 0–100:
   - academic foundation (CGPA, SSC, HSC composite)
   - academic consistency
   - aptitude readiness
   - communication readiness
   - portfolio readiness (internships, projects, certifications composite)
2. Explicitly exclude `placement_status`, `placement_training`,
   `placement_training_binary`, and `support_index` from clustering;
   describe clusters using those fields only after assignment.
3. Fit `StandardScaler` then K-means for `k = 2..6`, with `random_state=42`.
4. Select the highest silhouette-score solution subject to a minimum cluster
   size (at least 5% of the cohort).
5. Measure bootstrap stability with adjusted Rand index.
6. Generate interpretable labels from centroid deficits, e.g. “Academically
   strong, portfolio gap,” rather than exposing only cluster numbers.

### Integration

1. Tab 1: archetype distribution and cohort filter.
2. Tab 2: selected student's archetype and within-archetype peer benchmark.
3. Tab 3: allow archetype-based cohort targeting for scenario planning.
4. Cache models/data; do not re-fit clustering on every widget rerun.

**Acceptance:** every student has one deterministic archetype; no target
leakage occurs; profile names are actionable; cluster quality and stability are
recorded.

## Phase 4 — Observational Uplift Modelling

### Estimand

Estimate the conditional average treatment effect (CATE):

```text
tau(X) = E[Placement if trained - Placement if not trained | baseline profile X]
```

### Method

1. Create a treatment-ready baseline feature frame; reject missing treatment
   classes and invalid outcome labels.
2. Preserve the existing T-learner as an association baseline, labelled
   **not causal**.
3. Use K-fold cross-fitting for the causal-estimation candidate.
4. Fit a propensity model `P(training = Yes | X)` and an outcome model
   `E(placement | X)` on each training fold.
5. Fit a residualised R-learner CATE model on held-out residuals.
6. Calculate aggregate ATE, CATE distribution, and CATE by skill-gap
   archetype.
7. Use bootstrap confidence intervals for aggregate estimates; avoid
   individual “training-seat” recommendations until real experimental or
   time-stamped programme data is available.

### Diagnostics and failure gates

1. Treatment/control sample counts.
2. Propensity-score overlap distribution and clipped proportion.
3. Standardised mean differences before and after weighting.
4. Effective sample size after weighting.
5. Warning state if overlap, balance, or sample-size requirements fail.
6. Never display individual treatment recommendations as guaranteed outcomes.

**Acceptance:** outputs are reproducible, model assumptions and diagnostics
are visible, and insufficient evidence produces a clear non-result instead of
a misleading estimate.

## Phase 5 — Streamlit UX Integration

1. Keep the existing four user workflows intact:
   - departmental pulse/readiness
   - per-student diagnostic
   - cohort scenario planner
   - uploaded cohort analysis
2. Add a fifth tab, “Programme insights,” rather than replacing Tab 4. It is
   the clearest separation between uploaded-cohort operational analytics and
   built-in-cohort research evidence.
3. Present three distinct cards:
   - skill-gap archetype insights
   - scenario-planner result
   - observational programme-evidence estimate
4. Do not calculate Role 5 analyses merely because the tab is hidden:
   gate loading behind an explicit native Streamlit control and cache fitted
   resources/results once requested. This avoids slowing every interaction in
   the existing four tabs.
5. Use the project Streamlit conventions: native components, Material Symbols,
   sentence casing, the existing theme, bordered containers, no custom CSS,
   and `width="stretch"` rather than newly adding deprecated
   `use_container_width`.
6. Add an assumptions/limitations expander and methods citations. Keep
   methodology details secondary to the cohort-level findings.

**Acceptance:** no core dashboard regression, no unbounded reruns, and no
causal language attached to the scenario planner.

## Phase 6 — Monitoring and Evaluation Integrity

1. Rename the current PSI endpoint concept to prediction-distribution shift
   monitoring; it is not direct performance-drift measurement.
2. Correct the standard-deviation key mismatch in baseline metric handling.
3. Persist a real baseline probability distribution or bin counts instead of
   synthesising one.
4. Later add outcome-feedback capture for real performance monitoring:
   ROC-AUC, calibration, and false-negative rate.

**Acceptance:** monitoring documentation matches what the endpoint actually
measures.

## Phase 7 — Tests, Documentation, and Demo

1. Unit tests: feature preparation, deterministic clustering, no target
   leakage, invalid treatment states, no-overlap state, model-selection
   forwarding, and output-schema validation.
2. Integration tests: load Role 5 artifacts/models and render safe empty or
   warning states.
3. Add a Role 5 methodology document: estimand, assumptions, diagnostics,
   results, limitations, and citations.
4. Add a deck slide: “Scenario modelling is not causal uplift modelling.”
5. Rehearse the demo flow: archetype → individual profile → scenario planner
   → observational programme evidence → limitations.

**Acceptance:** live app, documentation, and presentation use the same schema,
metrics, and terminology.

## Literature to Cite

1. Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019).
   *Metalearners for estimating heterogeneous treatment effects using machine
   learning.* PNAS, 116(10), 4156–4165.
   https://doi.org/10.1073/pnas.1804597116
2. Nie, X., & Wager, S. (2021). *Quasi-oracle estimation of heterogeneous
   treatment effects.* Biometrika, 108(2), 299–319.
   https://doi.org/10.1093/biomet/asaa076
3. Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If.*
   Chapman & Hall/CRC. https://miguelhernan.org/whatifbook
4. Radcliffe, N., & Surry, P. (1999). *Differential response analysis:
   Modeling true responses by isolating the effect of a single action.*
   Credit Scoring and Credit Control IV.
5. MacQueen, J. B. (1967). *Some methods for classification and analysis of
   multivariate observations.* Proceedings of the Fifth Berkeley Symposium,
   281–297.
6. Rousseeuw, P. J. (1987). *Silhouettes: A graphical aid to the
   interpretation and validation of cluster analysis.* Journal of
   Computational and Applied Mathematics, 20, 53–65.
   https://doi.org/10.1016/0377-0427(87)90125-7
