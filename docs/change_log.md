# Change log

## 2026-08-20 — execution tracking initialized

- Added persistent implementation context, build log, and change log.
- Existing untracked inputs retained: `docs/implementation_plan.md` and
  `part6/`.
- No application, API, model, or test behavior has changed yet.

## 2026-08-20 — Phase 1 correctness, credibility, and defaults

- Scenario simulator now forwards the active dashboard model to both baseline
  and edited-profile scores.
- Tab 3 calls out modelled scenario scoring as non-causal and removes the
  deterministic cutoff-crossing claim.
- Added `frontend/cohort_upload.py`; Tab 4 still retains its uploader key,
  template, analytics, outcome comparison, and export, while header
  normalization and single-pass scoring are now regression-testable.
- Set Logistic Regression as the API and dashboard default while retaining
  Random Forest and XGBoost as selections. Updated current API examples.
- Added tests for each simulator model path, default-model schema behavior,
  raw/snake-case upload normalization, single-pass scoring, and missing-column
  validation.

## 2026-08-20 — Phases 3–5 Role 5 and dashboard integration

- Added `role5/features.py`, `role5/skill_gap_clustering.py`,
  `role5/uplift_modeling.py`, `role5/reporting.py`, and `role5/train_role5.py`.
- Cluster fitting now uses only five readiness dimensions and excludes the
  treatment, target, treatment-derived binary, and support index.
- Programme evidence now separates a non-causal association baseline from a
  cross-fitted observational candidate and suppresses an effect display when
  diagnostics fail.
- Added gated Tab 5 while preserving all four existing tabs. Archetype insight
  is available in Tabs 1–3 only after the explicit Role 5 load control is
  enabled; uploaded data does not enter Role 5 state or computation.
- Added `docs/role5_methodology.md` and marked the retained `part6/` prototype
  notes as superseded rather than using its legacy app replacement.
- Removed deprecated `use_container_width` calls from the maintained app.
