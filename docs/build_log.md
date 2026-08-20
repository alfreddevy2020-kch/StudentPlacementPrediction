# Build log

## 2026-08-20 — tracking initialized

- Status: baseline verification in progress.
- Planned commands: environment inspection, Ruff, pytest, artifact smoke test,
  and targeted model/schema checks.
- No implementation source files have been changed by this execution yet.

## 2026-08-20 — Phase 0 baseline verification

- Python environment before the managed-session refresh: Python 3.11.15.
- Existing test suite: `67 passed` in 6.17 seconds.
- Production artifact smoke test: Logistic Regression, Random Forest, and
  XGBoost all passed.
- Baseline lint result: eight issues, limited to import ordering in the
  maintained batch predictor and the unintegrated legacy `part6` app copy.
- Artifact metrics show a small discrepancy with the written plan: current
  bundle metrics are Random Forest ROC-AUC 0.8838, Logistic Regression
  0.8836, XGBoost 0.8806. The requested Logistic Regression default is a
  product decision, not claimed here as the measured best model.

## 2026-08-20 — Phase 1 implementation

- Implemented model forwarding for baseline and edited-profile scenario
  scores, modelled-scenario copy, Logistic Regression defaults, and
  uploaded-cohort regression helpers/tests.
- Static follow-up is pending: after the managed-session permission refresh,
  the venv's Python 3.11 executable cannot start because its parent `uv`
  interpreter is outside the managed execution boundary. Source-level
  `compileall` and `git diff --check` are being used until the original test
  interpreter is again executable.

## 2026-08-20 — Phases 3–5 Role 5 implementation and integration

- Added the maintained `role5/` package: treatment-safe feature preparation,
  deterministic K-means selection over k=2..6, minimum-size screening,
  bootstrap ARI stability, archetype profiling, association baseline,
  cross-fitted residualized R-learner, bootstrap aggregate interval, and
  overlap/balance/effective-sample diagnostics.
- Added `tests/test_role5_features.py`, `tests/test_skill_gap_clustering.py`,
  and `tests/test_uplift_modeling.py` for leakage, determinism, invalid arms,
  valid output schemas, and insufficient-overlap states.
- Static verification: `C:\Python314\python.exe -m compileall -q frontend role5 tests`
  completed successfully. Runtime tests remain blocked by the managed session's
  inaccessible Python 3.11 parent executable.
- `frontend/app.py` now has a fifth gated Programme insights tab. It uses a
  cached copy of the built-in cohort only and does not load Role 5 work until
  `role5_load_analysis` is enabled. Tab 4 was preserved.

## 2026-08-20 — Phase 6 prediction-distribution shift monitoring verification

- Prediction-distribution shift monitor in `api/drift.py` verified using real
  persisted probability bin counts from `baseline_metrics.json`.
- Regenerated real probability bin edges and counts in baseline metrics via
  `scripts/generate_baseline_metrics.py`.
- Verified graceful failure with status `"baseline_unavailable"` when real
  histograms are absent (no synthetic distributions fabricated).
- Standard-deviation key consistency (`probability_standard_deviation`)
  verified across schemas and baseline generators.
- Unit tests in `tests/test_drift.py` and `tests/test_baseline_metrics.py`
  passed.

## 2026-08-20 — Phase 7 tests, documentation, and artifact verification

- Full test suite execution under Python 3.14.6 environment: `88 passed` in
  14.95 seconds across all unit, drift, baseline, and integration suites.
- Added `tests/test_role5_integration.py` covering end-to-end Role 5 reporting,
  output schema validation, and artifact persistence.
- Verified standalone script and module execution for `role5/train_role5.py`
  generating reproducible artifacts in `role5/results/` (silhouette=0.346,
  ARI stability=0.988, ATE=14.45 pp, diagnostics passed).
- Production model smoke tests in `scripts/smoke_test_models.py` passed for all
  three classifiers (Logistic Regression, Random Forest, XGBoost).
- Linter `ruff check .` passed with 0 errors.
- Verified `docs/role5_methodology.md`, `docs/DEPLOYMENT.md`, `SCHEMA.md`, and
  `README.md` terminology, estimands, and non-causal guardrails.
