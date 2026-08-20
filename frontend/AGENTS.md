# AGENTS.md

## Project Structure

The Streamlit dashboard is **standalone**: it loads the trained artifacts from
disk and runs inference in-process. It does **not** call the FastAPI service in
`api/` — there is no HTTP client, no `BACKEND_URL`, and the dashboard runs with
the API stopped. `api/` is a separate deployable that happens to consume the
same artifacts.

```
frontend/
  app.py               # Streamlit UI — sections 1-5, then 3 tabs, then the benchmark expander
  batch_predictor.py   # BatchPredictor: artifact loading + cohort/single inference
  simulator.py         # CohortWhatIfSimulator + INTERVENTION_KNOBS
  AGENTS.md
  flow.md              # Data flow & module reference
```

Shared code lives at the repo root: `feature_engineering.py` (column
normalization + derived features) is imported by both `app.py` and
`batch_predictor.py`.

## Running

```bash
pip install -r requirements.txt      # from the repo root
streamlit run frontend/app.py
```

Requires the model artifacts and the dataset at `data/raw/student_placement.csv`.
`BatchPredictor` prefers the committed production bundles under
`artifacts/production/<model>/` and falls back to the local training outputs in
`part2/models/` and `part3/models/` (see `_resolve()` in `batch_predictor.py`).
Three models are loaded: Random Forest (the default), Logistic Regression, and
XGBoost.

## Key Gotchas

- **Streamlit reruns the entire script on every widget change.** `app.py` is
  top-level script code, not a function tree; there is no explicit state
  machine. Anything expensive must be cached.
- **`st.session_state["active_model"]` is the single source of truth for which
  model runs.** It is set by the sidebar selectbox in section 3 and every
  inference call must forward it:
  `predictor.predict_probabilities(df, model_name=st.session_state.get("active_model"))`.
  Panels that ignore it silently render a different model than the one the user
  picked — this was a real bug in the benchmark expander. It is the only
  non-widget session key the app uses.
- **`load_system()` is `@st.cache_resource`**, so the predictor, dataframe, and
  simulator are loaded once per process and shared across sessions. Never mutate
  the returned objects; set `active_model` in session state instead of
  reassigning `predictor.active_model_name`.
- **`compute_benchmark_suite()` is `@st.cache_data` keyed on dataset length
  alone.** Anything model-specific must be computed for *all* models inside it
  and selected at render time — adding the selected model to the cache key would
  re-run 5-fold CV plus the latency benchmark on every switch.
- **Expander open/close is client-side and does not trigger a rerun.** Checking
  an expander's state to gate expensive work never fires, so the benchmark suite
  sits behind an explicit `st.checkbox` instead.
- **A failed prediction calls `st.stop()` on purpose.** When the artifacts do
  not match the schema `feature_engineering.py` produces, the app stops with a
  remediation message rather than showing placeholder probabilities that would
  hide the mismatch. Do not "fix" this by substituting default values.
- **Raw columns are not model inputs.** The dataset carries 8 raw numerical and
  2 raw categorical columns; `engineer_features()` expands these to 29 numerical
  columns before the fitted `ColumnTransformer` sees them. Normalization maxima
  come from the production bundle's `normalization_stats.json` so served
  features match training scale.

## Conventions

- `ruff` is configured at the repo root (`pyproject.toml`): line length 100,
  rule set `E,W,F,I,UP,B,C4,SIM`. Run `ruff check .` before finishing.
- `E402` is per-file-ignored for `frontend/app.py` and `frontend/batch_predictor.py`
  because both must insert the repo root onto `sys.path` before importing
  `feature_engineering`. Keep that bootstrap above the local imports.
- Section headers use a `# ===` divider with a numbered title; tabs use the same
  divider with a `TAB N:` title. Keep the numbering contiguous when adding or
  removing a section.
- Material icons in labels (`:material/bar_chart:`) and sentence casing for UI
  copy.
