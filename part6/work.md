# Role 5 — Work Log

> **Superseded implementation note (2026-08-20):** This prototype was retained
> for audit history only. The maintained implementation is now `role5/`,
> integrated as the gated fifth tab in `frontend/app.py`, with its methods in
> `docs/role5_methodology.md`. It corrects the treatment-arm imbalance,
> excludes treatment-derived clustering features, and adds cross-fitting plus
> diagnostic gates. Do not import this directory into the production app.

**Role:** Innovation & Research — uplift modeling, skill-gap clustering,
literature citations
**Dataset:** `ruchikakumbhar/placement-prediction-dataset` (current, as of
the 2026-08-19 migration recorded in `SCHEMA.md`) — 10,000 rows, 12 raw
columns, 29 numerical + 2 categorical after `feature_engineering.py`.
**Written:** 2026-08-20. If this file and the current schema in
`feature_engineering.py` / `SCHEMA.md` ever disagree, trust the code —
that exact mismatch (a work.md describing an earlier, since-replaced
dataset) is what the team's own gap analysis flagged in `part3/work.md`.

## What was built

- `uplift_modeling.py` — T-learner (Künzel et al., 2019) estimating each
  student's counterfactual placement-probability lift from
  `placement_training`. Outputs `p_control`, `p_treated`, `uplift`, and a
  four-category label (Persuadable / Sure Thing / Lost Cause /
  Do-Not-Disturb / Moderate-Unclear) per student.
- `skill_gap_clustering.py` — K-means over five standardized skill/
  readiness composites, k chosen by silhouette search over 2–7,
  archetypes auto-named from centroid deviation, profiled against actual
  placement rate and (when available) uplift.
- `literature_review.md` — six sources, pulled and verified same-day, not
  from memory. Maps each to the specific function it justifies.
- `results/` — `uplift_scores.csv`, `covariate_balance.csv`,
  `cluster_assignments.csv`, `cluster_profile.csv`, `k_search.csv`. All
  regeneratable by rerunning the two scripts; not hand-edited.

## Why placement_training as the treatment variable, specifically

It was the only column on the roster that's an actual policy lever —
CGPA and test scores aren't things a placement cell can change before
next term, but who gets a training seat is. It also plugs into a result
Part 4 already reported (higher false-negative rate for untrained
students) rather than starting a disconnected thread: Part 4 says the
model under-catches at-risk students who skipped training; this says,
among the untrained, who training would actually move.

## Design decisions worth knowing before Q&A

- **LogisticRegression, not RF/XGBoost, as the T-learner's base learner.**
  `SCHEMA.md`'s own held-out comparison has LR narrowly ahead of RF and
  XGBoost on this dataset (0.8836 vs 0.8750 vs 0.8684 ROC-AUC) — evidence
  the relationship is close to linear. Splitting the data by treatment
  arm roughly halves what each model trains on, so the lower-variance
  choice was deliberate, not a shortcut.
- **`placement_training_binary` and `support_index` are excluded from the
  T-learner's own feature set.** Both encode the treatment itself (the
  second one literally sums it into another column). Leaving either in
  would let each arm-specific model partially re-derive which arm it's
  looking at instead of modeling the outcome surface — see the comment
  block at the top of `uplift_modeling.py` for the full explanation.
- **The five clustering features were chosen to avoid redundancy.** CGPA,
  SSC%, and HSC% are all "academic" and correlated; using all three as
  separate clustering axes would let academics dominate the distance
  metric three times over. `overall_academic_score` folds them into one
  axis on purpose, leaving aptitude, soft skills, portfolio, and
  institutional support as four more genuinely distinct dimensions.
- **k was chosen by silhouette, not by eye on the elbow plot.** Both are
  saved (`k_search.csv`) so the elbow curve can still go on a slide next
  to the silhouette curve — showing both is more defensible than showing
  either alone, per the playbook's "alternatives" scoring line.

## What this does *not* claim — say this before anyone asks

`placement_training` is an observed column in a Kaggle dataset, not a
randomized assignment. The uplift numbers are associational, estimated
under the assumption that every relevant confounder is already in the
feature set — an assumption the covariate-balance check can raise a flag
against but can never fully clear. Full reasoning and the citation trail
are in `literature_review.md`'s "What's explicitly not claimed" section.
Say the limitation out loud in the first pass through this section of
the presentation; don't wait for it to be asked.

## Handoff notes

- **To whoever owns the frontend (Tab 4 / dashboard):** both pipelines
  are read-only with respect to `part2`/`part3`/`part4` artifacts — they
  only read `feature_engineering.py`'s output, they don't touch the
  trained classifiers or preprocessor. Safe to wire in without risk to
  Tabs 1–3.
- **To whoever runs the real training pipeline:** both scripts assume
  `data/processed/normalization_stats.json` already exists (i.e.
  `preprocessing.py` has been run). Rerun `uplift_modeling.py` before
  `skill_gap_clustering.py` if you want the cluster profile's
  `avg_training_uplift` column populated — the clustering script merges
  in `results/uplift_scores.csv` if it's present and the row count
  matches, and explicitly skips the merge (rather than mis-pairing rows)
  if it doesn't.
- **Known limitation to flag, not hide:** everything above was built and
  smoke-tested against a synthetic stand-in dataset (see
  `_synthetic_smoke_test_notes.md` if present) because this environment
  can't reach Kaggle. Numbers in `results/*.csv` right now are NOT the
  real numbers — rerun both scripts against the actual downloaded
  dataset before anything in `results/` goes on a slide.
