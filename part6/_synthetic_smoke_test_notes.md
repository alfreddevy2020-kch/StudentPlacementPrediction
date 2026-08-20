# Testing note — read before trusting anything in `results/`

> **Superseded:** These notes cover the retained `part6/` prototype. Run the
> maintained test suite for `role5/`; do not execute the obsolete `part5/`
> commands below.

`uplift_modeling.py` and `skill_gap_clustering.py` were written and
smoke-tested in an environment with no access to Kaggle, so they were
run against a **synthetic stand-in** dataset (same 12 raw columns, same
value ranges as the real `ruchikakumbhar` dataset, 4,000 fake rows with
a deliberately-planted training→placement effect and a deliberately
self-selected treatment arm) instead of the real one.

That confirms: both scripts run end-to-end against the real schema
without crashing, produce correctly-shaped output, and the diagnostics
work — the covariate-balance check correctly flagged the synthetic
data's planted confounding, and the four uplift categories partition the
cohort exhaustively with no silent mislabeling.

That does **not** confirm anything about the actual numbers — silhouette
scores, archetype names, uplift magnitudes, and category counts in
`results/` right now came from fake data and will look nothing like the
real output. Before this goes anywhere near a slide:

```
python download_dataset.py      # if not already done
python preprocessing.py         # if not already done
python part5/uplift_modeling.py
python part5/skill_gap_clustering.py
```

Delete this file once that's been done for real — it stops being useful
the moment `results/` holds real numbers, and an unread testing note is
exactly the kind of leftover artifact the rest of this repo's gap
analysis already flagged as a recurring risk.
