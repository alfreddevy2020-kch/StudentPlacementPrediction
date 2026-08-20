# Role 5 — Literature Review

> **Superseded path note (2026-08-20):** The maintained Role 5 sources and
> methodology now live in `docs/role5_methodology.md` and are implemented in
> `role5/`. References to `part5/` below describe the retained prototype only.

Student Placement Prediction System · Batch 1 · Innovation & Research

Every source below was pulled and checked (title, authors, venue, DOI) on
2026-08-20 — none of this is from memory. Six sources, grouped by which
part of Role 5 they justify. Each entry says in one line why it's here,
not just what it's about — a reading list restates abstracts; this is
meant to be defensible if a mentor asks "why this method."

---

## 1. Uplift modeling

**Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019).**
"Metalearners for estimating heterogeneous treatment effects using
machine learning." *Proceedings of the National Academy of Sciences*,
116(10), 4156–4165. https://doi.org/10.1073/pnas.1804597116

Formalizes the family of meta-learners for estimating the conditional
average treatment effect (CATE) — S-learner, T-learner, X-learner — and
proves when each is preferable. `uplift_modeling.py` implements the
T-learner: two separately-trained models, one per treatment arm, scored
against every student and subtracted. This is the paper that gives that
approach a name and a formal justification, and the one to cite if asked
"why a T-learner and not something fancier" — the X-learner exists
specifically for imbalanced treatment arms, and this dataset's two arms
(`placement_training` Yes/No) are close to 50/50, so the simpler T-learner
is the appropriate choice, not a corner cut.

**Radcliffe, N. J., & Surry, P. D. (1999).** "Differential response
analysis: Modeling true response by isolating the effect of a single
action." *Credit Scoring and Credit Control VI.*

**Lo, V. S. Y. (2002).** "The true lift model: A novel data mining
approach to response modeling in database marketing." *ACM SIGKDD
Explorations Newsletter*, 4(2), 78–86.

The two-model idea predates the "T-learner" name by two decades — both
papers build one response model per arm (treated/control) and subtract,
in the direct-marketing literature, and Lo specifically used logistic
regression for both arms. Cited together because they establish this
isn't a technique invented for this project — it's ~25-year-old,
well-studied practice imported from marketing into an education-outcomes
setting. Lo's paper is also the direct precedent for
`uplift_modeling.py`'s choice of `LogisticRegression` as the base learner
for each arm.

## 2. Skill-gap clustering

**Liu, R. (2022).** "Data Analysis of Educational Evaluation Using
K-Means Clustering Method." *Computational Intelligence and
Neuroscience.* https://doi.org/10.1155/2022/3762431

Applies K-means to multi-dimensional student evaluation data to surface
latent groupings and support intervention decisions — the same shape of
problem `skill_gap_clustering.py` solves (five skill/readiness axes in,
archetype groups out), on a different dataset. Cited as evidence K-means
on standardized composite education features is established practice,
not a improvised choice.

**Santosa, R. G., Lukito, Y., & Chrismanto, A. R. (2021).**
"Classification and Prediction of Students' GPA Using K-Means Clustering
Algorithm to Assist Student Admission Process." *Journal of Information
Systems Engineering and Business Intelligence*, 7(1), 1–10.
https://doi.org/10.20473/jisebi.7.1.1-10

Uses K-means on academic/entrance-test features to segment students and
feed an admissions decision downstream — directly parallel to using
cluster archetypes to inform a *placement-readiness* decision here. Also
a useful precedent for method: they cluster first, then attach an
outcome-relevant label to each cluster after the fact, which is the same
order of operations `skill_gap_clustering.py` follows (cluster on
features only, report actual placement rate per cluster afterward — the
target is never part of the distance metric).

## 3. Student placement prediction (project-level grounding)

**Pathak, A., Matcha, M., Gopisetti, M., & Joshi, S. (2025).**
"A Machine Learning Framework for Predicting Student Placement
Outcomes." *Ingénierie des Systèmes d'Information*, 30(7), 1715–1721.
https://doi.org/10.18280/isi.300704

The closest published analogue to this entire project: same problem
(binary placement classification on a Kaggle placement dataset), same
core methods (Logistic Regression, Random Forest, XGBoost, plus a voting
ensemble), same preprocessing shape (standard scaling, categorical
encoding, 80/20 split). Their reported held-out numbers — LR 0.777
accuracy / 0.887 AUC, RF 0.886/0.935, XGBoost 0.892/0.945, best (voting)
0.892/0.943 — are useful external context for defending this project's
own held-out numbers (LR 0.884, RF 0.875, XGBoost 0.868 ROC-AUC, per
`SCHEMA.md`): this project's range sits inside the same band an
independently published, peer-reviewed study reports on comparable data.
That's a genuinely useful sentence to have ready if a judge questions
why the numbers aren't higher — 0.87–0.94 AUC is where this class of
problem lands in the literature, not a shortfall specific to this build.

Worth noting for calibration: their own citation [1] (Kumar et al.,
2023) reports a Random Forest with 96% *training* accuracy but only
64.6% *test* accuracy — a textbook overfitting gap the Pathak et al.
paper calls out directly. That's the same category of problem this
project's own gap analysis flagged in `part8/results/` (a Random Forest
and XGBoost run reporting a suspicious 1.0000 ROC-AUC, since traced to
evaluation against a rejected, near-deterministic dataset and scheduled
for cleanup) — the literature and this repo's own audit are pointing at
the same failure mode from two different directions, which is worth
saying out loud rather than treated as a coincidence.

---

## Prototype mapping (superseded by `role5/`)

| Literature | Implementation |
|---|---|
| Künzel et al. (2019) — T-learner | `uplift_modeling.py::fit_uplift_models`, `predict_uplift` |
| Radcliffe & Surry (1999); Lo (2002) — two-model uplift, LR base learner | `uplift_modeling.py` — LogisticRegression per arm |
| Radcliffe & Surry's four-fold target matrix | `uplift_modeling.py::categorize_uplift` |
| Liu (2022); Santosa et al. (2021) — K-means on education composites | `skill_gap_clustering.py::fit_clusters`, `name_archetypes` |
| Pathak et al. (2025) — comparable-problem benchmark | Context for defending `SCHEMA.md`'s held-out numbers |

## What's explicitly *not* claimed here

Two honest limitations, stated once here so they don't need re-deriving
live:

1. **The uplift estimate is associational, not certified causal.**
   `placement_training` is observational, not randomized. See the long
   comment block at the top of `uplift_modeling.py` and the covariate
   balance check it runs before fitting anything — that check can flag a
   confounding *risk*, it cannot rule confounding *out*. The rigorous
   next step, if this were carried past a hackathon, is a randomized
   pilot: offer training to a random half of an eligible group, withhold
   it from the other half, and measure the actual gap. That is the
   textbook design in the Radcliffe & Surry and Lo papers above — both
   are marketing papers precisely because marketers can run that
   experiment (an A/B test) in a way a placement cell would need to set
   up deliberately rather than get for free from historical records.

2. **Cluster archetypes describe correlation structure, not cause.**
   "Low Soft Skills, Low Support Engagement students have a lower actual
   placement rate" is a description of this cohort, not a claim that
   raising soft-skills scores in isolation would raise placement — the
   same confounding caveat as (1) applies to any intervention aimed at a
   cluster, not just at the uplift model specifically.
