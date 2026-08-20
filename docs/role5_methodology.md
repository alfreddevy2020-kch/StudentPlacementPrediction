# Role 5 methodology and limitations

## Purpose and separation of claims

CampusReady has two different tools that must not be conflated:

| Tool | Question | Permitted interpretation |
|---|---|---|
| Scenario planner | How does the selected placement classifier score an edited profile? | Modelled scenario estimate, not a causal effect |
| Programme insights | What association with placement training remains after adjustment for recorded baseline factors? | Observational estimated treatment effect, conditional on stated assumptions |

Neither tool automates student support eligibility or produces an individual
training-seat recommendation.

## Data contract

- Treatment: `placement_training` (`Yes` / `No`)
- Outcome: `placement_status` (`Placed` / `NotPlaced`)
- Adjustment covariates: CGPA, SSC/HSC marks, aptitude, soft-skills rating,
  internships, projects, workshops/certifications, and extracurricular status.

Both treatment arms and valid outcome labels are required. A missing arm,
invalid outcome, inadequate overlap, poor post-weighting balance, or too small
an effective sample produces a diagnostic warning/non-result rather than an
apparently precise effect estimate.

## Skill-gap archetypes

Clustering uses exactly five 0–100 readiness dimensions:

1. Academic foundation — CGPA, SSC, and HSC composite
2. Academic consistency — smaller SSC/HSC spread is higher
3. Aptitude readiness
4. Communication readiness — soft-skills rating rescaled to 0–100
5. Portfolio readiness — internships, projects, and certifications using the
   training-time normalization constants

`placement_status`, `placement_training`, `placement_training_binary`, and
`support_index` are excluded from the clustering frame. Training and outcome
rates are descriptive attributes attached only after an archetype is assigned.

K-means is fit with `random_state=42` after `StandardScaler`. Candidate values
of *k* are 2–6; the highest silhouette solution is selected only if every
cluster has at least 5% of the cohort. Bootstrap refits report adjusted Rand
index stability. Labels are reproducible centroid-deficit descriptions, not
manual cluster-number labels.

## Observational programme evidence

The page retains a two-model T-learner as an association-only comparison
baseline. The candidate observational estimate uses five-fold cross-fitting:

1. Fit `P(training = Yes | X)` and `E(placement | X)` outside each hold-out
   fold.
2. Compute held-out treatment and outcome residuals.
3. Fit a weighted residualized R-learner on those residuals.
4. Report the average CATE as an aggregate observational estimated effect, with
   a bootstrap interval over the cohort CATE distribution.

The interval reflects sampling variability of fitted cohort scores; it does
not remove confounding from variables that were not observed.

## Diagnostics and reporting gate

The app records treatment/control counts, propensity range and overlap,
clipped share, standardized mean differences before and after inverse-
probability weighting, and total/per-arm effective sample size.

Evidence is withheld when fewer than half the cohort lies in the configured
propensity-overlap region, more than 10% requires clipping, post-weighting
absolute SMD exceeds 0.10, or weighted effective sample size is too small.
These thresholds are diagnostics, not proof of causal identification.

> Estimates adjust for observed baseline factors only. Unmeasured selection
> factors may remain; use this evidence to prioritise programme evaluation,
> not to automate or deny student support.

## Recommended next step

Run a pre-registered randomized or time-stamped programme evaluation before
using results to scale a training programme. Capture programme eligibility,
enrolment timing, attendance/completion, and verified placement outcomes so
future monitoring can measure calibration, discrimination, and equity on real
feedback data.

## References

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
   Modeling true responses by isolating the effect of a single action.* Credit
   Scoring and Credit Control IV.
5. MacQueen, J. B. (1967). *Some methods for classification and analysis of
   multivariate observations.* Proceedings of the Fifth Berkeley Symposium,
   281–297.
6. Rousseeuw, P. J. (1987). *Silhouettes: A graphical aid to the interpretation
   and validation of cluster analysis.* Journal of Computational and Applied
   Mathematics, 20, 53–65. https://doi.org/10.1016/0377-0427(87)90125-7
