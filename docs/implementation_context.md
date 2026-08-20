# Implementation context

## Objective

Execute `docs/implementation_plan.md` while preserving the deployed
`frontend/app.py` entry point and every existing uploaded-cohort workflow.

## Guardrails

- Keep `Upload & analyze cohort` as Tab 4 with its existing widget key and
  independently processed uploaded data.
- Add Role 5 as a gated fifth tab, never by applying the legacy
  `part6/app_with_tab4.py` replacement.
- Use the sidebar-selected model for all scenario calculations.
- Describe profile edits as modelled scenarios, not causal effects.
- Describe programme evidence as observational and adjust only for recorded
  baseline covariates.
- Keep Role 5 session-state keys prefixed `role5_`.

## Implementation records

- Read this file, `docs/build_log.md`, and `docs/change_log.md` before each
  implementation phase.
- Append evidence and verification results to the build and change logs after
  each completed phase.
- Preserve pre-existing, untracked plan and `part6/` files unless a planned
  change explicitly replaces their contents.

## Current phase

Phase 7 complete — full test suite, linting, model smoke tests, and documentation verified.
