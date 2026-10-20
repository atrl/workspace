---
name: soxl-qlib-experiment
description: Use for SOXL single-factor, multi-factor, and model-selection experiments in the cleaned qlib workspace.
---

# SOXL Qlib Experiment Workflow

Use the repo-owned qlib-native workflow for reusable data loading, model
selection, recorder logging, and backtest analysis. Keep experiment-specific
choices, execution evidence, and the single-run conclusion inside the runtime
notebook.

## Required Shape

- Use the repository's active research/production notebook surfaces as defined
  in AGENTS.md; do not introduce a second runtime card hierarchy.
- Make `<experiment>.ipynb` the primary executable workflow and conclusion.
- Keep the card directory slim: `workflow.yaml`, `summary.json`, selected model
  metadata, risk/backtest/analysis summaries, and the notebook.
- Put large prediction, signal, portfolio report, position, and all-model
  prediction artifacts in the qlib recorder, not root card CSVs.
- `summary.json` must include `artifact_policy.policy:
  slim_card_recorder_first` and `artifact_policy.recorder_only_artifacts`.
- Reusable workflow code belongs in `src/workflow/`, `src/model/`, `src/data/`,
  `src/backtest/`, or `src/analysis/`; ordinary experiment choices stay in the
  notebook.
- Keep `soxl_backtest.reference` only as the legacy comparison benchmark when
  needed.

## Focused Validation

Follow the repository verification policy: use the smallest meaningful existing
check for the changed layer. Execute the relevant notebook for new research;
reuse unchanged verified Recorder evidence for presentation or delivery changes.
Do not rerun generic suites, refit models, or replay accounts just to update prose.

Every notebook should end with a concise conclusion cell covering what ran, key
metrics, decision, risk notes, artifact links, and next action.
