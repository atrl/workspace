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

- Prefer `runtime/<direction>/card/<experiment>/` for new runtime experiments.
  Existing promoted qlib cards may live under `research/<experiment>/` or
  `production/<experiment>/`.
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

## Default Validation

```bash
PYTHONPATH=. qlib_venv/bin/python -m pytest tests/test_soxlbacktest_framework.py -q
PYTHONPATH=. qlib_venv/bin/python -m pytest tests/test_qlib_native_workflow.py tests/test_qlib_model_selection_validation.py tests/test_qlib_native_notebook_cards.py -q
PYTHONPATH=. qlib_venv/bin/python -m nbconvert --execute runtime/model_selection/card/<experiment>/<experiment>.ipynb --to notebook --inplace
```

Every notebook should end with a concise conclusion cell covering what ran, key
metrics, decision, risk notes, artifact links, and next action.
