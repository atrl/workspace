---
name: soxl-qlib-experiment
description: Use for SOXL single-factor, multi-factor, and model-selection experiments in the cleaned qlib workspace.
---

# SOXL Qlib Experiment Workflow

Use `soxl_backtest.framework` for reusable data loading, reference replay, and
card-local artifact writing. Keep experiment-specific choices inside the runtime
notebook.

## Required Shape

- Put each experiment under `runtime/<direction>/card/<experiment>/`.
- Make `<experiment>.ipynb` the primary executable workflow and conclusion.
- Keep sidecars in the same card directory: metrics, daily outputs, trades,
  specs, and summaries.
- Import framework APIs from `soxl_backtest.framework` and reference logic from
  `soxl_backtest.reference`.
- Do not change framework code for ordinary signal tests. Change it only when a
  reusable interface gap is proven by a failing test.

## Default Validation

```bash
PYTHONPATH=. qlib_venv/bin/python -m pytest tests/test_soxlbacktest_framework.py -q
PYTHONPATH=. qlib_venv/bin/python -m nbconvert --execute runtime/model_selection/card/<experiment>/<experiment>.ipynb --to notebook --inplace
```

Every notebook should end with a concise conclusion cell covering what ran, key
metrics, decision, risk notes, artifact links, and next action.
