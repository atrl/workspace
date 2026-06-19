# AGENTS.md

This repository is the cleaned `SOXL` qlib research workspace.

Active surfaces are intentionally narrow:

- `data/`: canonical local data, qlib calendar/store files, and raw inputs.
- `soxl_backtest/`: reusable framework, reference policy, and stable APIs.
- `runtime/<direction>/card/<experiment>/`: executable notebooks and sidecars.
- `tests/`: framework, reference, and architecture boundary tests.

Production execution, dashboards, scheduler code, legacy RL training surfaces,
and scattered scripts are retired from this workspace.

## Startup

Before substantial work:

1. Read `AGENTS.md`.
2. Read `SPEC.md`.
3. Pick the relevant runtime card or create a new one under
   `runtime/<direction>/card/<experiment>/`.

Do not use chat history as durable state.

## Notebook-First Rule

Single-factor, multi-factor, model-selection, and diagnostic work should start
in an executable notebook. The notebook should hold the run-local logic,
metrics, artifact links, and final conclusion.

Framework code should change only when a reusable interface gap is proven by a
failing test. Ordinary experiments should import `soxl_backtest.framework` and
write sidecars beside the notebook.

## Framework Boundaries

- Use `soxl_backtest.framework` for qlib-calendar-backed data loading, experiment
  specs, runners, and card artifact writing.
- Use `soxl_backtest.reference` for the SOXLBacktest reference policy.
- Keep experiment-specific features, parameters, and comparisons in notebooks.
- Do not recreate production, dashboard, scheduler, or broker-execution code.

## Verification

Use the shared Python environment from the parent checkout when this repo is in a
worktree:

```bash
PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests/test_soxlbacktest_framework.py tests/test_soxlbacktest_architecture.py -q
PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m nbconvert --execute runtime/model_selection/card/soxlbacktest_qlib_reference_v1/soxlbacktest_qlib_reference_v1.ipynb --to notebook --inplace
```

Every file-changing task should finish with focused verification, commit, and
push unless network or permission failures prevent it.
