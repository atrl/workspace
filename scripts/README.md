# Scripts

Primary harness entrypoints:

- `build_backtest_dashboard.py`: rebuild the local static dashboard under `evidence/generalization/dashboard.html`.
- `check_harness.py`: repo-owned validation entrypoint that replaces the missing `pytest` dependency path.
- `checkpoint_review.py`: render a durable Markdown review from a saved summary JSON's selection history and gates.
- `execution_overlay_generalization.py`: evaluate execution overlay variants against the retained balanced actor.
- `generalization_backtest.py`: run the retained cross-ETF generalization backtest harness.
- `feature_analysis.py`: generate feature importance / IC analysis outputs under `evidence/feature_analysis/`.
- `cost_sensitivity_backtest.py`: replay retained artifacts under alternative cost assumptions for review.
- `export_profile_daily_backtest.py`: export daily replay rows from saved model artifacts for review diffs.
- `memory_search.py`: query normalized memory metadata without opening every memory file.
- `normalize_memory.py`: normalize `memory/*.md` into the shared front-matter schema and rebuild `memory/index.json`.
- `prune_artifacts.py`: generate and optionally apply the retained artifact manifest.
- `rank_counterfactual_variants.py`: sort fixed-actor counterfactual outputs by promotion-gate quality and write a Markdown ranking.

Operational utilities:

- `create_linear_soxl_issue.py`: issue/bootstrap helper for repo workflow.
- `fetch_3x_etf_data.py`: refresh external ETF source CSV inputs.
- `position_cap_counterfactual.py`: fixed-actor counterfactual sweep for cap-routing review.

Validation order before starting a new training loop:

1. `qlib_venv/bin/python scripts/memory_search.py balanced --status dropped --top-k 5`
2. `qlib_venv/bin/python scripts/check_harness.py`
3. only then run a new training or replay command
